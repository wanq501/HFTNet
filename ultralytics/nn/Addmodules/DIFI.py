# DIFI: dual-frequency intra-scale feature interaction (HFTNet, Section 3.3), implemented by AIFI_DF.
# The RT-DETR post-normalization encoder layer is kept. Its multi-head self-attention carries the low-frequency
# context, and a parallel 2x2 window-attention branch scaled by a zero-initialized channel-wise gate carries the
# high-frequency detail (Eqs. 6-7). HiLo and AIFI_HiLo are kept for comparison with the earlier design.
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.transformer import TransformerEncoderLayer, AIFI

__all__ = ['HiLo', 'AIFI_HiLo', 'WindowAttention', 'AIFI_DF']


class HiLo(nn.Module):
    """HiLo attention of LITv2 (Pan et al., NeurIPS 2022, https://github.com/ziplab/LITv2).
    Input tokens (B, N, C) with N = H * W; output (B, N, C)."""

    def __init__(self, dim, num_heads=8, window_size=2, alpha=0.5, lo_pool=None, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        head_dim = dim // num_heads
        self.dim, self.ws = dim, window_size
        self.lo_pool = window_size if lo_pool is None else int(lo_pool)   # pooling factor of the Lo-Fi keys and values
        self.l_heads = int(num_heads * alpha)          # low-frequency heads
        self.l_dim = self.l_heads * head_dim
        self.h_heads = num_heads - self.l_heads        # high-frequency heads
        self.h_dim = self.h_heads * head_dim
        self.scale = head_dim ** -0.5
        if self.l_heads > 0:
            if self.lo_pool > 1:
                self.sr = nn.AvgPool2d(kernel_size=self.lo_pool, stride=self.lo_pool)
            self.l_q = nn.Linear(dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

    def hifi(self, x):  # x: (B, H, W, C), H and W divisible by ws
        B, H, W, C = x.shape
        hg, wg = H // self.ws, W // self.ws
        x = x.reshape(B, hg, self.ws, wg, self.ws, C).transpose(2, 3)              # B, hg, wg, ws, ws, C
        qkv = self.h_qkv(x).reshape(B, hg * wg, -1, 3, self.h_heads, self.h_dim // self.h_heads)
        q, k, v = qkv.permute(3, 0, 1, 4, 2, 5)                                   # 3, B, groups, heads, ws*ws, hd
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x = (attn @ v).transpose(2, 3).reshape(B, hg, wg, self.ws, self.ws, self.h_dim)
        x = x.transpose(2, 3).reshape(B, hg * self.ws, wg * self.ws, self.h_dim)
        return self.h_proj(x)

    def lofi(self, x):  # x: (B, H, W, C)
        B, H, W, C = x.shape
        q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)
        if getattr(self, 'lo_pool', self.ws) > 1:   # checkpoints saved before lo_pool existed pool by ws
            x_ = self.sr(x.permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)   # pooled tokens
        else:
            x_ = x.reshape(B, H * W, C)
        kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        return self.l_proj(x)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.reshape(B, H, W, C)
        ph, pw = (self.ws - H % self.ws) % self.ws, (self.ws - W % self.ws) % self.ws
        if ph or pw:
            x = F.pad(x, (0, 0, 0, pw, 0, ph))
        outs = []
        if self.h_heads > 0:
            outs.append(self.hifi(x))
        if self.l_heads > 0:
            outs.append(self.lofi(x))
        x = torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]
        if ph or pw:
            x = x[:, :H, :W, :]
        return x.reshape(B, N, C)


class AIFI_HiLo(TransformerEncoderLayer):
    """DIFI encoder layer: E1 = LN(E + HiLo(E + P)), E2 = LN(E1 + FFN(E1)), M5 = Reshape(E2)."""

    def __init__(self, c1, cm=1024, num_heads=8, alpha=0.5, window_size=2, lo_pool=None, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        del self.ma
        self.hilo = HiLo(c1, num_heads, window_size, alpha, lo_pool)

    def forward(self, x):
        c, h, w = x.shape[1:]
        pos = AIFI.build_2d_sincos_position_embedding(w, h, c).to(device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1)                       # B, N, C
        if self.normalize_before:
            src2 = self.norm1(src)
            src = src + self.dropout1(self.hilo(self.with_pos_embed(src2, pos), h, w))
            src2 = self.norm2(src)
            src = src + self.dropout2(self.fc2(self.dropout(self.act(self.fc1(src2)))))
        else:
            src = self.norm1(src + self.dropout1(self.hilo(self.with_pos_embed(src, pos), h, w)))
            src = self.norm2(src + self.dropout2(self.fc2(self.dropout(self.act(self.fc1(src))))))
        return src.permute(0, 2, 1).view([-1, c, h, w]).contiguous()


class WindowAttention(nn.Module):
    """Multi-head self-attention inside non-overlapping ws x ws windows: the high-frequency (local detail) path."""

    def __init__(self, dim, num_heads=4, window_size=2, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.h, self.ws, self.hd = num_heads, window_size, dim // num_heads
        self.scale = self.hd ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):  # x: (B, N, C), N = H * W
        B, N, C = x.shape
        ws = self.ws
        x = x.reshape(B, H, W, C)
        ph, pw = (ws - H % ws) % ws, (ws - W % ws) % ws
        if ph or pw:
            x = F.pad(x, (0, 0, 0, pw, 0, ph))
        Hp, Wp = x.shape[1], x.shape[2]
        hg, wg = Hp // ws, Wp // ws
        x = x.reshape(B, hg, ws, wg, ws, C).transpose(2, 3).reshape(B, hg * wg, ws * ws, C)       # B, G, T, C
        q, k, v = self.qkv(x).reshape(B, hg * wg, ws * ws, 3, self.h, self.hd).permute(3, 0, 1, 4, 2, 5)  # B, G, h, T, hd
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x = (attn @ v).permute(0, 1, 3, 2, 4).reshape(B, hg, wg, ws, ws, C)                        # B, hg, wg, ws, ws, C
        x = x.transpose(2, 3).reshape(B, Hp, Wp, C)[:, :H, :W, :].reshape(B, N, C)
        return self.proj(x)


class AIFI_DF(TransformerEncoderLayer):
    """DIFI, dual-frequency intra-scale interaction. The full multi-head global attention of AIFI carries the
    low-frequency context, and a parallel window-attention branch on the same tokens carries the high-frequency detail.
    The window branch is scaled by a per-channel gate initialized to zero, so the layer starts exactly as AIFI and
    learns how much local detail to add:
        E1 = LN(E + MHSA(E + P) + gamma * WinAttn(E + P)),  E2 = LN(E1 + FFN(E1)).
    """

    def __init__(self, c1, cm=1024, num_heads=8, hi_heads=4, window_size=2, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.hifi = WindowAttention(c1, hi_heads, window_size)
        self.gamma = nn.Parameter(torch.zeros(c1))

    def forward(self, x):
        c, h, w = x.shape[1:]
        pos = AIFI.build_2d_sincos_position_embedding(w, h, c).to(device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1)
        q = k = self.with_pos_embed(src, pos)
        glob = self.ma(q, k, value=src)[0]
        loc = self.hifi(q, h, w)
        src = self.norm1(src + self.dropout1(glob) + self.gamma * loc)
        src = self.norm2(src + self.dropout2(self.fc2(self.dropout(self.act(self.fc1(src))))))
        return src.permute(0, 2, 1).view([-1, c, h, w]).contiguous()
