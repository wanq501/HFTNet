# DIFI: dual-frequency intra-scale feature interaction (HFTNet, Section 3.3), implemented by AIFI_DF.
# The RT-DETR post-normalization encoder layer is kept. Its multi-head self-attention carries the low-frequency
# context, and a parallel 2x2 window-attention branch scaled by a zero-initialized channel-wise gate carries the
# high-frequency detail (Eqs. 6-7).
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.transformer import TransformerEncoderLayer, AIFI

__all__ = ['WindowAttention', 'AIFI_DF']


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
