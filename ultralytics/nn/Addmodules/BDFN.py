# BDFN aggregation (HFTNet, Section 3.4.1): DFAL2, DFBN and a pure-PyTorch DCNv3 (Eqs. 10-12).
#   Agg(X)  = DFBN(W2 Concat(Za, Zb, Zc, Zd)), [Za, Zb] = Split(W1 X), Zc = W3 CSP(Zb), Zd = W4 CSP(Zc)
#   DFBN(Z) = Z + W8 DCN(Z), with the batch-normalization scale of W8 initialized to zero
# The split, the two CSP stages and the concatenation are the RepNCSPELAN4 block of YOLOv9.
# The pure-PyTorch DCNv3 follows InternImage (Wang et al., CVPR 2023), https://github.com/OpenGVLab/InternImage.
#   DFBN(Z) = Z + W6 DCN(Z)
# DCN is DCNv3 (Wang et al., 2023): grouped sampling with learned offsets and softmax-normalized
# modulation scalars, implemented with grid_sample so that no CUDA extension is required.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.Addmodules.RepNCSPELAN4 import RepNCSPELAN4

__all__ = ['DCNv3', 'DFBN', 'DFAL2']


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(c, eps=eps)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DCNv3(nn.Module):
    """DCNv3 core operator, input and output (B, C, H, W), resolution preserved."""

    def __init__(self, channels, kernel_size=3, dilation=1, group=4, offset_scale=1.0):
        super().__init__()
        assert channels % group == 0, 'channels must be divisible by group'
        self.k, self.d, self.g, self.gc = kernel_size, dilation, group, channels // group
        self.offset_scale = offset_scale
        pad = dilation * (kernel_size - 1) // 2
        self.dw_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 1, pad, dilation=dilation, groups=channels),
            LayerNorm2d(channels), nn.GELU())
        self.offset = nn.Conv2d(channels, group * kernel_size * kernel_size * 2, 1)
        self.mask = nn.Conv2d(channels, group * kernel_size * kernel_size, 1)
        self.input_proj = nn.Conv2d(channels, channels, 1)
        self.output_proj = nn.Conv2d(channels, channels, 1)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.zeros_(self.offset.weight), nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.mask.weight), nn.init.zeros_(self.mask.bias)
        nn.init.xavier_uniform_(self.input_proj.weight), nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight), nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.dw_conv(x)
        offset = self.offset(feat) * self.offset_scale                               # B, G*K*2, H, W
        mask = self.mask(feat).view(B, self.g, self.k * self.k, H, W).softmax(dim=2)  # B, G, K, H, W
        out = self._sample(self.input_proj(x), offset, mask)
        return self.output_proj(out)

    def _sample(self, x, offset, mask):
        B, C, H, W = x.shape
        K, G, dt, dev = self.k * self.k, self.g, x.dtype, x.device
        ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=dt), torch.arange(W, device=dev, dtype=dt), indexing='ij')
        ref = torch.stack((xs, ys), dim=-1)                                            # H, W, 2 (x, y)
        r = (self.k - 1) // 2
        ky, kx = torch.meshgrid(torch.arange(-r, r + 1, device=dev, dtype=dt) * self.d,
                                torch.arange(-r, r + 1, device=dev, dtype=dt) * self.d, indexing='ij')
        grid = torch.stack((kx, ky), dim=-1).reshape(K, 2)                            # K, 2 (dx, dy)
        offset = offset.view(B, G, K, 2, H, W).permute(0, 1, 4, 5, 2, 3)              # B, G, H, W, K, 2
        pos = ref[None, None, :, :, None, :] + grid[None, None, None, None, :, :] + offset
        norm = torch.tensor([max(W - 1, 1), max(H - 1, 1)], device=dev, dtype=dt)
        pos = 2.0 * pos / norm - 1.0
        sampled = F.grid_sample(x.reshape(B * G, self.gc, H, W), pos.reshape(B * G, H * W, K, 2),
                                mode='bilinear', padding_mode='zeros', align_corners=True)  # B*G, gc, H*W, K
        m = mask.permute(0, 1, 3, 4, 2).reshape(B * G, 1, H * W, K)
        return (sampled * m).sum(dim=-1).reshape(B, C, H, W)


class DFBN(nn.Module):
    """Deformable feature bottleneck: DFBN(Z) = Z + W6 DCN(Z)."""

    def __init__(self, c, group=4, k=1, zero_init=False):
        super().__init__()
        self.dcn = DCNv3(c, kernel_size=3, group=group)
        self.w6 = Conv(c, c, k)
        if zero_init:  # the branch outputs zero at initialization, so DFBN starts as the identity
            nn.init.zeros_(self.w6.bn.weight)

    def forward(self, z):
        return z + self.w6(self.dcn(z))


class DFAL2(nn.Module):
    """Aggregation node of BDFN, the Agg operator of Eq. 10: the RepNCSPELAN4 aggregation of YOLOv9 (split, two
    cascaded CSP stages, concatenation) followed by a zero-initialized deformable bottleneck DFBN(Z) = Z + W8 DCN(Z).
    At initialization the node equals the baseline aggregation; the deformable refinement is learned on top."""

    def __init__(self, c1, c2, n=3, g=8):
        super().__init__()
        self.agg = RepNCSPELAN4(c1, c2, n)
        self.dfbn = DFBN(c2, group=g, k=1, zero_init=True)

    def forward(self, x):
        if self.training and x.requires_grad and os.environ.get('HFTNET_CKPT') == '1':
            return cp.checkpoint(self._ckpt_forward, x)      # activation checkpointing: same math, less memory
        return self.dfbn(self.agg(x))

    def _ckpt_forward(self, x):
        # Reentrant checkpoint: the first pass runs under no_grad (BatchNorm statistics are updated once, as usual);
        # the recomputation inside backward runs with grad enabled, so freeze the running statistics for that pass.
        if torch.is_grad_enabled():
            bns = [m for m in self.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
            saved = [m.momentum for m in bns]
            for m in bns:
                m.momentum = 0.0
            try:
                return self.dfbn(self.agg(x))
            finally:
                for m, mo in zip(bns, saved):
                    m.momentum = mo
        return self.dfbn(self.agg(x))
