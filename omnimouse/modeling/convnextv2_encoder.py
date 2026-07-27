"""
ConvNeXtV2 video encoder — a drop-in alternative to the Hiera backbone.

Design goals (so it plugs into the existing `HieraFeatureExtractor` unchanged):
- Keep the SAME Conv3d patch-embed geometry as Hiera (kernel (6,7,7), stride (2,2,2),
  padding (2,1,3)). This is the ONLY temporal mixing, so each output token keeps a
  fixed 6-frame (0.2s) temporal receptive field — the contract the fusion stack relies on.
- ConvNeXtV2 blocks operate PER-FRAME in 2D (depthwise conv + GRN), so they never mix
  across time. This isolates the comparison to "local spatial attention (Hiera) vs
  local spatial convolution (ConvNeXtV2)" on top of an identical temporal tokenizer.
- Expose the same backbone attributes Hiera does (patch_embed.proj, patch_stride,
  q_pool, q_stride, blocks[-1].dim_out, tokens_spatial_shape) and return (B,T',H',W',C).

For input (1, 60, 36, 64) with the OmniMouse config this yields (30, 8, 16) = 3840 tokens,
matching Hiera exactly.
"""
import math
from typing import Tuple

import torch
import torch.nn as nn


def _spatial_shape(input_size, stride, kernel, pad):
    return [math.floor((i + 2 * p - k) / s + 1) for i, s, k, p in zip(input_size, stride, kernel, pad)]


def _drop_path(x, drop_prob: float, training: bool):
    if drop_prob == 0.0 or not training:
        return x
    keep = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep) * mask.floor_()


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        return _drop_path(x, self.p, self.training)


class GRN(nn.Module):
    """Global Response Normalization — the defining addition of ConvNeXt *V2*.
    Operates on channels-last (B, H, W, C)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)          # per-channel spatial L2
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)       # normalize across channels
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """One ConvNeXtV2 block, 2D (per-frame): depthwise 7x7 -> LN -> 1x1 expand -> GELU
    -> GRN -> 1x1 contract, with a residual + stochastic depth."""

    def __init__(self, dim, drop_path=0.0, mlp_ratio=4.0, norm_eps=1e-6):
        super().__init__()
        self.dim_out = dim  # so HieraFeatureExtractor can read blocks[-1].dim_out
        hidden = int(mlp_ratio * dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=norm_eps)
        self.pwconv1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.grn = GRN(hidden)
        self.pwconv2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):  # x: (B, C, H, W)
        inp = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)            # -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)            # -> (B, C, H, W)
        return inp + self.drop_path(x)


class _Downsample(nn.Module):
    """Spatial 2x2 downsample between stages (LN + stride-2 conv), doubling channels."""

    def __init__(self, dim_in, dim_out, stride, norm_eps=1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim_in, eps=norm_eps)  # LayerNorm over channels (channels-first)
        self.reduction = nn.Conv2d(dim_in, dim_out, kernel_size=stride, stride=stride)

    def forward(self, x):
        return self.reduction(self.norm(x))


class _PatchEmbed3D(nn.Module):
    """Conv3d patch embedding — IDENTICAL geometry to Hiera's, so the temporal window is preserved.
    Exposes `.proj` (the Conv3d) so downstream code can read kernel/stride/padding/in_channels."""

    def __init__(self, in_chans, embed_dim, kernel, stride, padding):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel, stride=stride, padding=padding)

    def forward(self, x):  # x: (B, C, T, H, W)
        return self.proj(x)


class ConvNeXtV2(nn.Module):
    """ConvNeXtV2 video backbone with a Hiera-compatible attribute interface.

    Accepts (and ignores) Hiera-only kwargs (num_heads, mask_unit_size, sep_pos_embed, ...)
    so it can be constructed through the same code path.
    """

    def __init__(
        self,
        input_size: Tuple[int, ...] = (60, 36, 64),
        in_chans: int = 1,
        embed_dim: int = 128,
        stages: Tuple[int, ...] = (2, 8),
        q_pool: int = 1,
        q_stride: Tuple[int, ...] = (1, 2, 2),
        patch_kernel: Tuple[int, ...] = (6, 7, 7),
        patch_stride: Tuple[int, ...] = (2, 2, 2),
        patch_padding: Tuple[int, ...] = (2, 1, 3),
        dim_mul: float = 2.0,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        norm_eps: float = 1e-6,
        **unused,  # swallow Hiera-only kwargs
    ):
        super().__init__()
        assert len(input_size) == 3, "ConvNeXtV2 video encoder expects (T,H,W) input_size"
        self.patch_stride = tuple(patch_stride)
        self.q_stride = tuple(q_stride)
        self.q_pool = q_pool

        # token grid right after the Conv3d patch embed
        self.tokens_spatial_shape = _spatial_shape(input_size, patch_stride, patch_kernel, patch_padding)

        self.patch_embed = _PatchEmbed3D(in_chans, embed_dim, patch_kernel, patch_stride, patch_padding)

        depth = sum(stages)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # spatial 2x2 downsample stride (drop the temporal dim of q_stride)
        spatial_stride = q_stride[-1] if len(q_stride) == 1 else q_stride[-2]  # 2

        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        dim = embed_dim
        i = 0
        for s, n_blocks in enumerate(stages):
            # downsample before every stage except the first (up to q_pool times)
            if s > 0 and s <= q_pool:
                dim_out = int(dim * dim_mul)
                self.downsamples.append(_Downsample(dim, dim_out, spatial_stride, norm_eps))
                dim = dim_out
            else:
                self.downsamples.append(None)
            for _ in range(n_blocks):
                self.blocks.append(ConvNeXtV2Block(dim, drop_path=dpr[i], mlp_ratio=mlp_ratio, norm_eps=norm_eps))
                i += 1
        self._stage_sizes = stages
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):  # x: (B, C, T, H, W)
        x = self.patch_embed(x)                       # (B, embed_dim, T', H0, W0)
        B, C, T, H, W = x.shape
        # merge batch & time -> process each frame independently in 2D
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        blk = 0
        for s, n_blocks in enumerate(self._stage_sizes):
            ds = self.downsamples[s]
            if ds is not None:
                x = ds(x)
            for _ in range(n_blocks):
                x = self.blocks[blk](x)
                blk += 1

        # x: (B*T, C_out, H', W') -> (B, T', H', W', C_out)
        Cout, Hf, Wf = x.shape[1], x.shape[2], x.shape[3]
        x = x.reshape(B, T, Cout, Hf, Wf).permute(0, 1, 3, 4, 2).contiguous()
        return x


# ---------------------------------------------------------------------------
# FeatureExtractor: reuse ALL of HieraFeatureExtractor's timestamp/window logic;
# only the backbone construction differs. Imported lazily to keep this file
# standalone-testable (no heavy package imports at module load).
# ---------------------------------------------------------------------------
def build_convnextv2_feature_extractor(*args, **kwargs):
    from omnimouse.modeling.model import HieraFeatureExtractor  # local import avoids cycle

    class ConvNeXtV2FeatureExtractor(HieraFeatureExtractor):
        def _make_backbone(self, *a, **kw):
            return ConvNeXtV2(*a, **kw)

    return ConvNeXtV2FeatureExtractor(*args, **kwargs)
