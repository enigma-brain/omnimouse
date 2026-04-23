# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles
#
# Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan,
# Po-Yao Huang, Vaibhav Aggarwal, Arkabandhu Chowdhury, Omid Poursaeed,
# Judy Hoffman, Jitendra Malik, Yanghao Li, Christoph Feichtenhofer.
#
# Paper: https://arxiv.org/abs/2306.00989/
#
# References:
# slowfast: https://github.com/facebookresearch/SlowFast
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------
#
# Modded version of Meta's Hiera implementation
# adds: flexible unroll, reroll, FlashAttention based MaskUnitAttention
#
# The enigma project, April 2025
# --------------------------------------------------------

import math
from functools import partial, cached_property
from typing import List, Tuple, Callable, Optional, Union
from unittest.mock import inplace

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Type, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm

from .nn import FeedForward, DropPath, Mlp, scaled_dot_product_attention


# optimized llama rms norm implementation
# https://github.com/meta-llama/llama-models/blob/01dc8ce46fecf06b639598f715efbb4ab981fb4c/models/llama4/model.py#L27
def rmsnorm(x, eps):
    def _norm(y):
        return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)

    return _norm(x.float()).type_as(x)


class MaskUnitAttention(nn.Module):
    """
    Computes either Mask Unit or Global Attention. Also is able to perform q pooling.

    Note: this assumes the tokens have already been flattened and unrolled into mask units.
    See `Unroll` for more details.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
        norm_eps: float = 1e-5,
    ):
        """
        Args:
        - dim, dim_out: The input and output feature dimensions.
        - heads: The number of attention heads.
        - q_stride: If greater than 1, pool q with this stride. The stride should be flattened (e.g., 2x2 = 4).
        - window_size: The current (flattened) size of a mask unit *after* pooling (if any).
        - use_mask_unit_attn: Use Mask Unit or Global Attention.
        """
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.heads = heads
        self.q_stride = q_stride
        self.norm_eps = norm_eps

        self.head_dim = dim_out // heads
        self.scale = (self.head_dim) ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Input should be of shape [Batch, Sequence, Dim]. """
        B, N, _ = x.shape  # [B, S, D]
        num_windows = (N // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1

        qkv = self.qkv(x).reshape(B * num_windows, -1, 3, self.heads, self.head_dim)
        q, k, v = torch.unbind(qkv, dim=2)  # [B*num_windows, S, H, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        q_processed = q
        if self.q_stride > 1:
            q_reshaped = q.reshape(B * num_windows, self.heads, -1, self.q_stride, self.head_dim)
            q_processed = q_reshaped.max(dim=3).values

        # Ensure contiguous regardless of whether pooling happened
        q_processed = q_processed.contiguous()

        # apply QK norm
        q_processed = rmsnorm(q_processed, self.norm_eps)
        k = rmsnorm(k, self.norm_eps)

        x = scaled_dot_product_attention(q_processed, k, v)
        x = x.view(B, num_windows, self.heads, -1, self.head_dim)
        x = x.permute(0, 3, 1, 2, 4).reshape(B, -1, self.dim_out).contiguous()
        x = self.proj(x)
        return x


class HieraBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
        use_gated_ffn: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.pre_norm1 = norm_layer(dim, eps=norm_eps)
        self.pre_norm2 = norm_layer(dim_out, eps=norm_eps)
        self.attn = MaskUnitAttention(
            dim=dim,
            dim_out=dim_out,
            heads=heads,
            q_stride=q_stride,
            window_size=window_size,
            use_mask_unit_attn=use_mask_unit_attn,
            norm_eps=norm_eps,
        )
        if use_gated_ffn:
            self.mlp = FeedForward(dim=dim_out, mult=mlp_ratio, multiple_of=128)
        else:
            self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio), act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.pre_norm1(x)
        if self.dim != self.dim_out:
            x = do_pool(self.proj(x_norm), stride=self.attn.q_stride)
        x = x + self.drop_path(self.attn(x_norm)) # Attention
        x = x + self.drop_path(self.mlp(self.pre_norm2(x))) # FeedForward
        return x


class PatchEmbed(nn.Module):
    """Patch embed that supports any number of spatial dimensions (1d, 2d, 3d)."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
    ):
        super().__init__()

        # Support any number of spatial dimensions
        self.spatial_dims = len(kernel)
        self.proj = conv_nd(self.spatial_dims)(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = do_masked_conv(x, self.proj, mask)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 1)
        return x


class Hiera(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, ...] = (224, 224),
        in_chans: int = 3,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        num_classes: int = 1000,
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, ...] = (2, 2),
        mask_unit_size: Tuple[int, ...] = (8, 8),  # must divide q_stride ** (#stages-1)
        # mask_unit_attn: which stages use mask unit attention?
        mask_unit_attn: Tuple[bool, ...] = (True, True, False, False),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        patch_kernel: Tuple[int, ...] = (7, 7),
        patch_stride: Tuple[int, ...] = (4, 4),
        patch_padding: Tuple[int, ...] = (3, 3),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.RMSNorm,
        sep_pos_embed: bool = True,
        use_gated_ffn: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        assert len(input_size) == len(patch_stride) == len(patch_kernel) == len(patch_padding), \
            "All spatial parameters must have the same dimensionality"

        depth = sum(stages)
        self.patch_stride = patch_stride
        self.tokens_spatial_shape = calculate_spatial_shape(
            input_size,
            patch_stride,
            patch_kernel,
            patch_padding
        )

        num_tokens = math.prod(self.tokens_spatial_shape)
        flat_mu_size = math.prod(mask_unit_size)
        flat_q_stride = math.prod(q_stride)

        assert q_pool < len(stages)
        self.q_pool, self.q_stride = q_pool, q_stride
        self.mu_size, self.mask_unit_size = flat_mu_size, mask_unit_size
        self.mask_spatial_shape = [
            i // s for i, s in zip(self.tokens_spatial_shape, self.mask_unit_size)
        ]
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]

        self.patch_embed = PatchEmbed(
            in_chans, embed_dim, patch_kernel, patch_stride, patch_padding
        )

        self.sep_pos_embed = sep_pos_embed
        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1,
                    self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                    embed_dim,
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.tokens_spatial_shape[0], embed_dim)
            )
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        # Setup roll and reroll modules
        self.unroll = Unroll(
            input_size=input_size,
            patch_stride=patch_stride,
            patch_kernel=patch_kernel,
            patch_padding=patch_padding,
            unroll_schedule=[q_stride] * len(self.stage_ends[:-1])
        )
        self.reroll = Reroll(
            input_size=input_size,
            patch_stride=patch_stride,
            patch_kernel=patch_kernel,
            patch_padding=patch_padding,
            unroll_schedule=[q_stride] * len(self.stage_ends[:-1]),
            stage_ends=self.stage_ends,
            q_pool=q_pool,
        )
        # q_pool locations
        q_pool_blocks = [x + 1 for x in self.stage_ends[:q_pool]]
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        cur_stage = 0
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # Mask unit or global attention.
            # Lag by 1 block, so that global attention,
            # applied post pooling on lower resolution
            use_mask_unit_attn = mask_unit_attn[cur_stage]

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
                if i in q_pool_blocks:
                    flat_mu_size //= flat_q_stride

            block = HieraBlock(
                dim=embed_dim,
                dim_out=dim_out,
                heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                q_stride=(flat_q_stride if i in q_pool_blocks else 1),
                window_size=flat_mu_size,
                use_mask_unit_attn=use_mask_unit_attn,
                use_gated_ffn=use_gated_ffn,
                norm_eps=norm_eps,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        # Initialize everything
        if sep_pos_embed:
            nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        else:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(partial(self._init_weights))

    def _init_weights(self, m, init_bias=0.02):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, init_bias)
        elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, init_bias)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.sep_pos_embed:
            return ["pos_embed_spatial", "pos_embed_temporal"]
        else:
            return ["pos_embed"]

    def get_random_mask(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """
        Generates a random mask, mask_ratio fraction are dropped.
        1 is *keep*, 0 is *remove*. Useful for MAE, FLIP, etc.
        """
        B = x.shape[0]
        # Tokens selected for masking at mask unit level
        num_windows = math.prod(self.mask_spatial_shape)  # num_mask_units
        len_keep = int(num_windows * (1 - mask_ratio))
        noise = torch.rand(B, num_windows, device=x.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the binary mask: 1 is *keep*, 0 is *remove*
        # Note this is opposite to original MAE
        mask = torch.zeros([B, num_windows], device=x.device)
        mask[:, :len_keep] = 1
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.bool()

    def get_pos_embed(self) -> torch.Tensor:
        if self.sep_pos_embed:
            return self.pos_embed_spatial.repeat(
                1, self.tokens_spatial_shape[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                dim=1,
            )
        else:
            return self.pos_embed

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_intermediates: bool = False, # unused, but kept for compatibility
    ) -> torch.Tensor:
        """
        mask should be a boolean tensor of shape [B, #MUt*#MUy*#MUx] where #MU are the number of mask units in that dim.
        Note: 1 in mask is *keep*, 0 is *remove*; mask.sum(dim=-1) should be the same across the batch.
        """
        x = self.patch_embed(
            x,
            mask=mask.view(
                x.shape[0], 1, *self.mask_spatial_shape
            )  # B, C, *mask_spatial_shape
            if mask is not None
            else None,
        )
        x = x + self.get_pos_embed()
        x = self.unroll(x)

        # Discard masked tokens
        if mask is not None:
            x = x[mask[..., None].tile(1, self.mu_size, x.shape[2])].view(
                x.shape[0], -1, x.shape[-1]
            )

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        return self.reroll(x, i, mask=mask)


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles
#
# Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan,
# Po-Yao Huang, Vaibhav Aggarwal, Arkabandhu Chowdhury, Omid Poursaeed,
# Judy Hoffman, Jitendra Malik, Yanghao Li, Christoph Feichtenhofer.
#
# Paper: https://arxiv.org/abs/2306.00989/
#
# References:
# slowfast: https://github.com/facebookresearch/SlowFast
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------
# Modded version of Meta's Hiera utils
#
# The enigma project, April 2025
# --------------------------------------------------------

def calculate_spatial_shape(input_size, patch_stride, kernel_size, padding):
    return [math.floor((i + 2*p - k) / s + 1)
            for i, s, k, p in zip(input_size, patch_stride, kernel_size, padding)]


def pretrained_model(checkpoints: Dict[str, str], default: str = None) -> Callable:
    """ Loads a Hiera model from a pretrained source (if pretrained=True). Use "checkpoint" to specify the checkpoint. """

    def inner(model_func: Callable) -> Callable:
        def model_def(pretrained: bool = False, checkpoint: str = default, strict: bool = True, **kwdargs) -> nn.Module:
            if pretrained:
                if checkpoints is None:
                    raise RuntimeError("This model currently doesn't have pretrained weights available.")
                elif checkpoint is None:
                    raise RuntimeError("No checkpoint specified.")
                elif checkpoint not in checkpoints:
                    raise RuntimeError(
                        f"Invalid checkpoint specified ({checkpoint}). Options are: {list(checkpoints.keys())}.")

                state_dict = torch.hub.load_state_dict_from_url(checkpoints[checkpoint], map_location="cpu")

                if "head.projection.weight" in state_dict["model_state"]:
                    # Set the number of classes equal to the state_dict only if the user doesn't want to overwrite it
                    if "num_classes" not in kwdargs:
                        kwdargs["num_classes"] = state_dict["model_state"]["head.projection.weight"].shape[0]
                    # If the user specified a different number of classes, remove the projection weights or else we'll error out
                    elif kwdargs["num_classes"] != state_dict["model_state"]["head.projection.weight"].shape[0]:
                        del state_dict["model_state"]["head.projection.weight"]
                        del state_dict["model_state"]["head.projection.bias"]

            model = model_func(**kwdargs)
            if pretrained:
                # Disable being strict when trying to load a encoder-decoder model into an encoder-only model
                if "decoder_pos_embed" in state_dict["model_state"] and not hasattr(model, "decoder_pos_embed"):
                    strict = False

                model.load_state_dict(state_dict["model_state"], strict=strict)

            return model

        # Keep some metadata so we can do things that require looping through all available models
        model_def.checkpoints = checkpoints
        model_def.default = default

        return model_def

    return inner


def conv_nd(n: int) -> Type[nn.Module]:
    """
    Returns a conv with nd (e.g., Conv2d for n=2). Work up to n=3.
    If you wanted a 4d Hiera, you could probably just implement this for n=4. (no promises)
    """
    return [nn.Identity, nn.Conv1d, nn.Conv2d, nn.Conv3d][n]


def do_pool(x: torch.Tensor, stride: int) -> torch.Tensor:
    # Refer to `Unroll` to see how this performs a maxpool-Nd
    return x.view(x.shape[0], stride, -1, x.shape[-1]).max(dim=1).values


def get_resized_mask(target_size: torch.Size, mask: torch.Tensor) -> torch.Tensor:
    # target_size: [(T), (H), W]
    # (spatial) mask: [B, C, (t), (h), w]
    if mask is None:
        return mask

    assert len(mask.shape[2:]) == len(target_size)
    if mask.shape[2:] != target_size:
        return F.interpolate(mask.float(), size=target_size)
    return mask


def do_masked_conv(
        x: torch.Tensor, conv: nn.Module, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Zero-out the masked regions of the input before conv.
    Prevents leakage of masked regions when using overlapping kernels.
    """
    if conv is None:
        return x
    if mask is None:
        return conv(x)

    mask = get_resized_mask(target_size=x.shape[2:], mask=mask)
    return conv(x * mask.bool())


def undo_windowing(
        x: torch.Tensor, shape: List[int], mu_shape: List[int]
) -> torch.Tensor:
    """
    Restore spatial organization by undoing windowed organization of mask units.

    Args:
        x: organized by mask units windows, e.g. in 2d [B, #MUy*#MUx, MUy, MUx, C]
        shape: current spatial shape, if it were not organized into mask unit
            windows, e.g. in 2d [B, #MUy*MUy, #MUx*MUx, C].
        mu_shape: current mask unit shape, e.g. in 2d [MUy, MUx]
    Returns:
        x: e.g. in 2d, [B, #MUy*MUy, #MUx*MUx, C]
    """
    D = len(shape)
    B, C = x.shape[0], x.shape[-1]
    # [B, #MUy*#MUx, MUy, MUx, C] -> [B, #MUy, #MUx, MUy, MUx, C]
    num_MUs = [s // mu for s, mu in zip(shape, mu_shape)]
    x = x.view(B, *num_MUs, *mu_shape, C)

    # [B, #MUy, #MUx, MUy, MUx, C] -> [B, #MUy*MUy, #MUx*MUx, C]
    permute = (
            [0]
            + sum(
        [list(p) for p in zip(range(1, 1 + D), range(1 + D, 1 + 2 * D))],
        [],
    )
            + [len(x.shape) - 1]
    )
    x = x.permute(permute).reshape(B, *shape, C)

    return x


class Unroll(nn.Module):
    """
    Reorders the tokens such that patches are contiguous in memory.
    E.g., given [B, (H, W), C] and stride of (Sy, Sx), this will re-order the tokens as
                           [B, (Sy, Sx, H // Sy, W // Sx), C]

    This allows operations like Max2d to be computed as x.view(B, Sx*Sy, -1, C).max(dim=1).
    Not only is this faster, but it also makes it easy to support inputs of arbitrary
    dimensions in addition to patch-wise sparsity.

    Performing this operation multiple times in sequence puts entire windows as contiguous
    in memory. For instance, if you applied the stride (2, 2) 3 times, entire windows of
    size 8x8 would be contiguous in memory, allowing operations like mask unit attention
    computed easily and efficiently, while also allowing max to be applied sequentially.

    Note: This means that intermediate values of the model are not in HxW order, so they
    need to be re-rolled if you want to use the intermediate values as a HxW feature map.
    The last block of the network is fine though, since by then the strides are all consumed.
    """

    def __init__(
            self,
            input_size: Tuple[int, ...],
            patch_stride: Tuple[int, ...],
            patch_kernel: Tuple[int, ...],
            patch_padding: Tuple[int, ...],
            unroll_schedule: List[Tuple[int, ...]],
    ):
        super().__init__()
        self.size = calculate_spatial_shape(input_size, patch_stride, patch_kernel, patch_padding)
        self.schedule = unroll_schedule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: Flattened patch embeddings [B, N, C]
        Output: Patch embeddings [B, N, C] permuted such that [B, 4, N//4, C].max(1) etc. performs MaxPoolNd
        """
        B, _, C = x.shape

        cur_size = self.size
        x = x.view(*([B] + cur_size + [C]))

        for strides in self.schedule:
            # Move patches with the given strides to the batch dimension

            # Create a view of the tensor with the patch stride as separate dims
            # For example in 2d: [B, H // Sy, Sy, W // Sx, Sx, C]
            cur_size = [i // s for i, s in zip(cur_size, strides)]
            new_shape = [B] + sum([[i, s] for i, s in zip(cur_size, strides)], []) + [C]
            x = x.view(new_shape)

            # Move the patch stride into the batch dimension
            # For example in 2d: [B, Sy, Sx, H // Sy, W // Sx, C]
            L = len(new_shape)
            permute = (
                    [0] + list(range(2, L - 1, 2)) + list(range(1, L - 1, 2)) + [L - 1]
            )
            x = x.permute(permute)

            # Now finally flatten the relevant dims into the batch dimension
            x = x.flatten(0, len(strides))
            B *= math.prod(strides)

        x = x.reshape(-1, math.prod(self.size), C)
        return x


class Reroll(nn.Module):
    """
    Undos the "unroll" operation so that you can use intermediate features.
    """

    def __init__(
            self,
            input_size: Tuple[int, ...],
            patch_stride: Tuple[int, ...],
            patch_kernel: Tuple[int, ...],
            patch_padding: Tuple[int, ...],
            unroll_schedule: List[Tuple[int, ...]],
            stage_ends: List[int],
            q_pool: int,
    ):
        super().__init__()
        self.size = calculate_spatial_shape(input_size, patch_stride, patch_kernel, patch_padding)

        # The first stage has to reverse everything
        # The next stage has to reverse all but the first unroll, etc.
        self.schedule = {}
        size = self.size
        for i in range(stage_ends[-1] + 1):
            self.schedule[i] = unroll_schedule, size
            # schedule unchanged if no pooling at a stage end
            if i in stage_ends[:q_pool]:
                if len(unroll_schedule) > 0:
                    size = [n // s for n, s in zip(size, unroll_schedule[0])]
                unroll_schedule = unroll_schedule[1:]

    def forward(
            self, x: torch.Tensor, block_idx: int, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Roll the given tensor back up to spatial order assuming it's from the given block.

        If no mask is provided:
            - Returns [B, H, W, C] for 2d, [B, T, H, W, C] for 3d, etc.
        If a mask is provided:
            - Returns [B, #MUs, MUy, MUx, C] for 2d, etc.
        """
        schedule, size = self.schedule[block_idx]
        B, N, C = x.shape

        D = len(size)
        cur_mu_shape = [1] * D

        for strides in schedule:
            # Extract the current patch from N
            x = x.view(B, *strides, N // math.prod(strides), *cur_mu_shape, C)

            # Move that patch into the current MU
            # Example in 2d: [B, Sy, Sx, N//(Sy*Sx), MUy, MUx, C] -> [B, N//(Sy*Sx), Sy, MUy, Sx, MUx, C]
            L = len(x.shape)
            permute = (
                    [0, 1 + D]
                    + sum(
                [list(p) for p in zip(range(1, 1 + D), range(1 + D + 1, L - 1))],
                [],
            )
                    + [L - 1]
            )
            x = x.permute(permute)

            # Reshape to [B, N//(Sy*Sx), *MU, C]
            for i in range(D):
                cur_mu_shape[i] *= strides[i]
            x = x.reshape(B, -1, *cur_mu_shape, C)
            N = x.shape[1]

        # Current shape (e.g., 2d: [B, #MUy*#MUx, MUy, MUx, C])
        x = x.view(B, N, *cur_mu_shape, C)

        # If masked, return [B, #MUs, MUy, MUx, C]
        if mask is not None:
            return x

        # If not masked, we can return [B, H, W, C]
        x = undo_windowing(x, size, cur_mu_shape)

        return x


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor / keep_prob
    return x * random_tensor


# TODO: test and benchmark DropPath with out inplace operation
#class DropPath(nn.Module):
#    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#    """
#    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
#        super(DropPath, self).__init__()
#        self.drop_prob = drop_prob
#        self.scale_by_keep = scale_by_keep
#
#    def forward(self, x):
#        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
#
#    def extra_repr(self):
#        return f'drop_prob={round(self.drop_prob,3):0.3f}'