from typing import Optional, Tuple, Union, Iterable, Literal
from jaxtyping import Float
import os
from math import prod
from functools import partial
from itertools import repeat
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
)


"""
NOTE: [On ``init_weights`` vs. ``reset_parameters``] Modules may define
``reset_parameters`` to initialize parameter values. ``reset_parameters`` is
meant to only initialize directly owned parameters/buffers, not those of their
child modules, and it can be used to give the initial values for these tensors.
Separately, users may want custom initialization for their modules, different
from that in ``reset_parameters``. For this, we define ``init_weights``.
"""

# Define type constants for supported norms and activations and layout modes
SupportedNorm = Literal['layer', 'layer_unbiased', 'rms', 'softplus']
SupportedActivation = Literal['silu', 'gelu']
SupportedLocalGlobalLayout = Literal['interleaved', 'local_first']
SupportedDepthInit = Literal['current', 'global']

# TODO: Compiling flex attention errors out on L40 GPUs. Seems to be related to
#  `this issue<https://github.com/pytorch/pytorch/issues/133254__>`, and possibly fixed
#  by adding default device config for the L40 GPU (see `here <https://github.com/pytorch/pytorch/blob/47c8aa80905f8cc9ea5488b5a3a209bee62d8409/torch/_inductor/kernel/flex_attention.py#L595>__`).


def scaled_dot_product_attention(q, k, v, *args, **kwargs):
    all_backends = [torch.nn.attention.SDPBackend(i) for i in range(5)]
    with torch.nn.attention.sdpa_kernel(all_backends):
        return F.scaled_dot_product_attention(q, k, v, *args, **kwargs)

# Set up flex attention
if os.environ.get("FLEX_ATTN_UNSUPPORTED") == "1":
    # Use fallback attention
    def compiled_attention(q, k, v, *args, **kwargs):
        enable_gqa = kwargs.get("enable_gqa", False)
        return scaled_dot_product_attention(q, k, v, enable_gqa=enable_gqa)
else:
    # Default to flex_attention
    compiled_attention = flex_attention

def set_compiled_attention(config):
    """Set the compiled attention function."""
    global compiled_attention

    # Skip if using fallback
    if os.environ.get("FLEX_ATTN_UNSUPPORTED") == "1":
        return

    # Skip on L40 GPUs
    if torch.cuda.is_available() and "L40" in torch.cuda.get_device_name(0):
        return

    compiled_attention = torch.compile(flex_attention, **config)


# optimized llama rms norm implementation
# https://github.com/meta-llama/llama-models/blob/01dc8ce46fecf06b639598f715efbb4ab981fb4c/models/llama4/model.py#L27
def rmsnorm(x, eps):
    def _norm(y):
        return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)

    return _norm(x.float()).type_as(x)


def build_norm(
    norm_type: SupportedNorm,
    dim: int,
    enable: bool = True,
    **kwargs,
) -> nn.Module | None:
    """
    Builds the specified normalization layer based on the norm_type.
     Adapted from `TorchTitan <https://github.com/pytorch/torchtitan/blob/40a10263c5b3468ffa53b3ac98d80c9267d68155/torchtitan/models/norms.py#L21>`__.

    Args:
        norm_type (str): The type of normalization layer to build.
            Supported types: layernorm, np_layernorm, rmsnorm, fused_rmsnorm
        dim (int): The dimension of the normalization layer.

    Returns:
        The built normalization layer.

    Raises:
        NotImplementedError: If an unknown norm_type is provided.
    """
    if not enable:
        return None
    if norm_type == "layer":
        kwargs.pop('bias', None)
        return nn.LayerNorm(dim, bias=True, **kwargs)
    elif norm_type == "layer_unbiased":
        kwargs.pop('bias', None)
        return nn.LayerNorm(dim, bias=False, **kwargs)
    elif norm_type == "rms":
        return nn.RMSNorm(dim, **kwargs)
    elif norm_type == "softplus":
        return nn.Softplus(dim, **kwargs)
    else:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")


def apply_norm(
    norm: nn.Module | None,
    x: Float[torch.Tensor, "batch seq dim"],
) -> Float[torch.Tensor, "batch seq dim"]:
    """
    Apply normalization to input tensor, or no-op if
     no normalization is provided.
    """
    if norm is None:
        return x
    return norm(x)


# Type hint for modules/parameters (or None)
MOP_OR_NOOP = Union[nn.Module, nn.Parameter, None, Iterable]


def init_weights_util(
    modules_and_params: MOP_OR_NOOP,
    init_std: Optional[float] = None,
    depth: Optional[int] = None,
):
    """
    Initialize weights for given modules using truncated normal distribution.

    This helper can be used within ``init_weights`` to standardize weight
    initialization across different modules.

    Initialization strategies adapted from weight initialization used for LlaMa
    in the `Lingua repo <https://github.com/facebookresearch/lingua/blob/47d921518520fc99205aab7746d14446ab389565/lingua/transformer.py#L588>`__.
    Parameters are initialized with a truncated normal distribution with a specified base
    standard deviation (e.g., 0.02), or with the inverse of the squared "input" dimension
    (i.e. such that for `x @ W`, the resulting activations maintain approximately unit
     variance. See more `here <"https://ai.stackexchange.com/questions/30491/is-there-a-proper-initialization-technique-for-the-weight-matrices-in-multi-head">`__).
    """
    # If module is an iterable, recursively call on all members
    do_recursion = (
        # Check for non-module iterables
        (
            isinstance(modules_and_params, Iterable)
            and not isinstance(modules_and_params, (nn.Module, nn.Parameter))
        )
        # Check for iterable modules
        or isinstance(modules_and_params, (nn.ModuleList, nn.Sequential))
    )
    if do_recursion:
        for mop in modules_and_params:
            init_weights_util(mop, init_std, depth)
    # If module/parameter is None or identity, skip initialization
    elif modules_and_params is None or isinstance(modules_and_params, nn.Identity):
        return
    # Reset parameters for normalization layers
    elif isinstance(modules_and_params, (nn.RMSNorm, nn.LayerNorm)):
        modules_and_params.reset_parameters()
    # Initialize weights/biases for convolutional layers
    elif isinstance(modules_and_params, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        # NOTE: fan-in = in_channels * numel(kernel)
        _init_std = init_std or prod(modules_and_params.weight.shape[1:]) ** (-0.5)
        nn.init.trunc_normal_(modules_and_params.weight, std=_init_std)
        if modules_and_params.bias is not None:
            modules_and_params.bias.data.zero_()
    # Initialize weights/biases for linear layers
    elif isinstance(modules_and_params, (nn.Linear,)):
        _init_std = init_std or modules_and_params.in_features ** (-0.5)
        if depth is not None:
            _init_std *= (2 * (depth + 1)) ** (-0.5)
        nn.init.trunc_normal_(modules_and_params.weight, std=_init_std)
        if modules_and_params.bias is not None:
            modules_and_params.bias.data.zero_()
    # Initialize weights or embeddings
    elif isinstance(modules_and_params, nn.Embedding):
        _init_std = init_std or modules_and_params.embedding_dim ** (-0.5)
        nn.init.trunc_normal_(modules_and_params.weight, std=_init_std)
        if modules_and_params.padding_idx is not None:
            modules_and_params.weight.data[modules_and_params.padding_idx].zero_()
    # Initialize parameters
    elif isinstance(modules_and_params, nn.Parameter):
        # TODO: This is a HACK to get the embedding dimension! Verify that the final
        #  dimension is always the embedding or in_feature dimension!
        _init_std = init_std or modules_and_params.shape[-1] ** (-0.5)
        nn.init.trunc_normal_(modules_and_params, std=_init_std)
    else:
        raise NotImplementedError(f"Unsupported module or parameter type: {type(modules_and_params)}")


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    
    .. note::
        Adapted from `timm` library: https://huggingface.co/spaces/Roll20/pet_score/blame/main/lib/timm/models/layers/drop.py

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
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    
    .. note::
        Adapted from `timm` library: https://huggingface.co/spaces/Roll20/pet_score/blame/main/lib/timm/models/layers/drop.py
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


def to_2tuple(x):
    """Convert a single value or iterable to a 2-tuple.

    .. note::
        Adapted from `timm` library: https://huggingface.co/spaces/Roll20/pet_score/blame/main/lib/timm/models/layers/helpers.py
    """
    if isinstance(x, Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.

    .. note::
        Adapted from `timm` library: https://huggingface.co/spaces/Roll20/pet_score/blame/main/lib/timm/models/layers/mlp.py
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim_rot: int = 16,
        context_window_len_s: float = 1.,
        temporal_precision_s: float = 1.e-3,
        use_radians: bool = True,
        clip_timestamps: bool = False,
        base: Optional[int] = None,
    ):
        """
        TODO: when registering the frequency buffers, persistent should be set to false,
          since this buffer can be recomputed. however, we set it to true for 2 reasons.
          (1) due to pytorch/pytorch#123411, compile or pipeline-tracer will not correctly
          handle non-persistent buffers, so we need to fix that. (2) if we initialize
          pipeline-parallel models from a seed checkpoint rather than calling init_weights,
          we need freqs_cis to be initialized by the checkpoint, or we need to add a separate
          initializer for just the non-persistent buffers that is called after loading checkpoints.
        TODO: Support dynamic `base` update during inference to facilitate inference on longer
          sequence lengths than seen during training.
        """
        super().__init__()
        self.dim_rot = dim_rot
        self.context_window_len_s = context_window_len_s
        self.temporal_precision_s = temporal_precision_s
        self.use_radians = use_radians
        self.clip_timestamps = clip_timestamps
        self.base = base

        # Compute frequencies
        inv_freq = self._precompute_freqs()
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    @property
    def device(self):
        return self.inv_freq.device

    @torch.autocast('cuda', enabled = False)
    def _precompute_freqs(self) -> Tuple[
        Float[torch.Tensor, "n_positions dim_rot"],
        Float[torch.Tensor, "n_positions dim_rot"]
    ]:
        """
        Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

        We use the *temporal* rotary embedding described in appendix A.1 of 
        `Poyo <https://arxiv.org/abs/2310.16046>`__. They formulate rotary embeddings
        in terms of timestamps, where the rotation for a given timestamp is given by:

        .. math::

        R_{2 \\times 2}(t, T) = \\begin{bmatrix}
        \\cos(2\\pi t/T) & -\\sin(2\\pi t/T) \\\\
        \\sin(2\\pi t/T) & \\cos(2\\pi t/T)
        \\end{bmatrix}

        When constructing the full rotary matrix, for :math:`D`, :math:`T_\\text{min}`, and :math:`T_\\text{max}`,
        they generate the periods of each dimension, :math:`T_i`, with the following equation:

        .. math::

        T_i = T_\\text{min} \\left(\\frac{T_\\text{max}}{T_\\text{min}}\\right)^{2i/D}

        where :math:`i`, i.e. the dimension index, ranges from 0 to :math:`D/2 - 1`.

        We reformulate this implementation in terms of the more standard implementation 
        of rotary embeddings (e.g., :ref:`Lucidrains <https://github.com/lucidrains/x-transformers/blob/a24bf3eef72e53f29636d07c1017aa1244fd48bb/x_transformers/x_transformers.py#L493>`).
        In this reformulation:

        1. :math:`\\frac{T_\\text{max}}{T_\\text{min}} = \\frac{\\text{context\\_window\\_len\\_s}}{\\text{temporal\\_precision\\_s}}`
        serves as the exponential base (commonly set to :math:`10000` in literature).
        
        2. (Optionally,) :math:`\\frac{1}{2\\pi}` is the *"interpolation factor"*, which "slows down" 
        (if :math:`> 1`) or "speeds up" (if :math:`< 1`) the rotation of the embeddings,
        making the model treat positions as if they were farther apart or closer together,
        respectively. Here it essentially converts the frequencies to radians.

        Note, the term "interpolation" is used because this factor allows for smooth 
        transitioning between different effective sequence lengths. It's like interpolating 
        between the original positional encoding and a scaled version of it.

        3. We create *"positions"* by linearly interpolating between :math:`0` and 
        :math:`\\text{context\\_window\\_len\\_s}` with steps of :math:`\\text{temporal\\_precision\\_s}`.
        Thus, during `apply` for a given timestamp, :math:`t`, the position becomes 
        :math:`\\frac{t}{\\text{temporal\\_precision\\_s}}` (or :math:`\\frac{t}{T_\\text{min}}` in the
        original formulation).
        """
        # Calculate base, i.e. T_max / T_min
        base = self.base or self.context_window_len_s / self.temporal_precision_s
        # Ensure even rotary dimension
        assert self.dim_rot % 2 == 0, "Dimension of rotary embeddings must be even!"
        # Generate inverse frequencies for each dimension (maintain float64 precision for accuracy)
        inv_freq = 1. / base ** (torch.arange(0, self.dim_rot, 2, dtype=torch.float64) / self.dim_rot)
        # Interpolation factor is used to convert from "positional" to "temporal" space
        interpolation_factor = self.temporal_precision_s
        # Adjust interpolation factor to convert to radians, if specified
        if self.use_radians:
            interpolation_factor *= 1. / (2 * torch.pi)
        # Scale frequencies inversely by interpolation factor, and convert to float32
        inv_freq = (inv_freq / interpolation_factor)
        return inv_freq
    
    def init_weights(self):
        with torch.device(self.device):
            self.inv_freq = self._precompute_freqs()

    # @torch.no_grad()  # TODO: Is this necessary? (see `HuggingFace's LlaMa <https://github.com/huggingface/transformers/blob/4fdf58afb72b0754da30037fc800b6044e7d9c99/src/transformers/models/llama/modeling_llama.py#L131>`__)
    @torch.autocast('cuda', enabled = False)
    def forward(
        self,
        ts: Optional[Float[torch.Tensor, "batch seq"]],
    ) -> Optional[Tuple[
        Float[torch.Tensor, "batch seq dim_rot"],
        Float[torch.Tensor, "batch seq dim_rot"],
    ]]:
        """
        Apply rotary embeddings to input tensors according to given timestamps,
         and reshape to match input dimensions (i.e. broadcast over multi-head 
         dimension).
        """
        # No-op if no timestamps are provided
        if ts is None:
            return None
        # Set 0.0s timestamps to "temporal precision" before validation.
        # TODO: this is kinda hacky, requires re-thinking if sample period < self.temporal_precision_s
        # TODO: Is this neccessary? Isn't 0 a valid "position"?
        if self.clip_timestamps:
            ts[ts < self.temporal_precision_s] = self.temporal_precision_s
        # TODO: Verify that the `contiguous` call is necessary for performance
        # Generate frequencies for given timestamps at each dimension: * -> batch x seq x dim_rot//2
        freqs = torch.einsum("..., f -> ... f", ts.contiguous(), self.inv_freq).float()
        # Repeat (interleaved) freqs along rotary dimension, and save cosine/sine
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)
        # Take cos and sin of each frequency
        return freqs.cos(), freqs.sin()


from einops import rearrange
def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


# @torch._dynamo.disable()
@torch.autocast('cuda', enabled = False)
def apply_rotary_emb(
    x: Float[torch.Tensor, "batch num_heads seq dim_head"],
    pos_cos_sin: Optional[Tuple[
        Float[torch.Tensor, "batch seq dim_rot"],
        Float[torch.Tensor, "batch seq dim_rot"],
    ]],
    negate: bool = False,
) -> Float[torch.Tensor, "batch num_heads seq dim_head"]:
    """
    Apply rotary embeddings to input tensors using given freqs_cos and freqs_sin tensors.
    """
    # No-op if no positional information is provided
    if pos_cos_sin is None:
        return x
    # Unpack positional tensors
    pos_cos, pos_sin = pos_cos_sin
    # Tensor shapes
    (B, Nh, S, Dh), Dr = x.shape, pos_cos.shape[-1]
    assert (B, S, Dr) == pos_cos.shape == pos_sin.shape and Dr <= Dh, \
        "Incompatible input and rotary embedding dimensions!"
    # Split into rotated and unrotated components (i.e. partial rotary embeddings, Wang et al. GPT-J)
    x_rot, x_unrot = x[..., :Dr], x[..., Dr:]
    # Negate sine (i.e. reverse rotation) if specified
    pos_sin = -pos_sin if negate else pos_sin
    # Unsqueeze position tensors to broadcast over multi-head dimension
    pos_cos, pos_sin = pos_cos.view(B, 1, S, Dr), pos_sin.view(B, 1, S, Dr)
    # Apply rotary embeddings
    x_rot = (x_rot * pos_cos + rotate_half(x_rot) * pos_sin)
    # Concatenate rotated and unrotated components
    x_rot = torch.cat((x_rot, x_unrot), dim=-1).type_as(x)
    return x_rot


class RotaryAttention(nn.Module):
    """
    Multi-head attention block with rotary embeddings.
    """
    def __init__(
        self,
        dim: int,
        dim_kv: Optional[int] = None,
        dim_head: int = 128,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        rotate_values: bool = False,
        use_projection_bias: bool = False,
        use_qk_norm: bool = True,
        is_cross_attention: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        # Fallback for `dim_kv` if not provided
        if dim_kv is None:
            dim_kv = dim
        # Attention hyperparameters
        self.dim = dim
        self.dim_kv = dim_kv
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        assert self.num_heads % self.num_kv_heads == 0, \
            "Number of heads must be divisible by number of kv heads!"
        self.enable_gqa = self.num_heads != self.num_kv_heads
        self.q_size = self.num_heads * self.dim_head
        self.kv_size = self.num_kv_heads * self.dim_head
        self.rotate_values = rotate_values
        self.is_cross_attention = is_cross_attention
        self.use_qk_norm = use_qk_norm
        self.norm_eps = norm_eps
        # Initialize projection layers  # TODO: Compare with single projection layer and chunking for matching values, i.e. k,v = self.to_kv(x).chunk(2, dim=-1)
        if self.is_cross_attention:
            self.wqkv = None
            # query projection
            self.wq = nn.Linear(dim, num_heads * dim_head, bias=False)
            # key/value projection
            self.wkv = nn.Linear(dim_kv, 2 * self.num_kv_heads * dim_head, bias=False)
        else:
            self.wq, self.wkv = None, None
            self.wqkv = nn.Linear(dim, (num_heads + 2 * self.num_kv_heads) * dim_head, bias=False)
        self.wo = nn.Linear(dim_head * num_heads, dim, bias=use_projection_bias)

    def init_weights(self, init_std: Optional[float] = None, depth: Optional[int] = None):
        # Initialize projection layers
        init_weights_util((self.wq, self.wkv, self.wqkv), init_std)
        # Initialize output layer (optionally, scaled by depth)
        init_weights_util(self.wo, init_std=init_std, depth=depth)

    def forward(
        self,
        q: Float[torch.Tensor, "batch q_seq dim"],
        q_pos: Optional[Tuple[
            Float[torch.Tensor, "batch q_seq dim_rot"],
            Float[torch.Tensor, "batch q_seq dim_rot"],
        ]],
        kv: Float[torch.Tensor, "batch kv_seq dim"],
        kv_pos: Optional[Tuple[
            Float[torch.Tensor, "batch kv_seq dim_rot"],
            Float[torch.Tensor, "batch kv_seq dim_rot"],
        ]],
        attn_mask: Optional[BlockMask] = None,
        qk_norm: Optional[nn.Module] = None,
    ) -> Float[torch.Tensor, "batch q_seq dim"]: 

        # Tensor shapes
        batch, q_seq, _ = q.shape
        # Project inputs: (batch, seq, dim) -> (batch, seq, num_heads*dim_head)
        if self.is_cross_attention:
            # Check that context positions and query positions are either both provided or both None
            assert kv is not None and (q_pos is None) == (kv_pos is None), \
                "Cross-attention requires key-value tensor (and positions, if query positions are provided)!"
            q, (k, v) = self.wq(q), self.wkv(kv).chunk(2, dim=-1)
        else:
            q, k, v = self.wqkv(q).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            # Alias key/value position tensor to query position tensor
            kv_pos = q_pos
        # Separate into multiple heads: (batch, seq, num_heads*dim_head) -> (batch, num_heads, seq, dim_head),
        q = q.view(batch, -1, self.num_heads, self.dim_head).transpose(1, 2)
        k = k.view(batch, -1, self.num_kv_heads, self.dim_head).transpose(1, 2)
        v = v.view(batch, -1, self.num_kv_heads, self.dim_head).transpose(1, 2)
        # Apply query/key normalization
        if self.use_qk_norm:
            q, k = rmsnorm(q, self.norm_eps), rmsnorm(k, self.norm_eps)
        # Apply rotary embeddings to token positions:
        #  (batch, num_heads, seq, dim_head) -> (batch, num_heads, seq, dim_head)
        q = apply_rotary_emb(q, q_pos)
        k = apply_rotary_emb(k, kv_pos)
        if self.rotate_values:
            v = apply_rotary_emb(v, kv_pos)
        # Apply optimized kernel for scaled dot-product attention: * -> (batch, num_heads, q_seq, dim_head)
        o = compiled_attention(q, k, v, block_mask=attn_mask, enable_gqa=self.enable_gqa)
        # Apply inverse rotary embeddings to output (if specified) to capture relative 
        #  positional information. See explanation in appendix A.1 of `Poyo <https://arxiv.org/abs/2310.16046>`__.
        if self.rotate_values:
            o = apply_rotary_emb(o, q_pos, negate=True)
        # Combine head outputs: (batch, num_heads, q_seq, dim_head) -> (batch, q_seq, num_heads*dim_head)
        o = o.transpose(1,2).contiguous().view(batch, q_seq, self.num_heads * self.dim_head)
        # Project outputs: (batch, q_seq, num_heads*dim_head) -> (batch, q_seq, dim)
        o = self.wo(o)
        return o


class FeedForward(nn.Module):
    """
    Two-layer feed forward network with gated SiLU function. 
    Adapted from `TorchTitan <https://github.com/pytorch/torchtitan/blob/40a10263c5b3468ffa53b3ac98d80c9267d68155/torchtitan/models/llama/model.py#L218>`__.
    """
    def __init__(
        self,
        dim: int,
        mult: float = 1.,
        multiple_of: int = 128,
        activation: SupportedActivation = 'silu',
        use_projection_bias: bool = False,
    ):
        super().__init__()
        # Ensure hidden dimension is a multiple of `multiple_of`
        hidden_dim = int(multiple_of * ((dim * mult + multiple_of - 1) // multiple_of))
        # Initialize projection layers
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=use_projection_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=use_projection_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=use_projection_bias)
        # Hyperparameters
        self.activation = activation
    
    def init_weights(self, init_std: Optional[float] = None, depth: Optional[int] = None):
        # Initialize projection layers
        init_weights_util((self.up_proj, self.gate_proj), init_std)
        # Initialize gate/output layers (optionally, scaled by depth)
        init_weights_util(self.down_proj, init_std=init_std, depth=depth)

    def activation_fn(self, x: Float[torch.Tensor, "batch seq dim"]):
        if self.activation == 'silu':
            return F.silu(x)
        elif self.activation == 'gelu':
            return F.gelu(x)
        else:
            raise NotImplementedError(f"Unknown activation function: '{self._activation}'")

    def forward(
        self, x: Float[torch.Tensor, "batch seq dim"]
    ) -> Float[torch.Tensor, "batch seq dim"]:
        return self.down_proj(
            self.activation_fn(self.up_proj(x)) * self.gate_proj(x)
        )


class RotaryTransformerBlock(nn.Module):
    """
    Transformer block with an attention and feedforward layer, as 
    well as layer normalization and residual connections.
    """
    def __init__(
        self,
        dim: int = 128,
        dim_kv: Optional[int] = None,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        dim_head: int = 16,
        rotate_values: bool = False,
        is_cross_attention: bool = True,
        ffn_hidden_mult: float = 1.,
        ffn_multiple_of: int = 128,
        ffn_activation: SupportedActivation = 'silu',
        use_projection_bias: bool = False,
        use_query_residual: bool = True,
        norm_type: SupportedNorm = 'rms',
        use_pre_norm: bool = True,
        use_post_norm: bool = True,
        use_qk_norm: bool = True,
        norm_eps: float = 1.e-5,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        # Fallback for `dim_kv` if not provided
        if dim_kv is None:
            dim_kv = dim
        # Block hyperparameters
        self.use_query_residual = use_query_residual
        # Define normalization layers
        self.attn_pre_norm_context = build_norm(norm_type, dim_kv, eps=norm_eps, enable=(is_cross_attention and use_pre_norm))
        self.attn_pre_norm = build_norm(norm_type, dim, eps=norm_eps, enable=use_pre_norm)
        self.attn_post_norm = build_norm(norm_type, dim, eps=norm_eps, enable=use_post_norm)
        self.ffn_pre_norm = build_norm(norm_type, dim, eps=norm_eps, enable=use_pre_norm)
        self.ffn_post_norm = build_norm(norm_type, dim, eps=norm_eps, enable=use_post_norm)
        # Define attention and feedforward layers
        self.attention = RotaryAttention(
            dim=dim,
            dim_kv=dim_kv,
            dim_head=dim_head,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rotate_values=rotate_values,
            use_projection_bias=use_projection_bias,
            is_cross_attention=is_cross_attention,
            use_qk_norm=use_qk_norm,
            norm_eps=norm_eps,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            mult=ffn_hidden_mult,
            multiple_of=ffn_multiple_of,
            activation=ffn_activation,
            use_projection_bias=use_projection_bias,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def init_weights(self, init_std: Optional[float] = None, depth: Optional[int] = None):
        # Reset normalization layers
        init_weights_util((
            self.attn_pre_norm_context,
            self.attn_pre_norm,
            self.attn_post_norm,
            self.ffn_pre_norm,
            self.ffn_post_norm,
        ),)
        # Initialize attention and feedforward layers
        self.attention.init_weights(init_std=init_std, depth=depth)
        self.feed_forward.init_weights(init_std=init_std, depth=depth)
        
    def forward(
        self,
        x: Float[torch.Tensor, "batch q_seq dim"],
        pos: Optional[Tuple[
            Float[torch.Tensor, "batch q_seq dim_rot"],
            Float[torch.Tensor, "batch q_seq dim_rot"],
        ]],
        context: Optional[Float[torch.Tensor, "batch kv_seq dim"]] = None,
        context_pos: Optional[Tuple[
            Float[torch.Tensor, "batch kv_seq dim_rot"],
            Float[torch.Tensor, "batch kv_seq dim_rot"],
        ]] = None,
        attn_mask: Optional[BlockMask] = None,
    ) -> Float[torch.Tensor, "batch q_seq dim"]: 

        # Alias residual before attention
        residual = x
        # Apply pre-normalization (or no-op, if not `use_pre_norm`) to queries...
        q = apply_norm(self.attn_pre_norm, x)
        # ...and context, if cross attention (otherwise, a no-op).
        kv = apply_norm(self.attn_pre_norm_context, context)
        # Apply attention
        hidden = self.attention(
            q,
            pos,
            kv,
            context_pos,
            attn_mask=attn_mask,
        )
        # Apply post-attention normalization (or no-op
        hidden = apply_norm(self.attn_post_norm, hidden)
        # Apply dropout
        hidden = self.drop_path(hidden)
        # Add residual connection if specified
        if self.use_query_residual:
            hidden = hidden + residual
        # Alias residual before feedforward
        residual = hidden
        # Apply pre-feedforward normalization (or no-op
        hidden = apply_norm(self.ffn_pre_norm, hidden)
        # Apply feedforward network
        hidden = self.feed_forward(hidden)
        # Apply post-feedforward normalization (or no-op)
        hidden = apply_norm(self.ffn_post_norm, hidden)
        # Apply dropout
        hidden = self.drop_path(hidden)
        # Add residual connection
        hidden = hidden + residual

        return hidden


class ELU1(nn.Module):
    """
    Adapted from `NeuralPredictors <https://github.com/sinzlab/neuralpredictors/blob/main/neuralpredictors/layers/activations.py>`__.
    
    Elu activation function shifted by 1 to ensure that the
    output stays positive. That is:
    Elu1(x) = Elu(x) + 1
    """

    def forward(self, x, inplace=False, eps=0.0):
        return F.elu(x, inplace=inplace) + 1.0 + eps


class ClippedLoss(nn.Module):
    """
    Wrapper around a loss function that clips the losses element-wise, before applying the reduction.
    """
    def __init__(
        self,
        base_criterion: nn.Module,
        magnitude: Optional[float] = None,
    ):
        super().__init__()
        # If magnitude is provided, we need to clip the losses element-wise, before applying the reduction.
        if magnitude is not None:
            self.clipped_reduction = base_criterion.reduction
            base_criterion.reduction = 'none'
        else:
            self.clipped_reduction = 'none'
        assert self.clipped_reduction in ('mean', 'sum', 'none'), \
            f"Base criterion must have reduction='mean', 'sum', or 'none'!"
        self.base_criterion = base_criterion
        self.magnitude = magnitude

    def forward(self, predictions, targets):
        # Get loss from base
        loss = self.base_criterion(predictions, targets)
        # If no magnitude is provided, we simply pass through the loss from the base criterion
        if self.magnitude is None:
            return loss
        # Clip the losses element-wise
        loss = torch.clamp(loss, min=(-self.magnitude), max=self.magnitude)
        # Now perform the reduction (mean/sum)
        if self.clipped_reduction == 'mean':
            loss = torch.mean(loss)
        elif self.clipped_reduction == 'sum':
            loss = torch.sum(loss)
        return loss