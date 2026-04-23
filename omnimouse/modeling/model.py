import math
import random
from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    BlockMask,
)

from omnimouse.masking import MaskingStrategy
from omnimouse.modeling import Model, ModelArgs, OMModelOutput, register_model
from omnimouse.modeling.mods import create_multimodal_temporal_sliding_window_block_mask
from omnimouse.modeling.nn import (
    ELU1,
    BlockMask,
    ClippedLoss,
    RotaryEmbedding,
    RotaryTransformerBlock,
    SupportedActivation,
    SupportedDepthInit,
    SupportedLocalGlobalLayout,
    SupportedNorm,
    init_weights_util,
    set_compiled_attention,
)
from omnimouse.utils.pylogger import RankedLogger
from omnimouse.utils.types import SessionMap
from omnimouse.utils.utils import omegaconf_to_masking_strategy

from .hiera_encoder import Hiera

log = RankedLogger(__name__)

"""
TODO:
- finish block mask wrapper for serialization
- look into TORCH DYNAMO CACHE path (make sure)
- dropout
- einops-ify
- consider configuring latent window size and stride in seconds
"""


def concat_notnone(xs: Sequence[Tensor | None], dim: int = 1) -> Tensor | None:
    """Concatenate a sequence of tensors, ignoring None's. Returns None if all are None."""
    filtered = tuple(x for x in xs if x is not None)
    if not len(filtered):
        return None
    return torch.cat(filtered, dim=dim)


def add_notnone(xs: Sequence[Tensor | None]) -> Tensor | None:
    """Add a sequence of tensors, ignoring None's. Returns None if all are None."""
    filtered = tuple(x for x in xs if x is not None)
    if not len(filtered):
        return None
    return torch.stack(filtered).sum(dim=0)


###############################
#  Session/Animal/etc. Params #
###############################

"""
NOTE: Both per-session and animal params are in embedding space, but while each
 `SessionParams` module contains only parameters for a single session, the `AnimalParams`
 module contains parameters for all animals (i.e. all sessions). This is done because multiple
 sessions can have the same animal, so the `AnimalParams` will be shared across sessions
 (and thus communicated across all GPUs) for convenience.
"""


class SessionParams(nn.Module):
    def __init__(
        self,
        neural_population_size: int,
        d_embedding: int,
        bias: bool = True,
    ):
        super().__init__()
        # Embedding modules
        #  Unique session embedding vector
        self._session_embedding = nn.Parameter(torch.zeros(d_embedding))
        #  Per-neuron embedding vectors
        self._neuron_embeddings = nn.Embedding(neural_population_size, d_embedding)
        #  Per-neuron biases. NOTE: Final singular dimension broadcasts to output feature dimension.
        self._neuron_biases = (
            nn.Parameter(torch.zeros(neural_population_size, 1)) if bias else None
        )

    def init_weights(self, init_std: Optional[float] = None):
        init_weights_util(
            (
                self._session_embedding,
                self._neuron_embeddings,
            ),
            init_std,
        )
        if self._neuron_biases is not None:
            self._neuron_biases.data.zero_()

    @property
    def session_embedding(self) -> Float[Tensor, "d_embedding"]:
        return self._session_embedding

    @property
    def neuron_embeddings(self) -> nn.Embedding:
        return self._neuron_embeddings

    def neuron_embedding_weights(self, B: int) -> Float[Tensor, "B N d_embedding"]:
        return self._neuron_embeddings.weight.expand(B, -1, -1).contiguous()

    @property
    def neuron_biases(self) -> Float[Tensor, "batch neuron_ids"]:
        return self._neuron_biases


class AnimalParams(nn.Module):
    def __init__(
        self,
        session_key_to_animal_id_map: Mapping[str, str],
        d_embedding: int,
    ):
        super().__init__()
        # Store mapping from session keys to animal IDs
        self.session_key_to_animal_id_map = session_key_to_animal_id_map
        # Create embeddings for each unique animal
        unique_animal_ids = sorted(set(session_key_to_animal_id_map.values()))
        self._animal_embeddings = nn.ParameterDict(
            {
                animal_id: nn.Parameter(torch.zeros(d_embedding))
                for animal_id in unique_animal_ids
            }
        )

    def init_weights(self, init_std: Optional[float] = None):
        """Initialize animal embeddings using truncated normal distribution."""
        init_weights_util(self._animal_embeddings.values(), init_std)

    def animal_embedding(self, session_key: str) -> Float[Tensor, "d_embedding"]:
        """Return the embedding for the animal associated with the given session key."""
        animal_id = self.session_key_to_animal_id_map[session_key]
        animal_embed = self._animal_embeddings[animal_id]
        return animal_embed


class AbsoluteExperimentTimeEmbedding(nn.Module):
    """
    Implement a sinusoidal positional embedding function. Adapted from the
     `annotated transformer<http://nlp.seas.harvard.edu/annotated-transformer/#positional-encoding>`__
    """

    def __init__(
        self,
        d_model: int = 256,
        max_recording_len_s: float = 18000,  # 3 hours
        temporal_precision_s: float = 1.0,
        base: Optional[int] = 100000,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_recording_len_idcs = math.ceil(
            max_recording_len_s / temporal_precision_s
        )
        self.temporal_precision_s = temporal_precision_s
        self.base = base or self.max_recording_len_idcs
        positional_embeddings = self._precompute()
        self.register_buffer("positional_embeddings", positional_embeddings)

    def _precompute(self) -> Float[Tensor, "max_recording_len_idcs d_model"]:
        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.max_recording_len_idcs, self.d_model)
        position = torch.arange(0, self.max_recording_len_idcs).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2) * -(math.log(self.base) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, ts: Float[Tensor, "B"]) -> Float[Tensor, "B D"]:
        ts = (ts // self.temporal_precision_s).long()
        return self.positional_embeddings[ts]


#########################################
#    Shared Latent Params/Embeddings    #
#########################################


class ResponseLatentEmbeddings(nn.Module):
    def __init__(
        self,
        num_samples_per_block: int,
        num_samples_per_s: float,
        context_window_len_s: float,
        latent_window_size_samples: int,
        latent_window_stride_samples: int,
        num_latent_groups: int,
        num_global_latents: int,
        dim_latents: int,
    ):
        """
        Module encapsulating learned *response* latent vectors (and corresponding timestamps).
         Grouped latents are each repeated `latent_group_size` times, with timestamps
         spanning the context window and a smaller window size. Global latents are
         assigned a timestamp of 0 and attend to the whole context window. We determine
         the number of latents per group as the minimum number which cover the entire
         context window, given a specified stride and window size.

        NOTE: Adapted from Poyo, see Appendix A.3 of `Azabou et al., 2023 <https://arxiv.org/abs/2310.16046>`__.)
        TODO: Rename from "response" embeddings
        """
        super().__init__()
        num_samples_per_s = float(num_samples_per_s)
        latent_group_size = (
            math.ceil(
                (num_samples_per_block - latent_window_size_samples)
                / latent_window_stride_samples
            )
            + 1
        )
        self._latent_embeddings = nn.Embedding(
            num_global_latents + num_latent_groups, dim_latents
        )
        # NOTE: `_latent_ids` is a registered buffer (Int[Tensor,
        #  "num_latent_groups*latent_group_size+num_global_latents"])
        self.register_buffer(
            "_latent_ids",
            torch.cat(
                (
                    #  grouped_latent_ids: num_global_latents + [0, 1, ..., num_latent_groups-1,
                    #  ..., 0, 1, ..., num_latent_groups-1 ]
                    torch.arange(
                        num_latent_groups,
                    ).repeat(latent_group_size),
                    #  global_latent_ids: [0, 1, 2, ..., num_global_latents-1]
                    torch.arange(
                        num_latent_groups, num_latent_groups + num_global_latents
                    ),
                )
            ),
        )
        # NOTE: `_latent_timestamps` is a registered buffer (Float[Tensor,
        #  "num_latent_groups*latent_group_size+num_global_latents"])
        latent_window_stride_s = latent_window_stride_samples / num_samples_per_s
        latent_timestamps = torch.arange(latent_group_size) * latent_window_stride_s
        self.register_buffer(
            "_latent_timestamps",
            torch.cat(
                (
                    #  grouped_latent_timestamps:  context_window_len_s / latent_group_size * [0,
                    #   ..., 0, 1, ..., 1, ..., latent_group_size-1, ..., latent_group_size-1]
                    latent_timestamps.repeat_interleave(num_latent_groups),
                    #  global_latent_timestamps: [0, 0, 0, ..., 0]
                    torch.zeros(num_global_latents),
                )
            ),
        )
        # Calculate window sizes in seconds (NOTE: global latents attend to entire block)
        global_window_size_s = num_samples_per_block / num_samples_per_s
        grouped_window_size_s = latent_window_size_samples / num_samples_per_s
        # NOTE: `_latent_window_sizes` is a registered buffer (Float[Tensor,
        #  "num_latent_groups*latent_group_size+num_global_latents"])
        self.register_buffer(
            "_latent_window_sizes",
            torch.cat(
                (
                    #  grouped_latent_window_sizes: [`grouped_wss`, ..., `grouped_wss`]
                    torch.full(
                        (num_latent_groups * latent_group_size,), grouped_window_size_s
                    ),
                    #  global_latent_window_sizes: [`global_wss`, ..., `global_wss`]
                    torch.full((num_global_latents,), global_window_size_s),
                )
            ),
        )

        # Hyperparameters
        self.num_samples_per_s = num_samples_per_s
        self.num_global_latents = num_global_latents

    def init_weights(self, init_std: Optional[float] = None):
        init_weights_util(self._latent_embeddings, init_std)

    def _context_mask(
        self,
        max_response_context_samples: int,
        response_context_start_idx: int = 0,
    ) -> Bool[Tensor, "batch n_context_latents"]:
        response_context_start_seconds = (
            response_context_start_idx / self.num_samples_per_s
        )
        max_response_context_seconds = (
            max_response_context_samples / self.num_samples_per_s
        )
        response_context_end_seconds = (
            response_context_start_seconds + max_response_context_seconds
        )
        if max_response_context_seconds == 0:
            mask = torch.zeros_like(self._latent_timestamps, dtype=torch.bool)
        else:
            mask = torch.logical_and(
                response_context_start_seconds <= self._latent_timestamps,
                self._latent_timestamps < response_context_end_seconds,
            )
        # Set global latents to True
        mask[-self.num_global_latents :] = True
        return mask

    def embeddings(
        self,
        batch_size: int,
        max_response_context_samples: int,
        response_context_start_idx: int = 0,
    ) -> Float[Tensor, "batch n_groups*group_size+n_global dim_latents"]:
        # TODO: Is it faster to use repeated ID's, or repeat on each forward call?
        context_mask = self._context_mask(
            max_response_context_samples,
            response_context_start_idx,
        )
        latent_ids = self._latent_ids[context_mask]
        latent_embeddings = self._latent_embeddings(latent_ids)
        return latent_embeddings.expand(batch_size, -1, -1).contiguous()

    def timestamps(
        self,
        batch_size: int,
        max_response_context_samples: int,
        response_context_start_idx: int = 0,
    ) -> Float[Tensor, "batch_size n_groups*group_size+n_global"]:
        context_mask = self._context_mask(
            max_response_context_samples,
            response_context_start_idx,
        )
        return self._latent_timestamps[context_mask].expand(batch_size, -1)

    def window_sizes(
        self,
        batch_size: int,
        max_response_context_samples: int,
        response_context_start_idx: int = 0,
    ) -> Float[Tensor, "batch_size n_groups*group_size+n_global"]:
        context_mask = self._context_mask(
            max_response_context_samples,
            response_context_start_idx,
        )
        return self._latent_window_sizes[context_mask].expand(batch_size, -1)


#########################################################
#  Modality -> Feature / Hidden -> Modality Projectors  #
#########################################################


class ResponseFeatureExtractor(nn.Module):
    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str | int | Tuple[int, int] = 0,
    ):
        super().__init__()
        if padding not in ("valid", 0, (0, 0)):
            raise NotImplementedError(
                "Padding not yet supported for response embedding!"
            )
        # Projection from response to neural embedding space
        out_channels = d_model
        self._feature_proj = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # Hyperparameters
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

    def init_weights(self, init_std: Optional[float] = None):
        """
        TODO: Consider alternative initialization schemes.
        """
        init_weights_util(self._feature_proj, init_std)

    def forward(
        self,
        responses: Float[Tensor, "B S N"],
        timestamps: Float[Tensor, "B S N"],
        neuron_embeddings: Int[Tensor, "B N D"],
    ) -> Tuple[
        Float[Tensor, "B T*N D"],
        Float[Tensor, "B T*N"],
    ]:
        """
        Embed neural activity and corresponding timestamps.

        TODO: Pad timestamps before downsampling to support non-zero conv padding
        """
        # Get shape of raw responses
        B, S, N = responses.shape
        # Move sequence dimension to the end, flattent batch/neuron dimensions and create
        #  "channel" dimension: (B, S, N) -> (B, N, S) -> (B*N, 1, S)
        responses = responses.transpose(1, 2).contiguous().view(B * N, 1, S)
        # Apply conv1d along sequence dimension, reducing
        #  samples to "tokens": (B*N, 1, S) -> (B*N, D, T)
        feats = self._feature_proj(responses)
        # Get shape of "embedded" response features
        _, D, T = feats.shape
        # Permute "token" dimension, such that neurons at the
        #  same position will be contiguous in memory after
        #  flattening: (B*N, D, T) -> (B, T, N, D)
        feats = feats.view(B, N, D, T).permute(0, 3, 1, 2).contiguous()
        # Add neuron embeddings to response features: (B, T, N, D) +
        #  (B, 1, N, D)  -> (B, T, N, D)
        feats = feats + neuron_embeddings.unsqueeze(1)
        # Flatten "token" and "population" dimensions: (B, T, N, D) -> (B, T*N, D)
        feats = feats.view(B, T * N, D)
        # Downsample timestamps to match embedding stride, and truncate
        #  to match embedded response length: (B, S, N) -> (B, T, N)
        feat_timestamps = timestamps[:, :: self.stride, :][:, :T, :]
        # Flatten timestamps dimension: (B, T, N) -> (B, T*N)
        feat_timestamps = feat_timestamps.reshape(B, T * N)
        return feats, feat_timestamps


class HieraFeatureExtractor(nn.Module):
    """
    Wrapper for Hiera model from `Ryali et al.<https://github.com/facebookresearch/hiera/tree/main>`__.
    """

    def __init__(
        self,
        *args,
        feature_dim: Optional[int] = None,
        relativize_timestamps: bool = True,
        **kwargs,
    ):
        super().__init__()
        # Initialize Hiera model as backbone
        self.backbone = Hiera(*args, **kwargs)
        # Validate that Hiera is a video model (and supports timestamp patchifying)
        assert len(self.backbone.tokens_spatial_shape) == 3, (
            "VideoHiera only valid for temporal inputs (i.e. videos)!"
        )

        # Construct projection from final intermediate features
        #  to desired feature dimension, if provided
        self.feature_proj = (
            nn.Linear(
                in_features=self.backbone.blocks[-1].dim_out,
                out_features=feature_dim,
            )
            if feature_dim is not None
            else None
        )

        # Attributes
        self.relativize_timestamps = relativize_timestamps

        # Dimension Properties:
        #  input channels expected by Hiera patch embedding
        self.in_chans = self.backbone.patch_embed.proj.in_channels
        #  final intermediate dimension of Hiera model (i.e. before head)
        self.final_intermediate_dim = self.backbone.blocks[-1].dim_out
        #  output channels of Hiera feature extractor (i.e. feature projection if
        #   provided, otherwise same as `final_intermediate_dim`)
        self.out_chans = (
            self.final_intermediate_dim
            if self.feature_proj is None
            else self.feature_proj.out_features
        )
        #  combined stride of query pooling stages
        self.final_q_pool_stride = tuple(
            qs**self.backbone.q_pool for qs in self.backbone.q_stride
        )
        # NOTE: Currently, Hiera assumes that the patch_kernel and padding aren't used, so the
        #  tokens_spatial_shape is inaccurate. We compute the accurate shape here. TODO: Integrate
        #  this into the Hiera model.
        accurate_tokens_spatial_shape = [
            int((i + 2 * p - k) // s + 1)
            for i, s, k, p in zip(
                kwargs["input_size"],
                self.backbone.patch_stride,
                self.backbone.patch_embed.proj.kernel_size,
                self.backbone.patch_embed.proj.padding,
            )
        ]
        #  shape of final multimodal features (i.e. input_shape // patch_stride // q_pool_stride)
        self.final_feat_shape = tuple(
            t // s
            for t, s in zip(accurate_tokens_spatial_shape, self.final_q_pool_stride)
        )
        #  combined stride of patch embedding + pooling stages
        self.final_feat_stride = tuple(
            ps * qs
            for ps, qs in zip(self.backbone.patch_stride, self.final_q_pool_stride)
        )
        #  total sequence length of final multimodal features
        self.final_feat_seqlen = math.prod(self.final_feat_shape)
        #   sequence length "per-frame" of final multimodal features
        self.final_feat_frames_group_size = math.prod(self.final_feat_shape[-2:])
        #  stride of final multimodal features in the temporal dimension
        self.final_feat_frames_stride = self.final_feat_stride[0]
        #  stride of final multimodal features in the temporal dimension,
        #   exclusively due to query pooling stages
        self.final_feat_frames_q_pool = self.final_q_pool_stride[0]
        #  padding of final multimodal features in the temporal dimension
        self.final_feat_frames_padding = self.backbone.patch_embed.proj.padding[0]
        #  kernel size of final multimodal features in the temporal dimension
        self.final_feat_frames_window_size = self.backbone.patch_embed.proj.kernel_size[
            0
        ]

    def init_weights(self, init_std: Optional[float] = None):
        # NOTE: Hiera already calls `Hiera._init_weights` in its `__init__`
        init_weights_util(self.feature_proj, init_std)

    def final_feat_timestamps(
        self,
        timestamps: Float[Tensor, "batch n_frames"],
    ) -> Float[Tensor, "batch n_frame_feats"]:
        """
        Convert full sequence timestamps to sequence of timestamps corresponding to
         features output by Hiera, selecting the latest timestamp from each features
         receptive field.
        """
        # Hiera's patch embedding parameters
        pad, window_size, stride, q_pool = (
            self.final_feat_frames_padding,  # patch embedding padding
            self.final_feat_frames_window_size,  # kernel size patch embedding kernel
            self.final_feat_frames_stride,  # patch embedding stride
            self.final_feat_frames_q_pool,  # query pooling stride
        )
        # Calculate *index* of last sample for first final feature. NOTE: Last sample
        #  of the first token is the `window_size - 1`. You must then consider the
        #  additional tokens merged into the first feature by the query pooling stages.
        patch_stride = stride // q_pool
        offset = window_size - 1 + patch_stride * (q_pool - 1)
        # Pad timestamps to match feature shape
        timestamps = F.pad(
            timestamps,
            (
                pad,
                pad,
            ),
            mode="replicate",
        )
        # Stride timestamps original timestamps to match feature pooling, offset by initial
        #  patch embedding's kernel size to get the *latest* timestamp from each receptive field
        timestamps = timestamps[..., offset::stride]
        return timestamps

    def n_feats_for_n_visible_frames(
        self,
        n_visible_frames: int,
        visible_frames_start: int = 0,
    ) -> Tuple[int, int]:
        """
        Calculate the start index and number of output features whose temporal receptive fields
        overlap with the specified range of visible frames.

        Args:
            n_visible_frames: The number of frames in the visible window
            visible_frames_start: The starting frame index of the visible window

        Returns:
            Tuple of (visible_feats_start, n_visible_feats):
            - visible_feats_start: Index of first feature with receptive field overlapping visible range
            - n_visible_feats: Number of features with receptive fields overlapping visible range

        TODO: Validate this thoroughly.
        """
        # Extract parameters affecting the temporal receptive field
        ff_stride, ff_window, ff_pad, ff_group_size = (
            self.final_feat_frames_stride,
            self.final_feat_frames_window_size,
            self.final_feat_frames_padding,
            self.final_feat_frames_group_size,
        )

        # TODO: Why the extra +1 ?
        # Calculate earliest feature whose receptive field overlaps with visible range
        visible_feats_start = (
            2 + (visible_frames_start + ff_pad - ff_window) // ff_stride
        )
        visible_feats_start = max(0, visible_feats_start) * ff_group_size
        # Calculate latest feature whose receptive field overlaps with visible range
        visible_frames_end = visible_frames_start + n_visible_frames
        visible_feats_end = 2 + (visible_frames_end + ff_pad - ff_window) // ff_stride
        visible_feats_end = max(0, visible_feats_end) * ff_group_size

        return visible_feats_start, visible_feats_end

    def forward(
        self,
        screen: Float[Tensor, "batch channels n_frames height width"],
        timestamps: Optional[Float[Tensor, "batch n_frames"]] = None,
        n_visible_frames: Optional[int] = None,
        visible_frames_start: int = 0,
    ) -> Tuple[
        Float[Tensor, "batch seq_feats dim"],
        Optional[Float[Tensor, "batch seq_feats"]],
    ]:
        """
        Forward pass for HieraVideoFeatureExtractor:
        - Extract multimodal features from video
        - Project multimodal features to embedding dimension
        - Assign timestamps to each feature based on its position in the video
         and temporal receptive field

        TODO: Support multi-level feature extraction.
        """
        # Tensor shapes. NOTE: Here, `T` (i.e. temporal) is the number of frames
        B, C, T, H, W = screen.shape
        # Swap temporal/channel dimensions and get multimodal features with wrapped Hiera
        #  (reduced over temporal/spatial dimensions by patch embedding + pooling):
        #  ((B, C, T, H, W) -> (B, T_redux, H_redux, W_redux, D_hiera)
        final_feats = self.backbone.forward(screen)
        # Project (final) multimodal features to feature (i.e. model) dimension:
        #  (B, T_redux, H_redux, W_redux, D_hiera) -> (B, T_redux, H_redux, W_redux, D)
        if self.feature_proj is not None:
            final_feats = self.feature_proj(final_feats)
        # Get final feature shapes
        B, T_redux, H_redux, W_redux, D = final_feats.shape
        # Flatten spatial dimension of multimodal features:
        #  (B, T_redux, H_redux, W_redux, D | D_hiera) -> (B, F, D | D_hiera)
        final_feats = final_feats.view(B, -1, D)
        # "Relativize" timestamps within block and convert
        timestamps = (
            timestamps - timestamps.amin(dim=-1, keepdim=True)
            if self.relativize_timestamps
            else timestamps
        )
        # Get *latest* timestamp for each feature
        feat_timestamps = self.final_feat_timestamps(timestamps)
        # Repeat timestamps for each spatial feature:
        #  (B, S_redux) -> (B, S_redux, 1, 1) -> (B, S_redux, H_redux, W_redux)
        feat_timestamps = feat_timestamps[..., None, None].repeat(
            1, 1, H_redux, W_redux
        )
        # Flatten spatial dimension of multimodal timestamps:
        #  (B, T_redux, H_redux, W_redux) -> (B, F)
        feat_timestamps = feat_timestamps.view(B, -1)
        # If only a subset of frames are visible, apply "video masking"
        if n_visible_frames is not None and n_visible_frames < T:
            # Determine downsampled number of visible frames
            visible_feats_start, visible_feats_end = self.n_feats_for_n_visible_frames(
                n_visible_frames,
                visible_frames_start,
            )
            # Remove feats corresponding to hidden frames
            final_feats = final_feats[..., visible_feats_start:visible_feats_end, :]
            feat_timestamps = feat_timestamps[
                ..., visible_feats_start:visible_feats_end
            ]

        return final_feats, feat_timestamps


@dataclass
class OmniMouseArgs(ModelArgs):
    """
    Model configuration arguments for OmniMouse.

    TODO: Session frequencies should be bassed by `session_map`
    """

    # Link to `OmniMouseModel`
    model_type: ClassVar[str] = "omnimouse"
    model_tags: Optional[List[str]] = None

    # Model Configuration
    num_samples_per_s: int = 8  # i.e., 8 Hz
    num_samples_per_block: int = 16
    skip_n_samples: int = (
        0  # num samples from start to ignore when only image processed
    )
    interpolation_buffer: int = 0  # number of samples to interpolate between responses
    max_population_size: Optional[int] = None
    n_neurons_multiple_of: Optional[int] = None
    d_embedding: int = 128  # size of neuron embeddings
    always_use_animal_embeddings: bool = False  # whether to always use animal embeddings, even if only a single animal is present
    add_meta_to_embeddings: bool = True  # whether to add session/animal/etc. to session-associated embeddings (i.e. neurons, behavior, etc.)
    add_meta_to_latents: bool = (
        True  # whether to add session/animal/etc. to response latents
    )
    add_meta_to_video_feats: bool = (
        False  # whether to add session/animal/etc. to video features
    )
    add_exp_time_embedding: bool = (
        True  # whether to add experiment time embedding to meta-embedding
    )
    max_recording_len_s: float = 18000  # 5 hours
    absolute_experiment_time_precision_s: float = 1.0
    d_model: Optional[int] = (
        None  # size of latent space / model layers (defaults to d_embedding)
    )
    per_modality_decoders: bool = (
        False  # whether to use a separate decoder for each modality
    )
    masking_strategies: MaskingStrategy | Sequence[MaskingStrategy] = field(
        default_factory=lambda: MaskingStrategy()
    )
    response_window_size_samples: int = 3
    response_stride_samples: int = 1
    response_query_lookback_samples: int = 0
    response_query_lookahead_samples: Optional[int] = None
    num_response_latent_groups: int = 256
    # NOTE: Latent "group size" determined by number of latents required to
    #  cover context window with specified window size and stride
    # NOTE: Default configuration yields exactly 2048 "grouped" response latents
    response_latent_window_size_samples: int = 3
    response_latent_window_stride_samples: int = 2
    num_global_latents: int = 256
    rope_use_radians: bool = True
    rope_base: Optional[int] = None
    rotate_values: bool = False  # whether to rotate values before attention calculation
    num_heads: int = 8
    num_kv_heads: Optional[int] = None
    num_blocks: int = 1  # weight sharing in depth (see :class:`RotaryPerceiverIO`)
    num_self_attends_per_block: int = 6
    ffn_hidden_mult: float = 4.0
    decoder_ffn_hidden_mult: Optional[float] = None
    ffn_activation: SupportedActivation = "silu"
    norm_type: SupportedNorm = "rms"
    use_pre_norm: bool = True
    use_qk_norm: bool = False
    use_post_norm: bool = True
    norm_eps: float = 1e-5
    drop_path_rate: float = 0.0
    drop_path_depth_decay: bool = True
    local_to_global_ratio: Optional[Tuple[int, int]] = (
        5,
        1,
    )  # (1, 0) or None -> all local, (0, 1) -> all global
    local_global_layout: SupportedLocalGlobalLayout = "interleaved"
    use_global_nope: bool = False
    response_output_activation: Optional[nn.Module] = (
        None  # falls back to `omnimouse.modeling.nn.ELU1`
    )
    response_criteria: Optional[nn.Module] = (
        None  # falls back to `nn.PoissonNLLLoss(log_input=False, reduction='mean')`
    )
    response_poisson_loss_eps: float = 1e-8  # Override default eps value for Poisson NLL loss, if `response_criteria` not provided manually
    response_loss_clamp_value: Optional[float] = (
        None  # if `None`, no clipping is applied
    )
    init_std: Optional[float] = None  # if `None`, use inverse sqrt of input dimensions
    init_depth_scaling: Optional[SupportedDepthInit] = (
        None  # enable depth-scaling of `init_std` for attn/feedforward out-proj's
    )
    # Initialization std dev for specific components. Defaults to `init_std` if not provided.
    embeddings_init_std: Optional[float] = None
    embed_to_model_proj_init_std: Optional[float] = None
    shared_embeddings_init_std: Optional[float] = None
    response_proj_init_std: Optional[float] = None
    behavior_proj_init_std: Optional[float] = None
    hiera_proj_init_std: Optional[float] = None
    readout_init_std: Optional[float] = None

    # Additional modality configuration
    behavior_as_channels: bool = False
    video_channels: int = 1
    num_frames_per_s: int = 30  # i.e., 30 Hz
    num_frames_per_block: int = 60  # number of frames of video/behavior per block
    behavior_channels: int = 5  # 4 eye_tracker, 1 treadmill
    num_behavior_samples_per_s: int = 20  # i.e., 20 Hz
    num_behavior_samples_per_block: int = 40  # number of behavior samples per block
    decode_behavior: bool = True
    behavior_output_activation: Optional[nn.Module] = None
    behavior_criteria: Optional[nn.Module] = None  # falls back to `nn.MSELoss()`
    behavior_loss_clamp_value: Optional[float] = (
        None  # if `None`, no clipping is applied
    )
    behavior_loss_factor: float = 1.0  # relative weight of behavior loss
    per_channel_behavior_readout: bool = False

    # Hiera feature extractor config
    hiera_num_heads: int = 3
    hiera_embed_dim: int = 96
    hiera_stages: Tuple[int, ...] = (2, 2)
    hiera_q_pool: int = 1
    hiera_q_stride: Tuple[int, ...] = (1, 2, 2)
    hiera_mask_unit_size: Tuple[int, ...] = (1, 8, 8)
    hiera_patch_kernel: Tuple[int, ...] = (6, 7, 7)
    hiera_patch_stride: Tuple[int, ...] = (2, 2, 2)
    hiera_patch_padding: Tuple[int, ...] = (2, 3, 3)
    hiera_sep_pos_embed: bool = True
    hiera_drop_path_rate: float = 0.1
    hiera_mlp_ratio: float = 4.0

    # flex attention configs
    flex_attention_compile_config: Optional[Dict[str, Any]] = None
    flex_attn_block_size: int = _DEFAULT_SPARSE_BLOCK_SIZE

    # TODO: Rescaling is no longer supported by experanto, so this should be sourced from
    #  the data config instead
    video_resolution: Tuple[int, int] = (36, 64)

    enable_fallback_sensorium_behavior: bool = False

    def __post_init__(self):
        super().__post_init__()

        # Only set if it's empty
        if not self.model_tags:
            self.model_tags = ["omnimouse"]

        # Default to `d_embedding` if not provided
        self.d_model = self.d_model or self.d_embedding

        # Convert local_to_global_ratio to tuple if not provided
        if self.local_to_global_ratio is not None:
            self.local_to_global_ratio = tuple(self.local_to_global_ratio)

        # Validate that convolutional response embedding parameters will capture all samples
        assert self.response_stride_samples <= self.response_window_size_samples, (
            f"Stride ({self.response_stride_samples}) must be <= kernel size"
            f"({self.response_window_size_samples}) to cover all samples!"
        )
        # Check for non-zero padding
        if (
            self.num_samples_per_block - self.response_window_size_samples
        ) % self.response_stride_samples != 0:
            log.warning(
                f"With sequence length {self.num_samples_per_block}, kernel size "
                f"({self.response_window_size_samples}), and stride "
                f"({self.response_stride_samples}), some samples will not be seen!"
            )
        if (
            self.num_samples_per_block - self.skip_n_samples
        ) % self.response_stride_samples != 0:
            log.warning(
                f"With sequence length {self.num_samples_per_block}, skip {self.skip_n_samples} samples, and stride"
                f"({self.response_stride_samples}), some samples will not be reconstructed!"
            )

        # Verify all literal fields with Literal type hints
        if self.local_global_layout not in (
            supp := SupportedLocalGlobalLayout.__args__
        ):
            raise ValueError(
                f"local_global_layout must be one of {supp}, "
                f"got '{self.local_global_layout}'"
            )
        if self.norm_type not in (supp := SupportedNorm.__args__):
            raise ValueError(
                f"norm_type must be one of {supp}, "
                f"or 'softplus', got '{self.norm_type}'"
            )
        if self.ffn_activation not in (supp := SupportedActivation.__args__):
            raise ValueError(
                f"ffn_activation must be one of {supp}, got '{self.ffn_activation}'"
            )
        if self.init_depth_scaling is not None:
            if self.init_depth_scaling not in (supp := SupportedDepthInit.__args__):
                raise ValueError(
                    f"init_depth_scaling must be one of {supp}, "
                    f"got '{self.init_depth_scaling}'"
                )

        # Make sure masking strategies are tuples
        at_least_tuple = lambda x: (x,) if not isinstance(x, Sequence) else tuple(x)
        self.masking_strategies = at_least_tuple(self.masking_strategies)
        # TODO: This is a HACK-ey fix, because Hydra for some reason fails to convert the
        #  list of DictConfig's to MaskingStrategy instances, so we do it manually here.
        self.masking_strategies = tuple(
            map(omegaconf_to_masking_strategy, self.masking_strategies)
        )

        # Convert OmegaConf.ListConfig to Tuple's
        self.video_resolution = tuple(self.video_resolution)
        self.hiera_stages = tuple(self.hiera_stages)
        self.hiera_q_stride = tuple(self.hiera_q_stride)
        self.hiera_mask_unit_size = tuple(self.hiera_mask_unit_size)
        self.hiera_patch_kernel = tuple(self.hiera_patch_kernel)
        self.hiera_patch_stride = tuple(self.hiera_patch_stride)
        self.hiera_patch_padding = tuple(self.hiera_patch_padding)


@register_model("omnimouse")
class OmniMouseModel(Model):
    """
    Combines neural preprocessing, model inference, response regression and loss calculation.

    TODO: head pruning? (see :meth:`transformers.modeling_utils.PreTrainedModel.prune_heads`)
    TODO: Support label subsampling, i.e. attempting to predict subset of input timestamps/neurons to save compute (see :meth:`OmniMouseModel.forward`)
    TODO: Support different video embedding architectures
    TODO: Implement custom caching of BlockMasks so that they can be added to the state dict
    TODO: Consider small positional embedding (i.e. sinusoidal) of absolute time, upprojected
     to d_model and added to neuron embeddings
    TODO: Rename "response" latent embeddings
    """

    def __init__(
        self,
        config: OmniMouseArgs,
        session_map: SessionMap,
        **data_kwargs: Any,
    ):
        super().__init__(config, session_map)

        flex_compile_config = config.flex_attention_compile_config or {}
        set_compiled_attention(flex_compile_config)

        # Calculate context window lengths, in seconds (used for response latents and rotary embeddings)
        response_context_len_s = config.num_samples_per_block / config.num_samples_per_s
        video_context_len_s = config.num_frames_per_block / config.num_frames_per_s
        behavior_context_len_s = (
            config.num_behavior_samples_per_block / config.num_behavior_samples_per_s
        )
        self.context_window_len_s = max(
            response_context_len_s,
            video_context_len_s,
            behavior_context_len_s,
        )

        # --------------------- Per-session parameters ---------------------

        # Neuron/Session-specific (i.e. neural population, session embedding, etc.) embedding
        #  parameters, and up-projections from embedding to model dimension
        self.set_sess_params(session_map)
        # Up projection from session embedding to model dimension (added to latents and neuron queries)
        self.sess_to_model_proj = nn.Linear(
            in_features=config.d_embedding,
            out_features=config.d_model,
        )
        # Up projection from neuron embedding to model dimension (for neuron queries)
        self.neuron_to_model_proj = nn.Linear(
            in_features=config.d_embedding,
            out_features=config.d_model,
        )

        # Animal-specific (i.e. animal embedding) parameters and up-projections.
        #  NOTE: We create parameters for all sessions, even if they are not on the current rank,
        #   since animal embedding is shared across ranks.
        #  NOTE: We only add animal embeddings if multiple sessions use the same animal, otherwise
        #   we just use the session embedding.
        self.animal_params, self.animal_to_model_proj = None, None
        sess_to_animal = {sk: meta.animal_id for sk, meta in session_map.items()}
        add_animal_embeddings = len(set(sess_to_animal.values())) < len(sess_to_animal)
        if add_animal_embeddings or config.always_use_animal_embeddings:
            self.animal_params = AnimalParams(
                session_key_to_animal_id_map=sess_to_animal,
                d_embedding=config.d_embedding,
            )
            # Up projection from animal embedding to model dimension (for animal queries)
            self.animal_to_model_proj = nn.Linear(
                in_features=config.d_embedding,
                out_features=config.d_model,
            )
        # Absolute experiment time embedding
        self.exp_time_embedding = AbsoluteExperimentTimeEmbedding(
            d_model=config.d_embedding,
            max_recording_len_s=config.max_recording_len_s,
            temporal_precision_s=config.absolute_experiment_time_precision_s,
            # base=100000, # TODO: Would increasing the base help with positional resoultion?
        )
        # Up projection from experiment time embedding to model dimension (for experiment time queries)
        self.exp_time_to_model_proj = nn.Linear(
            in_features=config.d_embedding,
            out_features=config.d_model,
        )
        # Behavior channel embeddings (shared per-model)
        self.behavior_channel_embeddings = nn.Parameter(
            torch.zeros(config.behavior_channels, config.d_model)
        )
        # Missing modality embedding (shared per-model). NOTE: We use 3 channels to represent
        #  the three modalities: response, video, behavior.
        self.missing_modality_embeddings = nn.Parameter(
            torch.zeros(
                3,
                config.d_model,
            )
        )

        # --------------------- Learned response latent embeddings ---------------------

        # Response latent response embeddings + timestamps (shared per-model)
        self.response_latent_embedding = ResponseLatentEmbeddings(
            num_samples_per_block=config.num_samples_per_block,
            num_samples_per_s=config.num_samples_per_s,
            context_window_len_s=self.context_window_len_s,
            latent_window_size_samples=config.response_latent_window_size_samples,
            latent_window_stride_samples=config.response_latent_window_stride_samples,
            num_latent_groups=config.num_response_latent_groups,
            num_global_latents=config.num_global_latents,
            dim_latents=config.d_model,
        )

        # --------------------- Response feature extractor / readouts --------------------

        # Module for embedding neural responses
        self.response_featx = ResponseFeatureExtractor(
            d_model=config.d_model,
            kernel_size=config.response_window_size_samples,
            stride=config.response_stride_samples,
        )
        # Projection from response query outputs to output logits: (D,) -> (1,).
        #  NOTE: We set bias=False so that we can manually apply the per-neuron biases.
        self.response_readout_proj = nn.Linear(
            config.d_model, config.response_stride_samples, bias=False
        )
        # Output activation function. Defaults to ELU1.
        self.response_output_act = config.response_output_activation or ELU1()
        # NOTE: Default to Poisson NLL loss.
        # HACK: Override for default eps value of Poisson NLL only used when `response_criteria` not provided manually
        base_response_criteria = config.response_criteria or nn.PoissonNLLLoss(
            log_input=False, eps=config.response_poisson_loss_eps
        )
        self.response_criteria = ClippedLoss(
            base_criterion=base_response_criteria,
            magnitude=config.response_loss_clamp_value,
        )

        # ------------------------- Video feature extractor -------------------------

        # Hiera video processing backbone
        self.hiera = HieraFeatureExtractor(
            input_size=(config.num_frames_per_block, *config.video_resolution),
            num_heads=config.hiera_num_heads,
            embed_dim=config.hiera_embed_dim,
            stages=config.hiera_stages,
            q_pool=config.hiera_q_pool,
            in_chans=(
                config.video_channels + config.behavior_channels
                if config.behavior_as_channels
                else config.video_channels
            ),
            q_stride=config.hiera_q_stride,
            mask_unit_size=config.hiera_mask_unit_size,
            patch_kernel=config.hiera_patch_kernel,
            patch_stride=config.hiera_patch_stride,
            patch_padding=config.hiera_patch_padding,
            sep_pos_embed=config.hiera_sep_pos_embed,
            drop_path_rate=config.hiera_drop_path_rate,
            mlp_ratio=config.hiera_mlp_ratio,
            feature_dim=config.d_model,
            norm_eps=config.norm_eps,
        )

        # ------------------- Behavior feature extractor / readout ----------------------

        # Behavior feature extractor. See NOTE on behavior processing in `forward`.
        self.behavior_featx = nn.Linear(
            config.num_behavior_samples_per_block, config.d_model
        )
        # NOTE: We create artificial timestamps buffer to avoid recomputing them in `forward`
        self.register_buffer(
            "behavior_timestamps", torch.zeros(config.behavior_channels)
        )
        # Module for projecting behavior queries to predictions. NOTE: We do a different
        #  projection for each channel.
        behavior_readout_channels = (
            config.behavior_channels if config.per_channel_behavior_readout else 1
        )
        self.behavior_readout_proj = nn.Parameter(
            torch.zeros(
                behavior_readout_channels,  # per-channel dim
                config.num_behavior_samples_per_block,  # out features
                config.d_model,  # in features
            )
        )
        # self.behavior_readout_bias = nn.Parameter(torch.zeros(
        #    1, # broadcastable batch dim
        #    behavior_readout_channels, # per-channel dim
        #    config.num_behavior_samples_per_block, # out features
        # ))
        # Output activation function for behavior logits. Defaults to raw logits.
        self.behavior_output_act = config.behavior_output_activation or nn.Identity()
        # Loss function for behavior predictions. Default to MSELoss
        base_behavior_criteria = config.behavior_criteria or nn.MSELoss()
        self.behavior_criteria = ClippedLoss(
            base_criterion=base_behavior_criteria,
            magnitude=config.behavior_loss_clamp_value,
        )

        # --------------------- Backbone + additional attributes ---------------------

        # Rotary embedding hyperparameters
        self.rotary_emb = RotaryEmbedding(
            dim_rot=(config.d_model // config.num_heads),
            context_window_len_s=self.context_window_len_s,
            use_radians=config.rope_use_radians,
            base=config.rope_base,
        )
        # Block hyperparameters
        block_kwargs = dict(
            num_heads=config.num_heads,
            dim_head=(config.d_model // config.num_heads),
            ffn_hidden_mult=config.ffn_hidden_mult,
            ffn_activation=config.ffn_activation,
            norm_type=config.norm_type,
            use_pre_norm=config.use_pre_norm,
            use_qk_norm=config.use_qk_norm,
            use_post_norm=config.use_post_norm,
            norm_eps=config.norm_eps,
        )
        # Drop path rate schedule
        dpr, depth = config.drop_path_rate, config.num_self_attends_per_block
        dpr_schedule = (
            [x.item() for x in torch.linspace(0.0, dpr, depth)]
            if config.drop_path_depth_decay
            else [dpr] * depth
        )
        # Response X-attn encoder (cross-attention -> FFN) block
        self.enc = RotaryTransformerBlock(
            dim=config.d_model,
            is_cross_attention=True,
            drop_path_rate=dpr,  # TODO: Should we zero-out dpr here when `drop_path_depth_decay == True`?
            **block_kwargs,
        )
        # Multimodal latent space processing (self-attention -> FFN) tower
        self.proc = nn.ModuleList(
            [
                RotaryTransformerBlock(
                    dim=config.d_model,
                    is_cross_attention=False,
                    num_kv_heads=config.num_kv_heads,  # NOTE: GQA is only used for self-attention
                    drop_path_rate=dpr_schedule[i],
                    **block_kwargs,
                )
                for i in range(config.num_self_attends_per_block)
            ]
        )
        # Update FFN hidden multiplier for decoder, to optionally use larger multiplier
        block_kwargs["ffn_hidden_mult"] = (
            config.decoder_ffn_hidden_mult or config.ffn_hidden_mult
        )
        # Multimodal X-attn decoder (cross-attention -> FFN) block(s). TODO: Add support for more than 2 modality decoders
        n_decs = 1 if not config.per_modality_decoders else 2
        self.decs = nn.ModuleList([])
        for _ in range(n_decs):
            self.decs.append(
                RotaryTransformerBlock(
                    dim=config.d_model,
                    is_cross_attention=True,
                    drop_path_rate=dpr,
                    **block_kwargs,
                )
            )

        # Determine local/global layer distribution and construct local/global schedule
        #  Fall back to all local if no ratio is provided.
        local_to_global_ratio = config.local_to_global_ratio or (1, 0)
        #  If either value is zero, set the other to one.
        if not all(local_to_global_ratio):
            local_to_global_ratio = tuple(map(bool, local_to_global_ratio))
        #  Calculate cycle size, i.e. pattern to be repeated
        cycle_size = sum(local_to_global_ratio)
        assert 0 < cycle_size <= config.num_self_attends_per_block, (
            "Sum of local/global ratio must be between 1 and `num_self_attends_per_block`!"
        )
        n_locals, n_globals = local_to_global_ratio
        #  Calculate number of times to repeat pattern
        n_cycles = config.num_self_attends_per_block // cycle_size
        #  If cycle size doesn't evenly divide `num_self_attends_per_block`,
        #   we prepend a prefix of purely local layers.
        local_prefix = config.num_self_attends_per_block % cycle_size
        #  Construct local/global schedule, i.e. `True if local else False for _ in
        #   range(num_self_attends_per_block)`
        if config.local_global_layout == "interleaved":
            #  Interleaved pattern: [L * n_locals, G * n_globals, L * n_locals, ..., G * n_globals, ...]
            local_global_schedule = (1,) * local_prefix + (
                (1,) * n_locals + (0,) * n_globals
            ) * n_cycles
        elif config.local_global_layout == "local_first":
            #  Local-first pattern: [L * n_locals * n_cycles, ..., G * n_globals * n_cycles]
            local_global_schedule = (
                (1,) * local_prefix
                + (1,) * n_locals * n_cycles
                + (0,) * n_globals * n_cycles
            )
        else:
            raise ValueError(
                f"Invalid local/global layout: '{config.local_global_layout}'"
            )
        self.local_global_schedule = local_global_schedule
        self.use_global_nope = config.use_global_nope

        # Calculate modality window sizes, in seconds (used for constructing attention masks)
        self.response_window_len_s = (
            config.response_window_size_samples / config.num_samples_per_s
        )
        self.response_query_lookback_s = (
            config.response_query_lookback_samples / config.num_samples_per_s
        )
        # NOTE: Response query "lookahead" (i.e. window size) defaults to response window size
        self.response_query_lookahead_s = self.response_window_len_s
        if (rqls := config.response_query_lookahead_samples) is not None:
            self.response_query_lookahead_s = rqls / config.num_samples_per_s
        # NOTE: Video window size determined by receptive field of output Hiera features
        self.video_window_len_s = (
            self.hiera.final_feat_frames_window_size / config.num_frames_per_s
        )
        # NOTE: Behavior can attent to full context window
        self.behavior_window_len_s = (
            config.num_behavior_samples_per_block / config.num_behavior_samples_per_s
        )

        # Extract masking strategies to sample and sampling weights from config
        self.masking_strategies = config.masking_strategies
        self.masking_strategy_sampling_weights = tuple(
            strategy.weight for strategy in self.masking_strategies
        )

        # Cache of temporal block masks
        self._block_mask_cache: Dict[
            Hashable,
            Tuple[
                Optional[BlockMask],
                BlockMask,
                Tuple[Optional[BlockMask], ...],
            ],
        ] = {}

        # Initialize model weights
        self.init_weights()

    def _init_sess_params(
        self,
        sess_params: nn.ModuleDict,
        init_std: Optional[float] = None,
    ) -> nn.ModuleDict:
        for sp in sess_params.values():
            sp.init_weights(init_std)
        return sess_params

    def set_sess_params(
        self,
        session_map: Optional[SessionMap] = None,
        on_rank_only: bool = True,
        init_weights: bool = False,
        init_std: Optional[float] = None,
    ):
        """
        Uitlity for reconstructing session parameters from a session map.

        Args:
            session_map: Session map. If not provided, use the model's session map.
            on_rank_only: Whether to only initialize session parameters on the current rank
            init_weights: Whether to initialize the session parameters
        """
        session_map = session_map or self.session_map
        self.sess_params = nn.ModuleDict()
        for sk, meta in session_map.items():
            if meta.on_rank or not on_rank_only:
                self.sess_params[sk] = SessionParams(
                    neural_population_size=meta.n_neurons,
                    d_embedding=self.c.d_embedding,
                    bias=True,
                )
        if init_weights:
            self._init_sess_params(self.sess_params, init_std)

    def select_session_params(self, session_keys: str | Sequence[str]) -> nn.ModuleDict:
        """
        Select session parameters for a given session key or list of session keys.

        Args:
            session_keys: Session key or list of session keys to select
        """
        if isinstance(session_keys, str):
            session_keys = [session_keys]
        return nn.ModuleDict({sk: self.sess_params[sk] for sk in session_keys})

    def init_weights(self):
        # Override `init_std` for specific components (if specified)
        embeddings_init_std = self.c.embeddings_init_std or self.c.init_std
        embed_to_model_proj_init_std = (
            self.c.embed_to_model_proj_init_std or self.c.init_std
        )
        shared_embeddings_init_std = (
            self.c.shared_embeddings_init_std or self.c.init_std
        )
        response_proj_init_std = self.c.response_proj_init_std or self.c.init_std
        behavior_proj_init_std = self.c.behavior_proj_init_std or self.c.init_std
        hiera_proj_init_std = self.c.hiera_proj_init_std or self.c.init_std
        readout_init_std = self.c.readout_init_std or self.c.init_std
        # Session-specific parameters
        self._init_sess_params(
            self.sess_params,
            embeddings_init_std,
        )
        # Animal-specific parameters
        if self.animal_params is not None:
            self.animal_params.init_weights(embeddings_init_std)
        # Embedding->model projections
        init_weights_util(
            (
                self.sess_to_model_proj,
                self.neuron_to_model_proj,
                self.animal_to_model_proj,
                self.exp_time_to_model_proj,
            ),
            embed_to_model_proj_init_std,
        )
        # Behavior embeddings (per-model):
        init_weights_util(self.behavior_channel_embeddings, shared_embeddings_init_std)
        # Missing modality embeddings (per-model):
        init_weights_util(self.missing_modality_embeddings, shared_embeddings_init_std)
        # Response latent embeddings (per-model):
        self.response_latent_embedding.init_weights(shared_embeddings_init_std)
        # Response embedding and readout. TODO: Avoid hard-coded init std dev for `nn.Conv1d`.
        self.response_featx.init_weights(response_proj_init_std)
        init_weights_util(self.response_readout_proj, readout_init_std)
        # Hiera feature extractor (if enabled)
        self.hiera.init_weights(hiera_proj_init_std)
        # Behavior embedding and readout
        init_weights_util(self.behavior_featx, behavior_proj_init_std)
        init_weights_util(self.behavior_readout_proj, readout_init_std)
        # Initialize behavior readout bias to zero
        # self.behavior_readout_bias.data.zero_()
        # Rotary embedding
        self.rotary_emb.init_weights()
        # Response X-attn encoder
        self.enc.init_weights(self.c.init_std)
        # Multimodal self-attn blocks
        depth = len(self.proc) if self.c.init_depth_scaling == "global" else None
        for i, block in enumerate(self.proc):
            depth = i if self.c.init_depth_scaling == "current" else depth
            block.init_weights(init_std=self.c.init_std, depth=depth)
        # Multimodal X-attn decoder
        [dec.init_weights(self.c.init_std) for dec in self.decs]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def shared_named_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        """Parameters which must be shared across all ranks"""
        for n, p in self.named_parameters():
            if "sess_params" not in n:
                yield n, p

    def shared_named_buffers(self) -> Iterable[Tuple[str, Tensor]]:
        """Registered buffers which must be shared across all ranks"""
        for n, b in self.named_buffers():
            if "sess_params" not in n:
                yield n, b

    def rank_named_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        """Parameters which are unique to each rank"""
        for n, p in self.named_parameters():
            if "sess_params" in n:
                yield n, p

    def rank_named_buffers(self) -> Iterable[Tuple[str, Tensor]]:
        """Registered buffers which are unique to each rank"""
        for n, b in self.named_buffers():
            if "sess_params" in n:
                yield n, b

    def core_named_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        """Named parameters which should be frozen for fine-tuning"""
        for n, p in self.shared_named_parameters():
            # Exclude animal parameters from being frozen
            if "animal_params" not in n:
                yield n, p

    def _sample_and_process_masking_strategies(
        self,
        neural_population_size: int,
        num_samples_per_block: int,
        num_frames_per_block: int,
        masking_override: Optional[MaskingStrategy] = None,
    ) -> MaskingStrategy:
        """
        Sample a masking strategy and realize its parameters based on actual data dimensions.

        Args:
            neural_population_size: Total number of neurons in the population
            num_samples_per_block: Total number of response samples in the block
            num_frames_per_block: Total number of video frames in the block
            masking_override: Optional joint masking strategy to use

        Returns:
            Realized MaskingStrategy instance
        """
        # Use provided strategy or sample one using weighted sampling
        strategy = (
            masking_override
            or random.choices(
                population=self.masking_strategies,
                weights=self.masking_strategy_sampling_weights,
            )[0]
        )

        # Re-instantiate the strategy with concrete values
        strategy = strategy.instantiate(
            neural_population_size=neural_population_size,
            num_samples_per_block=num_samples_per_block,
            num_frames_per_block=num_frames_per_block,
        )

        return strategy

    def _get_missing_modality_embedding(
        self,
        batch_size: int,
        n_visible_neurons: int,
        n_visible_frames: int,
        behavior_encoded: bool,
    ) -> Float[Tensor, "batch d_model"]:
        response_missing, video_missing, behavior_missing = None, None, None
        if n_visible_neurons == 0:
            response_missing = self.missing_modality_embeddings[0]
        if n_visible_frames == 0:
            video_missing = self.missing_modality_embeddings[1]
        if not behavior_encoded:
            behavior_missing = self.missing_modality_embeddings[2]
        missing_modality_embedding = add_notnone(
            (
                response_missing,
                video_missing,
                behavior_missing,
            )
        )
        if missing_modality_embedding is None:
            return None
        return missing_modality_embedding.expand(batch_size, 1, -1)

    def _preprocess_input_responses(
        self,
        responses: Float[Tensor, "batch seq n_neurons"],
        timestamps: Float[Tensor, "batch seq n_neurons"],
        neuron_embeddings: Float[Tensor, "batch n_neurons d_embedding"],
        neuron_ids: Int[Tensor, "batch n_neurons"],
        sample_positions: Int[Tensor, "batch seq"],
        masking_strategy: MaskingStrategy,
        skip_n_samples: int = 0,
        interpolation_buffer: int = 0,
    ) -> Tuple[
        Float[Tensor, "batch unmasked_seq d_model"],  # unmasked embeds
        Float[Tensor, "batch unmasked_seq"],  # unmasked timestamps
        Float[Tensor, "batch masked_seq"],  # masked labels
        Float[Tensor, "batch masked_seq"],  # masked timestamps
        Int[Tensor, "batch masked_seq"],  # masked neuron IDs
        Int[Tensor, "batch masked_seq"],  # masked positions
        Float[Tensor, "batch masked_seq d_model"],  # masked queries
    ]:
        """
                    Preprocess inputs by handling different masking configurations.

                    This method implements the general masking approach using multiple parameters:
                     - n_visible_neurons: Number of neurons visible to encoder (population masking)
                     - prefix_start_idx: Start position of visible prefix samples
                     - prefix_len: Length of visible prefix samples (causal masking)
                     - full_population_prefix_len: Portion of prefix for which all neurons are visible,
                         irrespective of `n_visible_neurons`.
                     - response_context_start_idx: Start position of "response context window". Response
                         context window is used to determine which samples are available for encoding, and
                         therefore which "latents" must be made available. The prefix region must be a subset
                         of the response context window. See :mod:`omnimouse.modeling.model` for more details.
                     - max_response_context_samples: Maximum samples in response context
                     - max_n_reconstructed_neurons: Upper bound on number of neurons for reconstruction.
                     - suffix_start_idx: Start position of reconstruction period
                     - suffix_len: Length of reconstruction period

                    Ultimately in the model, we flatten the sequence/neuron dimensions such that all
                     neurons with the same position are consecutive, hence we extract blocks with
                     different num neurons-per-position separately, and then stitch them together.
                     For example, in the general case diagramed below, we process the block by:

                    1. Construct model inputs (i.e. embeddings and timestamps) for full population
                     visible prefix region samples, if full population preifix is specified, and
                     where the remainder of the prefix will have less than the full population visible.
                    2. Construct inputs for remainder of the prefix region samples, for subset of neurons
                     which are designated to be visible. The resulting tensors (which have been flattened)
                     will be concatenated with the full population prefix tensors (if they exist)
                    3. Construct output queries and reconstruction targets for region of suffix
                     which overlaps in time with the prefix region where some subset of the neurons
                     available for reconstruction are already included in the visible prefix.
                    4. Construct output queries and reconstruction targets for region of suffix where
                     all neurons < `max_n_reconstructed_neurons` are reconstructed, and concatenate
                     with result from the previous region, if it exists.
                   ```
                                        num_samples_per_block
                                        <--------------------------------------------------------------->
                                                ┌> response_context_start_idx
                                                : max_response_context_samples
                                                :<---------------------------------------->:
                                                :   ┌> prefix_start_idx                    :
                                                :   : prefix_len                           :
                                                :   :<----------------------------->:      :
                                                :   : full_population_prefix_len    :      :
                                                :   :<---------->                   :      :
                       A            A  ┌────────────┬──────────────────────-─–─–––––┬─────────–───────────┐
                       │  n_visible │  │        :   │ ########## : ################ │      :              │
                       │  _neurons  │  │        :   │ ########## : #    prefix    # │      :              │
                       │            V  │        :   │ # full_  # : ################ │      :              │
         neural_       │               │        :   │ # popula # ┌──────────┬───-––─┴—————————————————————┼
        population_    │            A  │        :   │ #  tion_ # │          │ ########################### │
          size         │            │  │        :   │ # prefix # │          │ ########################### │
                       │  n_max_rec │  │        :   │ ########## │          │ #          suffix         # │
                       │  onstructe │  │        :   │ ########## │          │ ########################### │
                       │  d_neurons V  │        :   | ########## │          | ########################### │
                       V               └────────────┴────────────┴──────────┴───––––––-─––────────────────┘
                                                                            : <--------------------------->
                                                                            :          suffix_len
                                                                            └> suffix_start_idx
            ```
                    For each section, we use slices to extract and process the relevant parts independently.
                    When only certain masking parameters are active, some sections may be skipped entirely.

                    To capture population masking using the region-slicing, we first shuffle the
                        neurons in each *batch item* such that which neurons are visible is random.

                    We also enforce the `skip_n_samples` parameter, to remove the first `skip_n_samples`
                    samples from reconstruction targets (since they aren't predictable from video),
                    as well as the `interpolation_buffer` parameter, to avoid reconstructing samples
                    which are simply the result of interpolating between underlying samples.

                    Args:
                        session_key: Key identifying the session
                        responses: Neural response tensor
                        timestamps: Timestamps for each response
                        neuron_ids: Neuron IDs
                        meta_embedding: Optional embedding for session/animal/etc., which will be
                         added to neuron embeddings if provided
                        n_visible_neurons: Number of neurons to keep visible (population masking)
                        prefix_len: Number of initial samples to keep visible (causal masking)
                        skip_n_samples: Number of samples to skip in the masked region

                    Returns:
                        Preprocessed data combining all visible and masked sections
        """
        # Tensor shapes
        B, S, N = responses.shape

        # Get masking strategy parameters
        psi = masking_strategy.prefix_start_idx
        pl = masking_strategy.prefix_len
        pe = psi + pl  # prefix end
        rcs = masking_strategy.response_context_start_idx
        mrc = masking_strategy.max_response_context_samples
        rce = rcs + mrc  # response context end
        fppl = masking_strategy.full_population_prefix_len
        fpe = psi + fppl  # full population prefix end
        ssi = masking_strategy.suffix_start_idx
        sl = masking_strategy.suffix_len
        se = ssi + sl  # suffix end
        nvn = min(masking_strategy.n_visible_neurons, N)
        mnrn = min(masking_strategy.max_n_reconstructed_neurons, N)

        # Check that encoding/decoding configuration is valid
        assert 0 <= rcs <= rce <= S, (
            f"Response context ({rcs} - {rce}) must be contained within full window ({0} - {S})!"
        )
        assert rcs <= psi <= pe <= rce, (
            f"Prefix ({psi} - {pe}) must be contained within response context ({rcs} - {rce})!"
        )
        assert fpe <= pe, (
            f"Full population prefix ({psi} - {fpe}) must be contained within prefix ({psi} - {pe})!"
        )
        assert 0 <= ssi <= se <= S, (
            f"Suffix ({ssi} - {se}) must be contained within full window ({0} - {S})!"
        )

        # Define helper function to construct embeddings for a given subset
        def _construct_response_subset_embeddings(
            sample_slice: slice, neuron_slice: slice
        ) -> Tuple[
            Float[Tensor, "batch tok_s*n_neurons_s*stride_s"],  # labels
            Float[Tensor, "batch tok_s*n_neurons_s*stride_s"],  # timestamps
        ]:
            responses_s = responses[:, sample_slice, neuron_slice]
            timestamps_s = timestamps[:, sample_slice, neuron_slice]
            neuron_embeddings_s = neuron_embeddings[:, neuron_slice, :]
            _, seq_s, n_neurons_s = responses_s.shape
            stride_s, kernel_size = (
                self.c.response_stride_samples,
                self.c.response_window_size_samples,
            )
            if 0 in (seq_s, n_neurons_s):
                raise ValueError(
                    f"Empty slice (seq_s={seq_s}, n_neurons_s={n_neurons_s})"
                )
            if seq_s < kernel_size:
                raise ValueError(
                    f"Insufficient samples (n_samples={seq_s} < kernel={kernel_size})"
                )
            if (seq_s - kernel_size) % stride_s != 0:
                raise ValueError(
                    "Sequence length is not divisible by stride, responses will be dropped!"
                )
            return self.response_featx(
                responses_s,
                timestamps_s,
                neuron_embeddings_s,
            )

        # Encode unmasked samples
        unmasked_embeds_timestamps = None
        # "full population prefix"
        if fppl > 0 and 0 < nvn < N:
            unmasked_sample_slice = slice(psi, fpe)
            unmasked_neuron_slice = slice(0, None)
            unmasked_embeds_timestamps = _construct_response_subset_embeddings(
                unmasked_sample_slice,
                unmasked_neuron_slice,
            )
        # `n_visible` prefix
        if pl > 0 and nvn > 0 and fpe < pe:
            unmasked_sample_slice = slice(fpe, pe)
            unmasked_neuron_slice = slice(0, nvn)
            unmasked_embeds_timestamps_ar = _construct_response_subset_embeddings(
                unmasked_sample_slice,
                unmasked_neuron_slice,
            )
            # combine with full population prefix, if it exists
            if unmasked_embeds_timestamps is not None:
                unmasked_embeds_timestamps = map(
                    lambda x: torch.cat((x[0], x[1]), dim=1),
                    zip(unmasked_embeds_timestamps, unmasked_embeds_timestamps_ar),
                )
            else:
                unmasked_embeds_timestamps = unmasked_embeds_timestamps_ar
        # Create placeholder for unmasked embeds and timestamps if no unmasked samples
        if unmasked_embeds_timestamps is None:
            unmasked_embeds_timestamps = (None, None)

        # Define helper function to construct labels and queries for a given subset
        def _construct_response_subset_labels_and_queries(
            sample_slice: slice,
            neuron_slice: slice,
        ) -> Tuple[
            Float[Tensor, "batch tok_s*n_neurons_s*stride_s"],  # labels
            Float[Tensor, "batch tok_s*n_neurons_s*stride_s"],  # timestamps
            Float[Tensor, "batch tok_s*n_neurons_s d_model"],  # queries
            Float[Tensor, "batch tok_s*n_neurons_s d_model"],  # query timestamps
            Int[Tensor, "batch tok_s*n_neurons_s"],  # query neuron IDs
            Int[Tensor, "batch tok_s*n_neurons_s*stride_s"],  # (label) positions
            Int[Tensor, "batch seq_s*n_neurons_s*stride_s"],  # (label) neuron IDs
        ]:
            # Extract masked samples/info
            responses_s = responses[:, sample_slice, neuron_slice]
            timestamps_s = timestamps[:, sample_slice, neuron_slice]
            neuron_embeddings_s = neuron_embeddings[:, neuron_slice, :]
            neuron_ids_s = neuron_ids[:, neuron_slice]
            positions_s = sample_positions[sample_slice]
            # selected tensor's shapes
            batch_s, seq_s, n_neurons_s = responses_s.shape
            stride_s = self.c.response_stride_samples
            if 0 in (seq_s, n_neurons_s):
                raise ValueError(
                    f"Empty slice (seq_s={seq_s}, n_neurons_s={n_neurons_s})"
                )
            if seq_s % stride_s != 0:
                raise ValueError("Sequence length is not divisible by stride")
            tok_s = seq_s // stride_s
            # Rearange strided groups and flatten sequence & population dimensions of responses/timestamps:
            #  (batch_s, seq_s, n_neurons_s) -> (batch_s, tok_s, stride_s, n_neurons_s) ->
            #  (batch_s, tok_s*n_neurons_s, stride_s). NOTE: Keeps neurons with same position
            #  consecutive.
            subset_labels = (
                responses_s.reshape(batch_s, tok_s, stride_s, n_neurons_s)
                .permute(0, 1, 3, 2)
                .reshape(batch_s, tok_s * n_neurons_s * stride_s)
            )
            subset_timestamps = (
                timestamps_s.reshape(batch_s, tok_s, stride_s, n_neurons_s)
                .permute(0, 1, 3, 2)
                .reshape(batch_s, tok_s * n_neurons_s * stride_s)
            )
            # Repeat neuron embeddings for each sample position to use as queries:
            #  (batch_s, n_neurons_s, d_model) -> (batch_s, seq_s*n_neurons_s, d_model)
            subset_queries = neuron_embeddings_s.repeat(1, tok_s, 1)
            # Extract every `stride_s` timestamp: (batch_s, seq_s*neurons_s) -> (batch_s, tok_s*neurons_s)
            subset_query_timestamps = subset_timestamps[:, ::stride_s].contiguous()
            # Repeat neuron IDs for each token position: (batch_s, n_neurons_s) ->
            #  (batch_s, seq_s*n_neurons_s)
            subset_query_neuron_ids = neuron_ids_s.repeat(1, tok_s)
            # Repeat (interleaved) sample positions for each neuron and create batch
            #  dimension: (seq_s) -> (seq_s*n_neurons_s) -> (batch_s, seq_s*n_neurons_s)
            subset_positions = (
                positions_s.view(tok_s, stride_s)
                .repeat_interleave(n_neurons_s, dim=0)
                .view(-1)
                .expand(batch_s, -1)
            )
            # Repeat interleave neuron IDs for each token group: (batch_s, seq_s*n_neurons_s) ->
            #  (batch_s, seq_s*n_neurons_s*stride_s)
            subset_neuron_ids = subset_query_neuron_ids.repeat_interleave(
                stride_s, dim=-1
            )
            # TODO: Return dataclass encapsulation of preprocessed slices
            return (
                subset_labels,
                subset_timestamps,
                subset_queries,
                subset_query_timestamps,
                subset_query_neuron_ids,
                subset_positions,
                subset_neuron_ids,
            )

        # Decode masked samples
        if sl == 0 or mnrn == 0:  # if no masked samples, fill in with `None`s
            labels_and_queries = (None, None, None, None, None, None, None)
        else:
            # Verify that we account for skip_n_samples buffer
            if ssi < skip_n_samples:
                raise ValueError("Suffix start index is less than skip_n_samples")
            # We select the last `max_n_reconstructed_neurons` neurons to reconstruct,
            n_start = N - mnrn
            # Adjust prefix buffers for interpolated *visible* neuron samples
            fpe_plus_ib = min(fpe + interpolation_buffer, S) if fppl > 0 else 0
            pe_plus_ib = min(pe + interpolation_buffer, S) if pl > 0 else 0
            # if the visible population subset doesn't overlap with the reconstructed
            #  poplation, or if the suffix doesn't overlap with the prefix range, we can
            #  reconstruct the continuous block of samples (for `mnrn` neurons) from
            #  suffix start (or the interpolation buffer of the full population prefix,
            #  if it exists) to the suffix end.
            if n_start >= nvn or ssi >= pe_plus_ib:
                masked_sample_slice = slice(max(ssi, fpe_plus_ib), se)
                masked_neuron_slice = slice(n_start, None)
                labels_and_queries = _construct_response_subset_labels_and_queries(
                    masked_sample_slice,
                    masked_neuron_slice,
                )
            # if the visible population subset does overlap with the reconstructed
            #  poplation, we first reconstruct the non-visible neurons within the prefix
            #  range. NOTE: We flatten the sequence/neuron dimensions such that all
            #  neurons with the same position are consecutive, hence we require two
            #  sections since different positions will have a different number of neurons.
            else:
                masked_sample_slice = slice(max(ssi, fpe_plus_ib), min(pe_plus_ib, se))
                masked_neuron_slice = slice(nvn, None)
                labels_and_queries = _construct_response_subset_labels_and_queries(
                    masked_sample_slice,
                    masked_neuron_slice,
                )
                # if the suffix range extends beyond the prefix range, we reconstruct all
                #  `mnrn` neurons for the remaining suffix samples.
                if se > pe_plus_ib:
                    masked_sample_slice = slice(pe_plus_ib, se)
                    masked_neuron_slice = slice(n_start, None)
                    labels_and_queries_ = _construct_response_subset_labels_and_queries(
                        masked_sample_slice,
                        masked_neuron_slice,
                    )
                    # combine with existing result
                    labels_and_queries = map(
                        lambda x: torch.cat((x[0], x[1]), dim=1),
                        zip(labels_and_queries, labels_and_queries_),
                    )

        return (*unmasked_embeds_timestamps, *labels_and_queries)

    def _generate_cache_key(
        self,
        session_key: str,
        masking_strategy: MaskingStrategy,
    ) -> Hashable:
        """
        Create a hashable/serializable key for the block mask cache.

        NOTE: Since all the arguments are primitive types, we can just use a tuple
          of the arguments as the key.
        """
        return (
            session_key,
            *masking_strategy.parameters_tuple(),
        )

    def _create_temporal_block_masks_cached(
        self,
        feat_timestamps: Float[Tensor, "batch seq_f"] | None,
        feat_window_ends: Float[Tensor, "batch seq_f"] | None,
        latent_timestamps: Float[Tensor, "batch seq_l"],
        latent_window_ends: Float[Tensor, "batch seq_l"],
        screen_feat_timestamps: Float[Tensor, "batch seq_s"] | None,
        screen_feat_window_ends: Float[Tensor, "batch seq_s"] | None,
        query_timestamps: Tuple[Float[Tensor, "batch seq_q"], ...],
        query_window_ends: Tuple[Float[Tensor, "batch seq_q"], ...],
        are_batch_items_identical: bool = True,
        cache_key: Optional[Hashable] = None,
    ) -> Tuple[Optional[BlockMask], BlockMask, Tuple[Optional[BlockMask], ...]]:
        """
        Create temporal block masks for encoder, latents, and decoder.
        """
        # If cache key is provided, and block mask is cached, return cached block mask
        if cache_key is not None and cache_key in self._block_mask_cache:
            return self._block_mask_cache[cache_key]
        # Timestamps for each batch are identical, so we can use the first batch item to
        #  create the block mask, which is more efficient
        if are_batch_items_identical:
            if feat_timestamps is not None:
                feat_timestamps = feat_timestamps[0]
                feat_window_ends = feat_window_ends[0]
            if screen_feat_timestamps is not None:
                screen_feat_timestamps = screen_feat_timestamps[0]
                screen_feat_window_ends = screen_feat_window_ends[0]
            latent_timestamps = latent_timestamps[0]
            latent_window_ends = latent_window_ends[0]
            query_timestamps = tuple(
                None if qts is None else qts[0] for qts in query_timestamps
            )
            query_window_ends = tuple(
                None if qwe is None else qwe[0] for qwe in query_window_ends
            )
        # Create BlockMask for perceiver response encoder (i.e. X-attn), if responses exist
        encoder_mask = None
        if feat_timestamps is not None:
            encoder_mask = create_multimodal_temporal_sliding_window_block_mask(
                q_window_bound_1=latent_timestamps,
                q_window_bound_2=latent_window_ends,
                kv_window_bound_1=feat_timestamps,
                kv_window_bound_2=feat_window_ends,
                device=self.device,
                BLOCK_SIZE=self.c.flex_attn_block_size,
            )
        # Latents consist of response latent space and multimodal context, so we concatenate
        #  the timestamps and window ends along the sequence dimension. NOTE: We order the sequence
        #  dimension such that video features come first, then response latents + globals, then
        #  behavior features, leading to optimal block allignment for flex-attention.
        seq_dim = 0 if are_batch_items_identical else 1
        latent_timestamps = concat_notnone(
            (
                latent_timestamps,
                screen_feat_timestamps,
            ),
            dim=seq_dim,
        )
        latent_window_ends = concat_notnone(
            (
                latent_window_ends,
                screen_feat_window_ends,
            ),
            dim=seq_dim,
        )
        # Create BlockMask for latent space self-attention's. NOTE: queries = keys/values
        latent_space_mask = create_multimodal_temporal_sliding_window_block_mask(
            q_window_bound_1=latent_timestamps,
            q_window_bound_2=latent_window_ends,
            device=self.device,
            BLOCK_SIZE=self.c.flex_attn_block_size,
        )
        # Create BlockMask for perceiver decoder(s) (i.e. X-attn)
        decoder_masks = []
        for qts, qwe in zip(query_timestamps, query_window_ends):
            if qts is not None:
                mask = create_multimodal_temporal_sliding_window_block_mask(
                    q_window_bound_1=qts,
                    q_window_bound_2=qwe,
                    kv_window_bound_1=latent_timestamps,
                    kv_window_bound_2=latent_window_ends,
                    device=self.device,
                    BLOCK_SIZE=self.c.flex_attn_block_size,
                )
                decoder_masks.append(mask)
            else:
                decoder_masks.append(None)
        decoder_masks = tuple(decoder_masks)
        # Construct return tuple
        block_masks = (encoder_mask, latent_space_mask, decoder_masks)
        # Cache block masks if cache key is provided
        if cache_key is not None:
            self._block_mask_cache[cache_key] = block_masks
        return block_masks

    def forward(
        self,
        session_key: str,
        responses: Float[Tensor, "B S N"],
        timestamps: Float[Tensor, "B S N"],
        screen: Float[Tensor, "B C_vid S_vid H W"],
        screen_timestamps: Float[Tensor, "B S_vid"],
        behavior: Float[Tensor, "B C_beh S_beh"],
        behavior_timestamps: Float[Tensor, "B S_beh"],
        masking_override: Optional[MaskingStrategy] = None,
        all_neurons_override: Optional[bool] = None,
    ) -> OMModelOutput:
        """
        Forward pass.

        NOTE: We assume entire neural population of session is passed, and passed in the
         same order with every batch
        TODO: Support eval case where responses and timestamps not passed
         (and labels not needed).
        """
        # Tensor shapes
        (B, S, N), dev = responses.shape, responses.device
        _, C_vid, S_vid, H, W = screen.shape
        _, C_beh, S_beh = behavior.shape

        # Determine max number of neurons to consider
        max_neurons = N
        if (mn := self.c.max_population_size) is not None and not all_neurons_override:
            max_neurons = min(max_neurons, mn)
        if (
            nm := self.c.n_neurons_multiple_of
        ) is not None and not all_neurons_override:
            max_neurons = nm * (max_neurons // nm)

        # Sample modality masking behavior (or use provided strategies)
        strategy = self._sample_and_process_masking_strategies(
            neural_population_size=max_neurons,
            num_samples_per_block=S,
            num_frames_per_block=S_vid,
            masking_override=masking_override,
        )

        # Generate unique cache key for forward pass
        batch_cache_key = self._generate_cache_key(
            session_key=session_key,
            masking_strategy=strategy,
        )

        # Fetch session-specific parameters
        sess_params = self.sess_params[session_key]

        # -------------------- Build Session-Specific Embeddings --------------------
        # Get session-specific embeddings, and up-project to model dimension
        meta_embedding = sess_params.session_embedding  # (d_embedding)
        meta_embedding = self.sess_to_model_proj(
            meta_embedding
        )  # (d_embedding) -> (d_model)
        # Add animal embedding, if configured
        if self.animal_params is not None:
            animal_embedding = self.animal_params.animal_embedding(
                session_key
            )  # (d_embedding)
            animal_embedding = self.animal_to_model_proj(
                animal_embedding
            )  # (d_embedding) -> (d_model)
            meta_embedding = meta_embedding + animal_embedding  # (d_model) + (d_model)
        # Add "absolute" experiment time embedding, if configured
        if self.c.add_exp_time_embedding:
            # NOTE: We use the first timestamp of the first neuron from each batch item as
            #  experiment times, and add a singleton "sequence" dimension for broadcasting
            exp_time_embedding = self.exp_time_embedding(timestamps[:, 0, 0]).unsqueeze(
                1
            )  # (B, d_embedding) -> (B, 1, d_embedding)
            exp_time_embedding = self.exp_time_to_model_proj(
                exp_time_embedding
            )  # (B, 1, d_embedding) -> (B, 1, d_model)
            meta_embedding = (
                meta_embedding.expand(1, 1, -1) + exp_time_embedding
            )  # ((d_model) -> (1, 1, d_model)) + (B, 1, d_model) -> (B, 1, d_model)

        # -------------------- Responses Features / Queries ----------------------------
        # Generate random permutation of neuron IDs, and gather using the random indices
        if all_neurons_override is None or not all_neurons_override:
            neuron_ids = torch.argsort(torch.rand(B, N, device=dev), dim=-1)[
                ..., :max_neurons
            ]
            shuffle = neuron_ids.unsqueeze(1).expand(
                -1, S, -1
            )  # (B, max_neurons) -> (B, S, max_neurons)
            responses = torch.gather(responses, dim=-1, index=shuffle)
            timestamps = torch.gather(timestamps, dim=-1, index=shuffle)
        else:
            # if specifically requested, use all neurons in the population, no shuffling.
            neuron_ids = torch.arange(N, device=dev).unsqueeze(0).expand(B, -1)

        # Select neuron embeddings for neuron_ids: (B, n_visible_neurons) ->
        #  (B, n_visible_neurons, d_embedding)
        neuron_embeddings = sess_params.neuron_embeddings(neuron_ids)
        # Project neuron embeddings to model dimension: (B, max_neurons, d_embedding) -> (B, max_neurons, d_model)
        neuron_embeddings = self.neuron_to_model_proj(neuron_embeddings)
        # Add metadata embedding to neuron embeddings, if specified
        if self.c.add_meta_to_embeddings:
            # NOTE: We abuse the dimensionality of the metadata embedding to broadcast
            #  to the neuron dimension here, instead of the sequence dimension:
            #  (B, 1, d_model) -> (B, max_neurons, d_model)
            neuron_embeddings = neuron_embeddings + meta_embedding
        # Convert absolute timestamps to relative timestamps (i.e. subtract minimum
        #  timestamp per batch item): (B, max_neurons, S) -> (B, max_neurons, S)
        relative_timestamps = timestamps - timestamps.amin(dim=(1, 2), keepdim=True)
        # Construct relative "sample positions": (S,)
        sample_positions = torch.arange(S, device=dev)
        # Preprocess responses and timestamps. NOTE: `embeds` and `embed_timestamps`
        #  will be `None` if `process_responses` is False
        (
            response_feats,
            response_feat_timestamps,
            response_labels,
            response_label_timestamps,
            response_queries,
            response_query_timestamps,
            response_query_neuron_ids,
            response_label_positions,
            response_label_neuron_ids,
        ) = self._preprocess_input_responses(
            responses,
            relative_timestamps,
            neuron_embeddings,
            neuron_ids,
            sample_positions,
            masking_strategy=strategy,
            skip_n_samples=self.c.skip_n_samples,
            interpolation_buffer=self.c.interpolation_buffer,
        )

        # Calculate window ends for unmasked and masked responses
        (
            response_feat_window_ends,
            response_query_window_starts,
            response_query_window_ends,
        ) = None, None, None
        if response_feat_timestamps is not None:
            response_feat_window_ends = (
                response_feat_timestamps + self.response_window_len_s
            )
        n_response_queries = 0
        if response_queries is not None:
            # NOTE: We create window start to facilitate extended lookback for queries
            response_query_window_starts = (
                response_query_timestamps - self.response_query_lookback_s
            )
            response_query_window_ends = (
                response_query_timestamps + self.response_query_lookahead_s
            )
            n_response_queries = response_queries.size(
                1
            )  # track num queries belonging to responses

        # -------------------------- Video Features / Context --------------------------
        # Process video, if visible. NOTE: If `n_visible_frames` is not provided, we
        #  process video and use all frames.
        # TODO: Consider adding meta embedding to video features
        screen_feats, screen_feat_timestamps, screen_feat_window_ends = None, None, None
        if strategy.n_visible_frames > 0:
            screen_feats, screen_feat_timestamps = self.hiera(
                screen=screen,
                timestamps=screen_timestamps,
                n_visible_frames=strategy.n_visible_frames,
                visible_frames_start=strategy.n_visible_frames_start_idx,
            )
            if self.c.add_meta_to_video_feats:
                screen_feats = screen_feats + meta_embedding
            screen_feat_window_ends = screen_feat_timestamps + self.video_window_len_s

        # ----------------------- Behavior Features / Context ---------------------------
        # Process behavior: If visible, treat as multimodal context, else as decoder queries.
        n_behavior_queries = 0  # placeholder for num queries belonging to behavior
        behavior_embeddings = self.behavior_channel_embeddings.expand(
            B, -1, -1
        )  # (B, C_beh, d_model)
        if self.c.add_meta_to_embeddings:
            behavior_embeddings = behavior_embeddings + meta_embedding
        # NOTE: We use the same timestamp for all behavior features, each corresponding to
        #  the entire block
        behavior_timestamps = self.behavior_timestamps.expand(B, -1)  # (B, C_beh)
        behavior_window_ends = behavior_timestamps + self.behavior_window_len_s
        # Construct unmasked or masked behavior features/queries
        behavior_feats, behavior_feat_timestamps, behavior_feat_window_ends = (
            None,
            None,
            None,
        )
        behavior_queries, behavior_query_timestamps, behavior_query_window_ends = (
            None,
            None,
            None,
        )
        if strategy.behavior_encoded:
            # Transpose channel/temporal dimensions and project behavior along temporal
            #  dimension: (B, C_beh, S_beh) -> (B, C_beh, d_model)
            behavior_feats = self.behavior_featx(behavior)
            # Add channel embeddings: (B, C_beh, d_model) + (B, C_beh, d_model) -> (B, C_beh, d_model)
            behavior_feats = behavior_feats + behavior_embeddings
            behavior_feat_timestamps = behavior_timestamps
            behavior_feat_window_ends = behavior_window_ends
        elif strategy.behavior_reconstructed:
            # NOTE: We use the per-channel embeddings and timestamps directly
            behavior_queries = behavior_embeddings
            behavior_query_timestamps = behavior_timestamps
            behavior_query_window_ends = behavior_window_ends
            n_behavior_queries = behavior_queries.size(
                1
            )  # track num queries belonging to responses

        # Check that we are decoding at least some responses/behavior...
        assert any(
            x is not None
            for x in (
                response_queries,
                behavior_queries,
            )
        ), "Must decode at least a subset of one modality!"

        # ---------------------- Response/Global Latents -------------------------------
        # TODO: Rename since no longer just response latents
        # NOTE: Latents are always concatentated with multimodal context, even when all
        #  responses hidden
        # Prepare batched latents and latent timestamps/window sizes: (B, G*Z+O, D)
        latent_context_start = strategy.response_context_start_idx
        latent_context = strategy.max_response_context_samples
        response_latent_embeddings = self.response_latent_embedding.embeddings(
            B,
            latent_context,
            latent_context_start,
        )
        if self.c.add_meta_to_latents:
            response_latent_embeddings = response_latent_embeddings + meta_embedding
        response_latent_timestamps = self.response_latent_embedding.timestamps(
            B,
            latent_context,
            latent_context_start,
        )  # (B, G*Z+O)
        response_latent_window_ends = (
            response_latent_timestamps
            + self.response_latent_embedding.window_sizes(
                B,
                latent_context,
                latent_context_start,
            )  # (B, G*Z+O)
        )
        # Add missing modality embedding to latents if necessary: (B, G*Z+O, D) + (B, 1, D) -> (B, G*Z+O, D)
        missing_modality_embedding = self._get_missing_modality_embedding(
            batch_size=B,
            n_visible_neurons=strategy.n_visible_neurons,
            n_visible_frames=strategy.n_visible_frames,
            behavior_encoded=strategy.behavior_encoded,
        )
        if missing_modality_embedding is not None:
            response_latent_embeddings = (
                response_latent_embeddings + missing_modality_embedding
            )

        # --------------- Attention Masks & Backbone Processing ------------------------

        # Concatenate unmasked feats and timestamps for all modalities present
        unmasked_feats = concat_notnone(
            (
                response_feats,
                behavior_feats,
            )
        )
        unmasked_feat_timestamps = concat_notnone(
            (
                response_feat_timestamps,
                behavior_feat_timestamps,
            )
        )
        unmasked_feat_window_ends = concat_notnone(
            (
                response_feat_window_ends,
                behavior_feat_window_ends,
            )
        )

        # Construct masked queries and timestamps for all modalities present, and concatentate if using shared decoder
        masked_queries = (
            response_queries,
            behavior_queries,
        )
        masked_query_timestamps = (
            response_query_timestamps,
            behavior_query_timestamps,
        )
        masked_query_window_starts = (
            response_query_window_starts,
            behavior_query_timestamps,
        )
        masked_query_window_ends = (
            response_query_window_ends,
            behavior_query_window_ends,
        )
        # If using shared decoder, concatentate masked queries and timestamps and consturct len(1) tuple
        if not self.c.per_modality_decoders:
            masked_queries = (concat_notnone(masked_queries),)
            masked_query_timestamps = (concat_notnone(masked_query_timestamps),)
            masked_query_window_starts = (concat_notnone(masked_query_window_starts),)
            masked_query_window_ends = (concat_notnone(masked_query_window_ends),)
        else:
            assert len(masked_queries) == len(self.decs), (
                "Number of modalities and decoders must match for per-modality decoders!"
            )

        # Create FlexAttention BlockMask's (or retrieve from cache)
        (
            encoder_block_mask,
            latents_block_mask,
            decoder_block_masks,
        ) = self._create_temporal_block_masks_cached(
            feat_timestamps=unmasked_feat_timestamps,
            feat_window_ends=unmasked_feat_window_ends,
            latent_timestamps=response_latent_timestamps,
            latent_window_ends=response_latent_window_ends,
            screen_feat_timestamps=screen_feat_timestamps,
            screen_feat_window_ends=screen_feat_window_ends,
            # NOTE: We use explicitly created window starts instead of timestamps for "lookback"
            query_timestamps=masked_query_window_starts,
            query_window_ends=masked_query_window_ends,
            are_batch_items_identical=True,
            cache_key=batch_cache_key,
        )

        # Embed timestamps: (batch, seq) -> (batch, seq, dim_rot)
        unmasked_feat_pos = self.rotary_emb(unmasked_feat_timestamps)
        response_latent_pos = self.rotary_emb(response_latent_timestamps)
        screen_pos = self.rotary_emb(screen_feat_timestamps)
        masked_query_pos = tuple(map(self.rotary_emb, masked_query_timestamps))

        # Initialize the hidden state with the latent embeddings
        hidden, hidden_pos = response_latent_embeddings, response_latent_pos

        # Encoder: Input -> Latent (+ multimodal context)
        if (
            (response_feats is not None)
            if self.c.enable_fallback_sensorium_behavior
            else (unmasked_feats is not None)
        ):
            hidden = self.enc(
                x=hidden,
                pos=hidden_pos,
                context=unmasked_feats,
                context_pos=unmasked_feat_pos,
                attn_mask=encoder_block_mask,
            )

        # Concatenate multimodal context (along sequence dimension) if provided. NOTE: We
        #  use the order: screen, response latents/globals, behavior
        hidden = concat_notnone(
            (
                hidden,
                screen_feats,
            )
        )
        # NOTE: Positional embeddings are tuple of (cos, sin) tensors, or None if not present
        hidden_pos = tuple(
            map(
                concat_notnone,
                zip(*(x for x in (hidden_pos, screen_pos) if x is not None)),
            )
        )

        # Process: Latent -> Latent with (optional) multimodal fusion
        for _ in range(self.c.num_blocks):  # weights of each "block" are shared
            # Process each layer with appropriate masking
            for layer, is_local in zip(self.proc, self.local_global_schedule):
                nope = self.use_global_nope and not is_local
                # Apply the layer with or without mask based on local/global determination
                hidden = layer(
                    x=hidden,
                    pos=(None if nope else hidden_pos),
                    attn_mask=(latents_block_mask if is_local else None),
                )

        # Decoder: Latent -> Output
        if self.c.per_modality_decoders:
            # Separate decoder blocks for each modality. TODO: Support configuration of different modalities
            (rq, rqp, rm, rdec), (bq, bqp, bm, bdec) = zip(
                masked_queries,
                masked_query_pos,
                decoder_block_masks,
                self.decs,
            )
            responses_hidden, behavior_hidden = None, None
            if rq is not None:
                responses_hidden = rdec(
                    x=rq, pos=rqp, context=hidden, context_pos=hidden_pos, attn_mask=rm
                )
            if bq is not None:
                bm = (
                    None  # NOTE: Behavior queries are global, so no need for block mask
                )
                behavior_hidden = bdec(
                    x=bq, pos=bqp, context=hidden, context_pos=hidden_pos, attn_mask=bm
                )
        else:
            hidden = self.decs[0](
                x=masked_queries[0],
                pos=masked_query_pos[0],
                context=hidden,
                context_pos=hidden_pos,
                attn_mask=decoder_block_masks[0],
            )
            # Extract response/behavior hidden states
            feat_idx, responses_hidden, behavior_hidden = 0, None, None
            if n_response_queries > 0:
                responses_hidden = hidden[
                    :, feat_idx : feat_idx + n_response_queries, :
                ].contiguous()
                feat_idx += n_response_queries
            if n_behavior_queries > 0:
                behavior_hidden = hidden[
                    :, feat_idx : feat_idx + n_behavior_queries, :
                ].contiguous()
                feat_idx += n_behavior_queries

        # Track total loss
        loss = 0.0

        # -------------------- Decode Responses Features -------------------------------
        # Parse decoded *response* queries.
        responses_preds, response_loss = None, None
        if responses_hidden is not None:
            # Select per-neuron biases corresponding to queries: (B, seq, 1)
            neuron_biases = sess_params.neuron_biases[response_query_neuron_ids]
            # Project responses from decoded states with per-neuron biases to prediction
            #  sequence: (B, seq, D) -> (B, seq, stride)
            responses_logits = self.response_readout_proj(responses_hidden)
            # Add per-neuron biases: (B, seq, stride,) + (B, seq, 1,) -> (B, seq, stride,)
            responses_logits = responses_logits + neuron_biases
            # Apply output activation function: (B, seq, stride) -> (B, seq, stride)
            responses_preds = self.response_output_act(responses_logits).view(B, -1)
            # Calculate loss between response reconstruction prediction and labels
            response_loss = self.response_criteria(responses_preds, response_labels)
            # Accumulate response loss
            loss = loss + response_loss

        # -------------------- Decode Behavior Features -------------------------------
        # Parse decoded *behavior* queries
        behavior_preds, behavior_loss = None, None
        if behavior_hidden is not None:
            # Project behavior sequences from decoded behavior channel states using einsum:
            #  (B, C_beh, D) @ (C_beh, S_beh, D) -> (B, C_beh, S_beh)
            behavior_logits = torch.einsum(
                "bcd,csd->bcs", behavior_hidden, self.behavior_readout_proj
            )
            # Add per-channel, per-sample biases: (B, C_beh, S_beh) + (1, C_beh, S_beh) -> (B, C_beh, S_beh)
            # behavior_logits = behavior_logits + self.behavior_readout_bias
            # Apply output activation function: (B, C_beh, S_beh) -> (B, C_beh, S_beh)
            behavior_preds = self.behavior_output_act(behavior_logits).float()
            # Calculate loss between predictions and labels (i.e. input behavior)
            behavior_loss = self.behavior_criteria(behavior_preds, behavior)
            # Accumulate behavior loss
            loss = loss + self.c.behavior_loss_factor * behavior_loss

        return OMModelOutput(
            session_key=session_key,
            cache_key=batch_cache_key,
            loss=loss,
            response_loss=response_loss,
            response_preds=responses_preds,
            response_labels=response_labels,
            response_timestamps=response_label_timestamps,
            response_positions=response_label_positions,
            response_neuron_ids=response_label_neuron_ids,
            behavior_loss=behavior_loss,
            behavior_preds=behavior_preds,
            behavior_labels=behavior,
            behavior_timestamps=behavior_timestamps,
        )

    def forward_from_batch(
        self,
        batch: Tuple[str, Dict[str, Tensor | Dict[str, Tensor]]],
        masking_override: Optional[MaskingStrategy] = None,
        all_neurons_override: Optional[bool] = None,
    ) -> OMModelOutput:
        """
        Process a batch of data and run it through the model.

        Args:
            batch: Tuple containing:
                - session_key (str): Identifier for the session
                - batch_data (Dict): Dictionary containing:
                    - responses (Tensor): Neural response data
                    - timestamps (Dict): Dictionary of timestamps for different modalities
                    - screen (Tensor): Screen/video data
                    - eye_tracker (Tensor): Eye tracking data
                    - treadmill (Tensor): Treadmill data
            masking_override: Optional joint masking strategy to use

        Returns:
            OMModelOutput: Model output containing predictions and losses

        NOTE: We assume neurons are always in the same order for each batch of a given
            session, and that the entire population for the session is always passed. If
            this isn't the case, true neuron ids should be provided or generated here.
        TODO:
            - Work with `experanto` to remove need for transposing and float32 conversion
            - Work with `experanto` to add neuron ids to data loader
            - Raise error if "eye_tracker" and "treadmill" timestamps don't match
            - Support other batch structures with different datasets
        """
        # Unpack batch
        session_key, batch_data = batch

        # Unpack batch data
        responses = batch_data["responses"]
        response_timestamps = batch_data["timestamps"]["responses"]
        response_timestamps = response_timestamps.unsqueeze(-1).repeat(
            1, 1, responses.shape[-1]
        )
        screen = batch_data["screen"]
        screen_timestamps = batch_data["timestamps"]["screen"]

        # Construct behavior tensors from eye tracking and treadmill data, and
        #  transpose channel and sequence dimensions: (B, S_beh, C_beh) -> (B, C_beh, S_beh)
        behavior = (
            torch.cat(
                [
                    batch_data["eye_tracker"],
                    batch_data["treadmill"],
                ],
                dim=-1,
            )
            .transpose(1, 2)
            .contiguous()
        )
        behavior_timestamps = batch_data["timestamps"]["eye_tracker"]

        # Convert inputs to float32
        responses = responses.float()
        screen = screen.float()
        behavior = behavior.float()

        # Run model forward pass
        return self.forward(
            session_key=session_key,
            responses=responses,
            timestamps=response_timestamps,
            screen=screen,
            screen_timestamps=screen_timestamps,
            behavior=behavior,
            behavior_timestamps=behavior_timestamps,
            masking_override=masking_override,
            all_neurons_override=all_neurons_override,
        )

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """TODO: Add cached BlockMasks to state dict"""
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict

        TODO: Load cached BlockMasks from state dict
        """
        return super().load_state_dict(state_dict, strict=strict)
