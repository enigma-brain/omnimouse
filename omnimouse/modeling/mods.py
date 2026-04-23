from typing import Tuple, Optional, List, Dict, Callable, Mapping, Any
from jaxtyping import Float
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from torch import nn
from torch.nn.attention.flex_attention import (
    create_block_mask,
    BlockMask,
    _DEFAULT_SPARSE_BLOCK_SIZE,
)

"""
NOTE: See discussion on creating mask mods which access dense masks
 `here <https://discuss.pytorch.org/t/creating-the-flexattention-blockmask-from-a-mask/214175>`_.
"""

def create_multimodal_temporal_sliding_window_block_mask(
    q_window_bound_1: Float[Tensor, "*batch_heads seqlen"],
    q_window_bound_2: Float[Tensor, "*batch_heads seqlen"],
    kv_window_bound_1: Optional[Float[Tensor, "*batch_heads seqlen"]] = None,
    kv_window_bound_2: Optional[Float[Tensor, "*batch_heads seqlen"]] = None,
    treat_leading_dim_as_heads: bool = False,
    device: str | torch.device = "cuda",
    BLOCK_SIZE: int | Tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE,
):  
    """
    """
    # Input shapes validation
    assert q_window_bound_1.ndim <= 3, "Query window bounds must have at most 3 dimensions!"
    batch_heads = q_window_bound_1.shape[:-1]
    assert all(x.shape[:-1] == batch_heads for x in (q_window_bound_1, q_window_bound_2,
     kv_window_bound_1, kv_window_bound_2) if x is not None), \
        "All window bounds must have same batch and head dimensions!"
    assert q_window_bound_1.shape == q_window_bound_2.shape, \
        "Query window bounds must have same shape!"
    if kv_window_bound_1 is not None:
        assert kv_window_bound_1.shape == kv_window_bound_2.shape, \
            "Key/value window bounds must have same shape!"
    assert not treat_leading_dim_as_heads or q_window_bound_1.ndim <= 2, \
        "Head dimension must be second to last dimension!"
    # Calculate window start/end sample indices for query (and key/value,
    #  if specified) sequences...
    q_ts_to_start = torch.minimum(q_window_bound_1, q_window_bound_2)
    q_ts_to_end = torch.maximum(q_window_bound_1, q_window_bound_2)
    kv_ts_to_start, kv_ts_to_end = q_ts_to_start, q_ts_to_end
    if kv_window_bound_1 is not None:
        kv_ts_to_start = torch.minimum(kv_window_bound_1, kv_window_bound_2)
        kv_ts_to_end = torch.maximum(kv_window_bound_1, kv_window_bound_2)
    # Define mask shapes and indexing function, using appropriate batch/head dimensions
    #  to idx into tensors.
    Q_LEN, KV_LEN = q_ts_to_start.size(-1), kv_ts_to_start.size(-1)
    if len(batch_heads) == 2:
        B, H = batch_heads
        sel = lambda b, h, idx: (b, h, idx)
    elif len(batch_heads) == 1 and treat_leading_dim_as_heads:
        B, H = None, batch_heads[0]
        sel = lambda b, h, idx: (h, idx)
    elif len(batch_heads) == 1 and not treat_leading_dim_as_heads:
        B, H = batch_heads[0], None
        sel = lambda b, h, idx: (b, idx)
    else:
        B, H = None, None
        sel = lambda b, h, idx: (idx,)
    # Define "mask_mod" function, referencing window bound tensors and indexing function
    def sliding_window_mask_mod(b, h, q_idx, kv_idx):
        q_sel, kv_sel = sel(b, h, q_idx), sel(b, h, kv_idx)
        q_start, q_end = q_ts_to_start[*q_sel], q_ts_to_end[*q_sel]
        kv_start, kv_end = kv_ts_to_start[*kv_sel], kv_ts_to_end[*kv_sel]
        non_overlapping = (kv_end <= q_start) | (kv_start >= q_end)
        return ~non_overlapping
    
    # Create BlockMask. NOTE: Always agnostic to head dimension
    return create_block_mask(
        mask_mod=sliding_window_mask_mod,
        B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN,
        device=device, BLOCK_SIZE=BLOCK_SIZE
    )