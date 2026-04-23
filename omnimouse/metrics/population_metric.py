from typing import Optional, Union, Any, Tuple, List, Dict, Union, Sequence
from jaxtyping import Float, Int, Bool, Num
from typing_extensions import override
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class MaskablePositionalPopulationMetric(Metric, ABC):
    """Base class for metrics designed for maskable neural populations."""
    def __init__(
        self,
        population_size: int,
        num_samples_per_block: int,
        masked: Optional[bool] = None,
        positions_of_interest: Optional[List[int] | int] = None,
        **kwargs: Any,
    ) -> None:
        # Forcibly disable distributed syncing of metrics
        kwargs.update({
            "process_group": None,
            "sync_on_compute": False,
            "dist_sync_fn": None,
        })

        super().__init__(**kwargs)
        # Population attributes
        self._population_size = population_size
        self._num_samples_per_block = num_samples_per_block
        # Masking behavior
        self._masked = masked
        # Position reduction behavior
        if positions_of_interest is not None:
            if isinstance(positions_of_interest, int):
                positions_of_interest = [positions_of_interest]
            assert all([0 <= p < num_samples_per_block for p in positions_of_interest]), \
                f"Positions of interest must be in range [0, {num_samples_per_block})!"
        self._positions_of_interest = positions_of_interest
    
    @property
    def population_size(self) -> int:
        return self._population_size

    @property
    def num_samples_per_block(self) -> int:
        return self._num_samples_per_block

    @property
    def state_size(self) -> Tuple[int, ...]:
        return (2, self.num_samples_per_block, self.population_size)

    @property
    def masked_positional_population_metric_state(self) -> Dict[str, torch.Tensor]:
        """
        Wrapper for `Metric.metric_state`, which returns a dictionary of metric states,
         but only those which have mask/position/population dimensions.
        """
        mpp_metric_state = {}
        for k, v in self.metric_state.items():
            # Skip states that don't match the expected `state_size`. NOTE: An additional
            #  dimension may be prepended for multiple devices (i.e. when using DDP).
            if (not isinstance(v, torch.Tensor) or
                v.shape[-len(self.state_size):] != self.state_size):
                continue
            mpp_metric_state[k] = v
        return mpp_metric_state

    @property
    @abstractmethod
    def observed_population_size(self) -> Int[torch.Tensor, ""]:
        """The number of neurons that have been observed at least once."""
        pass

    @abstractmethod
    def _update(
        self,
        preds: Float[torch.Tensor, "n_reconstructed_samples"],
        target: Float[torch.Tensor, "n_reconstructed_samples"],
        neuron_ids: Int[torch.Tensor, "n_reconstructed_samples"],
        positions: Int[torch.Tensor, "n_reconstructed_samples"],
        mask: Optional[Bool[torch.Tensor, "n_reconstructed_samples"]] = None,
    ) -> None:
        """Update metric states for a single batch item (i.e. block)."""
        pass

    @override
    @torch.autocast('cuda', enabled = False)
    def update(
        self,
        preds: Float[torch.Tensor, "batch n_reconstructed_samples"],
        target: Float[torch.Tensor, "batch n_reconstructed_samples"],
        neuron_ids: Int[torch.Tensor, "batch n_reconstructed_samples"],
        positions: Int[torch.Tensor, "batch n_reconstructed_samples"],
        mask: Optional[Bool[torch.Tensor, "batch n_reconstructed_samples"]] = None,
        valid_samples: Optional[Bool[torch.Tensor, "batch n_reconstructed_samples"]] = None,
    ) -> None:
        """
        NOTE: We require that for each item in the batch, no two samples
         have the same neuron ID *and* position.
        """
        # Iterate over samples in batch. NOTE: batches might use overlapping
        #  neurons/positions so can't parallelize
        batch_size = preds.size(0)
        for i in range(batch_size):
            _preds, _target, _nids, _pos = (
                preds[i], target[i], neuron_ids[i], positions[i],
            )
            _mask = mask[i] if mask is not None else None
            if valid_samples is not None:
                _valid_samples = valid_samples[i]
                _preds, _target, _nids, _pos = (
                    _preds[_valid_samples],
                    _target[_valid_samples],
                    _nids[_valid_samples],
                    _pos[_valid_samples],
                )
                if _mask is not None:
                    _mask = _mask[_valid_samples]
            self._update(
                _preds, _target, _nids, _pos, mask=_mask,
            )
    
    @abstractmethod
    def _aggregation_function(
        self,
        **states: Dict[str, Num[torch.Tensor, "aggregated_dim *additional_dims"]]
    ) -> Dict[str, Num[torch.Tensor, "*additional_dims"]]:
        """Aggregate states across devices, mask values, and (optionally) positions"""
        pass

    def _aggregate_states(
        self,
        reduce_across_positions: bool = True,
    ) -> Dict[str, Num[torch.Tensor, "*computed_positions population_size"]]:
        """Aggregate states across devices, mask values, and (optionally) positions
        
        NOTE: `computed_positions` matches exactly 0 (if `reduce_across_positions`) or
         1 (if `not reduce_across_positions`) dimensions. If `not reduce_across_positions`,
          then `computed_positions` equals `len(positions_of_interest)` (if `positions_of_interest`
          is not None) or `num_samples_per_block` (if `positions_of_interest` is None).

          Reduction across devices in torch.multiprocssing is disabled here!
          Metrics will be fully local.
        """
        # Initialize placeholder for aggregated states
        aggregated_states = self.masked_positional_population_metric_state
        # The check for multi-device dimensionality and the subsequent
        # call to _aggregation_function for cross-device reduction are removed.
        # The method now proceeds directly assuming aggregated_states is local.

        # Handle mask dimension (either aggregating or selecting)
        if self._masked is None:
            # Aggregate values across masked and unmasked states if mask is ignored
            # This call now operates only on the local state tensor dimensions.
            aggregated_states = self._aggregation_function(**aggregated_states)
        else:
            # Select states according to mask (i.e. masked 0th index of first dimension, unmasked in 1st index)
            aggregated_states = {k: v[(0 if self._masked else 1)] for k, v in aggregated_states.items()}
        # If positions of interest are specified, select states according to positions of interest
        if self._positions_of_interest is not None:
            aggregated_states = {k: v[self._positions_of_interest] for k, v in aggregated_states.items()}
        # Aggregate values across positions of interest (or all), if not computing per-position
        if reduce_across_positions:
            # This call now operates only on the local state tensor dimensions.
            aggregated_states = self._aggregation_function(**aggregated_states)
        return aggregated_states
        

    @abstractmethod
    def _compute_from_states_with_counts(
        self,
        **states: Dict[str, Num[torch.Tensor, "*state_dimensions"]],
    ) -> Tuple[
        Float[torch.Tensor, "*state_dimensions"],  # (per-position) per-neuron metric values
        Int[torch.Tensor, "*state_dimensions"],  # (per-position) per-neuron observation counts
    ]:
        """
        Compute metric values for all neurons in the population, 
        and per-neuron observation counts, given aggregated states.
        """
        pass
    
    @override
    @torch.autocast('cuda', enabled = False)
    def compute(self) -> Float[torch.Tensor, ""]:
        # Aggregate states across devices, mask values, and positions
        aggregated_states = self._aggregate_states()
        # Compute aggregated, but unreduced (i.e. population averaged) metric values and observation counts
        ret, counts = self._compute_from_states_with_counts(**aggregated_states)
        # Take weighted average (by observation counts)
        weights = counts.float() / counts.sum(dim=-1)
        ret = torch.nansum(weights * ret, dim=-1)
        return ret
    
    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # TODO: Implement forward function for population metrics
        raise NotImplementedError
        
    @override
    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot the metric value.
        
        TODO: Implement more informative plot of population distribution (using `plotly`)
        """
        return self._plot(val, ax)

    # NOTE: `compute` is designed to return a single value accross positions for the entire population.
    #  This utility is a bit overly complex, but allows flexibility for computing without reducing
    #  dimensions. Can be ignored for most users/use cases.
    def compute_unreduced_with_counts(
        self,
        reduce_across_positions: bool = False,
        reduce_across_population: bool = False,
        remove_unobserved_neurons: bool = False,
        remove_unobserved_positions: bool = False,
    ) -> Tuple[
        Float[torch.Tensor, "*unreduced_dimensions"],
        Int[torch.Tensor, "*unreduced_dimensions"],
    ]:
        """Utility function for computing metric values and observation counts,
         optionally for each position and neuron.
        
        NOTE: `unreduced_dimensions` matches exactly 0 (if `reduce_across_positions and reduce_across_population`) or
         1 (if `reduce_across_positions ^ reduce_across_population`) or 2 (if `not (reduce_across_positions or
         reduce_across_population)`) dimensions. If `not reduce_across_positions`, the position dimension will be
         either `len(positions_of_interest)` (if `positions_of_interest` is not None) or `num_samples_per_block` (minus
         the number of unobserved positions, if `remove_unobserved_positions` is True). If `not reduce_across_population`,
         the population dimension will be `population_size` (minus the number of unobserved neurons, if `remove_unobserved_neurons`).
        NOTE: We infer "unobserved" if all neurons/positions at a given position/neuron have observation counts of 0.
        """
        # Aggregate states across devices, mask values, and (optionally) positions
        aggregated_states = self._aggregate_states(reduce_across_positions=reduce_across_positions)
        # Compute aggregated, but unreduced (i.e. population averaged) metric values and observation counts
        ret, counts = self._compute_from_states_with_counts(**aggregated_states)
        assert ret.shape == counts.shape, "Metric values and observation counts must have the same shape!"
        assert counts.ndim <= 2, "Observation counts must be a 1D or 2D tensor!"
        # Remove unobserved positions, if specified
        if remove_unobserved_positions and counts.ndim == 2:
            observed_positions = counts.sum(dim=-1) > 0
            ret, counts = ret[observed_positions], counts[observed_positions]
        # Remove unobserved neurons, if specified
        if remove_unobserved_neurons:
            observed_neurons = torch.atleast_2d(counts).sum(dim=0) > 0
            ret, counts = ret[..., observed_neurons], counts[..., observed_neurons]
        # Take weighted average (by observation counts)
        if reduce_across_population:
            weights = counts.float() / counts.sum(dim=-1)
            ret = torch.nansum(weights * ret, dim=-1)
            counts = counts.sum(dim=-1)
        # Return results
        return ret, counts