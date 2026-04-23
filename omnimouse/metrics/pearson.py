"""
Adapted from `torchmetrics.PearsonCorrCoef <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/regression/pearson.py>`__.
"""
from typing import Optional, Any, Tuple, List, Dict
from jaxtyping import Float, Float64, Int, Bool
from typing_extensions import override
import torch
from torchmetrics.regression.pearson import _final_aggregation
from torchmetrics.functional.regression.pearson import _pearson_corrcoef_compute

from omnimouse.metrics.population_metric import MaskablePositionalPopulationMetric


# NOTE: The current implementation of  :func:`torchmetrics.regression.pearson._final_aggregation` has a bug
#  where states are modified in place. This fixes that bug (and also cleans up the algorithm). Once issue is
#  fixed in torchmetrics, we can revert to using the torchmetrics implementation. Track issue `here<https://github.com/Lightning-AI/torchmetrics/issues/2893>`_.
def _final_aggregation(
    means_x: torch.Tensor,
    means_y: torch.Tensor,
    vars_x: torch.Tensor,
    vars_y: torch.Tensor,
    corrs_xy: torch.Tensor,
    nbs: torch.Tensor,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate the statistics from multiple devices.

    Formula taken from here: `Parallel algorithm for calculating variance <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm>`_

    NOTE: We use `eps` to avoid division by zero when `n1` and `n2` are both zero. Generally,
     the value of `eps` should not matter, as if `n1` and `n2` are both zero, all the states will
     also be zero.
    """
    if len(means_x) == 1:
        return means_x[0], means_y[0], vars_x[0], vars_y[0], corrs_xy[0], nbs[0]
    mx1, my1, vx1, vy1, cxy1, n1 = means_x[0], means_y[0], vars_x[0], vars_y[0], corrs_xy[0], nbs[0]
    for i in range(1, len(means_x)):
        mx2, my2, vx2, vy2, cxy2, n2 = means_x[i], means_y[i], vars_x[i], vars_y[i], corrs_xy[i], nbs[i]
        # count
        nb = torch.where(torch.logical_or(n1, n2), n1 + n2, eps)
        # mean_x
        mean_x = (n1 * mx1 + n2 * mx2) / nb
        # mean_y
        mean_y = (n1 * my1 + n2 * my2) / nb
        # intermediates for running variances
        n12_b = n1 * n2 / nb
        delta_x = mx2 - mx1
        delta_y = my2 - my1
        # var_x
        var_x = vx1 + vx2 + n12_b * delta_x ** 2
        # var_y
        var_y = vy1 + vy2 + n12_b * delta_y ** 2
        # corr_xy
        corr_xy = cxy1 + cxy2 + n12_b * delta_x * delta_y

        mx1, my1, vx1, vy1, cxy1, n1 = mean_x, mean_y, var_x, var_y, corr_xy, nb
    return mean_x, mean_y, var_x, var_y, corr_xy, nb


def _pearson_corrcoef_update(
    preds: Float[torch.Tensor, "n_updated_states"],
    target: Float[torch.Tensor, "n_updated_states"],
    mean_x: Float[torch.Tensor, "n_updated_states"],
    mean_y: Float[torch.Tensor, "n_updated_states"],
    var_x: Float[torch.Tensor, "n_updated_states"],
    var_y: Float[torch.Tensor, "n_updated_states"],
    corr_xy: Float[torch.Tensor, "n_updated_states"],
    num_prior: Int[torch.Tensor, "n_updated_states"],
) -> Tuple[
    Float[torch.Tensor, "n_updated_states"],
    Float[torch.Tensor, "n_updated_states"],
    Float[torch.Tensor, "n_updated_states"],
    Float[torch.Tensor, "n_updated_states"],
    Float[torch.Tensor, "n_updated_states"],
    Float[torch.Tensor, "n_updated_states"],
    Int[torch.Tensor, "n_updated_states"]
]:
    """Update and returns variables required to compute Pearson Correlation Coefficient,
        for subset of population neurons. Uses `Welford's online algorithm <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`__.
        Adapted (for population and masking) from :func:`torchmetrics.functional.regression.pearson._pearson_corrcoef_update`.
        Optimized based on the assumption that each sample corresponds to a single state entry, and thus
        gets only a single observation for each call.

    Args:
        preds: estimated scores
        target: ground truth scores
        mask: binary mask indicating which samples to include in update
        mean_x: current mean estimate of x tensor
        mean_y: current mean estimate of y tensor
        var_x: current variance estimate of x tensor
        var_y: current variance estimate of y tensor
        corr_xy: current covariance estimate between x and y tensor
        num_prior: current number of observed observations
    """
    # Update obs counts
    num_prior += 1
    # Calculate intermediates for mean updates
    delta_x, delta_y = preds - mean_x, target - mean_y
    # Running mean updaes
    mx_new = mean_x + delta_x / num_prior
    my_new = mean_y + delta_y / num_prior
    # Calculate intermediates for variance updates
    delta2_x, delta2_y = preds - mx_new, target - my_new
    # Running variance updates
    var_x += delta2_x * delta_x
    var_y += delta2_y * delta_y
    # Running covariance updates
    corr_xy += delta2_x * delta_y

    return mx_new, my_new, var_x, var_y, corr_xy, num_prior


class MaskablePositionalPopulationPearsonCorrCoef(MaskablePositionalPopulationMetric):
    is_differentiable: bool = True
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True
    plot_lower_bound: float = -1.0
    plot_upper_bound: float = 1.0

    # NOTE: `self.state_size` is `(2, self.num_samples_per_block, self.population_size)`
    mean_x: Float64[torch.Tensor, "{self.state_size}"]
    mean_y: Float64[torch.Tensor, "{self.state_size}"]
    var_x: Float64[torch.Tensor, "{self.state_size}"]
    var_y: Float64[torch.Tensor, "{self.state_size}"]
    corr_xy: Float64[torch.Tensor, "{self.state_size}"]
    n_total: Float64[torch.Tensor, "{self.state_size}"]

    def __init__(
        self,
        population_size: int,
        num_samples_per_block: int,
        masked: Optional[bool] = None,
        positions_of_interest: Optional[List[int] | int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            population_size=population_size,
            num_samples_per_block=num_samples_per_block,
            masked=masked,
            positions_of_interest=positions_of_interest,
            **kwargs
        )
        # Initialize states: (masked_and_unmasked, num_samples_per_block, population_size). NOTE: We use `float64`
        #  to avoid overflow during repeated aggregation, but `float32` gives similar results (within 1e-5).
        self.add_state("mean_x", default=torch.zeros(self.state_size, dtype=torch.float64), dist_reduce_fx=None)
        self.add_state("mean_y", default=torch.zeros(self.state_size, dtype=torch.float64), dist_reduce_fx=None)
        self.add_state("var_x", default=torch.zeros(self.state_size, dtype=torch.float64), dist_reduce_fx=None)
        self.add_state("var_y", default=torch.zeros(self.state_size, dtype=torch.float64), dist_reduce_fx=None)
        self.add_state("corr_xy", default=torch.zeros(self.state_size, dtype=torch.float64), dist_reduce_fx=None)
        self.add_state("n_total", default=torch.zeros(self.state_size, dtype=torch.float64), dist_reduce_fx=None)

    @property
    @override
    def observed_population_size(self) -> Int[torch.Tensor, ""]:
        """The number of neurons that have been observed at least once."""
        # NOTE: We sum observations across non-population dimensions
        summation_dims = tuple(range(self.n_total.ndim - 1))
        return self.n_total.sum(dim=summation_dims).bool().sum()

    @override
    def _update(
        self,
        preds: Float[torch.Tensor, "n_reconstructed_samples"],
        target: Float[torch.Tensor, "n_reconstructed_samples"],
        neuron_ids: Int[torch.Tensor, "n_reconstructed_samples"],
        positions: Int[torch.Tensor, "n_reconstructed_samples"],
        mask: Optional[Bool[torch.Tensor, "n_reconstructed_samples"]] = None,
    ) -> None:
        """Update metric states for a single batch item (i.e. block).
        
        NOTE: We treat each mask/position/neuron as a separate output, taking advantage
         of the fact that any combination of mask/position/neuron can only occur once per block.
        """
        # Convert mask from bool to index ∈ [0, 1]. Defaults to 0, i.e. masked, if `None`
        mask_idx = mask.int() if mask is not None else 0
        # Calculate new state values for neurons in block
        mean_x, mean_y, var_x, var_y, corr_xy, n_total = _pearson_corrcoef_update(
            preds, target,
            self.mean_x[mask_idx, positions, neuron_ids],
            self.mean_y[mask_idx, positions, neuron_ids],
            self.var_x[mask_idx, positions, neuron_ids],
            self.var_y[mask_idx, positions, neuron_ids],
            self.corr_xy[mask_idx, positions, neuron_ids],
            self.n_total[mask_idx, positions, neuron_ids],
        )
        # Update population state values for subset of neurons
        self.mean_x[mask_idx, positions, neuron_ids] = mean_x
        self.mean_y[mask_idx, positions, neuron_ids] = mean_y
        self.var_x[mask_idx, positions, neuron_ids] = var_x
        self.var_y[mask_idx, positions, neuron_ids] = var_y
        self.corr_xy[mask_idx, positions, neuron_ids] = corr_xy
        self.n_total[mask_idx, positions, neuron_ids] = n_total

    @override
    def _aggregation_function(
        self,
        mean_x: Float64[torch.Tensor, "aggregated_dim *additional_dims"],
        mean_y: Float64[torch.Tensor, "aggregated_dim *additional_dims"],
        var_x: Float64[torch.Tensor, "aggregated_dim *additional_dims"],
        var_y: Float64[torch.Tensor, "aggregated_dim *additional_dims"],
        corr_xy: Float64[torch.Tensor, "aggregated_dim *additional_dims"],
        n_total: Float64[torch.Tensor, "aggregated_dim *additional_dims"],
    ) -> Dict[str, Float64[torch.Tensor, "*additional_dims"]]:
        mean_x, mean_y, var_x, var_y, corr_xy, n_total = _final_aggregation(
            mean_x, mean_y, var_x, var_y, corr_xy, n_total
        )
        return {
            'mean_x': mean_x,
            'mean_y': mean_y,
            'var_x': var_x,
            'var_y': var_y,
            'corr_xy': corr_xy,
            'n_total': n_total
        }
    
    @override
    def _compute_from_states_with_counts(
        self,
        mean_x: Float64[torch.Tensor, "*state_dimensions"],
        mean_y: Float64[torch.Tensor, "*state_dimensions"],
        var_x: Float64[torch.Tensor, "*state_dimensions"],
        var_y: Float64[torch.Tensor, "*state_dimensions"],
        corr_xy: Float64[torch.Tensor, "*state_dimensions"],
        n_total: Float64[torch.Tensor, "*state_dimensions"],
    ) -> Tuple[
        Float[torch.Tensor, "*state_dimensions"],  # (per-position) per-neuron metric values
        Int[torch.Tensor, "*state_dimensions"],  # (per-position) per-neuron observation counts
    ]:
        # Compute Pearson Correlation Coefficient for entire population
        ret = _pearson_corrcoef_compute(var_x, var_y, corr_xy, n_total)
        # Alias `n_total` as `counts`
        counts = n_total
        # Return metric values and per-neuron observation counts
        return ret, counts

class UnmaskedPopulationPearsonCorrCoef(MaskablePositionalPopulationPearsonCorrCoef):
    def __init__(
        self,
        population_size: int,
        num_samples_per_block: int,
        **kwargs: Any,
    ) -> None:
        kwargs.pop('masked', None)
        kwargs.pop('positions_of_interest', None)
        super().__init__(
            population_size=population_size,
            num_samples_per_block=num_samples_per_block,
            masked=False,
            positions_of_interest=None,
            **kwargs
        )

class MaskedPopulationPearsonCorrCoef(MaskablePositionalPopulationPearsonCorrCoef):
    def __init__(
        self,
        population_size: int,
        num_samples_per_block: int,
        **kwargs: Any,
    ) -> None:
        kwargs.pop('masked', None)
        kwargs.pop('positions_of_interest', None)
        super().__init__(
            population_size=population_size,
            num_samples_per_block=num_samples_per_block,
            masked=True,
            positions_of_interest=None,
            **kwargs
        )

class PositionPopulationPearsonCorrCoef(MaskablePositionalPopulationPearsonCorrCoef):
    def __init__(
        self,
        population_size: int,
        num_samples_per_block: int,
        position: int,
        masked: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop('positions_of_interest', None)
        super().__init__(
            population_size=population_size,
            num_samples_per_block=num_samples_per_block,
            masked=masked,
            positions_of_interest=position,
            **kwargs
        )

class CausalFirst4PopulationPearsonCorrCoef(MaskablePositionalPopulationPearsonCorrCoef):
    def __init__(
        self,
        population_size: int,
        num_samples_per_block: int,
        next_sample: int,
        **kwargs: Any,
    ) -> None:
        kwargs.pop('masked', None)
        kwargs.pop('positions_of_interest', None)
        super().__init__(
            population_size=population_size,
            num_samples_per_block=num_samples_per_block,
            masked=True,
            positions_of_interest=list(range(next_sample, next_sample + 4)),
            **kwargs
        )
