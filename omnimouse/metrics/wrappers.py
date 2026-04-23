from typing import Union, Optional, Sequence, Dict, List, Any, Iterable, Tuple
from typing_extensions import override
from copy import deepcopy
import torch
from torchmetrics import MetricCollection
from torchmetrics.utilities.plot import _PLOT_OUT_TYPE

from omnimouse.metrics.population_metric import MaskablePositionalPopulationMetric as MPMetric


class MultiSessionMetricMUX(MetricCollection):
    """
    Multilayer metric collection designed for collections of `MaskablePopulationMetric`
     instances from multiple sessions.
     
     This class extends MetricCollection to handle metrics from multiple sessions
     by creating separate clones of the base metrics for each session and managing
     them under session-specific prefixes. The compute groups functionality is also
     extended to work across sessions.
    """
    def __init__(
        self,
        session_keys: Sequence[str],
        metrics: Union[MPMetric, Sequence[MPMetric], Dict[str, MPMetric]],
        *additional_metrics: MPMetric,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
        compute_groups: Union[bool, List[List[str]]] = True,
    ) -> None:
        # Initialize base metric collection
        solo = MetricCollection(
            metrics,
            *additional_metrics,
        )
        # Clone base metric collection for each session. NOTE: We replace underscores
        #  with dashes in session keys, to avoid conflicts with metric prefixes
        multi = {f"sess-{sk.replace('_', '-')}": solo.clone() for sk in session_keys}
        # Store unique updated session keys and metric names
        self._met_sess_keys, self._met_names =  list(multi.keys()), list(solo.keys())
        # Store mapping from originals to modified session keys
        self._session_key_map = {sk: mapped_sk for sk, mapped_sk in zip(session_keys, self._met_sess_keys)}
        # If provided, update `compute_groups` for all sessions, i.e. repeat each compute group,
        #  with session key prepended
        if isinstance(compute_groups, List):
            compute_groups = [
                [f'{sess}_{met}' for met in group]
                for sess in self._met_sess_keys
                for group in compute_groups
            ]
        # Initialize multi-session metric collection
        super().__init__(
            multi,
            prefix=prefix,
            postfix=postfix,
            compute_groups=compute_groups,
        )
        # Verify that all metrics are instances of `MaskablePopulationMetric`
        for metric in self.values(copy_state=False):
            if not isinstance(metric, MPMetric):
                raise TypeError(
                    f"{self.__class__.__name__} can only wrap instances of"
                    f"`{MPMetric.__module__}.{MPMetric.__name__}`"
                )
        # Maintain list of sessions with merged compute groups (only applicable for `compute_groups=True`)
        self._sessions_with_merged_compute_groups = set([])

    @property
    def device(self) -> torch.device:
        return next(iter(self.values(copy_state=False))).device

    def to(self, device: torch.device | str) -> 'MultiSessionMetricMUX':
        for metric in self.values(copy_state=False):
            metric.to(device)
        return self

    @override
    def items(self, *args, session_key: Optional[str] = None, **kwargs) -> Iterable[Tuple[str, MPMetric]]:
        """Enable filtering items by session key"""
        ret = super().items(*args, **kwargs)
        if session_key is not None:
            met_sess_key = self._session_key_map[session_key]
            ret = ((k,v) for k,v in ret if k.startswith(met_sess_key))
        return ret

    def _groups_by_session(self, session_key: str) -> Dict[int, List[str]]:
        """Get compute groups for specified session"""
        met_sess_key = self._session_key_map[session_key]
        return {idx: cg for idx, cg in self._groups.items() if cg[0].startswith(met_sess_key)}
    
    @override
    def _compute_groups_create_state_ref(self, *args: Any, **kwargs: Any) -> None:
        """
        TODO: Consider overriding this method to only create state references for a specified session,
         if a use case arises. For now, we use the original (i.e. create references for all groups on each call).
        """
        return super()._compute_groups_create_state_ref(*args, **kwargs)
    
    @override
    def update(self, session_key: str, *args: Any, **kwargs: Any) -> None:
        """Adapted from :meth:`torchmetrics.collections.MetricCollection.update`.
        
        Matches original implementation, but only updates metrics (and compute groups on
         first call, if enabled) for specified session. NOTE: Compute groups are merged
         by comparing metric states, so we merge a session only after its first update.
        """
        # NOTE: session keys are only added to `_sessions_with_merged_compute_groups`
        #  if compute groups are enabled and relevant groups have been merged (see
        #  `_merge_compute_groups`). Otherwise this condition will always be `False`.
        # Use compute groups if already initialized and checked.
        if self._groups_checked or session_key in self._sessions_with_merged_compute_groups:
            # Delete the cache of all metrics to invalidate the cache and therefore recent compute calls, forcing new
            # compute calls to recompute
            for _, m in self.items(session_key=session_key, keep_base=True, copy_state=False):
                m._computed = None
            for cg in self._groups_by_session(session_key).values():
                # only update the first member
                m0 = getattr(self, cg[0])
                m0.update(*args, **m0._filter_kwargs(**kwargs))
            if self._state_is_copy:
                # NOTE: This block is never entered! Calling `items` above already calls `_compute_groups_create_state_ref`
                #  with `copy_state=False`. Leaving to maintain alignment with original, and in case `items` changes).
                # NOTE: Links will be established for metrics from *all* sessions, not just the current one.
                # NOTE: Order switched from original code to fix bug. Track issue `here<https://github.com/Lightning-AI/torchmetrics/issues/2896>`_.
                # If we have deep copied state in between updates, reestablish link.
                self._state_is_copy = False
                self._compute_groups_create_state_ref()
        else:  # the first update always do per metric to form compute groups
            for _, m in self.items(session_key=session_key, keep_base=True, copy_state=False):
                m_kwargs = m._filter_kwargs(**kwargs)
                m.update(*args, **m_kwargs)

            if self._enable_compute_groups:
                self._merge_compute_groups(session_key=session_key)
                # create reference between states
                self._compute_groups_create_state_ref()
                # NOTE: Only set `self._groups_checked = True` if merging attempted for all sessions
                if set(self._session_key_map) == self._sessions_with_merged_compute_groups:
                    self._groups_checked = True
    
    @override
    def _merge_compute_groups(self, session_key: str) -> None:
        """Adapted from :meth:`torchmetrics.collections.MetricCollection._merge_compute_groups`.
        
        From the original implementation, we replace `self._groups` with `self._groups_by_session(session_key)`,
         such that we only consider groups from the specified session for merging. Before returning, we also add
         the session key to `_sessions_with_merged_compute_groups`.
        """
        num_groups = len(self._groups_by_session(session_key))  # Only consider groups for current session
        while True:
            for cg_idx1, cg_members1 in deepcopy(self._groups_by_session(session_key)).items():
                for cg_idx2, cg_members2 in deepcopy(self._groups_by_session(session_key)).items():
                    if cg_idx1 == cg_idx2:
                        continue

                    metric1 = getattr(self, cg_members1[0])
                    metric2 = getattr(self, cg_members2[0])

                    if self._equal_metric_states(metric1, metric2):
                        self._groups[cg_idx1].extend(self._groups.pop(cg_idx2))
                        break

                # Start over if we merged groups
                if len(self._groups_by_session(session_key)) != num_groups:
                    break

            # Stop when we iterate over everything and do not merge any groups
            if len(self._groups_by_session(session_key)) == num_groups:
                break
            num_groups = len(self._groups_by_session(session_key))

        # Re-index groups
        temp = deepcopy(self._groups)
        self._groups = {}
        for idx, values in enumerate(temp.values()):
            self._groups[idx] = values
        
        # Add session key to `_sessions_with_merged_compute_groups`
        self._sessions_with_merged_compute_groups.add(session_key)

    @override
    @torch.autocast('cuda', enabled = False)
    def _compute_and_reduce(
        self, *args: Any, **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """Adapted from :meth:`torchmetrics.collections.MetricCollection._compute_and_reduce`."""
        ret: Dict[str, torch.Tensor] = super()._compute_and_reduce(*args, **kwargs)

        # `ret` contains `compute` results from each metric, with keys of format:
        #   `f'{prefix}{met_sess_key}_{met_name}{postfix}'`. For each metric, we will
        #   compute a weighted average of results across sessions and add a new "overall"
        #   entry to the return dictionary.
        for met_name in self._met_names:
            results, counts = [], []
            # Iterate over all sessions for current metric
            for met_sess_key in self._met_sess_keys:
                base = f'{met_sess_key}_{met_name}'
                # Extract result from return dictionary
                results.append(ret[self._set_name(base)])
                # Get *population size* from corresponding metric
                counts.append(getattr(self, base).observed_population_size)
            # Convert to tensors. shape: (num_sessions,), (num_sessions, *metric_dims)
            results, counts = torch.stack(results), torch.stack(counts)
            # Calculate weights as counts normalized by total counts
            weights = counts / counts.sum(0)
            # Expand weights to match results dimensions for broadcasting (i.e. expand to match `metric_dims`)
            weights = weights.broadcast_to(results.T.shape).T
            # Take weighted average of results across sessions (excluding `nan` results)
            reduced = torch.nansum(results * weights, dim=0)
            # Add overall (i.e. weighted multi-session average) result to return dictionary
            ret[self._set_name(f'overall_{met_name}')] = reduced

        return ret
    
    @override
    def plot(
        self, *args, **kwargs,
    ) -> Sequence[_PLOT_OUT_TYPE]:
        """Plot a single or multiple values from the metric.
        
        TODO: Implement more informative plots of population distributions across sessions (using `plotly`)
        """
        return super().plot(*args, **kwargs)