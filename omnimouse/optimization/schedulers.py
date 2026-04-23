import math
import warnings
from typing import List, Optional, Dict, Any
import numpy as np

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from omnimouse.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class LinearWarmupLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr and
    base_lr.

    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.

    Example:
        >>> import torch.nn as nn
        >>> from torch.optim import Adam
        >>> #
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)

    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """            
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr

        super().__init__(optimizer, last_epoch)

    def _calc_lr(
        self,
        last_epoch: int,
        warmup_steps: int,
        warmup_start_lr: float,
        base_lrs: List[float],
        param_groups: List[Dict],
        get_lr_called_within_step: bool
    ) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler; please use `get_last_lr()`.",
                UserWarning,
            )

        if last_epoch == 0:
            return [warmup_start_lr] * len(base_lrs)
        if last_epoch < warmup_steps:
            return [
                group["lr"] + (base_lr - warmup_start_lr) / (warmup_steps - 1)
                for base_lr, group in zip(base_lrs, param_groups)
            ]
        return base_lrs

    def _calc_lr_closed_form(
        self,
        last_epoch: int,
        warmup_steps: int,
        warmup_start_lr: float,
        base_lrs: List[float]
    ) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if last_epoch < warmup_steps:
            return [
                warmup_start_lr + last_epoch * (base_lr - warmup_start_lr) / (warmup_steps - 1)
                for base_lr in base_lrs
            ]

        return base_lrs

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        return self._calc_lr(
            last_epoch=self.last_epoch,
            warmup_steps=self.warmup_steps,
            warmup_start_lr=self.warmup_start_lr,
            base_lrs=self.base_lrs,
            param_groups=self.optimizer.param_groups,
            get_lr_called_within_step=self._get_lr_called_within_step
        )

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        return self._calc_lr_closed_form(
            last_epoch=self.last_epoch,
            warmup_steps=self.warmup_steps,
            warmup_start_lr=self.warmup_start_lr,
            base_lrs=self.base_lrs
        )
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary into scheduler.
        
        Args:
            state_dict (Dict[str, Any]): State dictionary to load.
        """
        if 'eta_min' in state_dict or 'max_steps' in state_dict:
            raise ValueError(
                "Unrecognized state_dict keys: 'eta_min' and 'max_steps'. Are you trying "
                "to load a cosine annealing scheduler?"
            )
        assert self.warmup_steps == state_dict.get("warmup_steps", self.warmup_steps), \
            "Warmup steps must match"
        assert self.warmup_start_lr == state_dict.get("warmup_start_lr", self.warmup_start_lr), \
            "Warmup start lr must match"
        for blrs, _blrs in zip(self.base_lrs, state_dict.get("base_lrs", self.base_lrs)):
            assert blrs == _blrs, "Base lrs must match"
        
        # Additional validation checks for scheduler state compatibility
        loaded_last_epoch = state_dict.get("last_epoch", self.last_epoch)
        loaded_last_lr = state_dict.get("_last_lr", None)
        loaded_base_lrs = state_dict.get("base_lrs", self.base_lrs)

        # If _last_lr is present, verify it matches what we would compute
        if loaded_last_lr is not None:
            # Compute expected learning rate using our calculation function
            expected_lr = self._calc_lr_closed_form(
                last_epoch=loaded_last_epoch,
                warmup_steps=self.warmup_steps,
                warmup_start_lr=self.warmup_start_lr,
                base_lrs=loaded_base_lrs,
            )
            
            # Check that the loaded last_lr matches expected (with small tolerance for floating point errors)
            if isinstance(loaded_last_lr, list) and len(loaded_last_lr) == len(expected_lr):
                for loaded_lr, expected_lr_val in zip(loaded_last_lr, expected_lr):
                    assert abs(loaded_lr - expected_lr_val) < 1e-7, \
                        f"Loaded last_lr {loaded_lr} doesn't match expected {expected_lr_val} for given state"
                    log.info(f"Loading scheduler state with last_lr {loaded_lr} (expected = {expected_lr_val})")
            else:
                assert False, f"Loaded last_lr format is incompatible: {loaded_last_lr}"
        
        super().load_state_dict(state_dict)


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """COPIED FROM: `pl_bolts<https://github.com/Lightning-Universe/lightning-bolts/blob/2c4602aa684e7b90e7ffdcea1d3f93a20f9c2ead/src/pl_bolts/optimizers/lr_scheduler.py>`__
    
    Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr and
    base_lr followed by a cosine annealing schedule between base_lr and eta_min.

    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.

    Example:
        >>> import torch.nn as nn
        >>> from torch.optim import Adam
        >>> #
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)

    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        max_steps: Optional[int] = None,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            steps_per_epoch (Optional[int]): Number of steps per epoch. If provided, the scheduler will use this value to calculate the number of iterations.
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        assert max_epochs is not None or max_steps is not None, \
            "Either max_epochs or max_steps must be provided"
        assert steps_per_epoch is not None or max_epochs is None, \
            "`steps_per_epoch` must be provided if `max_epochs` is provided"

        if max_epochs is not None:
            max_steps_from_epochs = math.ceil(max_epochs * steps_per_epoch)
            if max_steps is not None:
                max_steps = min(max_steps_from_epochs, max_steps)
            else:
                max_steps = max_steps_from_epochs

        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def _calc_lr(
        self,
        last_epoch: int,
        warmup_steps: int,
        max_steps: int,
        warmup_start_lr: float,
        eta_min: float,
        base_lrs: List[float],
        param_groups: List[Dict],
        get_lr_called_within_step: bool
    ) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler; please use `get_last_lr()`.",
                UserWarning,
            )

        if last_epoch == 0:
            return [warmup_start_lr] * len(base_lrs)
        if last_epoch < warmup_steps:
            return [
                group["lr"] + (base_lr - warmup_start_lr) / (warmup_steps - 1)
                for base_lr, group in zip(base_lrs, param_groups)
            ]
        if last_epoch == warmup_steps:
            return base_lrs
        if (last_epoch - 1 - max_steps) % (2 * (max_steps - warmup_steps)) == 0:
            return [
                group["lr"]
                + (base_lr - eta_min) * (1 - math.cos(math.pi / (max_steps - warmup_steps))) / 2
                for base_lr, group in zip(base_lrs, param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (last_epoch - warmup_steps) / (max_steps - warmup_steps)))
            / (
                1
                + math.cos(
                    math.pi * (last_epoch - warmup_steps - 1) / (max_steps - warmup_steps)
                )
            )
            * (group["lr"] - eta_min)
            + eta_min
            for group in param_groups
        ]

    def _calc_lr_closed_form(
        self,
        last_epoch: int,
        warmup_steps: int,
        max_steps: int,
        warmup_start_lr: float,
        eta_min: float,
        base_lrs: List[float]
    ) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if last_epoch < warmup_steps:
            return [
                warmup_start_lr + last_epoch * (base_lr - warmup_start_lr) / (warmup_steps - 1)
                for base_lr in base_lrs
            ]

        return [
            eta_min
            + 0.5
            * (base_lr - eta_min)
            * (1 + math.cos(math.pi * (last_epoch - warmup_steps) / (max_steps - warmup_steps)))
            for base_lr in base_lrs
        ]

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        return self._calc_lr(
            last_epoch=self.last_epoch,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            warmup_start_lr=self.warmup_start_lr,
            eta_min=self.eta_min,
            base_lrs=self.base_lrs,
            param_groups=self.optimizer.param_groups,
            get_lr_called_within_step=self._get_lr_called_within_step
        )

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        return self._calc_lr_closed_form(
            last_epoch=self.last_epoch,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            warmup_start_lr=self.warmup_start_lr,
            eta_min=self.eta_min,
            base_lrs=self.base_lrs
        )
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary into scheduler.
        
        Args:
            state_dict (Dict[str, Any]): State dictionary to load.
        """
        assert self.warmup_steps == state_dict.get("warmup_steps", self.warmup_steps), \
            "Warmup steps must match"
        assert self.max_steps == state_dict.get("max_steps", self.max_steps), \
            "Max steps must match"
        assert self.warmup_start_lr == state_dict.get("warmup_start_lr", self.warmup_start_lr), \
            "Warmup start lr must match"
        assert self.eta_min == state_dict.get("eta_min", self.eta_min), \
            "Eta min must match"
        for blrs, _blrs in zip(self.base_lrs, state_dict.get("base_lrs", self.base_lrs)):
            assert blrs == _blrs, "Base lrs must match"
        
        # Additional validation checks for scheduler state compatibility
        loaded_last_epoch = state_dict.get("last_epoch", self.last_epoch)
        loaded_step_count = state_dict.get("_step_count", self._step_count)
        loaded_last_lr = state_dict.get("_last_lr", None)
        loaded_base_lrs = state_dict.get("base_lrs", self.base_lrs)

        # If _last_lr is present, verify it matches what we would compute
        if loaded_last_lr is not None:
            # Compute expected learning rate using our calculation function
            expected_lr = self._calc_lr_closed_form(
                last_epoch=loaded_last_epoch,
                warmup_steps=self.warmup_steps,
                max_steps=self.max_steps,
                warmup_start_lr=self.warmup_start_lr,
                eta_min=self.eta_min,
                base_lrs=loaded_base_lrs,
            )
            
            # Check that the loaded last_lr matches expected (with small tolerance for floating point errors)
            if isinstance(loaded_last_lr, list) and len(loaded_last_lr) == len(expected_lr):
                for loaded_lr, expected_lr_val in zip(loaded_last_lr, expected_lr):
                    assert abs(loaded_lr - expected_lr_val) < 1e-7, \
                        f"Loaded last_lr {loaded_lr} doesn't match expected {expected_lr_val} for given state"
                    log.info(f"Loading scheduler state with last_lr {loaded_lr} (expected = {expected_lr_val})")
            else:
                assert False, f"Loaded last_lr format is incompatible: {loaded_last_lr}"
        
        super().load_state_dict(state_dict)