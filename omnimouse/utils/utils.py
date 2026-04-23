from typing import Any, Callable, Dict, Optional, Literal, List, Sequence
import warnings
import hydra
import random
import hashlib
import math
from statistics import mean
from itertools import product
from importlib.util import find_spec
from functools import wraps
from collections import deque, defaultdict
from omegaconf import DictConfig, OmegaConf, ListConfig
from lightning import Trainer
from lightning.pytorch.loggers import Logger, MLFlowLogger, WandbLogger, CSVLogger
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch import distributed as dist

from omnimouse.masking.strategies import MaskingStrategy
from omnimouse.utils import pylogger, rich_utils
from omnimouse.distributed.utils import cleanup_distributed


log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def dict_config_to_list(**cfg: DictConfig) -> List[Any]:
    """
    Utility to convert a DictConfig to a list of it's values,
    Inteded to support nested instantiation of a `Trainer` object,
    which takes lists of callbacks, loggers, etc.

    TODO: Is this too hacky?
    """
    return list(cfg.values())


def omegaconf_to_dict(cfg: DictConfig, resolve: bool = True) -> Dict[str, Any]:
    if cfg is None:
        return None
    return OmegaConf.to_container(cfg, resolve=resolve)


def resolve_cfg(cfg: DictConfig | None) -> DictConfig | ListConfig | None:
    if cfg is None:
        return None
    return OmegaConf.create(omegaconf_to_dict(cfg, resolve=True))

# TODO: This is hack to address that Hydra isn't resolving dictconfigs!
def omegaconf_to_masking_strategy(cfg: Dict | DictConfig | MaskingStrategy) -> MaskingStrategy:
    if isinstance(cfg, MaskingStrategy):
        return cfg
    assert isinstance(cfg, (DictConfig, Dict)), \
        f"Invalid masking strategy config type: {type(cfg)}! Must be DictConfig or Dict"
    kwargs: Dict[str, Any] = {str(k): v for k, v in cfg.items() if k != '_target_'}
    return MaskingStrategy(**kwargs)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)



def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Any:

        # Intantiate logger (and status) before the task function is executed such that it
        #  can be cleaned up afterwards.
        logger: Optional[Logger] = None
        status: Optional[Literal['success', 'failed']] = None
        if "logger" in cfg.trainer and not cfg.trainer.logger is False:
            log.info("Instantiating loggers...")
            logger = hydra.utils.instantiate(cfg.trainer.logger)

        # execute the task
        try:
            # TODO: Verify call signature of task function
            ret = task_func(cfg=cfg, logger=logger)

            # Set status to success
            status = 'success'

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

            # always close mlflow run (even if exception occurs so multirun won't fail)
            if find_spec("mlflow"):  # check if mlflow is installed
                import mlflow

                if mlflow.active_run():
                    log.info("Closing mlflow!")
                    mlflow.end_run()
            
            # Cleanup logger if it was instantiated
            if logger is not None and isinstance(logger, (
                MLFlowLogger, WandbLogger, CSVLogger,
            )):
                # If status is None, it means the task function didn't succeed
                if status is None:
                    status = 'failed'
                log.info(f"Finalizing {logger.__class__.__name__} with status: {status}")
                logger.finalize(status)

            # handle distributed cleanup
            cleanup_distributed()

        return ret

    return wrap


def get_trainer_metrics(trainer: Trainer) -> Dict[str, Any]:
    """
    Helper to combine callback and logged metrics from a Lightning Trainer.
    """
    # Default to callback metrics
    metrics_dict = trainer.callback_metrics
    # Override with logged metrics
    metrics_dict.update(trainer.logged_metrics)
    return metrics_dict


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value

LOSS_TRACKER_MODES = Literal['mean', 'sum', 'max', 'min', 'last']


class LossTracker:
    def __init__(
        self,
        name: str,
        mode: LOSS_TRACKER_MODES = 'mean',
        window_size: int = -1
    ):
        self.name = name
        assert mode in LOSS_TRACKER_MODES.__args__, \
            f"Invalid mode: {mode}! Must be one of {LOSS_TRACKER_MODES.__args__}"
        self.mode = mode
        # For non-windowed operation
        self.running_loss = 0
        self.running_count = 0
        # For windowed operation if window_size > 0
        self.window_size = window_size
        self.loss_window = deque(maxlen=window_size) if window_size > 0 else None
    
    @property
    def updated(self) -> bool:
        return self.running_count > 0

    def update(self, loss: float | Tensor | None | List[float | Tensor | None]):
        # Handle list of losses (from gradient accumulation)
        if isinstance(loss, list):
            loss = [
                l.detach().item() if isinstance(l, Tensor) else l
                for l in loss if l is not None
            ]
            loss = mean(loss) if loss else None
        if loss is not None:
            if isinstance(loss, Tensor):
                loss = loss.detach().item()
            self.running_count += 1
            # Update the running statistics
            if self.mode == 'mean':
                self.running_loss += (loss - self.running_loss) / self.running_count
            elif self.mode == 'sum':
                self.running_loss += loss
            elif self.mode == 'max':
                self.running_loss = max(self.running_loss, loss)
            elif self.mode == 'min':
                self.running_loss = min(self.running_loss, loss)
            elif self.mode == 'last':
                self.running_loss = loss
            # Update the window
            if self.loss_window is not None:
                self.loss_window.append(loss)
    
    def compute(self) -> float:
        # If windowing is disabled, use the running stats
        if self.loss_window is None:
            if not self.running_count:
                return float('nan')
            return self.running_loss
        # Compute the windowed metric
        if not len(self.loss_window):
            return float('nan')
        if self.mode == 'mean':
            return sum(self.loss_window) / len(self.loss_window)
        elif self.mode == 'sum':
            return sum(self.loss_window)
        elif self.mode == 'max':
            return max(self.loss_window)
        elif self.mode == 'min':
            return min(self.loss_window)
        elif self.mode == 'last':
            return self.loss_window[-1]
        raise ValueError(
            f"Invalid mode: {self.mode}! Must be one of {LOSS_TRACKER_MODES.__args__}"
        )
    
    def reset(self):
        self.running_loss = 0
        self.running_count = 0
        if self.loss_window is not None:
            self.loss_window.clear()
    
    def state_dict(self) -> Dict[str, Any]:
        ret = {
            "name": self.name,
            "mode": self.mode,
            "running_loss": self.running_loss,
            "running_count": self.running_count,
            "window_size": self.window_size,
        }
        if self.loss_window is not None:
            ret["loss_window"] = list(self.loss_window)
            ret["loss_window_maxlen"] = self.loss_window.maxlen
        return ret
    
    # TODO: Make more `load_state_dict`-ish (i.e. only load "states" and support strict/unstrict loading)
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        self.name = state_dict["name"]
        self.mode = state_dict["mode"]
        self.running_loss = state_dict["running_loss"]
        self.running_count = state_dict["running_count"]
        self.window_size = state_dict.get("window_size", self.window_size)
        if "loss_window" in state_dict:
            assert state_dict["loss_window_maxlen"] == self.window_size, \
                f"Window size mismatch: {state_dict['loss_window_maxlen']} != {self.window_size}"
            self.loss_window = deque(
                state_dict["loss_window"],
                maxlen=state_dict["loss_window_maxlen"]
            )



class LossTrackerCollection:
    JOIN_CHAR = '/'
    RANK_PREFIX = 'rank_'
    SESS_PREFIX = 'sess_'
    
    def __init__(
        self,
        names: List[str],
        mode: LOSS_TRACKER_MODES = 'mean',
        window_size: int = -1,
        prefix: List[str] | str = '',
        rank: int = 0,
        session_names: Optional[List[str]] = None,
    ):
        assert len(set(names)) == len(names), \
            f"Duplicate names in names: {names}"
        assert all(self.JOIN_CHAR not in n for n in names), \
            f'Metric names cannot contain "{self.JOIN_CHAR}"'
        self.trackers = {}
        # NOTE: We prepend `None` for overall tracker
        for session_name, name in product([None] + (session_names or []), names):
            if session_name is not None:
                # Construct the session-specific tracker name
                name = self.join(f'{self.SESS_PREFIX}{session_name}', name)
            self.trackers[name] = LossTracker(name, mode, window_size)
        self._prefix = self._validated_prefix(prefix)
        self._rank = rank
        self._session_names = session_names
    
    @classmethod
    def _validated_prefix(cls, prefix: List[str] | str) -> List[str]:
        prefix = prefix if isinstance(prefix, List) else [prefix]
        reserved = [cls.RANK_PREFIX, cls.SESS_PREFIX]
        assert not any(p.startswith(r) for r in reserved for p in prefix), \
            f"Prefix item(s) has reserved prefix ({reserved}): {prefix}"
        return prefix
    
    @classmethod
    def join(cls, *args: str) -> str:
        return cls.JOIN_CHAR.join(args)
    
    def prefix(self, name: str, rank: Optional[int | str] = None) -> str:
        _rank = self._rank if rank is None else rank
        return self.join(
            *self._prefix,
            f'{self.RANK_PREFIX}{_rank}',
            name,
        )
    
    @property
    def window_size(self) -> int:
        return next(iter(self.trackers.values())).window_size
    
    @property
    def updated_trackers(self) -> Dict[str, LossTracker]:
        return {name: tracker for name, tracker in self.trackers.items() if tracker.updated}
        
    def update(
        self,
        session_name: Optional[str | List[str]] = None,
        **kwargs: float | Tensor | None | List[float | Tensor | None],
    ):
        for name, loss in kwargs.items():
            # Update the overall tracker for given metric
            self.trackers[name].update(loss)
            # If session name is provided, also update the session-specific tracker. NOTE:
            #  With gradient accumulation, we may have multiple sessions per batch.
            if session_name is not None:
                # Wrap single session name in list
                if not isinstance(session_name, List):
                    session_name = [session_name]
                # Verify that we have either one loss, or one loss per session
                if isinstance(loss, Sequence):
                    assert len(session_name) == len(loss), \
                        f"Num sess names ({len(session_name)}) != losses ({len(loss)})!"
                # Update the session-specific trackers with session-specific losses
                for i, s in enumerate(session_name):
                    assert s in (self._session_names or []), \
                        f"{session_name} not found in session names: {self._session_names}"
                    _loss = loss[i] if isinstance(loss, Sequence) else loss
                    # Construct the session-specific tracker name
                    _name = self.join(f'{self.SESS_PREFIX}{s}', name)
                    self.trackers[_name].update(_loss)
        
    def compute(self, updated_only: bool = False) -> Dict[str, float]:
        trackers = self.updated_trackers if updated_only else self.trackers
        return {
            self.prefix(name): tracker.compute() for name, tracker in trackers.items()
        }
    
    @property
    def loss_windows(self) -> Dict[str, List[float]]:
        return {
            name: list(tracker.loss_window) for name, tracker in self.trackers.items()
        }
        
    def reset(self):
        for tracker in self.trackers.values():
            tracker.reset()
    
    def state_dict(self) -> Dict[str, Any]:
        tracker_state_dicts = {}
        for name, tracker in self.trackers.items():
            tracker_state_dicts[name] = tracker.state_dict()
        return {
            "trackers": tracker_state_dicts,
            # These aren't really states, but we'll keep them here for convenience
            "prefix": self._prefix,
            "rank": self._rank,
            "session_names": self._session_names,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        if strict:
            keys = set(self.trackers.keys())
            state_keys = set(state_dict["trackers"].keys())
            assert keys == state_keys, \
                f"Trackers in state dict don't match: {keys} != {state_keys}"
        for name, tracker in self.trackers.items():
            _state_dict = state_dict["trackers"].get(name, None)
            if _state_dict is not None:
                tracker.load_state_dict(_state_dict, strict=strict)
        window_sizes = [tracker.window_size for tracker in self.trackers.values()]
        assert len(set(window_sizes)) == 1, \
            f"Window sizes don't match: {window_sizes}"

    @classmethod
    def summarize_distributed_results(cls, results: Dict[str, float]) -> Dict[str, float]:
        """Combine results from multiple distributed processes
        
        TODO: This is super hacky
        """
        combined_rank_results, combined_rank_counts = defaultdict(float), defaultdict(int)
        combined_sess_results, combined_sess_counts = defaultdict(float), defaultdict(int)
        for key, value in results.items():
            # Skip NaN values
            if math.isnan(value):
                continue
            key_parts = key.split(cls.JOIN_CHAR)
            rank_idx = [i for i,p in enumerate(key_parts) if p.startswith(cls.RANK_PREFIX)][0]
            key_parts.pop(rank_idx)
            sess_idx = [i for i,p in enumerate(key_parts) if p.startswith(cls.SESS_PREFIX)]
            if sess_idx:
                key_parts.pop(sess_idx[0])
                key_parts.insert(rank_idx, 'sess_overall')
                overall_prefix = cls.join(*key_parts)
                combined_sess_results[overall_prefix] += value
                combined_sess_counts[overall_prefix] += 1
            else:
                key_parts.insert(rank_idx, 'rank_overall')
                overall_prefix = cls.join(*key_parts)
                combined_rank_results[overall_prefix] += value
                combined_rank_counts[overall_prefix] += 1
        combined_rank_results = {
             key: value / combined_rank_counts[key]
             for key, value in combined_rank_results.items()
         }
        combined_sess_results = {
             key: value / combined_sess_counts[key]
             for key, value in combined_sess_results.items()
        }
        return dict(
            **combined_rank_results,
            **combined_sess_results
        )

def get_rng_state() -> Dict[str, Any]:
    """Save current RNG states"""
    return {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate()
    }


def set_rng_state(rng_state: Dict[str, Any]):
    """Restore RNG states"""
    # Helper in case tensors are on GPU
    def _move_to_cpu(state: Tensor | List[Tensor]) -> Tensor | List[Tensor]:
        """Ensure state is on CPU"""
        if isinstance(state, Tensor):
            return state.cpu()
        elif isinstance(state, list):
            return [_move_to_cpu(s) for s in state]
        raise ValueError(f"Invalid state type: {type(state)}")
    # Torch RNG state
    torch_rng_state = rng_state.get('torch_rng_state', None)
    if torch_rng_state is not None:
        torch_rng_state = _move_to_cpu(torch_rng_state)
        torch.set_rng_state(torch_rng_state)
    # CUDA RNG state (and full RNG state)
    cuda_rng_state = rng_state.get('cuda_rng_state', None)
    if cuda_rng_state is not None and torch.cuda.is_available():
        cuda_rng_state = _move_to_cpu(cuda_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
    cuda_rng_state_all = rng_state.get('cuda_rng_state_all', None)
    if (
        cuda_rng_state_all is not None and
        torch.cuda.is_available() and
        dist.is_initialized() and
        dist.get_world_size() == len(cuda_rng_state_all)
    ):
        cuda_rng_state_all = _move_to_cpu(cuda_rng_state_all)
        torch.cuda.set_rng_state_all(cuda_rng_state_all)
    # Numpy RNG state
    numpy_rng_state = rng_state.get('numpy_rng_state', None)
    if numpy_rng_state is not None:
        np.random.set_state(numpy_rng_state)
    # Python random RNG state
    python_rng_state = rng_state.get('python_rng_state', None)
    if python_rng_state is not None:
        random.setstate(python_rng_state)


def preserve_rng_state(func: Callable) -> Callable:
    """Decorator that preserves and restores all RNG states during function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Save current RNG states
        rng_state = get_rng_state()
        
        try:
            # Execute the wrapped function
            result = func(*args, **kwargs)
            return result
        finally:
            # Restore RNG states
            set_rng_state(rng_state)
    
    return wrapper


def ensure_state_dict_compilation_matches(
    model: nn.Module,
    state_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """This handles the case where the checkpoint was saved from a compiled model
    (with '_orig_mod.' prefix) but we're loading into a non-compiled model,
    or vice versa.

    Args:
        model: nn.Module to check compilation against
        state_dict: State dict to check
    """
    # Check if state dict has '_orig_mod.' prefix
    has_orig_mod_prefix = any(k.startswith('_orig_mod.') for k in state_dict.keys())

    # Check if current model is compiled (has _orig_mod attribute)
    is_compiled = hasattr(model, '_orig_mod')

    # If there's a mismatch, we need to adjust the keys
    if has_orig_mod_prefix and not is_compiled:
        # Remove '_orig_mod.' prefix from state dict keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v  # Remove '_orig_mod.' prefix (10 chars)
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    elif not has_orig_mod_prefix and is_compiled:
        # Add '_orig_mod.' prefix to state dict keys
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[f'_orig_mod.{k}'] = v
        state_dict = new_state_dict

    return state_dict