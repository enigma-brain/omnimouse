from typing import Any, Dict, Optional, Tuple, List
import os
import math
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from experanto.utils import LongCycler
from omnimouse.utils.pylogger import RankedLogger
from omnimouse.utils.types import SessionMap
from omnimouse.modeling.model import OmniMouseModel
from omnimouse.utils.utils import (
    LossTrackerCollection,
    get_rng_state,
    set_rng_state,
    ensure_state_dict_compilation_matches,
)
from omnimouse.distributed.utils import (
    broadcast_model_shared,
    broadcast_optimizer,
    broadcast_scheduler,
)

log = RankedLogger(__name__, rank_zero_only=True)


class ModelCheckpoint:
    """
    Model checkpoint handler, adapted from Lightning's
     :class:`~lightning.pytorch.callbacks.ModelCheckpoint` callback.
     Saves the model periodically by monitoring a quantity.
    """
    CHECKPOINT_JOIN_CHAR = "-"
    CHECKPOINT_EQUALS_CHAR = "="
    CHECKPOINT_NAME_BEST = "best"
    CHECKPOINT_NAME_LAST = "last"
    RANK_PREFIX = "rank_"
    FILE_EXTENSION = ".ckpt"

    def __init__(
        self,
        dirpath: str,
        checkpoint_dirname: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: bool = True,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        random_state_saving_enabled: bool = True,
        compile_cache_saving_enabled: bool = True,
    ):
        """Initialize ModelCheckpoint.
        
        Args:
            dirpath: directory to save checkpoints
            checkpoint_dirname: checkpoint directory name format (supports {epoch}, {step} and metric placeholders)
            monitor: quantity to monitor for determining best checkpoints
            verbose: whether to display messages when saving checkpoints
            save_last: when True, creates a symlink named "last.ckpt" to the last checkpoint;
            save_top_k: number of best checkpoints to keep based on monitored metric
            save_weights_only: if True, only save model weights, not optimizer/scheduler
            mode: "min" or "max" for metric monitoring
            every_n_epochs: save a checkpoint every N epochs
            every_n_steps: save a checkpoint every N steps
            enable_version_counter: append version counter to avoid overwriting files
            compile_cache_saving_enabled: whether to save PyTorch compile cache artifacts
            monitor_requires_validation: whether to require validation metric to save checkpoint
        """
        self.dirpath = dirpath
        self.checkpoint_dirname = checkpoint_dirname or "{epoch}-{step}"
        self.monitor = monitor
        self.verbose = verbose
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.random_state_saving_enabled = random_state_saving_enabled
        self.compile_cache_saving_enabled = compile_cache_saving_enabled

        # Initialize best/kth-best tracking variables
        self.best_k_models = {}  # checkpoint_dir -> score mapping
        self.best_model_dir = ""
        self.last_model_dir = ""

        # Default to rank 0
        self._rank = 0
        self._world_size = 1
        
        # Store references to model, optimizer, scheduler, and dataloaders
        self._model = None
        self._shared_optimizer = None
        self._rank_optimizer = None
        self._shared_scheduler = None
        self._rank_scheduler = None
        self._train_dataloader = None
        self._val_dataloaders = None
        self._running_trackers = None
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.dirpath, exist_ok=True)
    
    def assign_modules_and_rank(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        model: Optional[OmniMouseModel] = None,
        shared_optimizer: Optional[torch.optim.Optimizer] = None,
        rank_optimizer: Optional[torch.optim.Optimizer] = None,
        shared_scheduler: Optional[Any] = None,
        rank_scheduler: Optional[Any] = None,
        train_dataloader: Optional[LongCycler] = None,
        val_dataloaders: Optional[Dict[str, LongCycler]] = None,
        running_trackers: Optional[LossTrackerCollection] = None,
    ) -> None:
        """Assign model, optimizer, scheduler, and dataloader references to the checkpoint handler.
        
        Args:
            model: The model to save
            optimizer: The optimizer to save
            scheduler: The learning rate scheduler to save
            train_dataloader: The training dataloader (LongCycler) to save
            val_dataloader: The validation dataloader (LongCycler) to save
            test_dataloader: The test dataloader (LongCycler) to save
        """
        self._rank = rank or self._rank
        self._world_size = world_size or self._world_size
        self._model = model or self._model
        self._shared_optimizer = shared_optimizer or self._shared_optimizer
        self._rank_optimizer = rank_optimizer or self._rank_optimizer
        self._shared_scheduler = shared_scheduler or self._shared_scheduler
        self._rank_scheduler = rank_scheduler or self._rank_scheduler
        self._train_dataloader = train_dataloader or self._train_dataloader
        self._val_dataloaders = val_dataloaders or self._val_dataloaders
        self._running_trackers = running_trackers or self._running_trackers

    @property
    def inf(self) -> torch.Tensor:
        sign = 1 if self.mode == "min" else -1
        return (sign * torch.tensor(float("inf"))).item()
    
    @property
    def device(self) -> torch.device:
        if self._model is None:
            return torch.device("cpu")
        return self._model.device
    
    def _process_metrics(self, metrics: Dict[str, Any]) -> Dict[str, int | float]:
        processed_metrics = {}
        for k, v in metrics.items():
            # Handle tensor metrics
            if isinstance(v, torch.Tensor):
                assert v.numel() == 1, "Tensor metric must be a scalar"
                v = v.item()
            # Validate data type
            if not isinstance(v, (int, float)):
                log.warning(f"Checkpointer ignoring metric '{k}' due to unsupported type: {type(v)}")
                continue
            # Handle NaN scores
            if math.isnan(v):
                log.warning(f"Checkpointer replacing NaN metric '{k}' with infinity")
                v = self.inf
            processed_metrics[k] = v
        return processed_metrics
    
    def _format_checkpoint_dirname(self, metrics: Dict[str, int | float]) -> str:
        """Generate a directory name according to the defined template with optional version."""
        checkpoint_dirname = self.checkpoint_dirname
        # Format checkpoint directory name with metrics
        try:
            checkpoint_dirname = checkpoint_dirname.format(**metrics)
        except Exception as e:
            raise ValueError(
                f"Failed to format checkpoint directory name ({checkpoint_dirname}), "
                f"for metrics: {metrics} [{e}]"
            )
        # Construct full filepath
        checkpoint_dir = os.path.join(self.dirpath, f"{checkpoint_dirname}")

        return checkpoint_dir

    def _create_checkpoint_path(
        self,
        checkpoint_dir: str | os.PathLike,
        rank: int,
        check_exists: bool = False,
    ) -> str:
        # Create checkpoint path for the current rank
        filepath = os.path.join(checkpoint_dir, f"{self.RANK_PREFIX}{rank}{self.FILE_EXTENSION}")
        if check_exists and not os.path.isfile(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        return filepath

    def _remove_file_or_dir(self, path: str | os.PathLike, is_dir: bool = False) -> None:
        # Check if the file exists before attempting to remove it
        if not os.path.exists(path):
            return
        # Delete checkpoint directory
        try:
            if is_dir:
                # Remove "rank_X" files from inside the directory
                for file in os.listdir(path):
                    os.remove(os.path.join(path, file))
                # Remove the checkpoint directory
                os.rmdir(path)
            else:
                # Remove specific file (or symlink)
                os.remove(path)
            if self.verbose:
                log.info(f"Removed checkpoint directory or file: {path}")
        except:
            raise OSError(f"Checkpointer failed to remove checkpoint directory or file: {path}")
    
    def _create_link(self, path: str | os.PathLike, link_name: str, to_dir: bool = True) -> None:
        if not to_dir:
            link_name = f"{link_name}{self.FILE_EXTENSION}"
        # Create the link path in checkpoint directory
        linkpath = os.path.join(self.dirpath, f"{link_name}")
        # Remove the link if it exists
        self._remove_file_or_dir(linkpath, is_dir=False)
        try:
            relpath = os.path.relpath(path, os.path.dirname(linkpath))
            os.symlink(relpath, linkpath)
            if self.verbose:
                log.info(f"Created symlink from {linkpath} to {path}")
        except:
            raise OSError(f"Checkpointer failed to create symlink: {linkpath} -> {path}")

    def _update_top_k(
        self,
        checkpoint_dir: str | os.PathLike,
        metrics: Dict[str, int | float],
        rank: int,
    ) -> bool:
        """Check if the current model should be saved in the top k."""
        # If not saving any models, skip this step and return False
        if self.save_top_k == 0:
            return False
        # Validate that the monitor metric is present in the metrics dict
        current_score = metrics.get(self.monitor)
        # Validate current score
        if current_score is None:
            # TODO: This is a *HACK* to ignore missing monitor
            # NOTE: For self.save_top_k < 0, we don't require the monitor metric to be,
            #  but will have to skip saving a link to the best model below
            # Fill current score with a dummy value if monitor is missing and saving all models
            msg = f"Checkpointer couldn't find monitor ('{self.monitor}') in metrics: {metrics}"
            if self.save_top_k < 0:
                # Only warn if in verbose mode or if a monitor was explicitly provided
                if self.verbose or self.monitor is not None:
                    log.warning(msg + " No `best` link will be created!")
                current_score = self.inf
            else:
                raise ValueError(msg)
        # Create helper functions for determining best/worst models
        get_best, get_worst = (min, max) if self.mode == "min" else (max, min)
        # Insert the current model into the best_k_models dict
        self.best_k_models[checkpoint_dir] = current_score
        # Prune non-top k models (if `self.save_top_k` is not -1)
        while 0 < self.save_top_k < len(self.best_k_models):
            del_path = get_worst(self.best_k_models, key=self.best_k_models.get)
            self.best_k_models.pop(del_path)
            if rank == 0:
                self._remove_file_or_dir(del_path, is_dir=True)
        # Update the link to the best model path, if monitor is present (i.e. save_top_k > 0)
        if self.monitor in metrics:
            self.best_model_dir = get_best(self.best_k_models, key=self.best_k_models.get)
            if rank == 0:
                self._create_link(self.best_model_dir, self.CHECKPOINT_NAME_BEST)
        # Return True if the current model is in the top k
        return checkpoint_dir in self.best_k_models

    def _get_compile_cache(self) -> bytes | None:
        """Retrieve PyTorch compile cache artifacts if enabled."""
        try:
            ret = torch.compiler.save_cache_artifacts()
            assert ret is not None, \
                "`torch.compiler.save_cache_artifacts` returned `None`"
        except Exception as e:
            log.warning(f"Failed to retrieve compile cache artifacts: {e}")
            return None
        compile_artifacts_bytes, compile_artifacts_info = ret
        log.info(f"Retrieved compile cache artifacts for saving: {compile_artifacts_info}")
        return compile_artifacts_bytes

    def _create_and_save_checkpoint(
        self,
        rank: int,
        model: OmniMouseModel,
        epoch: int,
        step: int,
        shared_optimizer: Optional[torch.optim.Optimizer] = None,
        rank_optimizer: Optional[torch.optim.Optimizer] = None,
        shared_scheduler: Optional[Any] = None,
        rank_scheduler: Optional[Any] = None,
        train_dataloader: Optional[LongCycler] = None,
        val_dataloaders: Optional[Dict[str, LongCycler]] = None,
        running_trackers: Optional[LossTrackerCollection] = None,
        metrics: Optional[Dict[str, Any]] = None,
        loss_windows: Optional[Dict[str, List[float]]] = None,
        session_map: Optional[SessionMap] = None,
        filepath: Optional[os.PathLike | str] = None,
        save_random_states: bool = True,
        save_compile_cache: bool = True,
        save_weights_only: bool = False,
    ) -> Dict[str, Any]:

        # Build checkpoint dictionary
        checkpoint = {"epoch": epoch, "step": step}
        # Get state dict for the current rank
        state_dict = model.state_dict()
        # Remove shared parameters from state dict, if not rank 0
        if rank != 0:
            rank_state_dict = {}
            for name, _ in model.rank_named_parameters():
                # TODO: This is a *HACK* to handle compiled models! Fix this!
                if hasattr(model, '_orig_mod'):
                    name = '_orig_mod.' + name
                rank_state_dict[name] = state_dict[name]
            state_dict = rank_state_dict
        checkpoint["state_dict"] = state_dict
        # Get random states for the current rank
        if save_random_states:
            rng_state: Dict[str, Any] = get_rng_state()
            # TODO: Consider uncommenting this to support saving with non-weights only
            # # Modify numpy RNG state to be a tuple of tuples
            # rng_state["numpy_rng_state"] = tuple(
            #     tuple(s) if isinstance(s, np.ndarray)
            #     else s for s in rng_state["numpy_rng_state"]
            # )
            checkpoint["rng_state"] = rng_state
        # Save compile cache artifacts if enabled
        if save_compile_cache:
            compile_artifacts_bytes = self._get_compile_cache()
            if compile_artifacts_bytes is not None:
                checkpoint["compile_artifacts_bytes"] = compile_artifacts_bytes
        # Save optimizer, scheduler, and dataloader state dictionaries if not saving weights only
        if not save_weights_only:
            # NOTE: We only save the shared optimizer on rank 0 (since it's shared across all ranks)
            if shared_optimizer is not None and rank == 0:
                checkpoint["shared_optimizer_state_dict"] = shared_optimizer.state_dict()
            if rank_optimizer is not None:
                checkpoint["rank_optimizer_state_dict"] = rank_optimizer.state_dict()
            if shared_scheduler is not None:
                checkpoint["shared_scheduler_state_dict"] = shared_scheduler.state_dict()
            if rank_scheduler is not None:
                checkpoint["rank_scheduler_state_dict"] = rank_scheduler.state_dict()
            if train_dataloader is not None:
                checkpoint["train_dataloader_state_dict"] = train_dataloader.get_state()
            if val_dataloaders is not None:
                checkpoint["val_dataloader_state_dicts"] = {
                    vk: val_dl.get_state() if val_dl is not None else None
                    for vk, val_dl in val_dataloaders.items()
                }
            if running_trackers is not None:
                checkpoint["running_trackers_state_dict"] = running_trackers.state_dict()
            if metrics is not None:
                checkpoint["metrics"] = metrics
            if session_map is not None:
                checkpoint["session_map"] = session_map.to_dict()
            if loss_windows is not None:
                checkpoint["loss_windows"] = loss_windows
        # Save the checkpoint
        if filepath is not None:
            try:
                if self.verbose:
                    log.info(f"Saving checkpoint to {filepath} with keys: {checkpoint.keys()}")
                torch.save(checkpoint, filepath)
            except Exception as e:
                raise OSError(f"Checkpointer failed to save checkpoint: {filepath} [{e}]")

        return checkpoint
    
    def save_checkpoint(
        self, 
        rank: Optional[int] = None,
        model: Optional[OmniMouseModel] = None,
        session_map: Optional[SessionMap] = None,
        shared_optimizer: Optional[torch.optim.Optimizer] = None,
        rank_optimizer: Optional[torch.optim.Optimizer] = None,
        shared_scheduler: Optional[Any] = None,
        rank_scheduler: Optional[Any] = None,
        train_dataloader: Optional[LongCycler] = None,
        val_dataloaders: Optional[Dict[str, LongCycler]] = None,
        running_trackers: Optional[LossTrackerCollection] = None,
        metrics: Optional[Dict[str, Any]] = None,
        loss_windows: Optional[Dict[str, List[float]]] = None,
        epoch: Optional[int] = None, 
        step: Optional[int] = None,
        save_random_states: Optional[bool] = None,
        save_compile_cache: Optional[bool] = None,
        save_weights_only: Optional[bool] = None,
    ) -> None:
        """Save model checkpoint based on monitoring logic.
        
        Args:
            model: model to save (if not already assigned)
            epoch: current epoch
            step: current step
            metrics: metrics dict for checkpoint name formatting and monitoring
            optimizer: optimizer to save state (if save_weights_only=False)
            scheduler: scheduler to save state (if save_weights_only=False)
            train_dataloader: training dataloader to save state (if save_weights_only=False)
            val_dataloader: validation dataloader to save state (if save_weights_only=False)
            test_dataloader: test dataloader to save state (if save_weights_only=False)
        """
        # Use assigned modules if not provided
        rank = rank or self._rank
        model = model or self._model
        session_map = session_map or model.session_map
        shared_optimizer = shared_optimizer or self._shared_optimizer
        rank_optimizer = rank_optimizer or self._rank_optimizer
        shared_scheduler = shared_scheduler or self._shared_scheduler
        rank_scheduler = rank_scheduler or self._rank_scheduler
        train_dataloader = train_dataloader or self._train_dataloader
        val_dataloaders = val_dataloaders or self._val_dataloaders
        running_trackers = running_trackers or self._running_trackers
        # Use default values if not provided
        if save_random_states is None:
            save_random_states = self.random_state_saving_enabled
        if save_compile_cache is None:
            save_compile_cache = self.compile_cache_saving_enabled
        if save_weights_only is None:
            save_weights_only = self.save_weights_only
        # Validate that the model is assigned
        if model is None:
            raise ValueError("No model provided and no model assigned to checkpoint handler")
        # Process metrics and insert epoch and step
        metrics = self._process_metrics(metrics or {})
        metrics["epoch"] = epoch or 0
        metrics["step"] = step or 0
        # TODO: Implement version counter
        # Construct unique checkpoint filename
        checkpoint_dir = self._format_checkpoint_dirname(metrics)
        # TODO: Is it ok to do this on all ranks?
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Update tracked top k models with current model (or no-op if not in the top k)
        is_top_k = self._update_top_k(checkpoint_dir, metrics, rank)
        # Create a symlink to the last checkpoint (if enabled)
        if self.save_last:
            # Remove previous "last" checkpoint if not in the top k
            if self.last_model_dir not in self.best_k_models:
                if rank == 0:
                    self._remove_file_or_dir(self.last_model_dir, is_dir=True)
            self.last_model_dir = checkpoint_dir
            if rank == 0:
                self._create_link(checkpoint_dir, self.CHECKPOINT_NAME_LAST)
        # Save the checkpoint inside the checkpoint directory
        if is_top_k or self.save_last:
            # Create a checkpoint file for the current rank
            filepath = self._create_checkpoint_path(checkpoint_dir, rank)
            self._create_and_save_checkpoint(
                rank,
                model,
                epoch,
                step,
                shared_optimizer=shared_optimizer,
                rank_optimizer=rank_optimizer,
                shared_scheduler=shared_scheduler,
                rank_scheduler=rank_scheduler,
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloaders,
                running_trackers=running_trackers,
                metrics=metrics,
                loss_windows=loss_windows,
                session_map=session_map,
                filepath=filepath,
                save_random_states=save_random_states,
                save_compile_cache=save_compile_cache,
                save_weights_only=save_weights_only,
            )

    
    def load_session_map(self, checkpoint_dir: str | os.PathLike) -> SessionMap:
        """Load session map from checkpoint directory."""
        try:
            filepath = self._create_checkpoint_path(checkpoint_dir, 0, check_exists=True)
            checkpoint = torch.load(
                filepath,
                map_location=self.device,
                # TODO: Remove this ASAP!
                weights_only=False,
            )
            return SessionMap.from_dict(checkpoint["session_map"])
        except Exception as e:
            error_msg = f"Failed to load session map from checkpoint: {e}"
            log.warning(error_msg)
            raise RuntimeError(error_msg) from e

    def _load_compile_cache(self, compile_artifacts_bytes: bytes) -> None:
        """Load PyTorch compile cache artifacts if enabled."""
        try:
            info = torch.compiler.load_cache_artifacts(compile_artifacts_bytes)
        except Exception as e:
            log.warning(f"Failed to load compile cache artifacts: {e}")
        log.info(f"Successfully loaded compile cache artifacts: {info}")

    def _load_rank_checkpoint(
        self,
        checkpoint_dir: str | os.PathLike,
        rank: int,
        model: OmniMouseModel,
        shared_optimizer: Optional[torch.optim.Optimizer] = None,
        rank_optimizer: Optional[torch.optim.Optimizer] = None,
        shared_scheduler: Optional[Any] = None,
        rank_scheduler: Optional[Any] = None,
        train_dataloader: Optional[LongCycler] = None,
        val_dataloaders: Optional[Dict[str, LongCycler]] = None,
        running_trackers: Optional[LossTrackerCollection] = None,
        strict: bool = True,
        load_random_states: bool = True,
        load_compile_cache: bool = True,
        load_weights_only: bool = False,
        verbose: bool = False,
        load_start_counts: bool = False,
    ) -> None:
        # Build checkpoint filepath for current rank
        filepath = self._create_checkpoint_path(checkpoint_dir, rank, check_exists=True)
        # Load rank-specific checkpoint
        checkpoint = torch.load(
            filepath, map_location=self.device,
            # TODO: Remove this ASAP!
            weights_only=False,
        )
        # Get epoch and step from checkpoint
        if load_start_counts or not load_weights_only:
            epoch, step = checkpoint.get("epoch", 0), checkpoint.get("step", 0)
        else:
            epoch, step = 0, 0
        # Load random states from checkpoint
        if load_random_states:
            rng_state = checkpoint.get("rng_state", None)
            if rng_state is None:
                # NOTE: This is a HACK for backwards compatibility with old checkpoints
                rng_state = {}
                rng_state['torch_rng_state'] = checkpoint.get("torch_rng_state", None)
                rng_state['cuda_rng_state_all'] = checkpoint.get("torch_cuda_rng_state", None)
                rng_state['numpy_rng_state'] = checkpoint.get("numpy_rng_state", None)
                rng_state['python_rng_state'] = checkpoint.get("python_random_state", None)
            if rng_state is None or all(v is None for v in rng_state.values()):
                log.warning("No random states found in checkpoint, skipping")
            else:
                # TODO: This is a HACK to support back-compatibility loading with non-weights only
                # Modify numpy RNG state to be a tuple of np.ndarray's
                if "numpy_rng_state" in rng_state:
                    numpy_rng_state = rng_state["numpy_rng_state"]
                    rng_state["numpy_rng_state"] = tuple(
                        np.array(s) if isinstance(s, tuple) else s
                        for s in numpy_rng_state
                    )
                set_rng_state(rng_state)
        # Load compile cache artifacts if enabled
        if load_compile_cache:
            if "compile_artifacts_bytes" in checkpoint:
                self._load_compile_cache(checkpoint["compile_artifacts_bytes"])
            else:
                log.warning("No compile cache artifacts info found in checkpoint, skipping")
        # Ensure state dict `_orig_mod.` prefix matches model
        state_dict = ensure_state_dict_compilation_matches(model, checkpoint["state_dict"])
        # Load rank-specific parameters. NOTE: We disable strictness for non-zero rank, as
        #  shared parameters *should* be missing from checkpoint (see `save_checkpoint`).
        model.load_state_dict(state_dict, strict=(strict and (rank == 0)))
        # Load optimizer, scheduler, dataloader, and running trackers if not weights only
        if not load_weights_only:
            # NOTE: We only load the shared optimizer/scheduler on rank 0 (and sync across ranks after loading)
            if rank == 0 and shared_optimizer is not None and "shared_optimizer_state_dict" in checkpoint:
                shared_optimizer.load_state_dict(checkpoint["shared_optimizer_state_dict"])
            if rank == 0 and shared_scheduler is not None and "shared_scheduler_state_dict" in checkpoint:
                shared_scheduler.load_state_dict(checkpoint["shared_scheduler_state_dict"])
            if rank_optimizer is not None and "rank_optimizer_state_dict" in checkpoint:
                rank_optimizer.load_state_dict(checkpoint["rank_optimizer_state_dict"])
            if rank_scheduler is not None and "rank_scheduler_state_dict" in checkpoint:
                rank_scheduler.load_state_dict(checkpoint["rank_scheduler_state_dict"])
            if train_dataloader is not None and "train_dataloader_state_dict" in checkpoint:
                train_dataloader.set_state(checkpoint["train_dataloader_state_dict"])
            if val_dataloaders is not None:
                if "val_dataloader_state_dicts" in checkpoint:
                    for val_key, val_dl_state in checkpoint["val_dataloader_state_dicts"].items():
                        if val_dataloaders.get(val_key) is not None:
                            val_dataloaders[val_key].set_state(val_dl_state)
                        else:
                            log.warning(f"{val_key} dataloader not found, skipping loading state from checkpoint")
                    for val_key in val_dataloaders:
                        if checkpoint["val_dataloader_state_dicts"].get(val_key) is None:
                            log.warning(f"State for {val_key} dataloader not found in checkpoint!")
                else:
                    # Backwards compatibility with old checkpoints
                    if "val_dataloader_state_dict" in checkpoint and val_dataloaders.get('validation') is not None:
                        val_dataloaders['validation'].set_state(checkpoint["val_dataloader_state_dict"])
                    if 'test_dataloader_state_dict' in checkpoint and val_dataloaders.get('test') is not None:
                        val_dataloaders['test'].set_state(checkpoint["test_dataloader_state_dict"])
            if running_trackers is not None and "running_trackers_state_dict" in checkpoint:
                running_trackers.load_state_dict(checkpoint["running_trackers_state_dict"], strict=strict)
        
        return epoch, step

    def load_checkpoint(
        self,
        checkpoint_dir: str | os.PathLike,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        model: Optional[OmniMouseModel] = None,
        shared_optimizer: Optional[torch.optim.Optimizer] = None,
        rank_optimizer: Optional[torch.optim.Optimizer] = None,
        shared_scheduler: Optional[Any] = None,
        rank_scheduler: Optional[Any] = None,
        train_dataloader: Optional[LongCycler] = None,
        val_dataloaders: Optional[Dict[str, LongCycler]] = None,
        running_trackers: Optional[LossTrackerCollection] = None,
        strict: bool = True,
        load_random_states: Optional[bool] = None,
        load_compile_cache: Optional[bool] = None,
        load_weights_only: Optional[bool] = None,
        verbose: Optional[bool] = None,
        load_start_counts: bool = False,
    ) -> Tuple[int, int]:
        """Load model checkpoint.
        
        Args:
            checkpoint_dir: Directory containing the checkpoint file for the current rank
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            train_dataloader: Optional training dataloader to load state into
            val_dataloader: Optional validation dataloader to load state into
            test_dataloader: Optional test dataloader to load state into
            strict: Whether to enforce exact matching of state dict keys
            
        Returns:
            Tuple of (epoch, step) from the checkpoint
        """
        # Use assigned modules if not provided
        rank = rank or self._rank
        world_size = world_size or self._world_size
        model = model or self._model
        shared_optimizer = shared_optimizer or self._shared_optimizer
        rank_optimizer = rank_optimizer or self._rank_optimizer
        shared_scheduler = shared_scheduler or self._shared_scheduler
        rank_scheduler = rank_scheduler or self._rank_scheduler
        train_dataloader = train_dataloader or self._train_dataloader
        val_dataloaders = val_dataloaders or self._val_dataloaders
        running_trackers = running_trackers or self._running_trackers
        # Use default values if not provided
        if load_random_states is None:
            load_random_states = self.random_state_saving_enabled
        if load_compile_cache is None:
            load_compile_cache = self.compile_cache_saving_enabled
        if load_weights_only is None:
            load_weights_only = self.save_weights_only
        if verbose is None:
            verbose = self.verbose
        # Validate that the model is assigned
        if model is None:
            raise ValueError("No model provided and no model assigned to checkpoint handler")
        # Check that checkpoint directory exists
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        if verbose:
            log.info(f"Loading checkpoint from {checkpoint_dir}")
        # Validate world size
        ckpt_world_size = len([
            f for f in os.listdir(checkpoint_dir) if f.endswith(self.FILE_EXTENSION)
        ])
        assert world_size in (1, ckpt_world_size), \
            f"Provided world size ({world_size}) must be 1 or match the number of " \
            f"checkpoint files ({ckpt_world_size})"
        # Load checkpoint for world size 1
        if world_size == 1 < ckpt_world_size:
            assert rank == 0, "Rank must be 0 when world size is 1"
            # NOTE: We don't support loading non-model states from multiple ranks into
            #  a single rank, as `torch.optim.Optimizer.load_state_dict` with non-matching
            #  optimizer states will raise an error or exhibit undefined behavior.
            # TODO: Implement `torch.optim.Optimizer.register_load_state_dict_pre_hook`
            #  to handle loading multi-rank optimizer state dicts to a single rank.
            assert load_weights_only, \
                "Weights-only loading is required when world size is less than " \
                f"checkpointed world size (i.e. resuming training on a single rank " \
                "is not supported)"
            for ckpt_rank in range(ckpt_world_size):
                # NOTE: We just use the epoch/step from the last checkpoint file (they should all be the same)
                epoch, step = self._load_rank_checkpoint(
                    checkpoint_dir, ckpt_rank, model,
                    strict=False,
                    load_random_states=load_random_states,
                    load_compile_cache=load_compile_cache,
                    load_weights_only=True,
                    verbose=verbose,
                    load_start_counts=load_start_counts,
                )
        else:
            epoch, step = self._load_rank_checkpoint(
                checkpoint_dir, rank, model,
                shared_optimizer=shared_optimizer,
                rank_optimizer=rank_optimizer,
                shared_scheduler=shared_scheduler,
                rank_scheduler=rank_scheduler,
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloaders,
                running_trackers=running_trackers,
                strict=strict,
                load_random_states=load_random_states,
                load_compile_cache=load_compile_cache,
                load_weights_only=load_weights_only,
                verbose=verbose,
                load_start_counts=load_start_counts,
            )
        # If distributed, broadcast shared states loaded only to rank 0 to all ranks
        if dist.is_initialized():
            # Broadcast model shared parameters and buffers to all ranks
            broadcast_model_shared(model, src=0)
            # Broadcast optimizer state to all ranks. NOTE: We use `states_match=False`
            #  because we haven't taken any steps yet, so `state` will be empty on non-zero
            #  ranks.
            if shared_optimizer is not None:
                broadcast_optimizer(shared_optimizer, src=0, states_match=False, device=self.device,)
            if shared_scheduler is not None:
                broadcast_scheduler(shared_scheduler, src=0)

        return epoch, step
