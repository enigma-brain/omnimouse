import math
import os
import re
import time
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from os import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import hydra
import lightning as L
import mlflow
import numpy as np
import rootutils
import torch
import torch.distributed as dist
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import PolynomialLR
from torchmetrics import PearsonCorrCoef
from tqdm.auto import tqdm

# Handles initialization of project root dir
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from omnimouse.distributed.utils import (
    SyncedFastSessionDataLoader,
    gather_dataset_info,
    gather_validation_metrics,
    get_dataset_list,
    maybe_get_validation_concat_loader,
    prepare_synced_dataloaders,
    setup_distributed,
)
from omnimouse.experiments.pretraining_dataloaders import get_pretraining_dataloader
from omnimouse.experiments.scaling_laws import colossus
from omnimouse.masking.strategies import (
    MaskingStrategy,
)
from omnimouse.metrics import MaskedPopulationPearsonCorrCoef
from omnimouse.modeling import Model, ModelArgs, OMModelOutput, build_model
from omnimouse.optimization.schedulers import (
    LinearWarmupCosineAnnealingLR,
    LinearWarmupLR,
)
from omnimouse.utils import (
    LossTrackerCollection,
    ModelCheckpoint,
    RankedLogger,
    SessionMap,
    SessionMetadata,
    get_rng_state,
    log_hyperparameters,
    omegaconf_to_masking_strategy,
    preserve_rng_state,
    print_model_summary,
    set_rng_state,
)

log = RankedLogger(__name__, rank_zero_only=True)

# Setup precision settings
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
mlflow.enable_system_metrics_logging()

# Manually disable distributed syncing of torchmetrics metrics
TORCHMETRIC_DIST_KWARGS_OVERRIDE = {
    "process_group": None,
    "sync_on_compute": False,
    "dist_sync_fn": None,
}

# ------------------------------------------------------------
# Setup Functions
# ------------------------------------------------------------


def set_deterministic(deterministic: bool | None = None, benchmark: bool | None = None):
    """See `here <https://github.com/Lightning-AI/pytorch-lightning/blob/831870a15a17cca4152ffde5c8a3ebc535c68ab0/src/lightning/pytorch/trainer/connectors/accelerator_connector.py#L631>__`"""
    if deterministic:
        if benchmark is None:
            # Set benchmark to False to ensure determinism
            benchmark = False
        elif benchmark:
            log.warning(
                "You passed `deterministic=True` and `benchmark=True`. Note that PyTorch ignores"
                " torch.backends.cudnn.deterministic=True when torch.backends.cudnn.benchmark=True.",
            )
    if benchmark is not None:
        torch.backends.cudnn.benchmark = benchmark
    if isinstance(deterministic, bool):
        # do not call this if deterministic wasn't passed
        torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def setup_cuda(
    enable_cudnn_sdp: bool,
    deterministic: Optional[bool] = None,
    benchmark: Optional[bool] = None,
) -> bool:
    """Setup CUDA settings for precision / sdp / determinism"""
    if not torch.cuda.is_available():
        return False
    # precision settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch._dynamo.config.cache_size_limit = 4096
    # sdp settings
    if enable_cudnn_sdp:
        torch.backends.cuda.enable_cudnn_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
    # determinism settings
    set_deterministic(deterministic, benchmark)
    # return success
    return True


def setup_dataloaders_single_rank(
    data_root: str | os.PathLike,
    dataset_collection: str | List[str] | Dict[str, List[str]],
    dataloader_cfg: DictConfig,
    train_dataset_cfg: DictConfig,
    pretraining_datasets: Optional[list[str]] = None,
    validation_dataset_cfgs: Optional[Dict[str, DictConfig]] = None,
    validation_dataset_sets: Optional[Dict[str, list[str]]] = None,
    seed: Optional[int] = None,
) -> Tuple[SyncedFastSessionDataLoader, List[SyncedFastSessionDataLoader]]:
    """Setup dataloaders for a single rank."""
    if isinstance(dataset_collection, str):
        dataset_dict = get_dataset_list(dataset_collection)
    else:
        dataset_dict = dataset_collection
    # Flatten dataset dict if world_size == 1
    if isinstance(dataset_dict, dict):
        dataset_dict = [d for l in dataset_dict.values() for d in l]
    rank_paths = np.array([path.join(data_root, f) for f in dataset_dict])

    if pretraining_datasets is None:
        pretraining_datasets = colossus
    elif isinstance(pretraining_datasets, str):
        pretraining_datasets = get_dataset_list(pretraining_datasets)

    log.info(f"Creating dataloader with rank_paths: {rank_paths}")
    train_dataloader = get_pretraining_dataloader(
        paths=rank_paths,
        configs=deepcopy(train_dataset_cfg),
        seed=seed,
        pretrain_lists=pretraining_datasets,
        dataloader_config=dataloader_cfg,
    )
    # NOTE: We wrap the train dataloader in a dummy SyncedFastSessionDataLoader just to match
    #  distributed training setup
    train_dataloader = SyncedFastSessionDataLoader(
        original_dataloader=train_dataloader, synced_length=len(train_dataloader)
    )

    # Manually disable dataloader shuffling / drop last
    val_dataloader_cfg = deepcopy(dataloader_cfg)
    val_dataloader_cfg.shuffle = False
    val_dataloader_cfg.drop_last = False

    validation_dataloaders = {}
    if validation_dataset_sets is not None:
        assert validation_dataset_cfgs is not None, (
            "Validation dataset sets must come with a corresponding configs!"
        )
        assert set(validation_dataset_sets.keys()) == set(
            validation_dataset_cfgs.keys()
        ), "Keys for validation dataset sets and configs must match!"
        for val_key, val_set in validation_dataset_sets.items():
            val_set = get_dataset_list(val_set)
            val_ds_cfg = validation_dataset_cfgs[val_key]
            val_dl = maybe_get_validation_concat_loader(
                deepcopy(val_ds_cfg),
                rank_paths,
                fixed_validation_set=val_set,
                dl_config=val_dataloader_cfg,
                seed=seed,
            )
            validation_dataloaders[val_key] = val_dl

    log.info("Finished dataloader creation")
    msg = f"Assigned keys:\n\tTrain: {train_dataloader.dataset.session_names}"
    for val_key, val_dl in validation_dataloaders.items():
        val_sess = None
        if val_dl is not None:
            val_sess = val_dl.dataset.session_names
        msg += f"\n\t{val_key}: {val_sess}"
    log.info(msg)

    return train_dataloader, validation_dataloaders


def setup_dataloaders(
    data_root: str | os.PathLike,
    dataset_collection: str,
    dataloader_cfg: DictConfig,
    train_dataset_cfg: DictConfig,
    pretraining_datasets: Optional[list[str]] = None,
    validation_dataset_cfgs: Optional[Dict[str, DictConfig]] = None,
    validation_dataset_sets: Optional[Dict[str, list[str]]] = None,
    rank: int = 0,
    world_size: int = 1,
    seed: Optional[int] = None,
    init_sleep_time_per_rank: int | float = 0.0,
    max_val_sessions_per_rank: Optional[int] = None,
    flatten_datasets_if_single_rank: bool = False,
) -> Tuple[SyncedFastSessionDataLoader, List[SyncedFastSessionDataLoader]]:
    """Setup dataloaders for training, validation, and testing."""

    dataset_dict = get_dataset_list(dataset_collection)
    # Flatten dataset dict if world_size == 1
    if (
        flatten_datasets_if_single_rank
        and world_size == 1
        and isinstance(dataset_dict, dict)
    ):
        dataset_dict = [d for l in dataset_dict.values() for d in l]
    rank_paths = dataset_dict[rank] if type(dataset_dict) is dict else dataset_dict
    rank_paths = np.array([path.join(data_root, f) for f in rank_paths])
    log.info(f"Rank {rank} has {len(rank_paths)} datasets: {rank_paths}")

    sleep_duration = rank * init_sleep_time_per_rank
    time.sleep(sleep_duration)

    if pretraining_datasets is None:
        pretraining_datasets = colossus
    else:
        pretraining_datasets = get_dataset_list(pretraining_datasets)

    log.info(f"creating dataloader on rank {rank}, with rank_paths: {rank_paths}")
    train_dataloader = get_pretraining_dataloader(
        paths=rank_paths,
        configs=deepcopy(train_dataset_cfg),
        seed=seed,
        pretrain_lists=pretraining_datasets,
        dataloader_config=dataloader_cfg,
    )
    log.info(f"== {rank} getting synced loader")
    log.info(f"DL length before sync: {len(train_dataloader)}")
    train_dataloader = prepare_synced_dataloaders(
        train_dataloader,
        rank,
        world_size,
    )
    log.info(f"DL length after sync!! {len(train_dataloader)}")

    # Manually disable dataloader shuffling / drop last
    val_dataloader_cfg = deepcopy(dataloader_cfg)
    val_dataloader_cfg.shuffle = False
    val_dataloader_cfg.drop_last = False

    validation_dataloaders = {}
    if validation_dataset_sets is not None:
        assert validation_dataset_cfgs is not None, (
            "Validation dataset sets must come with a corresponding configs!"
        )
        assert set(validation_dataset_sets.keys()) == set(
            validation_dataset_cfgs.keys()
        ), "Keys for validation dataset sets and configs must match!"
        for val_key, val_set in validation_dataset_sets.items():
            val_set = get_dataset_list(val_set)
            val_ds_cfg = validation_dataset_cfgs[val_key]
            val_dl = maybe_get_validation_concat_loader(
                deepcopy(val_ds_cfg),
                rank_paths,
                fixed_validation_set=val_set,
                max_sessions=max_val_sessions_per_rank,
                dl_config=val_dataloader_cfg,
                seed=seed,
            )
            validation_dataloaders[val_key] = val_dl

    log.info(f"finished dataloader on rank {rank}")
    msg = (
        f"Rank {rank} Assigned keys:\n\tTrain: {train_dataloader.dataset.session_names}"
    )
    for val_key, val_dl in validation_dataloaders.items():
        val_sess = None
        if val_dl is not None:
            val_sess = val_dl.dataset.session_names
        msg += f"\n\t{val_key}: {val_sess}"
    log.info(msg)

    return train_dataloader, validation_dataloaders


def compile_model(model: Model, compile_cfg: DictConfig, disable_compile: bool = False):
    # No-op if compile disabling override flag is set, or compile is disabled in config
    if disable_compile or not compile_cfg.enabled:
        return model

    torch._dynamo.config.cache_size_limit = compile_cfg.dynamo_cache_size_limit
    torch._dynamo.config.accumulated_cache_size_limit = (
        compile_cfg.accumulated_cache_size_limit
    )
    compile_mode = compile_cfg.mode

    if compile_cfg.mode == "max-autotune":
        torch._inductor.config.benchmark_kernel = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.max_autotune = True
        torch._inductor.config.max_autotune_gemm = True
        torch._inductor.config.max_autotune_pointwise = True
        torch._inductor.config.triton.autotune_pointwise = True
        torch._inductor.config.triton.autotune_cublasLt = True
        torch._inductor.config.epilogue_fusion = True
        torch._inductor.config.prologue_fusion = True

    if compile_mode is None:
        compile_options = {
            "coordinate_descent_tuning": compile_cfg.coordinate_descent_tuning,
            "prologue_fusion": compile_cfg.prologue_fusion,
            "epilogue_fusion": compile_cfg.epilogue_fusion,
            "max_autotune": compile_cfg.max_autotune,
        }
    else:
        # if compile mode is set, then options have to be None
        compile_options = None

    log.info(f"Compiling model with options: {compile_options or compile_mode}")
    model = torch.compile(
        model,
        fullgraph=compile_cfg.fullgraph,
        dynamic=compile_cfg.dynamic,
        mode=compile_mode,
        options=compile_options,
    )
    return model


def setup_model(
    dataset_info: Dict[str, Dict[str, int | str]],
    model_cfg: Optional[DictConfig],
    trainer_cfg: Optional[DictConfig],
    local_session_names: Optional[Sequence[str]] = None,
    device: torch.device | str = "cpu",
    world_size: int = 1,
) -> Model:
    """
    Instantiate and initialize model.

    Args:
        dataset_info: Dictionary of dataset information with session keys as keys and dictionaries as values.
                     Each inner dictionary contains at least 'n_neurons' (int) and 'animal_id' (str) keys.
        model_cfg: Model configuration
        trainer_cfg: Trainer configuration
        local_session_names: Optional sequence of session names that are local to this rank
        device: Device to place the model on
        world_size: Number of processes in the distributed training
    """
    local_session_names = local_session_names or dataset_info.keys()
    session_map = SessionMap()
    for session_key, info in dataset_info.items():
        session_attrs = SessionMetadata(
            n_neurons=info.get(
                "n_neurons", 9000
            ),  # NOTE: If n_neurons not provided, use 9000 as "upper bound"
            animal_id=info.get(
                "animal_id", session_key
            ),  # NOTE: If animal id not provided, use session key as animal id, i.e. assume unique animals per session
            on_rank=(
                session_key in local_session_names or world_size == 1
            ),  # True only if this session is on the current rank
        )
        session_map.update(session_key, session_attrs)

    # Instantiate model
    log.info(f"Instantiating model <{model_cfg._target_}>")
    model_config: ModelArgs = hydra.utils.instantiate(model_cfg)

    # We initialize model weights directly on the device and compile, if requested. See
    #  `lightning docs<https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html#speed-up-model-initialization>__`
    #  See discussion on order of compile/device placement/ddp `here <https://discuss.pytorch.org/t/torch-compile-before-or-after-cuda/176031>`__
    with torch.device(device):
        model = build_model(model_config, session_map)
        # Should be a no-op with context manager, but just in case...
        model = model.to(device)
        # Compile model based on configuration
        model = compile_model(model, trainer_cfg.compile)
    return model


def parse_validation_dataset_cfgs(
    data_cfg: DictConfig,
) -> Tuple[Dict[str, DictConfig], Dict[str, list[str]]]:
    """Parse validation dataset configs and sets from config, handling new and legacy behavior."""
    # Construct validation dataset sets and configs dictionaries, while maintaining legacy behavior
    validation_dataset_sets, validation_dataset_cfgs = {}, {}
    if (_legacy_val_ds_cfg := data_cfg.get("validation_dataset")) is not None:
        validation_dataset_cfgs["validation"] = _legacy_val_ds_cfg
        if (_legacy_val_ds_set := data_cfg.get("validation_datasets")) is not None:
            validation_dataset_sets["validation"] = _legacy_val_ds_set
        else:
            validation_dataset_sets["validation"] = "sensorium_2023_test_set"
    if (_legacy_test_ds_cfg := data_cfg.get("test_dataset")) is not None:
        validation_dataset_cfgs["test"] = _legacy_test_ds_cfg
        if (_legacy_test_ds_set := data_cfg.get("test_datasets")) is not None:
            validation_dataset_sets["test"] = _legacy_test_ds_set
        else:
            validation_dataset_sets["test"] = "sensorium_2023_test_set"
    if (
        hasattr(data_cfg, "additional_validation_dataset_cfgs")
        and data_cfg.additional_validation_dataset_cfgs is not None
    ):
        assert (
            hasattr(data_cfg, "additional_validation_dataset_sets")
            and data_cfg.additional_validation_dataset_sets is not None
        ), "Additional validation dataset configs must come with a corresponding sets!"
        additional_validation_dataset_cfgs = data_cfg.additional_validation_dataset_cfgs
        additional_validation_dataset_sets = data_cfg.additional_validation_dataset_sets
        if _legacy_val_ds_cfg is not None:
            assert "validation" not in additional_validation_dataset_cfgs, (
                "If validation dataset explicitly provided, it cannot be in the "
                "additional validation dataset configs!"
            )
        if _legacy_test_ds_cfg is not None:
            assert "test" not in additional_validation_dataset_cfgs, (
                "If test dataset explicitly provided, it cannot be in the "
                "additional test dataset configs!"
            )
        validation_dataset_cfgs.update(additional_validation_dataset_cfgs)
        validation_dataset_sets.update(additional_validation_dataset_sets)

    return validation_dataset_sets, validation_dataset_cfgs


def setup_common(
    cfg: DictConfig,
    logger: Optional[Logger] = None,
    flatten_datasets_if_single_rank: bool = False,
) -> Tuple[
    SyncedFastSessionDataLoader,
    Dict[str, SyncedFastSessionDataLoader],
    Model,
    Logger,
    torch.device,
    int,
    int,
    int,
    Optional[Dict[str, Any]],
    Optional[Dict[str, Dict[str, Any]]],
]:
    """Common setup code for both training and evaluation."""
    rank, local_rank, world_size, device = setup_distributed()

    cuda_is_available = setup_cuda(
        enable_cudnn_sdp=cfg.trainer.enable_cudnn_sdp,
        deterministic=cfg.trainer.get("deterministic", None),
        benchmark=cfg.trainer.get("benchmark", None),
    )

    # Set seed for random number generators in pytorch, numpy and python.random
    seed = cfg.get("seed", None)
    if cfg.get("use_seed_per_rank", False):
        seed = seed + rank
    if seed is not None:
        L.seed_everything(seed, workers=True)
    # Get initial random state to use for evaluation consistency
    fix_eval_rng_state = cfg.get("fix_eval_rng_state", False)
    initial_rng_state = get_rng_state() if fix_eval_rng_state else None

    # Create output dir if it doesn't exist
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate dataloaders
    log.info("Instantiating dataloaders>")
    # Resolve the config before passing it to the dataloader setup
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))

    validation_dataset_sets, validation_dataset_cfgs = parse_validation_dataset_cfgs(
        data_cfg
    )

    train_dl, val_dls = setup_dataloaders(
        data_root=cfg.paths.data_dir,
        dataset_collection=data_cfg.dataset_collection,
        dataloader_cfg=data_cfg.dataloader,
        train_dataset_cfg=data_cfg.train_dataset,
        pretraining_datasets=data_cfg.get("pretraining_datasets", None),
        validation_dataset_cfgs=validation_dataset_cfgs,
        validation_dataset_sets=validation_dataset_sets,
        rank=rank,
        world_size=world_size,
        seed=seed,
        init_sleep_time_per_rank=cfg.distributed.init_sleep_time_per_rank,
        max_val_sessions_per_rank=cfg.distributed.max_val_sessions_per_rank,
        flatten_datasets_if_single_rank=flatten_datasets_if_single_rank,
    )
    # TODO: Standardize keys. Currently must match those used as `phase` in `run_evaluation`
    initial_dl_rng_states = None
    if fix_eval_rng_state:
        initial_dl_rng_states = {}
        initial_dl_rng_states["train"] = train_dl.get_state()
        for val_dl_key, val_dl in val_dls.items():
            val_dl_state = None
            if val_dl is not None:
                val_dl_state = val_dl.get_state()
            initial_dl_rng_states[val_dl_key] = val_dl_state
    # Gather dataset info
    combined_info = gather_dataset_info(train_dl, rank, world_size)

    # Now create SessionMap with the full info
    local_session_names = train_dl.dataset.session_names

    # Instantiate model
    model = setup_model(
        dataset_info=combined_info,
        model_cfg=cfg.model,
        trainer_cfg=cfg.trainer,
        local_session_names=local_session_names,
        device=device,
        world_size=world_size,
    )
    # Initialize loggers if provided in config
    if logger is None and "logger" in cfg.trainer and cfg.trainer.logger is not False:
        log.info("Instantiating loggers...")
        logger: Logger = hydra.utils.instantiate(cfg.trainer.logger)

    # Prepare object dict for hyperparameter logging
    object_dict = {
        "cfg": deepcopy(cfg),
        "model": model,
        "logger": logger,
        "root": root,
    }

    # Log hyperparameters
    log.info("Logging hyperparameters!")
    log_hyperparameters(object_dict)

    # Print model summary
    print_model_summary(model, cfg.trainer.get("model_summary_depth", 2))

    return (
        train_dl,
        val_dls,
        model,
        logger,
        device,
        rank,
        local_rank,
        world_size,
        initial_rng_state,
        initial_dl_rng_states,
    )


def create_optimizers(cfg, model) -> Tuple[Optimizer, Optimizer]:
    """Create optimizers for the model based config and rank overwrites"""

    base_opt_fn = hydra.utils.instantiate(cfg.trainer.optimizer)

    # Shared optimizer with base settings
    shared_optimizer = base_opt_fn(model.shared_named_parameters())

    # Rank optimizer with conditional overrides
    rank_cfg = OmegaConf.create(cfg.trainer.optimizer)

    # Only apply overrides if explicitly set (not None)
    if cfg.trainer.get("rank_lr") is not None:
        rank_cfg.lr = cfg.trainer.rank_lr

    if cfg.trainer.get("rank_weight_decay") is not None:
        rank_cfg.weight_decay = cfg.trainer.rank_weight_decay

    if cfg.trainer.get("rank_betas") is not None:
        rank_cfg.betas = cfg.trainer.rank_betas

    if cfg.trainer.get("rank_eps") is not None:
        rank_cfg.eps = cfg.trainer.rank_eps

    rank_opt_fn = hydra.utils.instantiate(rank_cfg)
    rank_optimizer = rank_opt_fn(model.rank_named_parameters())

    # Fix betas after creation
    for pg in chain(shared_optimizer.param_groups, rank_optimizer.param_groups):
        if "betas" in pg:
            pg["betas"] = tuple(pg["betas"])

    return shared_optimizer, rank_optimizer


def create_lwu_schedulers(
    cfg,
    shared_optimizer: Optimizer,
    rank_optimizer: Optimizer,
) -> Tuple[LinearWarmupLR, LinearWarmupLR]:
    """TODO: Move cfg access outside this function and just pass args"""
    trainer_cfg = cfg.trainer
    # Get scheduler config params
    warmup_steps = trainer_cfg.scheduler_warmup_steps
    warmup_start_lr = trainer_cfg.warmup_start_lr
    assert None not in [warmup_steps, warmup_start_lr], (
        "All scheduler config params must be provided!"
    )
    log.info(
        f"Linear-Warmup scheduler with:"
        f"\n\tWarmup steps: {warmup_steps}"
        f"\n\tWarmup start lr: {warmup_start_lr}"
    )
    _sched_gen_fn = lambda opt: LinearWarmupLR(
        optimizer=opt,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
    )
    shared_scheduler = _sched_gen_fn(shared_optimizer)
    rank_scheduler = _sched_gen_fn(rank_optimizer)

    return shared_scheduler, rank_scheduler


def create_lwc_schedulers(
    cfg,
    shared_optimizer: Optimizer,
    rank_optimizer: Optimizer,
    dataset_length: int,
    gradient_accumulation_steps: Optional[int] = 1,
) -> Tuple[LinearWarmupCosineAnnealingLR, LinearWarmupCosineAnnealingLR]:
    """TODO: Move cfg access outside this function and just pass args"""
    trainer_cfg = cfg.trainer
    # Get scheduler config params
    warmup_steps = trainer_cfg.scheduler_warmup_steps
    eta_min = trainer_cfg.eta_min
    warmup_start_lr = trainer_cfg.warmup_start_lr
    assert None not in [warmup_steps, eta_min, warmup_start_lr], (
        "All scheduler config params must be provided!"
    )
    log.info(
        f"Creating linear-Warmup, Cosine Annealing scheduler with:"
        f"\n\tWarmup steps: {warmup_steps}"
        f"\n\tEta min: {eta_min}"
        f"\n\tWarmup start lr: {warmup_start_lr}"
    )
    # Get training params for calculating decay length
    max_epochs = trainer_cfg.get("max_epochs", None)
    max_steps = trainer_cfg.get("max_steps", None)
    steps_per_epoch = math.ceil(dataset_length / gradient_accumulation_steps)
    _sched_gen_fn = lambda opt: LinearWarmupCosineAnnealingLR(
        optimizer=opt,
        warmup_steps=warmup_steps,
        max_epochs=max_epochs,
        steps_per_epoch=steps_per_epoch,
        max_steps=max_steps,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )
    shared_scheduler = _sched_gen_fn(shared_optimizer)
    rank_scheduler = _sched_gen_fn(rank_optimizer)

    return shared_scheduler, rank_scheduler


def create_ld_schedulers(
    shared_optimizer: Optimizer,
    rank_optimizer: Optimizer,
    total_iters: int,
    power: float = 1.0,
) -> Tuple[PolynomialLR, PolynomialLR]:
    """Create linear decaying learning rate schedulers for the model which go
    from current lr to 0 at `final_step`"""
    # Get scheduler config params
    log.info(f"Creating Linear-Decaying schedulers for {total_iters} total steps!")
    _sched_gen_fn = lambda opt: PolynomialLR(
        optimizer=opt,
        total_iters=total_iters,
        power=power,
    )
    shared_scheduler = _sched_gen_fn(shared_optimizer)
    rank_scheduler = _sched_gen_fn(rank_optimizer)

    return shared_scheduler, rank_scheduler


# ------------------------------------------------------------
# Data Processing Utilities
# ------------------------------------------------------------


def move_data_to_device(
    data: Any,
    device: Union[torch.device, str],
    dtype: Optional[torch.dtype] = None,
    non_blocking: bool = True,
) -> Any:
    """
    Recursively moves tensors in a nested structure (dict, list, tuple) to the target device,
    optionally casts them to a specific dtype, and uses non-blocking transfers.

    Args:
        data: The data structure (potentially nested) containing tensors.
        device: The target device (e.g., 'cuda', 'cpu', or torch.device object).
        dtype: Optional target dtype for tensor casting (e.g., torch.bfloat16).
        non_blocking: Whether to use non-blocking CUDA transfers. Recommended True
                      when source tensors are in pinned memory.

    Returns:
        A new data structure mirroring the input, with tensors moved to the device
        and potentially cast to the specified dtype. Other data types are preserved.
    """
    if isinstance(data, Tensor):
        # Move tensor to device, optionally change dtype, use non_blocking
        # Using data.to() is generally preferred over data.cuda() for device flexibility
        return data.to(device=device, dtype=dtype, non_blocking=non_blocking)
    elif isinstance(data, dict):
        # Recursively process dictionary values
        return {
            k: move_data_to_device(v, device, dtype, non_blocking)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        # Recursively process list elements
        return [move_data_to_device(elem, device, dtype, non_blocking) for elem in data]
    elif isinstance(data, tuple):
        # Recursively process tuple elements. Note: tuples are immutable, creates a new one.
        return tuple(
            move_data_to_device(elem, device, dtype, non_blocking) for elem in data
        )
    else:
        # Return non-tensor data types as is
        return data


# Make transfer_batch_to_device use the optimized helper
def transfer_batch_to_device(
    batch: Tuple[str, Dict[str, Any]],  # More general type hint for data dict
    device: Union[torch.device, str],
    dtype: Optional[torch.dtype] = None,  # Add dtype parameter
) -> Tuple[str, Any]:
    """
    Transfer a batch (session_key, data_dict) to the specified device,
    optionally casting tensor dtype using non-blocking transfers.

    Args:
        batch: A tuple of (session_key, data_dict) where data_dict contains tensors
               or nested dictionaries/lists/tuples of tensors.
        device: The target device.
        dtype: Optional target dtype for tensor casting (e.g., torch.bfloat16).

    Returns:
        The same batch structure with tensors moved/cast to the specified device.
    """
    session_key, data = batch
    # Use non_blocking=True by default, pass device and dtype
    data = move_data_to_device(data, device, dtype=dtype, non_blocking=True)
    return session_key, data


def calculate_gradient_accumulation_steps(
    world_size: int,
    local_batch_size: int,
    gradient_accumulation_steps: Optional[int] = None,
    global_batch_size: Optional[int] = None,
) -> int:
    """
    Calculate gradient accumulation steps from configuration.

    Args:
        cfg: Configuration containing gradient accumulation settings
        world_size: Number of distributed processes
        local_batch_size: Batch size per process

    Returns:
        Number of gradient accumulation steps
    """
    # Set defaults for gradient accumulation steps and global batch size
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = 1
    if global_batch_size is None:
        global_batch_size = -1
    # Re-calculate gradient accumulation steps based on global batch size, if provided
    if global_batch_size > 0:
        effective_batch_size = local_batch_size * world_size
        calculated_steps = global_batch_size // effective_batch_size
        assert global_batch_size % effective_batch_size == 0, (
            f"global_batch_size ({global_batch_size}) must be divisible by "
            f"effective_batch_size (local_batch_size * world_size = {effective_batch_size})"
        )
        gradient_accumulation_steps = calculated_steps

    assert gradient_accumulation_steps > 0, (
        "gradient_accumulation_steps must be positive"
    )

    log.info("Gradient accumulation configuration:")
    log.info(f"  - Local batch size: {local_batch_size}")
    log.info(f"  - World size: {world_size}")
    log.info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
    log.info(
        f"  - Effective global batch size: {local_batch_size * world_size * gradient_accumulation_steps}"
    )

    return gradient_accumulation_steps


def create_accumulated_dataloader(
    dataloader: SyncedFastSessionDataLoader,
    gradient_accumulation_steps: int,
    limit_train_batches: Optional[int] = None,
):
    """
    Group batches for gradient accumulation.

    Args:
        dataloader: The SyncedFastSessionDataLoader to group batches from
        gradient_accumulation_steps: Number of batches per accumulation group
        limit_batches: Optional limit on total batches to process

    Yields:
        List of batches for each accumulation group
    """
    # NOTE: If the SyncedFastSessionDataLoader is exhausted before we yield all batches
    #  (i.e. if length is not a multiple of `gradient_accumulation_steps`), the dataloader
    #  will automatically reset before yielding the last batches, resulting in a checkpoint
    #  with an incorrect dataloader state. We use `limit_train_batches` to manually break
    #  before exhausting the iterator, and reset the dataloader after yielding the last
    #  partial batch.
    if limit_train_batches is None or limit_train_batches > len(dataloader):
        limit_train_batches = len(dataloader)

    # Initialize collection of accumulated batches
    batch_group = []

    for batch in dataloader:
        # Add batch to group and get latest index (according to dataloader)
        batch_group.append(batch)
        batch_idx = dataloader.current_batch
        # Yield complete group or when we reach the limit
        if len(batch_group) == gradient_accumulation_steps:
            yield batch_group
            batch_group = []
        # Stop if we've reached the batch limit
        if batch_idx >= limit_train_batches:
            # Yield partial group if it exists
            if batch_group:
                yield batch_group
                batch_group = []
            # Manually reset the dataloader and break (before raising `StopIteration`)
            dataloader.reset_state()
            break

    # We should never have any batches left over here, see note above.
    assert len(batch_group) == 0, (
        "Batch group is not empty at the end of the accumulated dataloader!"
    )


# ------------------------------------------------------------
# Step Implementations
# ------------------------------------------------------------


@preserve_rng_state
def run_evaluation(
    model: Model | List[Model],
    dataloader: Optional[SyncedFastSessionDataLoader],
    amp: bool = True,
    device: torch.device | str = "cpu",
    limit_eval_batches: Optional[int] = None,
    eval_masking_strategies: Optional[
        Dict[str, Dict | DictConfig | MaskingStrategy]
    ] = None,
    rank: int = 0,
    phase: str = "validation",
    initial_rng_state: Optional[torch.Tensor] = None,
    initial_dl_rng_states: Optional[Dict[str, Dict[str, Any]]] = None,
    all_neurons_override: bool = False,
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    """
    Run validation loop.

    TODO: Make wrapper for behavior metrics and avoid hardcoding channels
    TODO: Fix distributed metrics
    """

    if dataloader is None:
        log.info(f"No dataloader provided on rank {rank}, skipping evaluation!")
        return None, None

    # Preserve the rng state of the dataloader (which are managed internally, instead of
    #  with the overall process rng states)
    inital_dataloader_state = dataloader.get_state()

    # Set initial random state to use for evaluation consistency
    if initial_rng_state is not None:
        set_rng_state(deepcopy(initial_rng_state))
    if initial_dl_rng_states is not None:
        if phase in initial_dl_rng_states:
            initial_dl_rng_state = initial_dl_rng_states[phase]
            dataloader.set_state(deepcopy(initial_dl_rng_state))
        else:
            log.warning(
                f"No initial dataloader rng state provided for phase {phase}, can't "
                f"gaurantee consistency!"
            )

    # dist.barrier()

    # Initialize validation loss dictionary
    eval_loss = defaultdict(float)
    dtype = torch.bfloat16 if amp else torch.float32
    device = torch.device(device)

    # Create ensemble of models if not already a list
    if not isinstance(model, List):
        model = [model]
    # Set model to eval mode
    for _model in model:
        _model.eval()

    eval_metrics = {}

    # Define validation step function
    @torch.no_grad()
    def evaluation_step(b, *args, **kwargs) -> OMModelOutput:
        outputs_ensemble = []
        for _model in model:
            # Forward pass and loss calculation with no gradients
            with autocast(device_type=device.type, dtype=dtype, enabled=amp):
                outputs_ensemble.append(_model.forward_from_batch(b, *args, **kwargs))
        outputs = outputs_ensemble[0]
        if len(outputs_ensemble) == 1:
            return outputs
        if outputs.response_preds is not None:
            outputs.response_preds = torch.stack(
                [o.response_preds for o in outputs_ensemble], dim=0
            ).mean(dim=0)
        if outputs.behavior_preds is not None:
            outputs.behavior_preds = torch.stack(
                [o.behavior_preds for o in outputs_ensemble], dim=0
            ).mean(dim=0)
        return outputs

    # Add dummy strategy configuration if not provided
    if eval_masking_strategies is None:
        eval_masking_strategies = {"random_builtin_": None}
    else:
        # TODO: This is a HACK-ey fix, because Hydra for some reason fails to convert the
        #  list of DictConfig's to MaskingStrategy instances, so we do it manually here.
        eval_masking_strategies = {
            name: omegaconf_to_masking_strategy(strategy)
            for name, strategy in eval_masking_strategies.items()
        }
    # Get session meta's for sessions on current rank
    rank_session_map = {
        sk: sm
        for sk, sm in model[0].session_map.items()
        if sk in dataloader.dataset.session_names and sm.on_rank
    }
    # Iterate over masking strategies and corresponding metrics
    progress_bar = tqdm(
        eval_masking_strategies.items(),
        desc="Processing strategies",
        disable=rank != 0,
    )
    for name, masking_strategy in progress_bar:
        prefix = f"{phase}/{name}/"
        # TODO: This is a *HACK* to avoid reporting metrics that are never used
        responses_decoded, behavior_decoded = False, False
        # Build response metrics
        response_metrics = {
            f"{prefix}{sess_key}/response_xcorr": MaskedPopulationPearsonCorrCoef(
                population_size=sess_meta.n_neurons,
                num_samples_per_block=model[0].c.num_samples_per_block,
                **TORCHMETRIC_DIST_KWARGS_OVERRIDE,
            ).to(device)
            for sess_key, sess_meta in rank_session_map.items()
        }

        # Build behavior metrics
        behavior_metrics = {}
        for sess_key in rank_session_map.keys():
            behavior_metrics[f"{prefix}{sess_key}/eye_tracker_pos_xcorr"] = (
                PearsonCorrCoef(
                    num_outputs=2,
                    **TORCHMETRIC_DIST_KWARGS_OVERRIDE,
                ).to(device)
            )
            behavior_metrics[f"{prefix}{sess_key}/eye_tracker_vel_xcorr"] = (
                PearsonCorrCoef(
                    num_outputs=2,
                    **TORCHMETRIC_DIST_KWARGS_OVERRIDE,
                ).to(device)
            )
            behavior_metrics[f"{prefix}{sess_key}/treadmill_xcorr"] = PearsonCorrCoef(
                num_outputs=1,
                **TORCHMETRIC_DIST_KWARGS_OVERRIDE,
            ).to(device)
        # Initialize validation loss tracker
        strat_eval_losses = {
            sess_key: LossTrackerCollection(
                ["loss", "response_loss", "behavior_loss"],
                mode="mean",
                window_size=-1,
                prefix=f"{prefix}{sess_key}",
                rank=rank,
            )
            for sess_key in rank_session_map.keys()
        }
        # Run validation steps
        if limit_eval_batches is None or limit_eval_batches > len(dataloader):
            limit_eval_batches = len(dataloader)
        _progress_bar = tqdm(
            dataloader,
            desc=f"Validating {name}",
            leave=False,
            total=limit_eval_batches,
            disable=rank != 0,
        )
        for batch_idx, batch in enumerate(_progress_bar):
            # Skip if we've reached the batch limit
            if batch_idx >= limit_eval_batches:
                dataloader.reset_state()
                break
            # if batch[-1]['responses'].shape[-1] < 4096:
            #     log.warning(f"Skipping batch {batch_idx} because it has more than 4096 neurons")
            #     continue
            # Move batch to device. TODO: We avoid manually casting dtype here, to allow
            #  intelligent casting by autocast.
            batch = transfer_batch_to_device(batch, device, dtype=None)  # dtype)
            outputs = evaluation_step(
                batch,
                masking_override=masking_strategy,
                all_neurons_override=all_neurons_override,
            )
            sess_key = batch[0]
            sess_prefix = f"{prefix}{sess_key}/"
            # Update response metrics (for this session)
            if outputs.response_preds is not None:
                responses_decoded = True
                response_metrics[f"{sess_prefix}response_xcorr"].update(
                    preds=outputs.response_preds,
                    target=outputs.response_labels,
                    neuron_ids=outputs.response_neuron_ids,
                    positions=outputs.response_positions,
                )
            # Update behavior metrics (for this session)
            if outputs.behavior_preds is not None:
                behavior_decoded = True
                # Concatenate along the batch dimension: (B, C_beh, S_beh) ->
                #   (1, C_beh, B*S_beh) -> (C_beh, B*S_beh) -> (B*S_beh, C_beh)
                # TODO: Chunk
                behavior_preds = (
                    torch.cat(outputs.behavior_preds.split(1), dim=-1)
                    .squeeze(0)
                    .T.contiguous()
                )
                behavior_labels = (
                    torch.cat(outputs.behavior_labels.split(1), dim=-1)
                    .squeeze(0)
                    .T.contiguous()
                )
                behavior_metrics[f"{sess_prefix}eye_tracker_pos_xcorr"].update(
                    preds=behavior_preds[:, 0:2].contiguous(),
                    target=behavior_labels[:, 0:2].contiguous(),
                )
                behavior_metrics[f"{sess_prefix}eye_tracker_vel_xcorr"].update(
                    preds=behavior_preds[:, 2:4].contiguous(),
                    target=behavior_labels[:, 2:4].contiguous(),
                )
                behavior_metrics[f"{sess_prefix}treadmill_xcorr"].update(
                    preds=behavior_preds[:, 4:5].contiguous(),
                    target=behavior_labels[:, 4:5].contiguous(),
                )
            # Update overall losses
            strat_eval_losses[sess_key].update(
                loss=outputs.loss,
                response_loss=outputs.response_loss,
                behavior_loss=outputs.behavior_loss,
            )
        with torch.no_grad():
            # Average loss over number of batches, and update `eval_loss` dict
            for strat_eval_loss in strat_eval_losses.values():
                eval_loss.update(strat_eval_loss.compute(updated_only=True))
            # Compute metrics. FIX: distributed metrics here.
            # Response metrics
            if responses_decoded:
                response_computed = {
                    k: m.compute() for k, m in response_metrics.items()
                }
                eval_metrics.update(response_computed)
            # Behavior metrics. NOTE: We average across multiple behavior channels
            if behavior_decoded:
                behavior_computed = {
                    k: m.compute().mean() for k, m in behavior_metrics.items()
                }
                eval_metrics.update(behavior_computed)

        # Log current strategy metrics
        [s.reset() for s in strat_eval_losses.values()]
        [m.reset() for m in response_metrics.values()]
        [m.reset() for m in behavior_metrics.values()]
        torch.cuda.empty_cache()

    [m.train() for m in model]

    # dist.barrier()

    # Restore the rng state of the dataloader
    dataloader.set_state(inital_dataloader_state)

    return eval_loss, eval_metrics


# ------------------------------------------------------------
# Checkpointing / Logging Functions
# ------------------------------------------------------------


def log_to_loggers(
    loggers: Logger | List[Logger],
    metrics: Dict[str, Any],
    prefix: str = "",
    step: Optional[int] = None,
) -> None:
    """Helper function to log metrics to all loggers with optional prefix.

    Args:
        loggers: List of Lightning loggers
        metrics: Dictionary of metrics to log
        prefix: Optional prefix to add to metric names (e.g., 'train/', 'val/', 'test/')
        step: Current step for logging
    """
    if isinstance(loggers, Logger):
        loggers = [loggers]

    if not loggers:
        return

    # Add prefix to metric names if specified
    if prefix and not prefix.endswith("/"):
        prefix = f"{prefix}/"

    prefixed_metrics = {}
    for k, v in metrics.items():
        if prefix and not k.startswith(prefix):
            k = f"{prefix}{k}"
        prefixed_metrics[k] = v

    # Log to all loggers
    for logger in loggers:
        try:
            logger.log_metrics(prefixed_metrics, step=step)
        except Exception as e:
            # TODO: Log to disk instead
            log.error(f"Error logging metrics to logger: {e}")


def add_overall_metrics(metrics_dict, session_map: Optional[SessionMap] = None):
    """
    Add overall metrics that average over sessions for each mode/name/metric combination.

    Args:
        metrics_dict: Dictionary of metrics with keys in the format '{mode}/{name}/{session_key}/{metric_type}'
        session_map: SessionMap object, containing session metadata (i.e. number of neurons), used for weighted averaging by population size

    Returns:
        Updated dictionary with additional '{mode}/{name}/overall/{metric_type}' entries
    """
    if not metrics_dict:
        return metrics_dict

    # Make a copy of the input dictionary to avoid modifying the original during iteration
    result_dict = metrics_dict.copy()

    # Pattern to extract mode, name, session_key, and metric_type
    pattern = r"^([^/]+)/([^/]+)/([^/]+)/([^/]+)$"

    # Group metrics by mode, name, and metric_type
    grouped_metrics = defaultdict(list)

    for key, value in metrics_dict.items():
        match = re.match(pattern, key)
        if match:
            mode, name, session_key, metric_type = match.groups()
            # Group by mode, name, metric_type
            group_key = (mode, name, metric_type)
            grouped_metrics[group_key].append((session_key, value))

    # Calculate averages and add new entries
    for (mode, name, metric_type), values in grouped_metrics.items():
        overall_key = f"{mode}/{name}/overall/{metric_type}"
        if session_map is not None:
            avg, total_n_neurons = 0, 0
            for sk, v in values:
                n_neurons = session_map[sk].n_neurons
                avg += v * n_neurons
                total_n_neurons += n_neurons
            result_dict[overall_key] = avg / total_n_neurons
        else:
            result_dict[overall_key] = sum(values) / len(values)

    return result_dict


@preserve_rng_state
def handle_metrics(
    eval_loss: Optional[Dict[str, float | Tensor]] = None,
    eval_metrics: Optional[Dict[str, float | Tensor]] = None,
    combined_running_losses: Optional[Dict[str, float | Tensor]] = None,
    session_map: Optional[SessionMap] = None,
    logger: Optional[Logger] = None,
    global_step: int = 0,
    rank: int = 0,
    world_size: int = 1,
    prefix: Optional[str] = None,
) -> Tuple[Dict[str, float | Tensor], Dict[str, float | Tensor]]:
    dist.barrier()

    # Gather metrics/losses and compute overall metrics
    combined_val_metrics = gather_validation_metrics(eval_metrics, rank, world_size)
    combined_val_loss = gather_validation_metrics(eval_loss, rank, world_size)
    combined_val_metrics = add_overall_metrics(combined_val_metrics, session_map)
    combined_val_loss = add_overall_metrics(combined_val_loss, session_map)

    # TODO: This is redundant with the prefixing done in `log_to_loggers`, but we want the
    #  outputs to for sure have the prefix
    # Add prefix to metric names if specified
    if prefix is None:
        prefix = ""
    if prefix and not prefix.endswith("/"):
        prefix = f"{prefix}/"
    prefixed_metrics, prefixed_losses = {}, {}
    for k, v in combined_val_metrics.items():
        if prefix and not k.startswith(prefix):
            k = f"{prefix}{k}"
        prefixed_metrics[k] = v
    for k, v in combined_val_loss.items():
        if prefix and not k.startswith(prefix):
            k = f"{prefix}{k}"
        prefixed_losses[k] = v

    # Log metrics to loggers if needed
    if logger is not None:
        dist.barrier()
        log_to_loggers(logger, prefixed_metrics, step=global_step, prefix=prefix)
        log_to_loggers(logger, prefixed_losses, step=global_step, prefix=prefix)
        # TODO: Log training losses from other ranks
        if combined_running_losses is not None:
            log_to_loggers(
                logger, combined_running_losses, step=global_step, prefix=prefix
            )

    dist.barrier()

    return prefixed_metrics, prefixed_losses


@preserve_rng_state
def handle_checkpointing(
    checkpointer: ModelCheckpoint,
    global_step: int = 0,
    epoch: int = 0,
    combined_val_metrics: Optional[Dict[str, float | Tensor]] = None,
    combined_val_loss: Optional[Dict[str, float | Tensor]] = None,
    combined_running_losses: Optional[Dict[str, float | Tensor]] = None,
    combined_loss_windows: Optional[Dict[str, List[float]]] = None,
) -> None:
    # Update checkpoint if needed.
    dist.barrier()
    # TODO: Why are some metrics not complete for some ranks here?
    ckpt_metrics = dict(
        **(combined_val_metrics or {}),
        **(combined_val_loss or {}),
        **(combined_running_losses or {}),
    )
    checkpointer.save_checkpoint(
        metrics=ckpt_metrics,
        epoch=epoch,
        step=global_step,
        loss_windows=combined_loss_windows,
    )
    dist.barrier()
