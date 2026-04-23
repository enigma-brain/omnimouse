from typing import Optional, Sequence
import os
import hydra
import rootutils
from copy import deepcopy

import torch
import torch.distributed as dist
from lightning.pytorch.loggers import Logger
import mlflow
from omegaconf import DictConfig

# Handles initialization of project root dir
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from omnimouse.utils import (
    RankedLogger,
    extras,
    task_wrapper,
    ModelCheckpoint,
    resolve_cfg,
)

from omnimouse.distributed.utils import broadcast_model_shared

from omnimouse.tasks.task_utils import (
    setup_common,
    run_evaluation,
    handle_metrics,
)

from omnimouse.modeling import Model


log = RankedLogger(__name__, rank_zero_only=True)

# Setup precision settings
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
mlflow.enable_system_metrics_logging()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch._dynamo.config.cache_size_limit = 4096

# Manually disable distributed syncing of torchmetrics metrics
TORCHMETRIC_DIST_KWARGS_OVERRIDE = {
    "process_group": None,
    "sync_on_compute": False,
    "dist_sync_fn": None,
}


def load_checkpoint(
    ckpt_path: str | os.PathLike,
    checkpointer_cfg: DictConfig,
    checkpoint_loading_cfg: DictConfig,
    rank: int,
    world_size: int,
    model: Model,
):
    # Model checkpoint callback
    log.info(f"Initializing checkpointer for checkpoint path: {ckpt_path}")
    checkpointer: ModelCheckpoint = hydra.utils.instantiate(checkpointer_cfg)
    # Assign modules to checkpointer
    checkpointer.assign_modules_and_rank(
        rank=rank, world_size=world_size, model=model,
    )
    log.info(f"Loading checkpoint from: {ckpt_path}")
    start_epoch, global_step = checkpointer.load_checkpoint(
        checkpoint_dir=ckpt_path,
        strict=checkpoint_loading_cfg.get("strict", True),
        load_random_states=checkpoint_loading_cfg.get("load_random_states", True),
        load_compile_cache=checkpoint_loading_cfg.get("load_compile_cache", True),
        load_weights_only=True, # NOTE: We only load weights from checkpoint by not passing other modules
        load_start_counts=True,
    )

    return start_epoch, global_step



@hydra.main(version_base="1.3", config_path=f"{root}/configs", config_name="eval.yaml")
@task_wrapper
def eval(cfg: DictConfig, logger: Optional[Logger] = None, **kwargs):
    """Training function that replaces the Lightning Trainer.
    
    Args:
        cfg: A DictConfig configuration composed by Hydra.
        
    Returns:
        A tuple with metrics and dict with all instantiated objects.
    """
    # Apply extra utilities
    extras(cfg)

    # Setup common components
    (
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
     ) = setup_common(cfg, logger, flatten_datasets_if_single_rank=True)
    # TODO: Add train dl to `val_dls`. This is HACKY!
    val_dls['train'] = train_dl

    dist.barrier()

    # Extract configurations from cfg
    limit_eval_batches = cfg.trainer.get("limit_val_batches", None)
    amp = cfg.trainer.get("amp", True)
    dtype = torch.bfloat16 if amp else torch.float32
    all_neurons_override = cfg.get("all_neurons_override", False)

    shared_parameters = []
    for p in model.shared_parameters():
        shared_parameters.append(p)

    rank_parameters = []
    for p in model.rank_parameters():
        rank_parameters.append(p)
    
    dist.barrier()


    # Check if we should load from checkpoint
    start_epoch, global_step = 0, 0
    loaded_models = []

    ckpt_path = cfg.get("ckpt_path", None)
    ckpt_loading_cfg = cfg.get("ckpt_loading", {})
    if ckpt_path is not None:
        if isinstance(ckpt_path, (str, os.PathLike)):
            ckpt_path = [ckpt_path]
        for i, _ckpt_path in enumerate(ckpt_path):
            assert isinstance(_ckpt_path, (str, os.PathLike)), \
                f"Checkpoint path must be a string, os.PathLike, or sequence of " \
                f"strings/os.PathLike, got {type(_ckpt_path)}"
            _model = model if i == 0 else deepcopy(model)
            start_epoch, global_step = load_checkpoint(
                _ckpt_path, cfg.trainer.checkpointer, ckpt_loading_cfg,
                rank, world_size, _model,
            )
            loaded_models.append(_model)
    else:
        loaded_models.append(model)
    
    for i, _model in enumerate(loaded_models):
        broadcast_model_shared(_model, src=0)
        log.info(f"Synchronized shared parameters across ranks from rank 0 for model {i}")
        dist.barrier()

    # Get eval datasets
    dataset_phases = cfg.get("eval_datasets", ["validation"])
    for dataset_phase in dataset_phases:
        assert dataset_phase in val_dls, f"Validation dataset {dataset_phase} not found!"
        eval_dl = val_dls[dataset_phase]
        # TODO: This is super hacky!
        mask_key = f"{dataset_phase}_masking_strategies"          
        # Get eval regimes
        eval_masking_strategies = resolve_cfg(cfg.get("evaluation", {}).get(mask_key, None))
        if eval_masking_strategies is None:
            raise ValueError(f"No masking strategies provided for {dataset_phase} set!")

        # TODO: Is this necessary?
        dist.barrier()

        log.info(f"Validating at step {global_step}")
        # Run validation on "validation" set
        val_loss, val_metrics = run_evaluation(
            loaded_models, eval_dl, amp, device, limit_eval_batches,
            eval_masking_strategies, rank, dataset_phase, initial_rng_state,
            initial_dl_rng_states, all_neurons_override=all_neurons_override,
        )
        dist.barrier()
        # Log "validation" metrics
        handle_metrics(
            val_loss, val_metrics, None, model.session_map, logger,
            global_step, rank, world_size, prefix=dataset_phase
        )
        dist.barrier()