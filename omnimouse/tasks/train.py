import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import hydra
import mlflow
import rootutils
import torch
import torch.distributed as dist
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torch import autocast
from tqdm import tqdm

# Handles initialization of project root dir
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from omnimouse.distributed.utils import (
    SyncedFastSessionDataLoader,
    broadcast_model_shared,
    broadcast_optimizer,
    calc_divergence,
    collect_task_ids,
    gather_validation_metrics,
    initialize_buckets,  # begin custom gradient syncing
    reduce_gradients,
    unpack_gradients,
)
from omnimouse.modeling import OMModelOutput
from omnimouse.tasks.task_utils import (
    calculate_gradient_accumulation_steps,
    create_ld_schedulers,
    create_lwc_schedulers,
    create_lwu_schedulers,
    create_optimizers,
    handle_checkpointing,
    handle_metrics,
    log_to_loggers,
    run_evaluation,
    setup_common,
    transfer_batch_to_device,
)
from omnimouse.utils import (
    LossTrackerCollection,
    ModelCheckpoint,
    RankedLogger,
    extras,
    resolve_cfg,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

# Setup precision settings
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
mlflow.enable_system_metrics_logging()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch._dynamo.config.cache_size_limit = 4096

# Manually disable distributed syncing of torchmetrics metrics
TORCHMETRIC_DIST_KWARGS_OVERRIDE = {
    "process_group": None,
    "sync_on_compute": False,
    "dist_sync_fn": None,
}

DEFAULT_LOG_EVERY_N_STEPS = 500


@hydra.main(version_base="1.3", config_path=f"{root}/configs", config_name="train.yaml")
@task_wrapper
def train(cfg: DictConfig, logger: Optional[Logger] = None, **kwargs):
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
    ) = setup_common(cfg, logger)

    dist.barrier()

    # Get eval regimes
    val_masking_strategies_all_sets = {}
    eval_cfg = cfg.get("evaluation", {})
    default_val_masking_strategies = resolve_cfg(
        eval_cfg.get("default_masking_strategies", None)
    )
    for val_dl_key in val_dls:
        # TODO: Standardize keys- this is super hacky!
        val_masking_key = f"{val_dl_key}_masking_strategies"
        val_masking_strategies = resolve_cfg(
            eval_cfg.get(val_masking_key)
        )  # , default_val_masking_strategies))
        if val_masking_strategies is None and val_dls[val_dl_key] is not None:
            raise ValueError(f"No masking strategies provided for {val_dl_key} set!")
        val_masking_strategies_all_sets[val_dl_key] = val_masking_strategies

    # Extract training configuration from cfg
    max_epochs = cfg.trainer.get("max_epochs", None)
    max_steps = cfg.trainer.get("max_steps", None)
    assert max_epochs is not None or max_steps is not None, (
        "Either `max_epochs` or `max_steps` must be provided!"
    )
    limit_train_batches = cfg.trainer.get("limit_train_batches", None)
    limit_val_batches = cfg.trainer.get("limit_val_batches", None)
    clip_val = cfg.trainer.gradient_clip_val
    amp = cfg.trainer.get("amp", True)
    dtype = torch.bfloat16 if amp else torch.float32
    all_neurons_override = cfg.get("all_neurons_override", False)
    # Validation interval settings
    val_every_n_steps = cfg.trainer.get("val_every_n_steps", None) or 0  # steps
    val_every_n_epochs = cfg.trainer.get("val_every_n_epochs", 1) or 0  # steps
    ckpt_every_n_steps = cfg.trainer.get("ckpt_every_n_steps", None) or 0  # steps
    ckpt_every_n_epochs = cfg.trainer.get("ckpt_every_n_epochs", 1) or 0  # steps
    # Logging interval settings
    log_every_n_steps = cfg.trainer.get(
        "log_every_n_steps", DEFAULT_LOG_EVERY_N_STEPS
    )  # steps
    pbar_enabled = cfg.trainer.get("pbar_enabled", True)
    pbar_every_n_steps = cfg.trainer.get("pbar_every_n_steps", 1)  # steps
    # Sync interval settings
    sync_params_every_n_steps = cfg.trainer.get(
        "sync_params_every_n_steps", None
    )  # steps
    sync_optimizers_every_n_steps = cfg.trainer.get(
        "sync_optimizers_every_n_steps", None
    )  # steps
    calc_divergence_every_n_steps = cfg.trainer.get(
        "calc_divergence_every_n_steps", None
    )  # steps
    divergence_threshold = cfg.trainer.get("divergence_threshold", 1e-8)
    divergence_mode = cfg.trainer.get("divergence_mode", "max")
    log_task_every_n_steps = cfg.trainer.get("log_task_every_n_steps", None)  # steps

    # Initialize optimizer directly
    log.info("Initializing optimizer and scheduler")
    # Create two parameter groups - shared and rank-specific
    shared_optimizer, rank_optimizer = create_optimizers(cfg, model)

    # Calculate gradient accumulation steps (default to 1 if not provided)
    gradient_accumulation_steps = calculate_gradient_accumulation_steps(
        world_size=world_size,
        local_batch_size=cfg.data.dataloader.batch_size,
        gradient_accumulation_steps=cfg.trainer.get("gradient_accumulation_steps"),
        global_batch_size=cfg.trainer.get("global_batch_size"),
    )

    # Initialize scheduler directly with hardcoded parameters
    if cfg.trainer.get("disable_cosine_annealing", False):
        # Create linear warmup with *no decay*
        shared_scheduler, rank_scheduler = create_lwu_schedulers(
            cfg, shared_optimizer, rank_optimizer
        )
    else:
        # Create linear warmup with cosine annealing, based on dataset length (and max epochs)
        dataset_length = len(train_dl)
        if limit_train_batches is not None:
            dataset_length = min(dataset_length, limit_train_batches)
        shared_scheduler, rank_scheduler = create_lwc_schedulers(
            cfg,
            shared_optimizer,
            rank_optimizer,
            dataset_length,
            gradient_accumulation_steps,
        )

    # setup running train loss trackers for logging to mlflow
    running_losses = LossTrackerCollection(
        names=["loss", "response_loss", "behavior_loss"],
        mode="mean",
        window_size=30,
        prefix="train",
        rank=rank,
        session_names=train_dl.dataset.session_names,
    )
    # setup collections for step-wise traces to put in checkpoints. NOTE: We abuse
    #  the `LossTrackerCollection` class to store step-wise traces.
    trace_names = [
        "loss",
        "response_loss",
        "behavior_loss",
        "rank_grad_norm",
        "shared_grad_norm",
        "rank_lr",
        "shared_lr",
        "batch_key",
    ]
    ckpt_traces = LossTrackerCollection(
        names=trace_names,
        mode="last",
        window_size=ckpt_every_n_steps,
        prefix="train",
        rank=rank,
        session_names=train_dl.dataset.session_names,
    )

    # TODO: Is this necessary?
    dist.barrier()

    # Model checkpoint callback
    log.info("Initializing checkpointer")
    checkpointer: ModelCheckpoint = hydra.utils.instantiate(cfg.trainer.checkpointer)
    # Assign modules to checkpointer
    checkpointer.assign_modules_and_rank(
        rank=rank,
        world_size=world_size,
        model=model,
        shared_optimizer=shared_optimizer,
        rank_optimizer=rank_optimizer,
        shared_scheduler=shared_scheduler,
        rank_scheduler=rank_scheduler,
        train_dataloader=train_dl,
        val_dataloaders=val_dls,
        running_trackers=running_losses,
    )
    # Check if we should load from checkpoint
    start_epoch, global_step = 0, 0

    ckpt_path = cfg.get("ckpt_path", None)
    resume_training = cfg.get("resume_training", None)
    finetuning = cfg.get("finetuning", None)
    assert not (resume_training and finetuning), (
        "Cannot resume training and finetune at the same time!"
    )
    if ckpt_path is None:
        last_ckpt_path = Path(checkpointer.dirpath) / checkpointer.CHECKPOINT_NAME_LAST
        if last_ckpt_path.exists():
            ckpt_path = last_ckpt_path
            log.warning(
                f"Found latest checkpoint at {ckpt_path}, resuming training! If you didn't "
                f"intend to resume training, please rerun with a new run name (i.e., "
                f"`cfg.paths.run_name`)."
            )
        resume_training = True
        if finetuning:
            raise ValueError(
                "Cannot finetune from latest checkpoint! Please provide a checkpoint path "
                "and rerun with a new run name (i.e., `cfg.paths.run_name`)."
            )

    if ckpt_path is not None:
        log.info(f"Resuming from checkpoint: {ckpt_path}")
        assert checkpointer is not None, (
            "Checkpointer config must be provided (i.e. `cfg.trainer.checkpointer`) "
            "to resume training!"
        )
        # TODO: Clean up configuration of checkpoint loading / configuration of checkpointer
        # Load checkpoint loading configuration
        ckpt_loading_cfg = cfg.get("ckpt_loading", {})
        # Check for special configurations
        if resume_training:
            strict, load_random_states, load_weights_only = (True, True, False)
        elif finetuning:
            strict, load_random_states, load_weights_only = (False, False, True)
        else:
            # Default to configuration in `ckpt_loading`
            strict = ckpt_loading_cfg.get("strict", True)
            load_random_states = ckpt_loading_cfg.get("load_random_states", True)
            load_weights_only = ckpt_loading_cfg.get("load_weights_only", False)
        # Load checkpoint
        start_epoch, global_step = checkpointer.load_checkpoint(
            checkpoint_dir=ckpt_path,
            strict=strict,
            load_random_states=load_random_states,
            load_weights_only=load_weights_only,
            load_compile_cache=ckpt_loading_cfg.get("load_compile_cache", True),
        )

    # Sync shared parameters across ranks
    log.info("Synchronizing shared parameters and optimizers across ranks from rank 0")
    broadcast_model_shared(model, src=0)
    broadcast_optimizer(shared_optimizer, src=0, states_match=True)

    dist.barrier()

    # Add this section for fine-tuning support
    if cfg.get("finetuning", False) or cfg.get("freeze_core", False):
        log.info("Fine-tuning mode: Freezing core parameters")
        # Freeze core parameters
        for param in model.core_parameters():
            param.requires_grad = False

    if (
        lr_linear_decay_until_step := cfg.get("lr_linear_decay_until_step")
    ) is not None:
        assert lr_linear_decay_until_step > global_step, (
            "`lr_linear_decay_until_step` must be greater than `global_step`!"
        )
        max_steps = lr_linear_decay_until_step
        total_iters = lr_linear_decay_until_step - global_step
        power = cfg.get("lr_linear_decay_poly_power", 1.0)
        log.info(
            f"Overriding schedulers with linear learning rate decay from step "
            f"{global_step} to {max_steps} (i.e. {total_iters} steps)!"
        )
        shared_scheduler, rank_scheduler = create_ld_schedulers(
            shared_optimizer,
            rank_optimizer,
            total_iters,
            power=power,
        )
        # Assign schedulers to checkpointer
        checkpointer.assign_modules_and_rank(
            shared_scheduler=shared_scheduler,
            rank_scheduler=rank_scheduler,
        )

    dist.barrier()

    shared_parameters = []
    for p in model.shared_parameters():
        shared_parameters.append(p)

    rank_parameters = []
    for p in model.rank_parameters():
        rank_parameters.append(p)

    dist.barrier()

    # Initialize gradient buckets for efficient all_reduce
    bucket_size_bytes = cfg.distributed.gradient_bucket_size_mb * 1000**2
    gradient_buckets = initialize_buckets(shared_parameters, bucket_size_bytes)
    log.info(f"Gradient buckets initialized with size {bucket_size_bytes} bytes")

    # TODO: Is this necessary?
    dist.barrier()

    # Main training loop
    log.info(f"Starting training from epoch {start_epoch} to {max_epochs}")

    # Set model to train mode
    model.train()

    if max_steps is not None and global_step >= max_steps:
        log.info(
            f"Already more than {max_steps} steps ({global_step}), stopping training!"
        )
        return

    for epoch in range(start_epoch, max_epochs):
        # Update model's epoch counter
        model._current_epoch = epoch

        # NOTE: We assume below that train_dl is a SyncedFastSessionDataLoader
        # TODO: Make this more robust to other dataloaders
        assert isinstance(train_dl, SyncedFastSessionDataLoader), (
            "train_dl must be a SyncedFastSessionDataLoader!"
        )
        # Get starting batch index from dataset
        batch_idx = train_dl.current_batch
        # Set limit_train_batches to the length of the dataset if not provided
        if limit_train_batches is None or limit_train_batches > len(train_dl):
            limit_train_batches = len(train_dl)
        # Skip if we've reached the batch limit
        if batch_idx >= limit_train_batches:
            train_dl.reset_state()
            continue

        # Training loop
        if pbar_enabled:
            progress_bar = tqdm(
                train_dl,
                initial=batch_idx,
                total=limit_train_batches,
                desc=f"Epoch {epoch}",
                disable=rank != 0,
            )
        else:
            progress_bar = train_dl

        # Initialize collection of accumulated batch losses, and additional batch keys
        #  (for checkpoint traces)
        accumulated_losses = defaultdict(list)
        accumulated_keys = defaultdict(list)

        # Iterate over remaining batches
        log.info(f"Starting epoch {epoch} with {limit_train_batches} batches...")
        for batch in progress_bar:
            # NOTE: SyncedFastSessionDataLoader.current_batch is updated by the iterator after yielding a batch
            batch_idx = train_dl.current_batch

            batch = transfer_batch_to_device(batch, device, dtype=None)  # dtype)

            # Forward pass and loss calculation
            with autocast(device_type=device.type, dtype=dtype, enabled=amp):
                # Run model forward pass + calculate loss
                output: OMModelOutput = model.forward_from_batch(
                    batch,
                    all_neurons_override=all_neurons_override,
                )
                loss = output.loss

            # Scale loss by accumulation steps to maintain correct gradient magnitude
            loss = loss / gradient_accumulation_steps

            # Backward pass with automatic mixed precision
            loss.backward()

            # Store losses for logging. NOTE: We store list of unscaled losses (which are
            #  averaged by `LossTracker.update()`)
            for key in ["loss", "response_loss", "behavior_loss"]:
                val = getattr(output, key)
                if val is not None and isinstance(val, torch.Tensor):
                    val = val.detach()
                accumulated_losses[key].append(val)
            # Collect session / batch keys for checkpoint traces
            accumulated_keys["sess_key"].append(batch[0])
            accumulated_keys["batch_key"].append(output.cache_key)

            # Determine if we're at an accumulation step or the last step
            is_accumulation_step = batch_idx % gradient_accumulation_steps == 0
            is_last_step = batch_idx >= limit_train_batches
            is_last_overall_step = (
                max_steps is not None and (global_step + 1) >= max_steps
            )

            # Skip if we're not at an accumulation step or the last step
            if not is_accumulation_step and not (is_last_step or is_last_overall_step):
                continue

            # -------- Custom all reduce bucketing and communication --------
            # Part 1: Sending gradients
            handles = reduce_gradients(gradient_buckets)
            # Sandwich computations while we wait for the handles to receive all buckets
            rank_grad_norm = torch.nn.utils.clip_grad_norm_(
                rank_parameters, max_norm=clip_val, norm_type=2
            )
            # ... step rank-specific optimizer and ...
            rank_optimizer.step()
            rank_optimizer.zero_grad()
            # ... step rank-specific scheduler
            rank_scheduler.step()
            # ... update latest learning rates (for logging)
            rank_lr = rank_scheduler.get_last_lr()[0]

            # Part 2: Unpacking gradients
            unpack_gradients(gradient_buckets, handles)
            # now we can clip the gradients of the shared params and ...
            shared_grad_norm = torch.nn.utils.clip_grad_norm_(
                shared_parameters, max_norm=clip_val, norm_type=2
            )
            # ... step shared optimizer and ...
            shared_optimizer.step()
            shared_optimizer.zero_grad()
            # ... step shared scheduler
            shared_scheduler.step()
            # ... update latest learning rates (for logging)
            shared_lr = shared_scheduler.get_last_lr()[0]
            # -------- Update trackers after step --------

            # Update running lossess
            running_losses.update(
                **accumulated_losses,
                session_name=accumulated_keys["sess_key"],
            )

            # Update step-wise loss collections
            ckpt_traces.update(
                **accumulated_losses,
                rank_grad_norm=rank_grad_norm,
                shared_grad_norm=shared_grad_norm,
                rank_lr=rank_lr,
                shared_lr=shared_lr,
                batch_key=tuple(accumulated_keys["batch_key"]),
                session_name=accumulated_keys["sess_key"],
            )

            # Reset accumulated losses
            accumulated_losses = defaultdict(list)
            accumulated_keys = defaultdict(list)

            # Only count step if we've done a full accumulation cycle or it's the last batch. TODO: Is this incremement premature?
            global_step += 1

            # Determine post-prossecing required for this batch
            is_divergence_step = (
                calc_divergence_every_n_steps is not None
                and global_step % calc_divergence_every_n_steps == 0
            )
            is_task_step = (
                log_task_every_n_steps is not None
                and global_step % log_task_every_n_steps == 0
            )
            is_sync_step = (
                sync_params_every_n_steps is not None
                and global_step % sync_params_every_n_steps == 0
            )
            is_sync_optimizers_step = (
                sync_optimizers_every_n_steps is not None
                and global_step % sync_optimizers_every_n_steps == 0
            )
            is_train_log_step = (
                log_every_n_steps > 0 and global_step % log_every_n_steps == 0
            )
            is_val_step = val_every_n_steps > 0 and global_step % val_every_n_steps == 0
            is_ckpt_step = (
                checkpointer is not None
                and ckpt_every_n_steps > 0
                and global_step % ckpt_every_n_steps == 0
            )
            is_pbar_step = (
                pbar_enabled
                and pbar_every_n_steps > 0
                and global_step % pbar_every_n_steps == 0
            )

            # Compute running losses
            if (
                is_pbar_step
                or is_train_log_step
                or is_val_step
                or is_ckpt_step
                or is_last_step
                or is_last_overall_step
            ):
                current_losses = running_losses.compute()

            # Update progress bar. NOTE: Progress bar is only visible on rank 0 anyways
            if is_pbar_step:
                progress_bar.set_postfix(current_losses)

            # Gather losses and meta info across ranks to rank 0 if needed for logging
            if (
                is_train_log_step
                or is_val_step
                or is_ckpt_step
                or is_last_step
                or is_last_overall_step
            ):
                # gather all losses across ranks to rank 0
                combined_losses_rank_0 = gather_validation_metrics(
                    current_losses, rank, world_size
                )
                # Calculate overall summary metrics (i.e. average accross ranks/sessions)
                losses_rank_0_summary = (
                    LossTrackerCollection.summarize_distributed_results(
                        combined_losses_rank_0
                    )
                )
                # Get latest learning from rank-specific and shared schedulers
                current_lr = {
                    f"rank_{rank}/rank_lr": rank_lr,
                    f"rank_{rank}/shared_lr": shared_lr,
                }
                # Gather latest learning rates across ranks to rank 0
                combined_lr_rank_0 = gather_validation_metrics(
                    current_lr, rank, world_size
                )
                # Gather grad norms from across ranks to rank 0
                current_grad_norms = {
                    f"rank_{rank}/rank_grad_norm": rank_grad_norm.item(),
                    f"rank_{rank}/shared_grad_norm": shared_grad_norm.item(),
                }
                combined_grad_norms_rank_0 = gather_validation_metrics(
                    current_grad_norms, rank, world_size
                )
                # Add gradient norms and learning rates
                combined_losses_and_meta_rank_0 = dict(
                    **combined_losses_rank_0,
                    **losses_rank_0_summary,
                    **combined_grad_norms_rank_0,
                    **combined_lr_rank_0,
                )

                # Log metrics to loggers every N steps (or if we're validating/checkpointing)
                log_to_loggers(
                    logger,
                    combined_losses_and_meta_rank_0,
                    prefix="train",
                    step=global_step,
                )

            # Log divergence metrics (before syncing) every N steps if configured
            if is_divergence_step:
                log.info("Calculating divergence across ranks from rank 0")
                divergence_stats = calc_divergence(
                    model,
                    shared_optimizer,
                    threshold=divergence_threshold,
                    mode=divergence_mode,
                    rank=rank,
                    world_size=world_size,
                )
                log_to_loggers(
                    logger, divergence_stats, prefix="train", step=global_step
                )

            if is_task_step:
                task_ids = collect_task_ids(output.cache_key, rank, world_size)
                log_to_loggers(logger, task_ids, prefix="train", step=global_step)

            # Sync parameters and optimizers every N steps if configured
            if is_sync_step:
                log.info("Syncing parameters and optimizers across ranks from rank 0")
                broadcast_model_shared(model, src=0)

            if is_sync_optimizers_step:
                broadcast_optimizer(shared_optimizer, src=0, states_match=True)

            # Handle validation during epoch if needed (e.g., check every N steps)
            combined_val_metrics_all_sets, combined_val_loss_all_sets = None, None
            if is_val_step and len(val_dls) > 0:
                log.info(f"Validating at step {global_step}")
                combined_val_metrics_all_sets, combined_val_loss_all_sets = {}, {}
                for val_dl_key, val_dl in val_dls.items():
                    dist.barrier()
                    # Run validation on "validation" set
                    val_masking_strategies = val_masking_strategies_all_sets[val_dl_key]
                    val_loss, val_metrics = run_evaluation(
                        model,
                        val_dl,
                        amp,
                        device,
                        limit_val_batches,
                        val_masking_strategies,
                        rank,
                        val_dl_key,
                        initial_rng_state,
                        initial_dl_rng_states,
                    )
                    dist.barrier()
                    # Log "validation" metrics and save checkpoint
                    combined_val_metrics, combined_val_loss = handle_metrics(
                        val_loss,
                        val_metrics,
                        None,
                        model.session_map,
                        logger,
                        global_step,
                        rank,
                        world_size,
                        prefix=val_dl_key,
                    )
                    dist.barrier()
                    combined_val_metrics_all_sets.update(combined_val_metrics)
                    combined_val_loss_all_sets.update(combined_val_loss)

            # Save checkpoint
            if is_ckpt_step:
                log.info(f"Saving checkpoint at the end of step {global_step}")
                combined_loss_windows_rank_0 = gather_validation_metrics(
                    ckpt_traces.loss_windows, rank, world_size
                )
                dist.barrier()
                handle_checkpointing(
                    checkpointer,
                    global_step,
                    epoch,
                    combined_val_metrics_all_sets,
                    combined_val_loss_all_sets,
                    combined_losses_and_meta_rank_0,
                    combined_loss_windows=combined_loss_windows_rank_0,
                )
                # Reset step-wise loss collections after saving checkpoint
                ckpt_traces.reset()
                dist.barrier()

            # Skip if we've reached the batch limit
            if is_last_step or is_last_overall_step:
                if is_last_step:
                    train_dl.reset_state()
                break

        # TODO: We shouldn't actually need to resave the checkpoint here, we should just save on last step above
        # Determine post-prossecing required for this epoch
        is_last_epoch = epoch + 1 == max_epochs
        is_val_epoch = val_every_n_epochs > 0 and epoch % val_every_n_epochs == 0
        is_ckpt_epoch = (
            checkpointer is not None
            and ckpt_every_n_epochs > 0
            and epoch % ckpt_every_n_epochs == 0
        )

        # Gather losses across ranks to rank 0 if needed for logging
        if is_val_epoch or is_ckpt_epoch or is_last_epoch or is_last_overall_step:
            # gather all losses across ranks to rank 0
            combined_losses_rank_0 = gather_validation_metrics(
                current_losses, rank, world_size
            )

        # Epoch-level validation if needed
        combined_val_metrics_all_sets, combined_val_loss_all_sets = None, None
        if (is_val_epoch or is_last_epoch or is_last_overall_step) and len(val_dls) > 0:
            log.info(f"Validating at the end of epoch {epoch}")
            combined_val_metrics_all_sets, combined_val_loss_all_sets = {}, {}
            for val_dl_key, val_dl in val_dls.items():
                dist.barrier()
                # Run validation on "validation" set
                val_masking_strategies = val_masking_strategies_all_sets[val_dl_key]
                val_loss, val_metrics = run_evaluation(
                    model,
                    val_dl,
                    amp,
                    device,
                    limit_val_batches,
                    val_masking_strategies,
                    rank,
                    val_dl_key,
                    initial_rng_state,
                    initial_dl_rng_states,
                )
                dist.barrier()
                # Log "validation" metrics and save checkpoint
                combined_val_metrics, combined_val_loss = handle_metrics(
                    val_loss,
                    val_metrics,
                    None,
                    model.session_map,
                    logger,
                    global_step,
                    rank,
                    world_size,
                    prefix=val_dl_key,
                )
                dist.barrier()
                combined_val_metrics_all_sets.update(combined_val_metrics)
                combined_val_loss_all_sets.update(combined_val_loss)

        # Increment epoch for checkpoint saving (using indicator of last step from above)
        if is_last_step:
            next_epoch = epoch + 1
        else:
            next_epoch = epoch

        # Save checkpoint
        if is_ckpt_epoch or is_last_epoch or is_last_overall_step:
            log.info(f"Saving checkpoint at the end of epoch {epoch}")
            combined_loss_windows_rank_0 = gather_validation_metrics(
                ckpt_traces.loss_windows, rank, world_size
            )
            dist.barrier()
            handle_checkpointing(
                checkpointer,
                global_step,
                next_epoch,
                combined_val_metrics_all_sets,
                combined_val_loss_all_sets,
                combined_losses_rank_0,
                combined_loss_windows=combined_loss_windows_rank_0,
            )
            # Reset step-wise loss collections after saving checkpoint
            ckpt_traces.reset()
            dist.barrier()

        if is_last_overall_step:
            log.info(f"Finished training for {global_step} steps!")
            return

        # NOTE: We maintain a *windowed* running loss accross epochs, so no need to reset
        # # Reset running losses
        # running_losses.reset()

    log.info(f"Finished training for {max_epochs} epochs!")
