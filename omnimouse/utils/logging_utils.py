import os
from typing import Any, Dict, Iterable, List, Tuple
from pathlib import Path
from subprocess import check_output, run  # nosec B404, B603, B607
import tempfile
import shutil
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
import yaml

import mlflow
from lightning_utilities.core.rank_zero import rank_zero_only
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger, MLFlowLogger, CSVLogger
from lightning.pytorch.utilities.model_summary import summarize
from lightning.pytorch.callbacks.rich_model_summary import RichModelSummary
from torch import nn
import wandb

from omnimouse.utils.pylogger import RankedLogger
from omnimouse.utils.utils import resolve_cfg, omegaconf_to_masking_strategy
from omnimouse.utils.git_utils import get_git_info, get_git_filtered_files

log = RankedLogger(__name__, rank_zero_only=True)

# Setup precision settings
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
mlflow.enable_system_metrics_logging()

def _prep_masking_strategies(masking_strategies: Iterable[DictConfig]) -> List[Tuple]:
    return list(map(str, map(omegaconf_to_masking_strategy, masking_strategies)))


def _to_dict(cfg: DictConfig | None, resolve: bool = False) -> Dict[str, Any]:
    if cfg is None:
        return None
    return OmegaConf.to_container(cfg, resolve=resolve)


def _defaultdict_to_dict(dd: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: _defaultdict_to_dict(v) if isinstance(v, Dict) else v for k, v in dd.items()
    }


def upload_artifact(
    logger,
    content,
    artifact_type: str,
    filename: str = None,
    artifact_name: str = None,
    artifact_path: str = None,
    description: str = None,
    resolve: bool = False,
    is_directory: bool = False,
):
    """
    Upload content as an artifact to wandb or mlflow.
    
    Args:
        logger: WandbLogger or MLFlowLogger instance
        content: Content to upload (DictConfig, string, list of file tuples for directories)
        artifact_type: Type of artifact ("config", "code", etc.)
        filename: Name of the file to save (for single files)
        artifact_name: Name of the artifact for WandB
        artifact_path: Path for MLFlow artifacts
        description: Description for WandB artifacts
        resolve: Whether to resolve OmegaConf configs
        is_directory: Whether content represents multiple files
    
    Examples:
        # Upload OmegaConf configuration
        upload_artifact(
            logger=logger,
            content=config,
            artifact_type="config",
            filename="config.yaml",
            artifact_name="config",
            artifact_path="config",
            description="OmegaConf configuration",
            resolve=True
        )
        
        # Upload git diff
        upload_artifact(
            logger=logger,
            content=git_diff_string,
            artifact_type="code",
            filename="git_diff.patch",
            artifact_name="git-diff",
            artifact_path="git",
            description="Git diff at experiment start"
        )
        
        # Upload source code directory
        filtered_files = get_git_filtered_files("/path/to/code")
        upload_artifact(
            logger=logger,
            content=filtered_files,
            artifact_type="code",
            artifact_name="project-source",
            artifact_path="code",
            description="Project source code",
            is_directory=True
        )
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            staging_dir = temp_path / "artifacts"
            staging_dir.mkdir()
            
            # Prepare files in staging directory
            if is_directory:
                # Multiple files: copy maintaining structure
                files_to_add = []
                for abs_path, rel_path in content:
                    dest_path = staging_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(abs_path), str(dest_path))
                    files_to_add.append((dest_path, rel_path))
            else:
                # Single file: save content
                file_path = staging_dir / filename
                
                if isinstance(content, DictConfig):
                    OmegaConf.save(content, file_path, resolve=resolve)
                elif isinstance(content, str):
                    file_path.write_text(content)
                elif isinstance(content, Dict):
                    content = _defaultdict_to_dict(content)
                    with open(file_path, 'w+') as f:
                        yaml.dump(content, f)
                else:
                    raise ValueError(f"Unsupported content type: {type(content)}")
                
                files_to_add = [(file_path, filename)]
            
            # Upload artifacts
            if isinstance(logger, WandbLogger):
                # WandB: Create artifact and add all files
                artifact = wandb.Artifact(
                    name=artifact_name or artifact_type,
                    type=artifact_type,
                    description=description
                )
                for file_path, name in files_to_add:
                    artifact.add_file(str(file_path), name=str(name))
                logger.experiment.log_artifact(artifact)
            elif isinstance(logger, MLFlowLogger):
                # MLFlow: Log the entire staging directory
                logger.experiment.log_artifacts(
                    run_id=logger.run_id,
                    local_dir=str(staging_dir),
                    artifact_path=artifact_path or artifact_type
                )
            elif isinstance(logger, CSVLogger):
                # CSV: Copy the staging directory to the logger's save_dir
                dst = os.path.join(logger.save_dir, 'artifacts', artifact_type)
                os.makedirs(dst, exist_ok=True)
                shutil.copytree(staging_dir, dst, dirs_exist_ok=True)
            else:
                log.warning(
                    f"Logger type {type(logger).__name__} is not supported. "
                    "Only WandbLogger and MLFlowLogger are supported."
                )
                    
    except Exception as e:
        log.warning(f"Failed to upload {artifact_type} to {type(logger).__name__}: {e}")


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any], save_code: bool = False) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    unresolved_cfg = deepcopy(object_dict["cfg"])
    cfg = resolve_cfg(deepcopy(unresolved_cfg))
    model = object_dict["model"]
    logger = object_dict.get("logger", None)
    root = object_dict.get("root", None)

    if not logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return
    
    # Top level params. TODO: Make this automatic, instead of manually adding each one!
    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = list(cfg.get("tags", []))
    hparams["tags"].extend(list(cfg.get("cli_tags", []))) # Allow for *additional* tags from command line
    hparams["tags"].extend([f'model-{t}' for t in cfg.model.get('model_tags', [])])
    hparams["seed"] = cfg.get("seed")
    hparams["use_seed_per_rank"] = cfg.get("use_seed_per_rank")
    hparams["fix_eval_rng_state"] = cfg.get("fix_eval_rng_state")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["resume_training"] = cfg.get("resume_training")
    hparams["finetuning"] = cfg.get("finetuning")
    hparams["freeze_core"] = cfg.get("freeze_core")
    hparams["lr_linear_decay_until_step"] = cfg.get("lr_linear_decay_until_step")
    hparams["ckpt_loading"] = _to_dict(cfg.get("ckpt_loading"))
    hparams["linked_experiment_name"] = cfg.get("linked_experiment_name")
    hparams["linked_run_name"] = cfg.get("linked_run_name")
    hparams["all_neurons_override"] = cfg.get("all_neurons_override")

    # Data config
    hparams["data"] = _to_dict(cfg["data"])

    # Model config
    model_cfg = deepcopy(cfg["model"])
    if 'masking_strategies' in model_cfg:
        model_cfg.masking_strategies = _prep_masking_strategies(model_cfg.masking_strategies)
    hparams["model"] = _to_dict(model_cfg)
    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # Trainer config
    hparams["trainer"] = _to_dict(cfg["trainer"])

    # Paths config
    paths_cfg = cfg.get("paths")
    hparams["paths"] = _to_dict(paths_cfg)
    exp_name, run_name = None, None
    if paths_cfg is not None:
        exp_name = paths_cfg.get('experiment_name')
        run_name = paths_cfg.get('run_name')
    hparams["experiment_name"] = exp_name
    hparams["run_name"] = run_name

    # Extras config
    hparams["extras"] = _to_dict(cfg.get("extras"))

    # Evaluation config
    eval_cfg = cfg.get("evaluation")
    if eval_cfg is not None:
        eval_cfg = deepcopy(eval_cfg)
        for k, v in eval_cfg.items():
            if "masking_strategies" in k and v is not None:
                prepped = _prep_masking_strategies(v.values())
                for _k, _v in zip(eval_cfg[k].keys(), prepped):
                    eval_cfg[k][_k] = _v
        hparams["evaluation"] = _to_dict(eval_cfg)
    
    # Distributed config
    hparams["distributed"] = _to_dict(cfg.get("distributed"))

    # Git information
    git_info, git_diff_string = get_git_info(root)
    hparams.update(git_info)

    # send hparams to all loggers
    logger.log_hyperparams(hparams)
    
    # Upload git diff
    if git_diff_string is not None:
        upload_artifact(
            logger=logger,
            content=git_diff_string,
            artifact_type="code",
            filename="git_diff.txt",
            artifact_name="git-diff",
            artifact_path="git",
            description="Git diff at experiment start"
        )

    # Upload OmegaConf configuration
    upload_artifact(
        logger=logger,
        content=unresolved_cfg,
        artifact_type="config",
        filename="config.yaml",
        artifact_name="config",
        artifact_path="config",
        description="OmegaConf configuration",
        resolve=True
    )

    # TODO: Support uploading code as artifact
    if save_code:
        raise NotImplementedError("Uploading code as artifact is not implemented yet!")


def print_model_summary(model: LightningModule | nn.Module, max_depth: int = 2) -> None:
    """Print a summary of the model architecture using Lightning's summarize utility.
    
    Args:
        model: The LightningModule to summarize
        max_depth: Maximum depth of layers to show in the summary
    """
    # TODO: We set these attributes so that the utility thinks its a
    #  LightningModule, just a hack to take advantage of their functionality
    if not hasattr(model, "_trainer"):
        model._trainer = None
    if not hasattr(model, "example_input_array"):
        model.example_input_array = None

    # Get the model summary
    model_summary = summarize(model, max_depth=max_depth)
    
    # Extract summary data
    summary_data = model_summary._get_summary_data()
    total_parameters = model_summary.total_parameters
    trainable_parameters = model_summary.trainable_parameters
    model_size = model_summary.model_size
    total_training_modes = model_summary.total_training_modes
    
    # Format the summary table
    summary_table = RichModelSummary.summarize(
        summary_data,
        total_parameters,
        trainable_parameters,
        model_size,
        total_training_modes,
    )
    
    # Print the summary
    log.info(f"\nModel Summary:\n{summary_table}")