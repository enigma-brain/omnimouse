from typing import Optional, Tuple, List
import os
from pathlib import Path
from omegaconf import DictConfig
import hydra
from hydra import compose
from hydra.core.global_hydra import GlobalHydra

from omnimouse.utils.utils import (
    extras,
)


def compose_omnimouse_config(
    project_root: str | os.PathLike,
    config_name: str,
    run_name: str,
    local_override: Optional[str] = None,
    experiment_override: Optional[str] = None,
    eval_experiment_override: Optional[str] = None,
    ckpt_path: Optional[str | os.PathLike] = None,
    print_config: bool = True,
    cli_overrides: Optional[List[str]] = None,
) -> Tuple[os.PathLike, DictConfig]:
    """
    Compose a Hydra config for the omnimouse project.
    """
    print(f"Composing config {config_name}")
    # Clear any existing Hydra initialization
    GlobalHydra.instance().clear()
    # Initialize hydra with the config dir
    configs_path = Path(project_root) / "configs"
    config_path_rel = configs_path.relative_to(Path(__file__).parent, walk_up=True)
    print(f"Config path: {__file__} {config_path_rel}")
    hydra.initialize(config_path=str(config_path_rel), version_base="1.1")
    # Collect "cli" overrides
    overrides=[f"paths.run_name={run_name}",]
    if local_override is not None:
        overrides.append(f"local={local_override}")
    if experiment_override is not None:
        overrides.append(f"experiment={experiment_override}")
    if eval_experiment_override is not None:
        overrides.append(f"eval_experiment={eval_experiment_override}")
    if ckpt_path is not None:
        overrides.append(f"ckpt_path={ckpt_path}")
    if not print_config:
        overrides.append("extras.print_config=false")
    if cli_overrides is not None:
        overrides.extend(cli_overrides)
    # Compose the config
    cfg = compose(
        config_name=config_name,
        overrides=overrides,
    )
    # Create output dir if it doesn't exist
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    # Run extra utilities
    extras(cfg)

    return output_dir, cfg