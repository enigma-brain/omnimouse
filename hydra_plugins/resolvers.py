"""
NOTE: This file is used to register custom resolvers for the Hydra config system.
We take advantage of the fact that the `hydra_plugins` path are automatically
loaded by Hydra before configuration parsing. For more on this, see this discussion on 
the `Hydra forums <https://github.com/facebookresearch/hydra/issues/2835>`_.
"""
from typing import Sequence, Any, Optional
from random_word import RandomWords
from datetime import datetime, timezone
import tempfile
import os
from pathlib import Path
from omegaconf import OmegaConf


def generate_run_name(num_random_words: int = 2) -> str:
    """
    Utility function to generate a run name composed of `n` random words
    and the current date and time.
    """
    # Initialize random words generator
    rwg = RandomWords()
    # Generate `n` random words
    words = [rwg.get_random_word() for _ in range(num_random_words)]
    # Get current date and time
    now = datetime.now(timezone.utc).strftime("%y%m%d%H%M")
    # Generate run name
    run_name = '-'.join(words) + '-' + now
    # Return run name
    return run_name

OmegaConf.register_new_resolver('generate_run_name', generate_run_name, use_cache=True)


def generate_temp_dir(prefix: str = "", slurm_mode: bool = False) -> str:
    """
    Utility function to generate a temporary directory with a prefix.
    The directory will be automatically removed when the process exits.
    """
    if slurm_mode:
        prefix += f"{os.getenv('SLURM_JOB_ID', 'local')}_"
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    # Create a TemporaryDirectory object that will clean up on process exit
    temp_obj = tempfile.TemporaryDirectory(dir=temp_dir)
    # Return just the path string
    return temp_obj.name

OmegaConf.register_new_resolver('generate_temp_dir', generate_temp_dir, use_cache=True)


def merge_lists(
    x: Optional[Sequence[Any]] = None,
    y: Optional[Sequence[Any]] = None,
) -> Optional[Sequence[Any]]:
    """
    Merge two lists.

    TODO: Raise error if `x` and `y` do not support `+`
    """
    if x is None and y is None:
        return None
    elif x is None:
        return y
    elif y is None:
        return x
    return x + y

OmegaConf.register_new_resolver("merge_lists", merge_lists, use_cache=False)


def convert_samples_to_frames(
    num_samples_per_s: int,
    num_samples_per_block: int,
    num_frames_per_s: int,
) -> int:
    """
    Convert number of samples per block to number of frames per block.
    """
    return int((num_samples_per_block / num_samples_per_s) * num_frames_per_s)

OmegaConf.register_new_resolver("convert_samples_to_frames", convert_samples_to_frames, use_cache=False)


def versioned_path(
    path: str | os.PathLike,
) -> str:
    """
    Convert number of samples per block to number of frames per block.
    """
    path = Path(path)
    version = 1
    while path.exists():
        path = path.parent / f"{path.name}_v{version}{path.suffix}"
        version += 1
    return str(path)

OmegaConf.register_new_resolver("versioned_path", versioned_path, use_cache=False)
