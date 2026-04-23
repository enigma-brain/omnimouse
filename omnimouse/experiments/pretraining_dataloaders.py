"""
Collection of dataset paths and configuration utils for running scaling laws experiments.
"""
from typing import Any, Dict, List, Optional, Union, Type
import hashlib
import time
import logging
from copy import deepcopy
from experanto.datasets import ChunkDataset
from experanto.utils import MultiEpochsDataLoader, LongCycler, SessionConcatDataset, FastSessionDataLoader
from omnimouse.experiments.scaling_laws import sensorium_test_set, validation_5

logger = logging.getLogger(__name__)


# TODO: Move this to a utils file (code duplicated in `omnimouse/distributed/utils.py`)
def get_path_hash(path: str) -> int:
    return int(hashlib.sha256(path.encode()).hexdigest(), 16) % 10000


def get_pretraining_dataloader(paths: List[str],
                               configs: Union[Dict, List[Dict]] = None,
                               seed: Optional[int] = 0,
                               pretrain_lists: List = [],
                               dataloader_config: Optional[Dict] = None,
                               **kwargs) -> 'FastSessionDataLoader':
    """
    Creates a multi-session dataloader using SessionConcatDataset and SessionDataLoader.
    Returns (session_key, batch) pairs during iteration.

    Args:
        paths: List of paths to dataset files
        configs: Configuration for datasets (single config or list of configs)
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
        prefetch_factor: Prefetch factor for data loading
        **kwargs: Additional arguments

    Returns:
        SessionDataLoader instance or None if no valid datasets found
    """
    if configs is None and "config" in kwargs:
        configs = kwargs.pop("config")

    # Convert single config to list for uniform handling
    if not isinstance(configs, list):
        configs = [configs] * len(paths)

    # Create datasets
    datasets = []
    session_names = []

    start_time = time.time()
    for i, (path, cfg) in enumerate(zip(paths, configs)):

        cfg = deepcopy(cfg)

        # Create dataset with deterministic seed
        path_hash = get_path_hash(path) % 10000
        dataset_seed = seed + path_hash if seed is not None else None

        # Set specific seed for this dataset if needed
        if "dataset" in cfg:
            cfg = cfg['dataset']
        assert hasattr(cfg, 'seed'), f"Seed not found in config for path {path}: {cfg}"
        if dataset_seed is not None:
            cfg['seed'] = dataset_seed

        for pretraining_scan in pretrain_lists:
            if pretraining_scan in path:
                logger.info(f"Pretraining scan found in {path}, will use full pretrain set")
                cfg.modality_config.screen.valid_condition = [dict(tier="train"), dict(tier="validation")]
                break

        # Assuming ChunkDataset is defined elsewhere
        dataset = ChunkDataset(path, **cfg)
        session_name = dataset.data_key

        # Only add datasets with non-zero length
        if len(dataset) > 0:
            datasets.append(dataset)
            session_names.append(session_name)

    if not datasets:
        return None

    # Create the concatenated dataset
    concat_dataset = SessionConcatDataset(datasets, session_names)

    # Get dataloader config from the first config
    if dataloader_config is None:
        dataloader_config = dict(configs[0].get('dataloader', {}))

    # Create the dataloader with our simplified implementation
    return FastSessionDataLoader(
        dataset=concat_dataset,
        seed=seed,
        **dataloader_config
    )