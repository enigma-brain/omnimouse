"""
Utils for distributed training via torchrun multiprocessing
"""
from typing import Callable, Dict, Any, Union, Hashable, Optional
import os
import importlib
import datetime
import json
import math
import logging
import hashlib
from chunk import Chunk
from collections import defaultdict
from copy import deepcopy
import traceback
from itertools import chain
from functools import wraps
import numpy as np
import torch
from torch import Tensor
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from experanto.utils import SessionConcatDataset, FastSessionDataLoader
from experanto.datasets import ChunkDataset

from omnimouse.experiments.scaling_laws import validation_5, sensorium_test_set, sensorium_2023_test_set
from omnimouse.modeling import Model

logger = logging.getLogger()


# TODO: Move this to a utils file (code duplicated in `omnimouse/experiments/pretraining_dataloaders.py`)
def get_path_hash(path: str) -> int:
    return int(hashlib.sha256(path.encode()).hexdigest(), 16) % 10000

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.info("No CUDA device available, returning None for device")
        return local_rank, local_rank, None, None

    # Set device before anything else
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Initialize process group with timeout
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=30), device_id=torch.device(local_rank))

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    logger.info(f"Initialized process group: rank {rank}/{world_size}, local_rank: {local_rank}")

    tensor = torch.tensor([rank], dtype=torch.int64, device=device)
    gathered = [torch.zeros(1, dtype=torch.int64, device=device)
                for _ in range(world_size)]

    # Add explicit barrier before all_gather
    logger.info(f"Rank {rank}: Before barrier")
    dist.barrier()
    logger.info(f"Rank {rank}: After barrier, before all_gather")

    dist.all_gather(gathered, tensor)
    gathered = [t.item() for t in gathered]

    logger.info(f"Rank {rank}: Can see processes {gathered}")
    return rank, local_rank, world_size, device


def cleanup_distributed():
    """Properly clean up the distributed process group to avoid resource leaks."""
    try:
        if dist.is_available() and dist.is_initialized():
            logger.info("Destroying distributed process group...")
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed successfully")
        else:
            # Log why we're not cleaning up, for debugging
            if not dist.is_available():
                logger.debug("Distributed not available, skipping cleanup")
            elif not dist.is_initialized():
                logger.debug("Distributed not initialized, skipping cleanup")
    except Exception as e:
        # Log the error but don't raise it, since this is cleanup code
        logger.warning(f"Error during distributed cleanup: {e}")
        # Still attempt to log success in case the cleanup partially worked
        logger.info("Attempted distributed cleanup despite errors")


def get_dataset_list(list_name: str):
    """
    Dynamically retrieves a dataset paths (expected to be a list) from either the
    'omnimouse.experiments.sandbox' or 'omnimouse.experiments.scaling_laws' module.

    Args:
        list_name: String name of the variable to retrieve.

    Returns:
        The value of the variable found in one of the specified modules.

    Raises:
        AttributeError: If the variable name is not found as an attribute in either module
                        after successful import.
        ImportError: If the specified modules themselves cannot be imported (this error
                     will propagate from importlib.import_module).
    """
    # List of module names to check, in order of priority
    module_names_to_check = [
        'omnimouse.experiments.sandbox',
        'omnimouse.experiments.scaling_laws',
        'omnimouse.experiments.legacy',
    ]

    found_module = None
    module_instance = None

    # Iterate through the modules to find the variable
    for module_name in module_names_to_check:
        try:
            # Attempt to import the module
            current_module = importlib.import_module(module_name)
            # Check if the module has the desired attribute
            if hasattr(current_module, list_name):
                found_module = module_name
                module_instance = current_module
                break # Stop searching once found
        except ImportError:
            print(f"Could not import module '{module_name}'.")

    # If found in one of the modules, return the attribute's value
    if found_module and module_instance:
        print(f"Found '{list_name}' in module: {found_module}") # Optional: for debugging/info
        return getattr(module_instance, list_name)
    else:
        # If the loop finishes without finding the attribute in any importable module
        raise AttributeError(
            f"Variable '{list_name}' not found in any of the specified modules: "
            f"{', '.join(module_names_to_check)}"
        )


def get_rank_paths(rank, datasets_per_rank, all_paths):
    """
    Distribute dataset paths across ranks.
    Each rank gets datasets_per_rank consecutive paths.
    """
    start_idx = rank * datasets_per_rank
    end_idx = min(start_idx + datasets_per_rank, len(all_paths))

    # If we run out of datasets, wrap around
    if start_idx >= len(all_paths):
        start_idx = start_idx % len(all_paths)
        end_idx = min(start_idx + datasets_per_rank, len(all_paths))

    paths = all_paths[start_idx:end_idx]

    # If we need more datasets to reach datasets_per_rank, wrap around
    if len(paths) < datasets_per_rank:
        additional_needed = datasets_per_rank - len(paths)
        paths = np.concatenate([paths, all_paths[:additional_needed]])

    return paths


def gather_dataset_info(train_dl, rank, world_size):
    """
    Gather dataset keys, animal IDs, and neuron counts from all distributed dataloaders.

    Args:
        train_dl: The local dataloader
        rank: Process rank
        world_size: Total number of processes

    Returns:
        dict: Combined dictionary with all keys, animal IDs, and neuron counts
    """
    # Get local dataset info
    local_info = {}
    for dataset in train_dl.dataset.datasets:
        session_key = dataset.data_key
        n_neurons = dataset._experiment.devices["responses"].n_signals
        animal_id = session_key.split('-')[0]

        local_info[session_key] = {
            'animal_id': animal_id,
            'n_neurons': n_neurons
        }

    # Convert local info to a serializable format
    # We'll use a list of tuples [(key, animal_id, n_neurons), ...]
    local_items = [(k, v['animal_id'], v['n_neurons']) for k, v in local_info.items()]

    # Determine maximum number of items across all ranks
    item_counts = [torch.tensor([len(local_items)], device="cuda") for _ in range(world_size)]
    dist.all_gather(item_counts, torch.tensor([len(local_items)], device="cuda"))
    max_items = max([count.item() for count in item_counts])

    # Pad local items list if needed
    if len(local_items) < max_items:
        # Use a special padding value that we can filter out later
        local_items.extend([("__pad__", "__pad__", -1) for _ in range(max_items - len(local_items))])

    # For each item, we need to gather its key, animal_id and neuron count
    all_data = []

    for i in range(max_items):
        if i < len(local_items):
            key, animal_id, neurons = local_items[i]
            # Encode strings as tensors - assuming keys and animal_ids won't exceed 50 chars
            key_tensor = torch.tensor([ord(c) for c in key] + [0] * (50 - len(key)), device="cuda")
            animal_tensor = torch.tensor([ord(c) for c in animal_id] + [0] * (20 - len(animal_id)), device="cuda")
            neuron_tensor = torch.tensor([neurons], device="cuda")
        else:
            # This shouldn't happen due to our padding above, but just in case
            key_tensor = torch.tensor([0] * 50, device="cuda")
            animal_tensor = torch.tensor([0] * 20, device="cuda")
            neuron_tensor = torch.tensor([-1], device="cuda")

        # Gather data from all ranks
        gathered_keys = [torch.zeros(50, dtype=torch.long, device="cuda") for _ in range(world_size)]
        gathered_animals = [torch.zeros(20, dtype=torch.long, device="cuda") for _ in range(world_size)]
        gathered_neurons = [torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(world_size)]

        dist.all_gather(gathered_keys, key_tensor)
        dist.all_gather(gathered_animals, animal_tensor)
        dist.all_gather(gathered_neurons, neuron_tensor)

        # Convert back to strings and add to our lists
        for r in range(world_size):
            key_chars = [chr(c.item()) for c in gathered_keys[r] if c.item() > 0]
            key = ''.join(key_chars)

            animal_chars = [chr(c.item()) for c in gathered_animals[r] if c.item() > 0]
            animal_id = ''.join(animal_chars)

            neurons = gathered_neurons[r].item()

            if key != "__pad__" and neurons != -1:
                all_data.append({
                    'data_key': key,
                    'animal_id': animal_id,
                    'n_neurons': neurons
                })

    # Combine into a dictionary with data_key as the key
    combined_info = {item['data_key']: {
        'animal_id': item['animal_id'],
        'n_neurons': item['n_neurons']
    } for item in all_data}

    if rank == 0:
        logger.info(f"Gathered dataset info: {combined_info}")

    return combined_info


def maybe_get_validation_concat_loader(
    cfg,
    paths,
    fixed_validation_set=None,
    max_sessions=None,
    dl_config=None,
    seed: int = None,
):
    """
    Creates validation dataloader using SessionConcatDataset approach.
    Returns (session_key, batch) pairs during iteration.

    Args:
        cfg: Configuration object
        paths: List of paths to dataset files
        max_sessions: Maximum number of sessions to load
        num_workers: Number of worker processes
        prefetch_factor: Prefetch factor for data loading

    Returns:
        SessionDataLoader instance or None if no validation loaders could be created
    """

    # Limit the number of paths
    if max_sessions is not None and len(paths) > max_sessions:
        paths = paths[:max_sessions]

    logger.info(f"Creating validation dataloader with paths: {paths}")
    valid_datasets = []
    valid_session_names = []

    if fixed_validation_set is None:
        # for backward compatibility: set only video sensorium 2023 as test set
        fixed_validation_set = sensorium_2023_test_set

    for i, path in enumerate(paths):
        for validation_scan in fixed_validation_set:
            if validation_scan in path:
                _cfg = deepcopy(cfg)

                # Create dataset with deterministic seed
                path_hash = get_path_hash(path) % 10000
                dataset_seed = seed + path_hash if seed is not None else None

                # Set specific seed for this dataset if needed
                if "dataset" in _cfg:
                    _cfg = _cfg['dataset']
                assert hasattr(_cfg, 'seed'), f"Seed not found in config for path {path}: {_cfg}"
                if dataset_seed is not None:
                    _cfg['seed'] = dataset_seed
                
                logger.info(f"Validation scan found in {path}, getting Dataset")
                dataset = ChunkDataset(path, **_cfg)
                session_name = dataset.data_key

                if len(dataset) == 0:
                    logger.warning(
                        f"Dataset {session_name} is empty (len = {len(dataset)}), skipping!"
                    )
                    continue

                valid_datasets.append(dataset)
                valid_session_names.append(session_name)

    if not valid_datasets:
        return None

    # Create the concatenated dataset
    concat_dataset = SessionConcatDataset(valid_datasets, valid_session_names)

    if len(concat_dataset) == 0:
        logger.warning("No valid sessions found in the concatenated dataset, returning None")
        return None
    # Get dataloader config
    if dl_config is None:
        if hasattr(cfg, 'dataloader'):
            dl_config = dict(cfg.dataloader)
        else:
            dl_config = {}
    dl_config = deepcopy(dl_config) # can't be too careful!
    # Create the dataloader
    return FastSessionDataLoader(
        dataset=concat_dataset,
        seed=seed,
        **dl_config
    )


def sync_dataloader_length(dataloader, rank, world_size):
    """
    Synchronize dataloader length across all ranks.

    Args:
        dataloader: FastSessionDataLoader instance
        rank: Process rank
        world_size: Total number of processes

    Returns:
        int: The synchronized number of batches per epoch
    """
    # Get local dataloader length in batches
    local_length = len(dataloader) if dataloader is not None else 0

    try:
        # Create tensors to hold lengths from all ranks
        all_lengths = [torch.tensor([0], device="cuda") for _ in range(world_size)]

        # Gather all lengths
        dist.all_gather(all_lengths, torch.tensor([local_length], device="cuda"))
        all_lengths = [length.item() for length in all_lengths]

        # Find maximum length across all ranks
        max_length = max(all_lengths)

        if rank == 0:
            logger.info(f"Dataset lengths across ranks (in batches): {all_lengths}")
            logger.info(f"Using synchronized length of {max_length} batches per epoch")

        return max_length
    except RuntimeError as e:
        # Handle distributed errors gracefully
        logger.error(f"Error in sync_dataloader_length: {e}")
        logger.warning("Falling back to local length")
        return local_length


class SyncedFastSessionDataLoader:
    """
    Wraps FastSessionDataLoader to support distributed training by
    ensuring all ranks process the same number of batches per epoch.
    """

    def __init__(self, original_dataloader, synced_length=None):
        """
        Initialize synchronized dataloader wrapper.

        Args:
            original_dataloader: FastSessionDataLoader instance to wrap
            synced_length: Synchronized number of batches per epoch
        """
        # Input validation
        if original_dataloader is None:
            raise ValueError("Cannot create SyncedFastSessionDataLoader with None dataloader")

        # Store the original dataloader
        self.dataloader = original_dataloader

        # Store original length
        self.original_length = len(original_dataloader)

        # Set synchronized length
        if synced_length is not None:
            if synced_length <= 0:
                raise ValueError(f"synced_length must be positive, got {synced_length}")
            self.synced_length = synced_length
        else:
            self.synced_length = self.original_length

        # For state tracking
        self.current_batch = 0
        self.session_batch_counts = defaultdict(int) # NOTE: Currently just for debugging

        # Copy useful attributes from the base dataloader
        self.batch_size = getattr(original_dataloader, 'batch_size', None)
        self.dataset = getattr(original_dataloader, 'dataset', None)
        self.shuffle = getattr(original_dataloader, 'shuffle', False)
        self.num_workers = getattr(original_dataloader, 'num_workers', 0)
        self.pin_memory = getattr(original_dataloader, 'pin_memory', False)

        logger.info(
            f"Created SyncedFastSessionDataLoader with original_length={self.original_length}, synced_length={self.synced_length}")

    def __len__(self):
        """Return the synchronized number of batches per epoch."""
        return self.synced_length

    def __getattr__(self, name):
        """Delegate attribute access to the underlying dataloader."""
        # Delegate any attributes we don't have to the base dataloader
        if hasattr(self.dataloader, name):
            return getattr(self.dataloader, name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

    def reset_state(self):
        """Reset the state of the dataloader."""
        self.current_batch = 0
        self.session_batch_counts = defaultdict(int)
        self.dataloader.reset_state()

    def get_state(self):
        """Return the current state of the dataloader."""
        # Get state from the underlying dataloader
        base_state = self.dataloader.get_state() if hasattr(self.dataloader, 'get_state') else {}

        # Add our own state tracking information
        synced_state = {
            'synced_current_batch': self.current_batch,
            'synced_length': self.synced_length,
            'original_length': self.original_length,
            'synced_session_batch_counts': self.session_batch_counts
        }

        # Merge states
        state = {**base_state, **synced_state}
        return state

    def set_state(self, state):
        """Set the state of the dataloader."""
        # Update our own state
        if 'synced_current_batch' in state:
            self.current_batch = state['synced_current_batch']
        if 'synced_session_batch_counts' in state:
            self.session_batch_counts = defaultdict(int)
            self.session_batch_counts.update(state['synced_session_batch_counts'])

        # Pass remaining state to the base dataloader
        if hasattr(self.dataloader, 'set_state'):
            # Filter out our custom state keys before passing to base dataloader
            base_state = {k: v for k, v in state.items()
                          if k not in {'synced_current_batch', 'synced_length', 'original_length'}}
            self.dataloader.set_state(base_state)

    def get_session_batch_counts(self):
        """Return the distribution of batches across sessions (if available)."""
        if hasattr(self.dataloader, 'batches_per_session'):
            return self.dataloader.batches_per_session
        return None

    def get_dataset_info(self):
        """Return information about the underlying dataset."""
        if self.dataset is not None:
            if hasattr(self.dataset, 'get_sessions_count'):
                sessions_count = self.dataset.get_sessions_count()
            else:
                # Create a simple summary if the method doesn't exist
                sessions_count = {}
                if hasattr(self.dataset, 'session_names'):
                    for session_name in self.dataset.session_names:
                        indices = self.dataset.get_indices_for_session(session_name)
                        sessions_count[session_name] = len(indices)

            return {
                'total_samples': len(self.dataset),
                'sessions_count': sessions_count
            }
        return None

    def __iter__(self):
        """
        Iterate through dataset, ensuring we produce exactly synced_length batches
        while maintaining correct session-to-batch correspondence.
        """

        # Get an iterator from the original dataloader
        dataloader_iterator = iter(self.dataloader)

        # Yield exactly synced_length batches
        while self.current_batch < self.synced_length:
            # Get batch with corresponding session key
            try:
                key, batch = next(dataloader_iterator)
            except StopIteration:
                # If the iterator is exhausted, create a new one
                logger.debug(f"Iterator exhausted at batch {self.current_batch}, restarting")
                dataloader_iterator = iter(self.dataloader)
                key, batch = next(dataloader_iterator)
            except Exception as e:
                # Log any other errors
                logger.error(f"Error at batch {self.current_batch}: {str(e)}")
                logger.error(traceback.format_exc())
                raise

            # Update counters
            self.current_batch += 1
            self.session_batch_counts[key] += 1

            # Debug logging for alignment verification
            if self.current_batch % 100 == 0 or self.current_batch == self.synced_length:
                logger.debug(
                    f"Overall: {self.current_batch}/{self.synced_length} [Session {key}] "
                    f"Loop: {self.current_batch // self.original_length + 1}: "
                    f"{self.current_batch % self.original_length} / {self.original_length}"
                )

            yield key, batch

        # Log session distribution for debugging
        if self.current_batch > 0:
            logger.info(f"Session distribution: {dict(self.session_batch_counts)}")
            logger.info(f"Total batches yielded: {self.current_batch + 1}")

        # Reset state
        self.reset_state()


def prepare_synced_dataloaders(dataloader, rank, world_size, **kwargs):
    """
    Prepares synchronized dataloaders for distributed training.

    Args:
        dataloader: FastSessionDataLoader instance
        rank: Process rank
        world_size: Total number of processes
        **kwargs: Additional arguments (ignored)

    Returns:
        SyncedFastSessionDataLoader: Synchronized dataloader for training
    """
    # First check if dataloader is None - synchronize this information
    has_dataloader = torch.tensor([1 if dataloader is not None else 0], device="cuda")
    all_has_dataloader = [torch.tensor([0], device="cuda") for _ in range(world_size)]

    try:
        # Debug information about the original dataloader
        if dataloader is not None and rank == 0:
            logger.info(f"Original dataloader: length={len(dataloader)}, "
                        f"batch_size={dataloader.batch_size if hasattr(dataloader, 'batch_size') else 'unknown'}")

        dist.all_gather(all_has_dataloader, has_dataloader)

        # Check if any rank is missing a dataloader
        if 0 in [dl.item() for dl in all_has_dataloader]:
            if rank == 0:
                logger.warning("Some ranks have no dataloader. Using dummy dataloader for synchronization.")

            # All ranks should return None or a dummy dataloader
            return None
    except Exception as e:
        logger.error(f"Error checking dataloader availability: {e}")
        if dataloader is None:
            return None

    # First synchronize all processes
    try:
        dist.barrier()
    except Exception as e:
        logger.warning(f"Barrier synchronization failed, continuing anyway: {e}")

    # Synchronize length across all ranks
    synced_length = sync_dataloader_length(dataloader, rank, world_size)

    # Create synchronized dataloader wrapper
    synced_dataloader = SyncedFastSessionDataLoader(
        original_dataloader=dataloader,
        synced_length=synced_length
    )

    # Ensure all processes have created their dataloaders before continuing
    try:
        dist.barrier()
    except Exception as e:
        logger.warning(f"Final barrier synchronization failed, continuing anyway: {e}")

    # Add debug report
    if rank == 0:
        logger.info(f"Created synced dataloader with length {synced_length}")

    return synced_dataloader


def gather_validation_metrics(val_corr, rank, world_size):
    """
    Gather validation correlation metrics from all ranks to rank 0

    Args:
        val_corr: Dictionary of {dataset_name: correlation_score} or None
        rank: Current process rank
        world_size: Total number of processes

    Returns:
        If rank 0, returns combined dictionary from all processes
        Otherwise, returns empty dict
    """
    # Handle None case
    if val_corr is None:
        local_items = []
    else:
        # Convert any tensor values to Python primitives and ensure they're on CPU
        processed_items = []
        for k, v in val_corr.items():
            with torch.no_grad():  # Ensure no gradients are tracked
                if isinstance(v, torch.Tensor):
                    # Detach, move to CPU, and convert to Python primitive
                    v = v.detach().cpu().item()
            processed_items.append((k, v))
        local_items = processed_items

    # Clear CUDA cache before communication
    torch.cuda.empty_cache()

    # Create a list to hold items from all ranks
    all_items = [None for _ in range(world_size)]

    # Gather all items to rank 0
    dist.all_gather_object(all_items, local_items)

    # Only rank 0 needs to process the gathered data
    if rank == 0:
        # Combine all dictionaries
        combined_metrics = {}
        for items in all_items:
            for k, v in items:
                combined_metrics[k] = v
        return combined_metrics
    else:
        return {}


def initialize_parameter_tracking(model):
    """
    Initialize tracking for specific parameters (with both "readout" and "query").
    Call this once at the beginning of training.

    Args:
        model: The PyTorch model

    Returns:
        Parameter tensor if found, None otherwise
    """
    target_params = []

    # Find the single parameter matching our criteria
    for name, param in model.named_parameters():
        if param.requires_grad and 'readout' in name and 'query' in name:
            target_params.append(param)

    return target_params


def compute_parameter_stats(target_params, rank):
    """
    Compute statistics for the tracked parameters.

    Args:
        target_params: List of parameter tensors to analyze
        rank: Process rank in distributed training

    Returns:
        Dictionary containing parameter statistics for this rank
    """
    stats_dict = {
        'rank': rank,
        'stats': {}
    }

    # Skip if no parameters
    if not target_params:
        return stats_dict

    # Detach and flatten all parameters into a single tensor
    flattened_params = torch.cat([param.detach().flatten() for param in target_params])
    param_count = flattened_params.numel()

    # Calculate statistics
    stats_dict['stats'] = {
        'mean': flattened_params.mean().item(),
        'std': flattened_params.std().item(),
        'min': flattened_params.min().item(),
        'max': flattened_params.max().item(),
        'l2norm': torch.norm(flattened_params, p=2).item() / math.sqrt(param_count),
        'abs_mean': torch.abs(flattened_params).mean().item()
    }

    return stats_dict


def gather_parameter_stats(local_stats, rank, world_size):
    """
    Gather parameter statistics from all ranks to rank 0.
    """
    # Convert dict to JSON string
    local_json = json.dumps(local_stats)
    local_tensor = torch.tensor(bytearray(local_json.encode('utf-8')), dtype=torch.uint8, device='cuda')

    # Get size of each rank's tensor
    size_tensor = torch.tensor([local_tensor.size(0)], dtype=torch.long, device='cuda')
    size_list = [torch.ones(1, dtype=torch.long, device='cuda') for _ in range(world_size)]

    # Gather sizes
    dist.all_gather(size_list, size_tensor)

    # Create tensors for gathering
    max_size = max(size.item() for size in size_list)

    # Pad local tensor to max_size
    if local_tensor.size(0) < max_size:
        padding = torch.zeros(max_size - local_tensor.size(0), dtype=torch.uint8, device='cuda')
        local_tensor = torch.cat((local_tensor, padding))

    # Prepare gather list
    tensor_list = [torch.zeros(max_size, dtype=torch.uint8, device='cuda') for _ in range(world_size)]

    # Gather all tensors
    dist.all_gather(tensor_list, local_tensor)

    # Only rank 0 processes the gathered data
    if rank == 0:
        all_stats = []
        for i, (tensor, size) in enumerate(zip(tensor_list, size_list)):
            json_bytes = tensor[:size.item()].cpu().numpy().tobytes()
            stats = json.loads(json_bytes.decode('utf-8'))
            all_stats.append(stats)
        return all_stats

    return None


def compute_and_gather_parameter_stats(target_params, rank=0, world_size=1):
    """
    Compute statistics for parameters and gather results from all ranks.
    This function only computes and gathers, NO logging.

    Args:
        target_params: List of parameter tensors to analyze
        rank: Process rank in distributed training
        world_size: Total number of ranks

    Returns:
        On rank 0: List of statistics dictionaries from all ranks
        On other ranks: None
    """
    # Skip if no parameters
    if not target_params:
        return None

    # Compute local stats
    local_stats = compute_parameter_stats(target_params, rank)

    # Gather stats from all ranks
    all_stats = gather_parameter_stats(local_stats, rank, world_size)

    # Only rank 0 gets the full stats
    if rank == 0:
        return all_stats

    return None


def all_gather_scalar(tensor):
    """
    Gather scalar tensors from all processes while preserving device and dtype.

    Args:
        tensor: A scalar tensor to be gathered from all processes

    Returns:
        A tensor containing values from all processes
    """
    # Ensure input is a tensor
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, device="cuda")

    # Ensure tensor is a scalar
    if tensor.numel() != 1:
        raise ValueError(f"Expected scalar tensor, got shape {tensor.shape}")

    # Create list to hold results
    world_size = dist.get_world_size()
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]

    # Gather the tensors
    dist.all_gather(gather_list, tensor)

    # Combine results while preserving dtype and device
    gathered_tensor = torch.stack(gather_list)

    return gathered_tensor


def initialize_buckets(parameters, bucket_size_bytes):
    """
    Create parameter buckets and pre-allocate buffers for all-reduce.

    Args:
        parameters: Iterable of parameters to bucket
        bucket_size_bytes: Target size for each bucket in bytes

    Returns:
        Dictionary with buckets and pre-allocated buffers
    """
    buckets = []
    current_bucket = []
    current_size = 0

    for param in parameters:
        # Skip frozen parameters
        if not param.requires_grad:
            continue

        # Calculate size in bytes
        param_size = param.numel() * param.element_size()

        # Start a new bucket if needed
        if current_size + param_size > bucket_size_bytes and current_bucket:
            buckets.append(current_bucket)
            current_bucket = []
            current_size = 0

        # Add parameter to current bucket
        current_bucket.append(param)
        current_size += param_size

    # Don't forget the last bucket
    if current_bucket:
        buckets.append(current_bucket)

    # Pre-allocate flat buffers for each bucket
    flat_buffers = []
    bucket_info = []

    for bucket in buckets:
        # Get gradient shapes and calculate total elements
        grad_shapes = [param.shape for param in bucket]
        total_elements = sum(param.numel() for param in bucket)

        # Create a flat buffer for this bucket
        device = bucket[0].device
        dtype = bucket[0].dtype
        flat_buffer = torch.zeros(total_elements, device=device, dtype=dtype)

        flat_buffers.append(flat_buffer)

        # Store metadata about this bucket
        offsets = []
        offset = 0
        for param in bucket:
            numel = param.numel()
            offsets.append((offset, offset + numel))
            offset += numel

        bucket_info.append({
            'params': bucket,
            'shapes': grad_shapes,
            'offsets': offsets
        })

    ret = {
        'bucket_info': bucket_info,
        'flat_buffers': flat_buffers,
    }

    return ret


def reduce_gradients(bucket_data):
    """
    Reduce gradients using pre-allocated buckets.

    Args:
        bucket_data: Dictionary from initialize_buckets

    Returns:
        List of handles
    """
    bucket_info = bucket_data['bucket_info']
    flat_buffers = bucket_data['flat_buffers']
    handles = []

    # Iterate over buckets (info and flat_buffer)
    for i, (info, flat_buffer) in enumerate(zip(bucket_info, flat_buffers)):
        # iterate over parameters in the bucket
        for (start, end), param in zip(info['offsets'], info['params']):
            # if param.grad doesn't exist, set it to zero
            if param.grad is None:
                param.grad = torch.zeros_like(param.data)
            # pack gradients into flat buffer
            flat_buffer[start:end].copy_(param.grad.view(-1))

        # Perform async all-reduce (We use SUM so that we can average based on "usage")
        handle = dist.all_reduce(flat_buffer, op=dist.ReduceOp.AVG, async_op=True)
        handles.append((handle, i))

    return handles


def unpack_gradients(bucket_data, handles):
    """
    Unpack reduced gradients from flat buffers back to parameters.

    Args:
        bucket_data: Dictionary from initialize_buckets
        handles: List of (handle, bucket_idx) tuples from reduce_gradients
    """
    bucket_info = bucket_data['bucket_info']
    flat_buffers = bucket_data['flat_buffers']

    # Iterate over buckets (i.e. all-reduce handles and bucket indices)
    for handle, bucket_idx in handles:
        # wait for this reduction to complete
        handle.wait()
        # get corresponding bucket info and flat buffer
        info = bucket_info[bucket_idx]
        flat_buffer = flat_buffers[bucket_idx]
        # iterate over parameters in the bucket
        for (start, end), param_shape, param in zip(
            info['offsets'], info['shapes'], info['params']
        ):
            # unpack gradients from flat buffer. NOTE: Params with no gradients were set
            #  to zero during `reduce_gradients`
            param.grad.copy_(flat_buffer[start:end].view(param_shape))


def blocking(func: Callable) -> Callable:

    """Decorator that wraps a function with dist.barrier() calls before and after"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Barrier before function execution
        if dist.is_initialized():
            dist.barrier()
        
        # Execute the original function
        result = func(*args, **kwargs)
        
        # Barrier after function execution
        if dist.is_initialized():
            dist.barrier()
        
        return result
    
    return wrapper


@blocking
def broadcast_model_shared(model: Model, src: int = 0) -> None:
    """
    Broadcast shared parameters and buffers to all ranks.
    """
    for x in chain(model.shared_parameters(), model.shared_buffers()):
        dist.broadcast(x.data, src=src)


@blocking
def broadcast_optimizer(
    optimizer: Optimizer,
    src: int = 0,
    states_match: bool = False,
    device: Union[torch.device, "cpu"] = "cpu",
) -> None:
    """Sync optimizer state across all ranks"""
    # If states may not match (i.e. src optimizer loaded from checkpoint, while others,
    #  haven't yet been stepped), we simply broadcast the optimizer state dict and load it
    if not states_match:
        optimizer_state_dict = None
        if dist.get_rank() == src:
            # Get state dict and move ALL tensors to CPU
            orig_state_dict = optimizer.state_dict()
            # Create CPU version of state dict
            optimizer_state_dict = {
                'state': {},
                'param_groups': orig_state_dict['param_groups']
            }
            # Move all state tensors to CPU before broadcasting
            for param_id, param_state in orig_state_dict['state'].items():
                optimizer_state_dict['state'][param_id] = {}
                for key, value in param_state.items():
                    if isinstance(value, torch.Tensor):
                        # Move to CPU to avoid device issues during broadcast
                        optimizer_state_dict['state'][param_id][key] = value.cpu()
                    else:
                        optimizer_state_dict['state'][param_id][key] = value

        # Now broadcast - all tensors are on CPU
        state_dict_list = [optimizer_state_dict]
        dist.broadcast_object_list(state_dict_list, src=src)

        # After broadcast, all ranks have CPU tensors
        optimizer_state_dict = state_dict_list[0]

        # Update state dict with correct device
        for param_state in optimizer_state_dict['state'].values():
            for key, value in param_state.items():
                if isinstance(value, torch.Tensor):
                    param_state[key] = value.to(device)

        optimizer.load_state_dict(optimizer_state_dict)
        # NOTE: This is a HACK to ensure that the step tensor is on CPU. Otherwise it stays
        #  on original device from checkpoint, i.e. `torch.device('cuda', index=src)`
        for k, v in optimizer.state.items():
            if 'step' in v:
                v['step'] = v['step'].cpu()
        return

    # If all the states match, we can just broadcast the tensors directly
    for group in optimizer.param_groups:
        for param in group['params']:
            param_state = optimizer.state.get(param, {})
            for key, value in param_state.items():
                if isinstance(value, Tensor):
                    # NOTE: `step` is stored on CPU, so we check here and skip syncing
                    if value.device.type == 'cpu':
                        continue
                    dist.broadcast(value, src=src)


@blocking
def broadcast_scheduler(
    scheduler: LRScheduler,
    src: int = 0,
) -> None:
    """Sync scheduler state across all ranks"""
    scheduler_state_dict = scheduler.state_dict() if dist.get_rank() == src else None
    state_dict_list = [scheduler_state_dict]
    dist.broadcast_object_list(state_dict_list, src=src)
    scheduler_state_dict = state_dict_list[0]
    scheduler.load_state_dict(scheduler_state_dict)


@blocking
def calc_divergence(
    model: Model,
    optimizer: Optional[torch.optim.Optimizer] = None,
    threshold: float = 1e-8,
    mode: str = 'max',
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, float]:
    """
    Check parameter and optimizer state divergence across ranks in DDP training.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer (optional)
        divergence_threshold: Threshold to consider a parameter/state as diverged
        mode: 'max' for element-wise max absolute difference, 'l2' for L2 norm
    
    Returns:
        Dictionary with divergence statistics for each rank
    """
    
    if world_size == 1:
        return {}

    result = {}
    
    # Collect all parameters (including frozen ones)
    param_list = []
    param_names = []
    for name, param in chain(
        model.shared_named_parameters(),
        model.shared_named_buffers()
    ):
        param_list.append(param.data.clone().contiguous())
        param_names.append(name)
    
    # Collect optimizer states
    opt_state_dict = {}
    opt_state_names = {}
    
    # Get optimizer state keys from the first parameter that has state
    state_keys = []
    if optimizer is not None:
        for param in optimizer.state.values():
            if param:
                state_keys = list(param.keys())
                break
    
    # Organize optimizer states by type
    for state_key in state_keys:
        opt_state_dict[state_key] = []
        opt_state_names[state_key] = []
        
        for param_name, param in chain(
            model.shared_named_parameters(),
            model.shared_named_buffers()
        ):
            if param in optimizer.state:
                state = optimizer.state[param]
                if state_key in state:
                    state_tensor = state[state_key]
                    # Skip CPU tensors (like 'step') when model is on GPU
                    # This avoids NCCL errors and is correct since CPU states are intentionally on CPU
                    if param.is_cuda and state_tensor.is_cpu:
                        continue
                    # Keep tensor on its original device and make contiguous
                    state_tensor = state_tensor.clone().contiguous()
                    opt_state_dict[state_key].append(state_tensor)
                    opt_state_names[state_key].append(f"{param_name}_{state_key}")
    
    # Function to calculate divergence
    def calc_divergence(tensor1, tensor2, mode='max'):
        diff = (tensor1 - tensor2).abs()
        if mode == 'max':
            return diff.max().item()
        elif mode == 'l2':
            return diff.norm(2).item()
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    # Gather all parameters and states to rank 0
    if rank == 0:
        # Parameters
        all_params = [[] for _ in range(world_size)]
        for param in param_list:
            # Create gather list on the same device as param
            gathered = [torch.zeros_like(param, device=param.device) for _ in range(world_size)]
            dist.gather(param, gathered, dst=0)
            for r in range(world_size):
                all_params[r].append(gathered[r])
        
        # Optimizer states
        all_opt_states = {}
        for state_key in state_keys:
            all_opt_states[state_key] = [[] for _ in range(world_size)]
            # Only gather if we have tensors for this state
            if opt_state_dict[state_key]:
                for state_tensor in opt_state_dict[state_key]:
                    # Create gather list on the same device as state_tensor
                    gathered = [torch.zeros_like(state_tensor, device=state_tensor.device) 
                               for _ in range(world_size)]
                    dist.gather(state_tensor, gathered, dst=0)
                    for r in range(world_size):
                        all_opt_states[state_key][r].append(gathered[r])
        
        # Calculate divergence for each rank compared to rank 0
        for r in range(1, world_size):
            # Parameter divergence
            n_params_diverged = 0
            max_param_div = 0.0
            sum_param_div = 0.0
            max_param_div_relative = 0.0
            sum_param_div_relative = 0.0
            n_params = len(param_list)
            
            for p_idx, (p0, pr) in enumerate(zip(all_params[0], all_params[r])):
                div = calc_divergence(p0, pr, mode)
                
                if div > threshold:
                    n_params_diverged += 1
                
                max_param_div = max(max_param_div, div)
                sum_param_div += div
                
                # Relative divergence
                p0_max = p0.abs().max().item()
                if p0_max > 0:
                    rel_div = div / p0_max
                    max_param_div_relative = max(max_param_div_relative, rel_div)
                    sum_param_div_relative += rel_div
            
            avg_param_div = sum_param_div / n_params if n_params > 0 else 0.0
            avg_param_div_relative = sum_param_div_relative / n_params if n_params > 0 else 0.0
            
            result[f'n_parameters_diverged_rank_{r}'] = n_params_diverged
            result[f'n_parameters_total_rank_{r}'] = n_params
            result[f'max_param_divergence_rank_{r}'] = max_param_div
            result[f'relative_max_param_divergence_rank_{r}'] = max_param_div_relative
            result[f'avg_param_divergence_rank_{r}'] = avg_param_div
            result[f'relative_avg_param_divergence_rank_{r}'] = avg_param_div_relative
            
            # Optimizer state divergence
            for state_key in state_keys:
                n_states_diverged = 0
                max_state_div = 0.0
                sum_state_div = 0.0
                max_state_div_relative = 0.0
                sum_state_div_relative = 0.0
                n_states = len(opt_state_dict[state_key])
                
                if n_states > 0:
                    for s_idx, (s0, sr) in enumerate(zip(all_opt_states[state_key][0], 
                                                         all_opt_states[state_key][r])):
                        div = calc_divergence(s0, sr, mode)
                        
                        if div > threshold:
                            n_states_diverged += 1
                        
                        max_state_div = max(max_state_div, div)
                        sum_state_div += div
                        
                        # Relative divergence
                        s0_max = s0.abs().max().item()
                        if s0_max > 0:
                            rel_div = div / s0_max
                            max_state_div_relative = max(max_state_div_relative, rel_div)
                            sum_state_div_relative += rel_div
                    
                    avg_state_div = sum_state_div / n_states
                    avg_state_div_relative = sum_state_div_relative / n_states
                    
                    result[f'n_{state_key}_diverged_rank_{r}'] = n_states_diverged
                    result[f'n_{state_key}_total_rank_{r}'] = n_states
                    result[f'max_{state_key}_divergence_rank_{r}'] = max_state_div
                    result[f'relative_max_{state_key}_divergence_rank_{r}'] = max_state_div_relative
                    result[f'avg_{state_key}_divergence_rank_{r}'] = avg_state_div
                    result[f'relative_avg_{state_key}_divergence_rank_{r}'] = avg_state_div_relative
                # else: skip this state entirely if no tensors to check
    
    else:
        # Non-zero ranks just send their data
        for param in param_list:
            dist.gather(param, dst=0)
        
        for state_key in state_keys:
            # Only gather if we have tensors for this state
            if opt_state_dict[state_key]:
                for state_tensor in opt_state_dict[state_key]:
                    dist.gather(state_tensor, dst=0)
    
    # Only rank 0 returns results, other ranks return empty dict
    return result if rank == 0 else {}


# Global dictionary to store cache key to task ID mapping
_GLOBAL_TASK_ID_MAPPING = {}
_NEXT_TASK_ID = 0

def collect_task_ids(cache_key: Hashable, rank: int = 0, world_size: int = 1) -> Dict[str, int]:
    """
    Collect task IDs (cache keys) from all ranks and assign global integer IDs.
    
    This function maintains a global mapping of cache keys to integer IDs across
    multiple calls. Each unique cache key gets assigned a unique integer ID that
    persists for the duration of the program.
    
    Args:
        cache_key: The cache key from the current forward pass
        rank: Current process rank
        world_size: Total number of processes
        
    Returns:
        On rank 0: Dictionary with keys 'task_id_rank_{i}' mapping to integer task IDs
        On other ranks: Empty dictionary
    """
    global _GLOBAL_TASK_ID_MAPPING, _NEXT_TASK_ID
    
    # Gather all cache keys to rank 0
    all_cache_keys = [None for _ in range(world_size)]
    dist.all_gather_object(all_cache_keys, cache_key)

    result = {}
    
    if rank == 0:
        # On rank 0, assign IDs to any new cache keys
        for i, key in enumerate(all_cache_keys):
            # Remove session key from cache key
            _key = key[1:]
            # Assign ID if this is a new cache key
            if _key not in _GLOBAL_TASK_ID_MAPPING:
                _GLOBAL_TASK_ID_MAPPING[_key] = _NEXT_TASK_ID
                _NEXT_TASK_ID += 1
                logger.debug(f"Assigned new task ID {_GLOBAL_TASK_ID_MAPPING[_key]} to cache key: {_key}")
            # Add to result dictionary
            result[f'task_id_rank_{i}'] = _GLOBAL_TASK_ID_MAPPING[_key]
        # Log if all ranks are doing the same task
        result['same_task_id'] = int(len(set(result.values())) == 1)
        
    return result
