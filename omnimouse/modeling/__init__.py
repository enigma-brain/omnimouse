from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Hashable, Iterable, List, Optional, Tuple, Type

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from omnimouse.utils.types import SessionMap


@dataclass
class ModelArgs:
    """
    Base class for model configuration arguments.
    """

    model_type: ClassVar[str] = field(init=False, default=None)
    # Optional interpretable tags, used only by top-level
    #  entrypoint for logging purposes
    model_tags: Optional[List[str]] = field(default_factory=list)

    def __post_init__(self):
        if self.model_type not in MODEL_REGISTRY:
            raise ValueError(
                f"[{self.__class__.__name__}] Unknown model "
                f"type: {self.model_type}. Options: ["
                f"{', '.join(MODEL_REGISTRY.keys())}]"
            )


class Model(nn.Module, ABC):
    """
    Base class for all models.

    Args:
        config: Model configuration arguments
        session_map: Structured mapping of session keys to their metadata.
    """

    def __init__(
        self,
        config: ModelArgs,
        session_map: SessionMap,
    ):
        super().__init__()
        self._config = config
        self._session_map = session_map

    @property
    def c(self) -> ModelArgs:
        """Alias for :attr:`_config`."""
        return self._config

    @property
    def session_map(self) -> SessionMap:
        """Alias for :attr:`_session_map`."""
        return self._session_map

    @abstractmethod
    def shared_named_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        """Named parameters which must be shared across all ranks"""
        pass

    def shared_parameters(self) -> Iterable[nn.Parameter]:
        """Parameters which must be shared across all ranks"""
        for _, p in self.shared_named_parameters():
            yield p

    @abstractmethod
    def shared_named_buffers(self) -> Iterable[Tuple[str, Tensor]]:
        """Named buffers which must be shared across all ranks"""
        pass

    def shared_buffers(self) -> Iterable[Tensor]:
        """Registered buffers which must be shared across all ranks"""
        for _, b in self.shared_named_buffers():
            yield b

    @abstractmethod
    def rank_named_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        """Named parameters which are unique to each rank"""
        pass

    def rank_parameters(self) -> Iterable[nn.Parameter]:
        """Parameters which are unique to each rank"""
        for _, p in self.rank_named_parameters():
            yield p

    @abstractmethod
    def rank_named_buffers(self) -> Iterable[Tuple[str, Tensor]]:
        """Named buffers which are unique to each rank"""
        pass

    def rank_buffers(self) -> Iterable[Tensor]:
        """Registered buffers which are unique to each rank"""
        for _, b in self.rank_named_buffers():
            yield b

    @abstractmethod
    def core_named_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        """Named parameters which should be frozen for fine-tuning"""
        pass

    def core_parameters(self) -> Iterable[nn.Parameter]:
        """Parameters which should be frozen for fine-tuning"""
        for _, p in self.core_named_parameters():
            yield p

    @abstractmethod
    def forward_from_batch(
        self,
        batch: Tuple[str, Dict[str, torch.Tensor | Dict[str, torch.Tensor]]],
        **kwargs: Any,
    ) -> "OMModelOutput":
        """Process a batch of data and run it through the model.

        Args:
            batch: Tuple containing:
                - session_key (str): Identifier for the session
                - batch_data (Dict): Dictionary containing model-specific data tensors
            **kwargs: Additional arguments specific to the model implementation

        Returns:
            OMModelOutput: Model output containing predictions and losses
        """
        pass


MODEL_REGISTRY: Dict[str, Type[Model]] = {}


def register_model(name: str):
    """Decorator to register a model class."""

    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_model(
    args: ModelArgs,
    session_map: SessionMap,
    **data_kwargs: Any,
) -> Model:
    """Build a model from the given arguments."""
    model_class = MODEL_REGISTRY[args.model_type]
    return model_class(args, session_map, **data_kwargs)


@dataclass
class OMModelOutput:
    """OmniMouse-modeling model output."""

    session_key: str
    cache_key: Hashable
    loss: Float[torch.Tensor, ""]
    # Response reconstruction
    response_loss: Float[torch.Tensor, ""]
    response_preds: Float[torch.Tensor, "batch n_reconstructed_samples"]
    response_labels: Float[torch.Tensor, "batch n_reconstructed_samples"]
    response_timestamps: Float[torch.Tensor, "batch n_reconstructed_samples"]
    response_positions: Int[torch.Tensor, "batch n_reconstructed_samples"]
    response_neuron_ids: Int[torch.Tensor, "batch n_reconstructed_samples"]

    # Behavior reconstruction
    behavior_loss: Float[torch.Tensor, ""]
    behavior_preds: Float[torch.Tensor, "batch n_behavior_channels n_behavior_samples"]
    behavior_labels: Float[torch.Tensor, "batch n_behavior_channels n_behavior_samples"]
    behavior_timestamps: Float[
        torch.Tensor, "batch n_behavior_channels n_behavior_samples"
    ]

    # NOTE: Above required attributes can be provided as `None`, but the below attributes
    #  aren't required
    response_mask: Optional[Bool[torch.Tensor, "batch n_reconstructed_samples"]] = None
    response_valid_samples: Optional[
        Bool[torch.Tensor, "batch n_reconstructed_samples"]
    ] = None

    def __post_init__(self):
        if self.response_preds is not None:
            if self.response_preds.grad_fn is not None:
                self.response_preds = self.response_preds.detach()
        if self.behavior_preds is not None:
            if self.behavior_preds.grad_fn is not None:
                self.behavior_preds = self.behavior_preds.detach()

    @property
    def device(self) -> torch.device:
        """Device of the model output."""
        return self.response_preds.device

    def reconstruct_responses(
        self,
        responses: Float[torch.Tensor, "batch n_neurons n_samples"],
        reconstruct_masked_only: bool = True,
    ) -> Float[torch.Tensor, "batch n_neurons n_samples"]:
        """
        Fill in response tensor with reconstruction predictions at a subset of sample indices.
        """
        # Validate required attributes
        if None in (
            self.response_preds,
            self.response_neuron_ids,
            self.response_positions,
        ):
            raise ValueError("Missing required attributes for response reconstruction.")
        # Get required tensors
        batch, _, _ = responses.shape
        batch_range = torch.arange(batch, device=self.device)[..., None]
        nids, pos, preds, mask = (
            self.response_neuron_ids,
            self.response_positions,
            self.response_preds,
            self.response_mask,
        )
        # Select only valid samples (if specified)
        if self.response_valid_samples is not None:
            nids, pos, preds = (
                nids[self.response_valid_samples],
                pos[self.response_valid_samples],
                preds[self.response_valid_samples],
            )
            if mask is not None:
                mask = mask[self.response_valid_samples]
        # Select only masked samples (if reconstructing masked-only responses)
        if reconstruct_masked_only and mask is not None:
            mask_flipped = mask.logical_not()
            if not mask_flipped.all():
                nids, pos, preds = (
                    nids[mask_flipped],
                    pos[mask_flipped],
                    preds[mask_flipped],
                )
        # Fill in reconstructed responses
        responses[batch_range, nids, pos] = preds
        # Return reconstructed responses
        return responses
