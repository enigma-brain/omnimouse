from typing import Optional, Tuple
from dataclasses import dataclass, replace

@dataclass
class MaskingStrategy:
    """
    Configuration class for masking patterns across neural responses, video, and behavior.
     We consider the activity/stimuli/behavior tensors for a context-window "block", and use
     the masking strategy to define the boundaries on visible / hidden regions. Visible 
     regions will be passed to the model. "reconstructed" regions serve as targets.

    - Responses:

    ```
                                num_samples_per_block
                                <---------------------------------------------------------------> 
                                        ┌> response_context_start_idx
                                        : max_response_context_samples
                                        :<---------------------------------------->:
                                        :   ┌> prefix_start_idx                    :
                                        :   : prefix_len                           :
                                        :   :<----------------------------->:      : 
                                        :   : full_population_prefix_len    :      :   
                                        :   :<---------->                   :      :
               A            A  ┌────────────┬──────────────────────-─–─–––––┬─────────–───────────┐
               │  n_visible │  │        :   │ ########## : ################ │      :              │
               │  _neurons  │  │        :   │ ########## : #    prefix    # │      :              │
               │            V  │        :   │ # full_  # : ################ │      :              │
 neural_       │               │        :   │ # popula # ┌──────────┬───-––─┴—————————————————————┼
population_    │            A  │        :   │ #  tion_ # │          │ ########################### │
  size         │            │  │        :   │ # prefix # │          │ ########################### │ 
               │  n_max_rec │  │        :   │ ########## │          │ #          suffix         # │
               │  onstructe │  │        :   │ ########## │          │ ########################### │
               │  d_neurons V  │        :   | ########## │          | ########################### │
               V               └────────────┴────────────┴──────────┴───––––––-─––────────────────┘
                                                                    : <--------------------------->
                                                                    :          suffix_len
                                                                    └> suffix_start_idx
    ```
    - Video:                              
    ```
                                                        ┌> n_visible_frames_start_idx
                                                        : n_visible_frames
                                                        :<----------------------->:
                               ┌────────────────────────┬──────────-─–─––––───────┬──–────────────┐
                               │                        │#########################│               │
                               └────────────────────────┴─────────────────––––––-─┴–───────–──────┘
    ```
    - Behavior:
    ```
                               behavior_encoded (True/False)     behavior_reconstructed (True/False)
                                      ┌──┐                                ┌──┐
                                      └──┘                                └──┘
    ```

    Key Parameters:
    - n_visible_neurons: Number of neurons visible to encoder (population masking)
    - prefix_start_idx: Start position of visible prefix samples  
    - prefix_len: Length of visible prefix samples (causal masking)
    - full_population_prefix_len: Portion of prefix for which all neurons are visible,
        irrespective of `n_visible_neurons`.
    - response_context_start_idx: Start position of "response context window". Response
        context window is used to determine which samples are available for encoding, and
        therefore which "latents" must be made available. The prefix region must be a subset
        of the response context window. See :mod:`omnimouse.modeling.model` for more details.
    - max_response_context_samples: Maximum samples in response context
    - max_n_reconstructed_neurons: Upper bound on number of neurons for reconstruction.
    - suffix_start_idx: Start position of reconstruction period
    - suffix_len: Length of reconstruction period
    - n_visible_frames_start_idx: Start position of visible video frames
    - n_visible_frames: Number of visible video frames
    - behavior_encoded/behavior_reconstructed: Behavior masking controls
    - weight: Sampling probability weight

    Common Configurations:
    - Population masking: n_visible_neurons < total, prefix_len = total
    - Causal masking: n_visible_neurons = total, prefix_len < total  
    - Two-stage causal: full_population_prefix_len > 0
    - With reconstruction: max_n_reconstructed_neurons > 0

    Video masking uses causal approach starting from n_visible_frames_start_idx.
    Behavior masking is binary: either encoded or reconstructed, not both.

    Examples:
    ```python
    # Basic population masking
    MaskingStrategy(n_visible_neurons=512)
    
    # Causal masking with custom start
    MaskingStrategy(prefix_start_idx=10, prefix_len=20)
    
    # Two-stage causal with reconstruction
    MaskingStrategy(
        n_visible_neurons=256,
        prefix_len=16, 
        full_population_prefix_len=8,
        max_n_reconstructed_neurons=128
    )
    ```
    """
    n_visible_neurons: Optional[int] = None
    prefix_start_idx: Optional[int] = None
    prefix_len: Optional[int] = None
    full_population_prefix_len: Optional[int] = None
    response_context_start_idx: Optional[int] = None
    max_response_context_samples: Optional[int] = None
    max_n_reconstructed_neurons: Optional[int] = None
    suffix_start_idx: Optional[int] = None
    suffix_len: Optional[int] = None
    n_visible_frames_start_idx: Optional[int] = None
    n_visible_frames: Optional[int] = None
    behavior_encoded: Optional[bool] = True
    behavior_reconstructed: Optional[bool] = False
    weight: float = 1.0
    
    def __post_init__(self):
        """Validate the joint masking strategy parameters."""
        # Ensure weight is positive
        if self.weight <= 0:
            raise ValueError(f"Weight must be positive, got {self.weight}")
        
        # If prefix_len is None, set it to max_response_context_samples
        if self.prefix_len is None:
            self.prefix_len = self.max_response_context_samples
        
        # Check that `n_visible_neurons` and `prefix_len` are consistent (i.e. we don't
        #  configure a prefix length without visible neurons)
        all_hidden = [x == 0 for x in (self.n_visible_neurons, self.prefix_len)]
        if any(all_hidden) and not all(all_hidden):
            raise ValueError("Cannot have hidden samples with visible neurons or vice versa.")
        
        # Set full_population_prefix_len to 0 if None or if n_visible_neurons is None
        #  (i.e. if the prefix is already full population)
        if self.full_population_prefix_len is None or self.n_visible_neurons is None:
            self.full_population_prefix_len = 0
        
        # Check that full_population_prefix_len < prefix_len when prefix_len is not None
        if (
            self.prefix_len is not None and
            self.full_population_prefix_len > 0 and
            self.full_population_prefix_len > self.prefix_len
        ):
            raise ValueError(
                f"full_population_prefix_len ({self.full_population_prefix_len}) must be <= "
                f"than prefix_len ({self.prefix_len})"
            )
        
        # If response_context_start_idx is None, set it to 0 (i.e. start of window)
        if self.response_context_start_idx is None:
            self.response_context_start_idx = 0
        
        # If prefix_start_idx is None, set it to 0 (i.e. start of window)
        if self.prefix_start_idx is None:
            self.prefix_start_idx = self.response_context_start_idx
        
        if self.prefix_len is not None and self.max_response_context_samples is not None:
            # Check that prefix samples are within the response temporal context
            prefix_ = self.prefix_start_idx # prefix start
            _prefix = prefix_ + self.prefix_len # prefix end
            context_ = self.response_context_start_idx # context start
            _context = context_ + self.max_response_context_samples # context end
            if prefix_ < context_ or _prefix > _context:
                raise ValueError(
                    f"response temporal context ({context_} - {_context}) must contain "
                    f"prefix samples ({prefix_} - {_prefix})"
                )
        
        # TODO: Validate response decoding parameters!
        
        # Ensure that behavior is not encoded and reconstructed at the same time
        if self.behavior_encoded and self.behavior_reconstructed:
            raise ValueError("Cannot have both behavior encoded and reconstructed!")
    
    def instantiate(
        self,
        neural_population_size: int,
        num_samples_per_block: int,
        num_frames_per_block: int,
    ) -> 'MaskingStrategy':
        """
        Create a new MaskingStrategy instance with concrete values based on data dimensions.
        
        This method returns a new instance with None values replaced by appropriate defaults
        based on the provided data dimensions. The original instance is not modified.
        
        Args:
            neural_population_size: Total number of neurons in the population
            num_samples_per_block: Total number of response samples in the block
            num_frames_per_block: Total number of video frames in the block
            
        Returns:
            New MaskingStrategy instance with concrete values
        """
        # If n_visible_neurons is None, set it to the total number of neurons
        n_visible_neurons = (
            self.n_visible_neurons if self.n_visible_neurons is not None 
            else neural_population_size
        )
        n_visible_neurons = min(n_visible_neurons, neural_population_size)
        # If response_context_start_idx is None, set it to 0
        response_context_start_idx = (
            self.response_context_start_idx if self.response_context_start_idx is not None
            else 0
        )
        response_context_start_idx = max(
            0, min(response_context_start_idx, num_samples_per_block)
        )
        # If max_response_context_samples is None, set it to the total remaining samples
        #  after response context start index
        max_response_context_samples = (
            self.max_response_context_samples if self.max_response_context_samples is not None
            else num_samples_per_block - response_context_start_idx
        )
        max_response_context_samples = min(
            max_response_context_samples, num_samples_per_block - response_context_start_idx
        )
        # Calculate the end index of the response context
        response_context_end_idx = response_context_start_idx + max_response_context_samples
        # If prefix_start_idx is None, set it to start of response context
        prefix_start_idx = (
            self.prefix_start_idx if self.prefix_start_idx is not None
            else response_context_start_idx
        )
        prefix_start_idx = max(
            response_context_start_idx, min(prefix_start_idx, response_context_end_idx)
        )
        # If prefix_len is None, set it to the remaining input context length after prefix
        #  start index
        prefix_len = (
            self.prefix_len if self.prefix_len is not None 
            else response_context_end_idx - prefix_start_idx
        )
        prefix_len = min(prefix_len, response_context_end_idx - prefix_start_idx)
        # If full_population_prefix_len is None, set it to 0
        full_population_prefix_len = (
            self.full_population_prefix_len if self.full_population_prefix_len is not None
            else 0
        )
        full_population_prefix_len = min(full_population_prefix_len, prefix_len)
        # If max_n_reconstructed_neurons is None, set it to the total number of neurons minus the number of visible neurons
        max_n_reconstructed_neurons = self.max_n_reconstructed_neurons
        if max_n_reconstructed_neurons is None:
            max_n_reconstructed_neurons = neural_population_size
        max_n_reconstructed_neurons = min(max_n_reconstructed_neurons, neural_population_size)
        # If suffix_start_idx is None, set it to start after the encoded prefix
        suffix_start_idx = (
            self.suffix_start_idx if self.suffix_start_idx is not None
            else full_population_prefix_len if n_visible_neurons < neural_population_size
            else prefix_len
        )
        suffix_start_idx = max(0, min(suffix_start_idx, num_samples_per_block))
        # If suffix_len is None, set it to the total number of samples minus the suffix start index
        suffix_len = (
            self.suffix_len if self.suffix_len is not None
            else num_samples_per_block - suffix_start_idx
        )
        suffix_len = min(suffix_len, num_samples_per_block - suffix_start_idx)
        # If n_visible_frames_start_idx is None, set it to 0
        n_visible_frames_start_idx = (
            self.n_visible_frames_start_idx if self.n_visible_frames_start_idx is not None
            else 0
        )
        n_visible_frames_start_idx = max(
            0, min(n_visible_frames_start_idx, num_frames_per_block)
        )
        # If n_visible_frames is None, set it to the remaining number of frames after
        #  n_visible_frames_start_idx
        n_visible_frames = (
            self.n_visible_frames if self.n_visible_frames is not None
            else num_frames_per_block - n_visible_frames_start_idx
        )
        n_visible_frames = min(n_visible_frames, num_frames_per_block - n_visible_frames_start_idx)
        # Create new instance with concrete values
        return replace(
            self,
            n_visible_neurons=n_visible_neurons,
            prefix_start_idx=prefix_start_idx,
            prefix_len=prefix_len,
            full_population_prefix_len=full_population_prefix_len,
            response_context_start_idx=response_context_start_idx,
            max_response_context_samples=max_response_context_samples,
            max_n_reconstructed_neurons=max_n_reconstructed_neurons,
            suffix_start_idx=suffix_start_idx,
            suffix_len=suffix_len,
            n_visible_frames_start_idx=n_visible_frames_start_idx,
            n_visible_frames=n_visible_frames,
        )
    
    def parameters_tuple(self) -> Tuple[
        int, int, int, int, int, int, int, int, int, int, int, bool, bool
    ]:
        """
        Convert the masking strategy parameters to a tuple.
        
        This is useful for creating cache keys or other scenarios where a 
        hashable representation is needed.
        
        Returns:
            Tuple containing all masking parameters in order
        """
        return (
            self.n_visible_neurons,
            self.prefix_start_idx,
            self.prefix_len,
            self.full_population_prefix_len,
            self.response_context_start_idx,
            self.max_response_context_samples,
            self.max_n_reconstructed_neurons,
            self.suffix_start_idx,
            self.suffix_len,
            self.n_visible_frames_start_idx,
            self.n_visible_frames,
            self.behavior_encoded,
            self.behavior_reconstructed,
        )