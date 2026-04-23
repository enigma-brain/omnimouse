from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, ItemsView, KeysView, ValuesView

DEFAULT_N_NEURONS = 9000

@dataclass
class SessionMetadata:
    """Metadata for a recording session.
    
    Attributes:
        n_neurons: Number of neurons recorded
        animal_id: Unique identifier for the subject/animal
        on_rank: Whether the session is on the current rank
    
    TODO:
        - Add other metadata attributes...
            subject_id: str: Unique identifier for the subject/animal
            date: str: Recording date
            recording_duration: float: Duration of recording in seconds
            sampling_rate: float: Sampling rate in Hz
            behavioral_features: List[str]: List of behavioral feature names
    """
    n_neurons: int
    animal_id: Optional[str] = None
    on_rank: Optional[bool] = True

    def update(self, data: Dict[str, Any] | "SessionMetadata") -> None:
        """Update metadata with new values.
        
        Args:
            data: Dictionary containing new values to update
        """
        # TODO: This is a hack to check for SessionMetadata object
        if not isinstance(data, Dict):
            data = data.to_dict()
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown metadata key: {key}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert SessionMetadata to dictionary format."""
        return {
            "n_neurons": self.n_neurons,
            "animal_id": self.animal_id,
            "on_rank": self.on_rank,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMetadata":
        """Create SessionMetadata from dictionary format."""
        return cls(
            n_neurons=data["n_neurons"],
            animal_id=data.get("animal_id", None),
            on_rank=data.get("on_rank", True),
        )

@dataclass
class SessionMap:
    """Maps session keys to their corresponding metadata.
    
    Attributes:
        sessions: Dictionary mapping session keys to their metadata
    """
    sessions: Dict[str, SessionMetadata] = field(default_factory=dict)

    def __getitem__(self, session_key: str) -> SessionMetadata:
        """Get session metadata for a given session key.
        
        Args:
            session_key: Key of the session to retrieve
            
        Returns:
            SessionMetadata for the requested session
            
        Raises:
            KeyError: If session_key does not exist
        """
        return self.sessions[session_key]

    def __setitem__(self, session_key: str, metadata: SessionMetadata) -> None:
        """Set session metadata for a given session key.
        
        Args:
            session_key: Key of the session to set
            metadata: SessionMetadata to set for the session
        """
        self.sessions[session_key] = metadata

    def items(self) -> ItemsView[str, SessionMetadata]:
        """Return a view of the session key-metadata pairs.
        
        Returns:
            A view object that displays a list of (session_key, metadata) pairs
        """
        return self.sessions.items()

    def keys(self) -> KeysView[str]:
        """Return a view of the session keys.
        
        Returns:
            A view object that displays a list of session keys
        """
        return self.sessions.keys()

    def values(self) -> ValuesView[SessionMetadata]:
        """Return a view of the session metadata.
        
        Returns:
            A view object that displays a list of session metadata
        """
        return self.sessions.values()

    @property
    def session_keys(self) -> List[str]:
        """List of session keys."""
        return list(self.sessions.keys())

    def _map_session_to(self, key: str, default=None) -> Dict[str, Any]:
        """Helper to map session keys to a specific attribute.
        
        Args:
            key: The key to extract from each session's metadata
            default: Default value to use if key is not present in session metadata
            
        Returns:
            Dictionary mapping session keys to the requested attribute
        """
        return {
            k: getattr(v, key, default)
            for k, v in self.sessions.items()
        }

    @property
    def n_neurons_map(self) -> Dict[str, int]:
        """Map from session keys to neural population sizes."""
        return self._map_session_to("n_neurons", DEFAULT_N_NEURONS)

    @property
    def animal_id_map(self) -> Dict[str, str]:
        """Map from session keys to animal IDs."""
        return self._map_session_to("animal_id")

    @property
    def on_rank_map(self) -> Dict[str, bool]:
        """Map from session keys to whether the session is on the current rank."""
        return self._map_session_to("on_rank")

    @property
    def max_n_neurons(self) -> int:
        """Maximum number of neurons across all sessions."""
        return max(self.n_neurons_map.values())

    @property
    def total_n_neurons(self) -> int:
        """Total number of neurons across all sessions."""
        return sum(self.n_neurons_map.values())

    def update(self, session_key: str, data: Dict[str, Any] | SessionMetadata) -> None:
        """Update metadata for a specific session.
        
        Args:
            session_key: Key of the session to update
            data: Dictionary containing new values to update
            
        Raises:
            KeyError: If session_key does not exist
        """
        if session_key not in self.sessions:
            if isinstance(data, Dict):
                data = SessionMetadata.from_dict(data)
            self.sessions[session_key] = data
        self.sessions[session_key].update(data)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert SessionMap to dictionary format for backward compatibility.
        
        Returns:
            Dictionary with the same structure as the original session_map
        """
        return {
            session_key: session_metadata.to_dict()
            for session_key, session_metadata in self.sessions.items()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, Any]]) -> "SessionMap":
        """Create SessionMap from dictionary format.
        
        Args:
            data: Dictionary with the original session_map structure
            
        Returns:
            SessionMap object
        """
        sessions = {}
        for session_key, session_data in data.items():
            sessions[session_key] = SessionMetadata.from_dict(session_data)
        return cls(sessions=sessions) 