"""
State Manager - Multi-Object, Multi-Light State Tracking

Manages independent state estimators for each (track_id, light_type) pair.
Provides clean API for pipeline integration with minimal coupling.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from .light_state_estimator import (
    LightStateEstimator,
    StateEstimatorConfig,
    StateEstimate,
    LightState
)


@dataclass
class DetectionInput:
    """Input detection from tracking pipeline"""
    track_id: int
    light_type: str  # e.g., "left_indicator", "brake_light"
    is_active: bool  # Binary activation
    confidence: float  # Detection confidence (not used in FSM)
    timestamp: float


class StateManager:
    """
    Manages state estimation for multiple tracked objects.
    
    Responsibilities:
    - Create/destroy state estimators per (track_id, light_type)
    - Route detections to appropriate estimators
    - Handle track lifecycle (appearance/disappearance)
    - Provide unified interface to pipeline
    
    Design:
    - Lazy initialization of estimators
    - Thread-safe operations (stateless from pipeline perspective)
    - Scalable to multi-vehicle scenarios
    """
    
    def __init__(self, config: StateEstimatorConfig, fps: float = 30.0):
        self.config = config
        self.fps = fps
        
        # Dictionary of state estimators: {(track_id, light_type): estimator}
        self.estimators: Dict[Tuple[int, str], LightStateEstimator] = {}
        
        # Track last update time for cleanup
        self.last_update: Dict[Tuple[int, str], float] = {}
        
        # Cleanup threshold (remove stale tracks)
        self.stale_threshold = 5.0  # seconds
    
    def update(self, detection: DetectionInput) -> StateEstimate:
        """
        Update state for a tracked object.
        
        Args:
            detection: Detection input with track_id, light_type, activation
            
        Returns:
            StateEstimate for this object
        """
        key = (detection.track_id, detection.light_type)
        
        # Lazy initialization of estimator
        if key not in self.estimators:
            self.estimators[key] = LightStateEstimator(self.config, self.fps)
        
        # Update estimator
        estimator = self.estimators[key]
        estimate = estimator.update(detection.is_active, detection.timestamp)
        
        # Track last update time
        self.last_update[key] = detection.timestamp
        
        return estimate
    
    def get_state(self, track_id: int, light_type: str) -> Optional[StateEstimate]:
        """Get current state for a track without updating"""
        key = (track_id, light_type)
        if key in self.estimators:
            estimator = self.estimators[key]
            return StateEstimate(
                state=estimator.current_state,
                confidence=estimator.confidence,
                blink_frequency=estimator._estimate_blink_frequency() if estimator.current_state == LightState.BLINK else None,
                activation_ratio=estimator._compute_activation_ratio()
            )
        return None
    
    def remove_track(self, track_id: int, light_type: Optional[str] = None):
        """
        Remove state estimator(s) for a track.
        
        Args:
            track_id: Track to remove
            light_type: Specific light type, or None to remove all lights for this track
        """
        if light_type is not None:
            key = (track_id, light_type)
            if key in self.estimators:
                del self.estimators[key]
                del self.last_update[key]
        else:
            # Remove all estimators for this track
            keys_to_remove = [k for k in self.estimators.keys() if k[0] == track_id]
            for key in keys_to_remove:
                del self.estimators[key]
                del self.last_update[key]
    
    def cleanup_stale_tracks(self, current_timestamp: float):
        """Remove estimators for tracks that haven't been updated recently"""
        stale_keys = [
            key for key, last_ts in self.last_update.items()
            if current_timestamp - last_ts > self.stale_threshold
        ]
        
        for key in stale_keys:
            del self.estimators[key]
            del self.last_update[key]
    
    def get_all_states(self) -> Dict[Tuple[int, str], StateEstimate]:
        """Get current state for all tracked objects"""
        states = {}
        for key, estimator in self.estimators.items():
            states[key] = StateEstimate(
                state=estimator.current_state,
                confidence=estimator.confidence,
                blink_frequency=estimator._estimate_blink_frequency() if estimator.current_state == LightState.BLINK else None,
                activation_ratio=estimator._compute_activation_ratio()
            )
        return states
    
    def reset(self):
        """Reset all estimators"""
        for estimator in self.estimators.values():
            estimator.reset()
    
    def reset_track(self, track_id: int, light_type: Optional[str] = None):
        """Reset specific track estimator(s)"""
        if light_type is not None:
            key = (track_id, light_type)
            if key in self.estimators:
                self.estimators[key].reset()
        else:
            # Reset all estimators for this track
            for key, estimator in self.estimators.items():
                if key[0] == track_id:
                    estimator.reset()
    
    def get_statistics(self) -> dict:
        """Get statistics about managed estimators"""
        return {
            'total_estimators': len(self.estimators),
            'unique_tracks': len(set(k[0] for k in self.estimators.keys())),
            'unique_light_types': len(set(k[1] for k in self.estimators.keys())),
            'state_distribution': self._compute_state_distribution()
        }
    
    def _compute_state_distribution(self) -> Dict[str, int]:
        """Compute distribution of current states"""
        distribution = {state.value: 0 for state in LightState}
        for estimator in self.estimators.values():
            distribution[estimator.current_state.value] += 1
        return distribution
    
    def __len__(self) -> int:
        """Return number of active estimators"""
        return len(self.estimators)
