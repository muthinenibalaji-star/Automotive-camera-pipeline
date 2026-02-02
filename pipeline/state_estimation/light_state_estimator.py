"""
Light State Estimator - Finite State Machine Implementation

This module implements deterministic state estimation for automotive lights using:
- Temporal activation buffers (sliding window)
- Finite State Machine (FSM) with controlled transitions
- Frequency-based blink detection
- Confidence modeling with temporal decay

States: UNKNOWN, OFF, ON, BLINK
Transitions: Based on temporal statistics with hysteresis
"""

import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


class LightState(Enum):
    """Automotive light states"""
    UNKNOWN = "UNKNOWN"
    OFF = "OFF"
    ON = "ON"
    BLINK = "BLINK"


@dataclass
class StateEstimatorConfig:
    """Configuration for state estimation"""
    # Temporal window
    window_size: int = 60  # frames (e.g., 2 seconds at 30 FPS)
    
    # Activation thresholds
    on_threshold: float = 0.5  # Ratio of ON frames to classify as ON
    off_threshold: float = 0.2  # Ratio of ON frames to classify as OFF
    
    # Blink detection
    min_blink_frequency: float = 0.5  # Hz (minimum automotive blink rate)
    max_blink_frequency: float = 3.0  # Hz (maximum automotive blink rate)
    min_blink_cycles: int = 2  # Minimum edges to confirm blinking
    blink_frequency_variance_threshold: float = 0.3  # Max variance for periodic signal
    
    # Confidence parameters
    confidence_gain: float = 0.1  # Increase per consistent frame
    confidence_decay: float = 0.05  # Decrease per missed detection
    confidence_reset_value: float = 0.3  # Reset value on state transition
    min_confidence_threshold: float = 0.6  # Minimum confidence to report state
    
    # State transition hysteresis
    state_change_debounce_frames: int = 5  # Minimum frames before state change


@dataclass
class StateEstimate:
    """Output of state estimation"""
    state: LightState
    confidence: float
    blink_frequency: Optional[float] = None
    activation_ratio: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'state': self.state.value,
            'confidence': round(self.confidence, 3),
            'blink_frequency': round(self.blink_frequency, 2) if self.blink_frequency else None,
            'activation_ratio': round(self.activation_ratio, 3)
        }


class LightStateEstimator:
    """
    Finite State Machine for automotive light state estimation.
    
    This estimator converts frame-wise binary detections into temporally
    consistent state classifications using:
    1. Sliding window activation buffer
    2. Deterministic FSM transitions
    3. Frequency-domain blink detection
    4. Exponentially weighted confidence
    
    Designed for:
    - Deterministic behavior (no ML in state logic)
    - Explainability and validation
    - Real-time performance
    - Safety-critical automotive applications
    """
    
    def __init__(self, config: StateEstimatorConfig, fps: float = 30.0):
        self.config = config
        self.fps = fps
        
        # Temporal activation buffer (sliding window)
        self.activation_buffer: deque = deque(maxlen=config.window_size)
        self.timestamp_buffer: deque = deque(maxlen=config.window_size)
        
        # FSM state
        self.current_state = LightState.UNKNOWN
        self.confidence = 0.0
        
        # State transition tracking
        self.frames_in_current_state = 0
        self.candidate_next_state: Optional[LightState] = None
        self.candidate_state_frames = 0
        
        # Debug/logging
        self.state_history: List[Tuple[float, LightState, float]] = []
        
    def update(self, is_active: bool, timestamp: float) -> StateEstimate:
        """
        Update state estimator with new detection.
        
        Args:
            is_active: Binary activation (True if light detected as ON)
            timestamp: Frame timestamp
            
        Returns:
            StateEstimate with current state, confidence, and metadata
        """
        # Add to temporal buffer
        self.activation_buffer.append(1.0 if is_active else 0.0)
        self.timestamp_buffer.append(timestamp)
        
        # Update confidence based on detection
        self._update_confidence(is_active)
        
        # Compute temporal statistics
        activation_ratio = self._compute_activation_ratio()
        blink_frequency = self._estimate_blink_frequency()
        
        # FSM state transition logic
        self._update_state(activation_ratio, blink_frequency)
        
        # Create state estimate
        estimate = StateEstimate(
            state=self.current_state,
            confidence=self.confidence,
            blink_frequency=blink_frequency if self.current_state == LightState.BLINK else None,
            activation_ratio=activation_ratio
        )
        
        # Log state history
        self.state_history.append((timestamp, self.current_state, self.confidence))
        
        return estimate
    
    def _compute_activation_ratio(self) -> float:
        """Compute ratio of ON frames in temporal window"""
        if len(self.activation_buffer) == 0:
            return 0.0
        return np.mean(self.activation_buffer)
    
    def _estimate_blink_frequency(self) -> Optional[float]:
        """
        Estimate blink frequency using rising-edge detection.
        
        Returns frequency in Hz if periodic signal detected, else None.
        """
        if len(self.activation_buffer) < self.config.min_blink_cycles * 2:
            return None
        
        # Convert buffer to numpy array
        signal = np.array(self.activation_buffer)
        
        # Detect rising edges (OFF → ON transitions)
        edges = self._detect_rising_edges(signal)
        
        if len(edges) < self.config.min_blink_cycles:
            return None
        
        # Compute inter-edge intervals
        edge_indices = np.where(edges)[0]
        if len(edge_indices) < 2:
            return None
        
        intervals = np.diff(edge_indices)
        
        # Check periodicity using variance
        if len(intervals) > 1:
            mean_interval = np.mean(intervals)
            variance = np.std(intervals) / mean_interval if mean_interval > 0 else 1.0
            
            # Reject non-periodic signals
            if variance > self.config.blink_frequency_variance_threshold:
                return None
            
            # Convert interval (frames) to frequency (Hz)
            frequency = self.fps / (2 * mean_interval)  # Full cycle = 2 edges
            
            # Validate frequency range
            if self.config.min_blink_frequency <= frequency <= self.config.max_blink_frequency:
                return frequency
        
        return None
    
    def _detect_rising_edges(self, signal: np.ndarray) -> np.ndarray:
        """Detect rising edges in binary signal"""
        # Threshold signal
        binary_signal = (signal > 0.5).astype(int)
        
        # Rising edge = transition from 0 to 1
        edges = np.diff(binary_signal, prepend=0)
        return edges > 0
    
    def _update_confidence(self, is_active: bool):
        """Update confidence with exponential weighting"""
        if is_active:
            # Increase confidence on detection
            self.confidence = min(1.0, self.confidence + self.config.confidence_gain)
        else:
            # Decay confidence on missed detection
            self.confidence = max(0.0, self.confidence - self.config.confidence_decay)
    
    def _update_state(self, activation_ratio: float, blink_frequency: Optional[float]):
        """
        FSM state transition logic with hysteresis.
        
        State Transition Rules:
        - UNKNOWN → OFF/ON/BLINK: Based on initial observations
        - OFF ↔ ON: Based on activation ratio with hysteresis
        - * → BLINK: When periodic signal detected
        - BLINK → ON/OFF: When periodicity lost
        """
        self.frames_in_current_state += 1
        
        # Determine candidate next state based on observations
        if blink_frequency is not None:
            candidate_state = LightState.BLINK
        elif activation_ratio >= self.config.on_threshold:
            candidate_state = LightState.ON
        elif activation_ratio <= self.config.off_threshold:
            candidate_state = LightState.OFF
        else:
            # In hysteresis region - maintain current state
            candidate_state = self.current_state
        
        # State transition with debouncing
        if candidate_state != self.current_state:
            if self.candidate_next_state == candidate_state:
                self.candidate_state_frames += 1
                
                # Transition if debounce threshold met
                if self.candidate_state_frames >= self.config.state_change_debounce_frames:
                    self._transition_to_state(candidate_state)
            else:
                # New candidate state
                self.candidate_next_state = candidate_state
                self.candidate_state_frames = 1
        else:
            # Candidate matches current state - reset candidate tracking
            self.candidate_next_state = None
            self.candidate_state_frames = 0
    
    def _transition_to_state(self, new_state: LightState):
        """Execute state transition with confidence reset"""
        self.current_state = new_state
        self.confidence = self.config.confidence_reset_value
        self.frames_in_current_state = 0
        self.candidate_next_state = None
        self.candidate_state_frames = 0
    
    def reset(self):
        """Reset estimator to initial state"""
        self.activation_buffer.clear()
        self.timestamp_buffer.clear()
        self.current_state = LightState.UNKNOWN
        self.confidence = 0.0
        self.frames_in_current_state = 0
        self.candidate_next_state = None
        self.candidate_state_frames = 0
        self.state_history.clear()
    
    def get_state_history(self) -> List[Tuple[float, str, float]]:
        """Return state history for debugging"""
        return [(ts, state.value, conf) for ts, state, conf in self.state_history]
