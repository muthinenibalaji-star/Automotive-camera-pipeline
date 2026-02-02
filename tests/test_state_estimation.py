"""
Unit Tests for State Estimation Module

Tests cover:
- FSM state transitions
- Blink frequency detection
- Confidence modeling
- Multi-object tracking
- Edge cases and boundary conditions
"""

import pytest
import numpy as np
from pipeline.state_estimation import (
    LightStateEstimator,
    StateEstimatorConfig,
    LightState,
    StateManager,
    DetectionInput
)


class TestLightStateEstimator:
    """Test cases for LightStateEstimator"""
    
    @pytest.fixture
    def config(self):
        """Default test configuration"""
        return StateEstimatorConfig(
            window_size=60,
            on_threshold=0.5,
            off_threshold=0.2,
            min_blink_frequency=0.5,
            max_blink_frequency=3.0,
            state_change_debounce_frames=5
        )
    
    @pytest.fixture
    def estimator(self, config):
        """Create estimator instance"""
        return LightStateEstimator(config, fps=30.0)
    
    def test_initialization(self, estimator):
        """Test proper initialization"""
        assert estimator.current_state == LightState.UNKNOWN
        assert estimator.confidence == 0.0
        assert len(estimator.activation_buffer) == 0
    
    def test_off_state_detection(self, estimator):
        """Test transition to OFF state"""
        # Send 60 frames of OFF
        for i in range(60):
            estimate = estimator.update(is_active=False, timestamp=i/30.0)
        
        # Should transition to OFF after debounce
        assert estimator.current_state == LightState.OFF
    
    def test_on_state_detection(self, estimator):
        """Test transition to ON state"""
        # Send 60 frames of ON
        for i in range(60):
            estimate = estimator.update(is_active=True, timestamp=i/30.0)
        
        # Should transition to ON after debounce
        assert estimator.current_state == LightState.ON
    
    def test_blink_detection(self, estimator):
        """Test blink frequency detection"""
        # Generate 1 Hz blink at 30 FPS (15 frames ON, 15 frames OFF)
        for cycle in range(4):  # 4 complete cycles
            # ON phase
            for i in range(15):
                frame_id = cycle * 30 + i
                estimator.update(is_active=True, timestamp=frame_id/30.0)
            
            # OFF phase
            for i in range(15):
                frame_id = cycle * 30 + 15 + i
                estimator.update(is_active=False, timestamp=frame_id/30.0)
        
        # Should detect BLINK state
        assert estimator.current_state == LightState.BLINK
        
        # Check frequency estimate
        freq = estimator._estimate_blink_frequency()
        assert freq is not None
        assert 0.9 <= freq <= 1.1  # ~1 Hz with tolerance
    
    def test_confidence_increase(self, estimator):
        """Test confidence increases with consistent detections"""
        initial_confidence = estimator.confidence
        
        # Send consistent ON signals
        for i in range(20):
            estimator.update(is_active=True, timestamp=i/30.0)
        
        # Confidence should increase
        assert estimator.confidence > initial_confidence
    
    def test_confidence_decay(self, estimator):
        """Test confidence decays with missed detections"""
        # Build up confidence
        for i in range(20):
            estimator.update(is_active=True, timestamp=i/30.0)
        
        high_confidence = estimator.confidence
        
        # Send missed detections
        for i in range(20):
            estimator.update(is_active=False, timestamp=(20+i)/30.0)
        
        # Confidence should decay
        assert estimator.confidence < high_confidence
    
    def test_state_transition_hysteresis(self, estimator, config):
        """Test that state changes require debounce frames"""
        # Initialize to OFF
        for i in range(60):
            estimator.update(is_active=False, timestamp=i/30.0)
        
        assert estimator.current_state == LightState.OFF
        
        # Send ON signals but less than debounce threshold
        for i in range(config.state_change_debounce_frames - 1):
            estimator.update(is_active=True, timestamp=(60+i)/30.0)
        
        # Should still be OFF (hysteresis)
        assert estimator.current_state == LightState.OFF
        
        # Send one more frame to exceed debounce
        estimator.update(is_active=True, timestamp=(60+config.state_change_debounce_frames)/30.0)
        
        # Now should transition
        # (may need a few more frames to fully transition)
        for i in range(10):
            estimator.update(is_active=True, timestamp=(70+i)/30.0)
        
        assert estimator.current_state == LightState.ON
    
    def test_reset(self, estimator):
        """Test reset functionality"""
        # Build up state
        for i in range(60):
            estimator.update(is_active=True, timestamp=i/30.0)
        
        # Reset
        estimator.reset()
        
        # Should be back to initial state
        assert estimator.current_state == LightState.UNKNOWN
        assert estimator.confidence == 0.0
        assert len(estimator.activation_buffer) == 0
    
    def test_activation_ratio_calculation(self, estimator):
        """Test activation ratio computation"""
        # Send mixed signals: 70% ON, 30% OFF
        for i in range(100):
            is_active = (i % 10) < 7  # 7 ON, 3 OFF
            estimator.update(is_active=is_active, timestamp=i/30.0)
        
        ratio = estimator._compute_activation_ratio()
        assert 0.65 <= ratio <= 0.75  # Should be ~0.7
    
    def test_edge_detection(self, estimator):
        """Test rising edge detection"""
        # Create signal with known edges
        signal = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0])
        edges = estimator._detect_rising_edges(signal)
        
        # Should detect edges at indices 2 and 6
        edge_indices = np.where(edges)[0]
        assert len(edge_indices) == 2
        assert 2 in edge_indices
        assert 6 in edge_indices


class TestStateManager:
    """Test cases for StateManager"""
    
    @pytest.fixture
    def config(self):
        return StateEstimatorConfig(window_size=60)
    
    @pytest.fixture
    def manager(self, config):
        return StateManager(config, fps=30.0)
    
    def test_initialization(self, manager):
        """Test manager initialization"""
        assert len(manager) == 0
        assert len(manager.estimators) == 0
    
    def test_lazy_estimator_creation(self, manager):
        """Test estimators are created on demand"""
        detection = DetectionInput(
            track_id=1,
            light_type="left_indicator",
            is_active=True,
            confidence=0.9,
            timestamp=0.0
        )
        
        manager.update(detection)
        
        # Estimator should be created
        assert len(manager) == 1
        assert (1, "left_indicator") in manager.estimators
    
    def test_multiple_tracks(self, manager):
        """Test managing multiple tracks"""
        # Add detections for multiple tracks
        for track_id in range(1, 4):
            for light_type in ["left_indicator", "brake_light"]:
                detection = DetectionInput(
                    track_id=track_id,
                    light_type=light_type,
                    is_active=True,
                    confidence=0.9,
                    timestamp=0.0
                )
                manager.update(detection)
        
        # Should have 3 tracks * 2 lights = 6 estimators
        assert len(manager) == 6
    
    def test_track_removal(self, manager):
        """Test removing tracks"""
        # Add track
        detection = DetectionInput(
            track_id=1,
            light_type="left_indicator",
            is_active=True,
            confidence=0.9,
            timestamp=0.0
        )
        manager.update(detection)
        
        assert len(manager) == 1
        
        # Remove track
        manager.remove_track(1, "left_indicator")
        
        assert len(manager) == 0
    
    def test_cleanup_stale_tracks(self, manager):
        """Test automatic cleanup of old tracks"""
        # Add detection at t=0
        detection = DetectionInput(
            track_id=1,
            light_type="left_indicator",
            is_active=True,
            confidence=0.9,
            timestamp=0.0
        )
        manager.update(detection)
        
        # Cleanup at t=10 (exceeds 5 second threshold)
        manager.cleanup_stale_tracks(current_timestamp=10.0)
        
        # Track should be removed
        assert len(manager) == 0
    
    def test_get_all_states(self, manager):
        """Test retrieving all current states"""
        # Add multiple tracks
        for track_id in range(1, 3):
            detection = DetectionInput(
                track_id=track_id,
                light_type="left_indicator",
                is_active=True,
                confidence=0.9,
                timestamp=0.0
            )
            manager.update(detection)
        
        states = manager.get_all_states()
        assert len(states) == 2
        assert all(state.state == LightState.UNKNOWN for state in states.values())
    
    def test_statistics(self, manager):
        """Test statistics computation"""
        # Add some tracks
        for i in range(3):
            detection = DetectionInput(
                track_id=i,
                light_type="left_indicator",
                is_active=True,
                confidence=0.9,
                timestamp=0.0
            )
            manager.update(detection)
        
        stats = manager.get_statistics()
        assert stats['total_estimators'] == 3
        assert stats['unique_tracks'] == 3
        assert stats['unique_light_types'] == 1


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_buffer(self):
        """Test behavior with empty buffer"""
        config = StateEstimatorConfig()
        estimator = LightStateEstimator(config, fps=30.0)
        
        ratio = estimator._compute_activation_ratio()
        assert ratio == 0.0
        
        freq = estimator._estimate_blink_frequency()
        assert freq is None
    
    def test_single_frame(self):
        """Test with single frame"""
        config = StateEstimatorConfig()
        estimator = LightStateEstimator(config, fps=30.0)
        
        estimate = estimator.update(is_active=True, timestamp=0.0)
        
        assert estimate.state == LightState.UNKNOWN
        assert estimate.confidence >= 0.0
    
    def test_non_periodic_signal(self):
        """Test that non-periodic signals don't detect blink"""
        config = StateEstimatorConfig()
        estimator = LightStateEstimator(config, fps=30.0)
        
        # Random activations
        np.random.seed(42)
        for i in range(120):
            is_active = np.random.random() > 0.5
            estimator.update(is_active=is_active, timestamp=i/30.0)
        
        # Should not detect periodic blink
        freq = estimator._estimate_blink_frequency()
        assert freq is None
    
    def test_rapid_state_changes(self):
        """Test behavior with rapid state changes"""
        config = StateEstimatorConfig(state_change_debounce_frames=3)
        estimator = LightStateEstimator(config, fps=30.0)
        
        # Alternate rapidly
        for i in range(60):
            is_active = (i % 2) == 0
            estimator.update(is_active=is_active, timestamp=i/30.0)
        
        # Should eventually stabilize or detect blink
        assert estimator.current_state in [LightState.UNKNOWN, LightState.BLINK]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
