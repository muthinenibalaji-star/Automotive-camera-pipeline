"""
Integrated Pipeline with State Estimation

This example shows how to integrate the state estimation module
into the existing perception pipeline with minimal code changes.
"""

import cv2
import yaml
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Import perception modules
from pipeline.state_estimation import (
    StateManager,
    DetectionInput,
    StateEstimatorConfig,
    LightState
)
from pipeline.visualization import PerceptionVisualizer
from pipeline.state_estimation import StateDebugger


class IntegratedPerceptionPipeline:
    """
    Complete perception pipeline with state estimation.
    
    Pipeline Flow:
    1. Camera Capture
    2. Detection (RTMDet)
    3. Tracking (ByteTrack)
    4. State Estimation (FSM) ← NEW
    5. Visualization ← ENHANCED
    6. Output Serialization ← ENHANCED
    """
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize FPS
        self.fps = self.config['cameras']['camera_0']['fps']
        
        # Initialize state estimation
        self._init_state_estimation()
        
        # Initialize visualization
        self._init_visualization()
        
        # Initialize debug tools
        self._init_debug_tools()
        
        # Placeholder for other components
        # self.camera_manager = CameraManager(...)
        # self.detector = LightDetector(...)
        # self.tracker = MultiObjectTracker(...)
        
        print("✓ Integrated pipeline initialized with state estimation")
    
    def _init_state_estimation(self):
        """Initialize state estimation module"""
        if not self.config['state_estimation']['enabled']:
            self.state_manager = None
            return
        
        # Create configuration
        state_config = StateEstimatorConfig(
            window_size=self.config['state_estimation']['window_size'],
            on_threshold=self.config['state_estimation']['on_threshold'],
            off_threshold=self.config['state_estimation']['off_threshold'],
            min_blink_frequency=self.config['state_estimation']['blink_detection']['min_frequency'],
            max_blink_frequency=self.config['state_estimation']['blink_detection']['max_frequency'],
            min_blink_cycles=self.config['state_estimation']['blink_detection']['min_cycles'],
            blink_frequency_variance_threshold=self.config['state_estimation']['blink_detection']['variance_threshold'],
            confidence_gain=self.config['state_estimation']['confidence']['gain'],
            confidence_decay=self.config['state_estimation']['confidence']['decay'],
            confidence_reset_value=self.config['state_estimation']['confidence']['reset_value'],
            min_confidence_threshold=self.config['state_estimation']['confidence']['min_threshold'],
            state_change_debounce_frames=self.config['state_estimation']['transition']['debounce_frames']
        )
        
        # Create state manager
        self.state_manager = StateManager(state_config, self.fps)
        print("✓ State estimation enabled")
    
    def _init_visualization(self):
        """Initialize visualization module"""
        if not self.config['visualization']['enabled']:
            self.visualizer = None
            return
        
        vis_config = self.config['visualization']
        self.visualizer = PerceptionVisualizer(
            show_labels=vis_config['display']['show_labels'],
            show_confidence=vis_config['display']['show_confidence'],
            show_frequency=vis_config['display']['show_frequency'],
            box_thickness=vis_config['rendering']['box_thickness'],
            font_scale=vis_config['rendering']['font_scale'],
            font_thickness=vis_config['rendering']['font_thickness']
        )
        print("✓ Visualization enabled")
    
    def _init_debug_tools(self):
        """Initialize debug tools"""
        if not self.config['debug']['enabled']:
            self.debugger = None
            return
        
        debug_config = self.config['debug']
        if debug_config['state_plots']['enabled']:
            output_dir = debug_config['state_plots']['output_dir']
            self.debugger = StateDebugger(output_dir)
            self.debug_save_interval = debug_config['state_plots']['save_interval']
            print(f"✓ Debug plots enabled (saving to {output_dir})")
        else:
            self.debugger = None
    
    def process_frame(self, frame, frame_id: int, timestamp: float) -> dict:
        """
        Process single frame through complete pipeline.
        
        Args:
            frame: Input image (numpy array)
            frame_id: Frame number
            timestamp: Frame timestamp
            
        Returns:
            Complete perception results
        """
        # Stage 1: Detection (placeholder - replace with actual detector)
        detections = self._mock_detections(frame_id, timestamp)
        
        # Stage 2: Tracking (placeholder - replace with actual tracker)
        tracked_objects = self._mock_tracking(detections)
        
        # Stage 3: State Estimation (NEW)
        state_results = {}
        if self.state_manager is not None:
            for obj in tracked_objects:
                detection_input = DetectionInput(
                    track_id=obj['track_id'],
                    light_type=obj['class'],
                    is_active=obj['is_active'],
                    confidence=obj['confidence'],
                    timestamp=timestamp
                )
                
                # Update state estimation
                state_estimate = self.state_manager.update(detection_input)
                
                # Store result
                key = (obj['track_id'], obj['class'])
                state_results[key] = state_estimate
                
                # Update object with state info
                obj['state'] = state_estimate.state.value
                obj['state_confidence'] = state_estimate.confidence
                obj['blink_frequency'] = state_estimate.blink_frequency
                obj['activation_ratio'] = state_estimate.activation_ratio
        
        # Stage 4: Visualization (ENHANCED)
        if self.visualizer is not None:
            frame = self._visualize_results(frame, tracked_objects, state_results, frame_id)
        
        # Stage 5: Debug Plots (if enabled)
        if self.debugger is not None and frame_id % self.debug_save_interval == 0:
            self._generate_debug_plots(frame_id)
        
        # Stage 6: Cleanup stale tracks
        if self.state_manager is not None:
            self.state_manager.cleanup_stale_tracks(timestamp)
        
        # Prepare output
        results = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'detections': tracked_objects,
            'num_tracks': len(set(obj['track_id'] for obj in tracked_objects))
        }
        
        if self.state_manager is not None:
            results['state_statistics'] = self.state_manager.get_statistics()
        
        return results
    
    def _mock_detections(self, frame_id: int, timestamp: float) -> List[dict]:
        """Mock detection results - replace with actual detector"""
        # Simulate detected lights with varying activation
        import random
        
        detections = []
        
        # Left indicator (blinking)
        is_blink_on = (frame_id % 30) < 15  # 1 Hz blink at 30 FPS
        detections.append({
            'bbox': [100, 200, 50, 30],
            'class': 'left_indicator',
            'confidence': 0.9,
            'is_active': is_blink_on
        })
        
        # Brake light (ON)
        detections.append({
            'bbox': [300, 200, 60, 35],
            'class': 'brake_light',
            'confidence': 0.95,
            'is_active': True
        })
        
        # Headlight (OFF)
        detections.append({
            'bbox': [500, 180, 70, 40],
            'class': 'headlight',
            'confidence': 0.85,
            'is_active': False
        })
        
        return detections
    
    def _mock_tracking(self, detections: List[dict]) -> List[dict]:
        """Mock tracking - replace with actual tracker"""
        # Add stable track IDs
        tracked = []
        for i, det in enumerate(detections):
            det['track_id'] = i + 1  # Stable IDs for demo
            tracked.append(det)
        return tracked
    
    def _visualize_results(
        self,
        frame,
        tracked_objects: List[dict],
        state_results: Dict,
        frame_id: int
    ):
        """Apply visualization overlays"""
        # Draw each detection
        for obj in tracked_objects:
            key = (obj['track_id'], obj['class'])
            if key in state_results:
                self.visualizer.draw_detection(
                    frame,
                    obj['bbox'],
                    state_results[key],
                    obj['class'],
                    obj['track_id']
                )
        
        # Draw info panel
        if self.config['visualization']['display']['show_info_panel']:
            state_dist = self.state_manager.get_statistics()['state_distribution'] if self.state_manager else None
            self.visualizer.draw_info_panel(
                frame,
                frame_id,
                self.fps,
                0.0,  # Placeholder latency
                len(tracked_objects),
                state_dist
            )
        
        return frame
    
    def _generate_debug_plots(self, frame_id: int):
        """Generate debug plots for all active estimators"""
        if self.state_manager is None:
            return
        
        for (track_id, light_type), estimator in self.state_manager.estimators.items():
            self.debugger.plot_state_timeline(
                estimator,
                track_id,
                light_type,
                save=True
            )
    
    def run(self, video_path: str = None):
        """Run complete pipeline"""
        print("Starting integrated pipeline...")
        
        # Mock video capture (replace with actual camera)
        frame_id = 0
        
        for _ in range(300):  # Process 300 frames for demo
            # Mock frame
            frame = self._create_mock_frame()
            timestamp = time.time()
            
            # Process frame
            results = self.process_frame(frame, frame_id, timestamp)
            
            # Display (optional)
            if self.visualizer is not None:
                cv2.imshow('Perception Pipeline', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_id += 1
            time.sleep(1/30)  # Simulate 30 FPS
        
        cv2.destroyAllWindows()
        print("Pipeline execution complete")
    
    def _create_mock_frame(self):
        """Create mock frame for demo"""
        return 255 * np.ones((720, 1280, 3), dtype=np.uint8)


def main():
    """Entry point"""
    config_path = "configs/pipeline_config.yaml"
    
    pipeline = IntegratedPerceptionPipeline(config_path)
    pipeline.run()


if __name__ == '__main__':
    import numpy as np
    main()
