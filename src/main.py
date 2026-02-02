"""
Automotive Camera Pipeline - Main Entry Point
"""

import cv2
import yaml
import time
import logging
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List

# Add project root to sys.path to allow imports from 'pipeline'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Detection
from pipeline.detection import DetectorFactory, DetectorOutput

# State Estimation
from pipeline.state_estimation import (
    StateManager,
    DetectionInput,
    StateEstimatorConfig
)

# Visualization
from pipeline.visualization import PerceptionVisualizer


def setup_logging(level_name='INFO'):
    logging.basicConfig(
        level=getattr(logging, level_name.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class AutomotivePerceptionPipeline:
    """
    Complete automotive perception pipeline.
    
    Pipeline Flow:
        Camera -> Detection -> Tracking -> State Estimation -> Visualization
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # Get system settings
        self.fps = self.config['cameras']['camera_0']['fps']
        self.max_latency_ms = self.config['system']['max_latency_ms']
        
        # Initialize components
        self.detector = None
        self.state_manager = None
        self.visualizer = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
    
    def initialize(self) -> bool:
        """Initialize all pipeline components."""
        try:
            logger.info("=" * 60)
            logger.info("Automotive Perception Pipeline")
            logger.info("=" * 60)
            
            # 1. Initialize Detector
            logger.info("\n[1/3] Initializing Detector...")
            self.detector = self._init_detector()
            
            # 2. Initialize State Manager
            logger.info("\n[2/3] Initializing State Estimation...")
            self.state_manager = self._init_state_manager()
            
            # 3. Initialize Visualizer
            logger.info("\n[3/3] Initializing Visualization...")
            self.visualizer = self._init_visualizer()
            
            logger.info("\nPipeline initialized successfully")
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return False
    
    def _init_detector(self):
        """Initialize detection module"""
        detection_config = self.config.get('detection', {})
        
        # Create detector using factory
        # Map 'model_type' from config to 'backend' expected by factory if needed
        backend = detection_config.get('backend', detection_config.get('model_type', 'rtmdet'))
        if backend == 'rtmdet':
             # Ensure config has 'backend' key if factory depends on it being explicit in dict
             detection_config['backend'] = 'rtmdet'

        detector = DetectorFactory.create(backend, detection_config)
        
        # Initialize detector (loads model)
        if not detector.initialize():
            raise RuntimeError("Detector initialization failed")
        
        # Log detector info
        stats = detector.get_statistics()
        logger.info(f" Detector: {stats.get('variant', backend).upper()}")
        logger.info(f"  Device: {stats.get('device', 'unknown')}")
        
        return detector
    
    def _init_state_manager(self):
        """Initialize state estimation module"""
        if not self.config['state_estimation'].get('enabled', True): # Default to True
            logger.info("State estimation disabled in config")
            return None
        
        se_config = self.config['state_estimation']
        
        # Handle config differences if any. 
        # The example used specific keys; we'll map from our pipeline_config.yaml
        
        # Create state estimation config
        state_config = StateEstimatorConfig(
            window_size=se_config.get('temporal_window', 5), # mapped from temporal_window
            on_threshold=se_config['on_threshold'],
            off_threshold=se_config['off_threshold'],
            min_blink_frequency=se_config['blink_frequency_hz'][0],
            max_blink_frequency=se_config['blink_frequency_hz'][1],
            min_blink_cycles=se_config.get('min_blink_cycles', 2),
            blink_frequency_variance_threshold=0.2, # Default
            confidence_gain=0.1, # Default
            confidence_decay=0.05, # Default
            confidence_reset_value=0.5, # Default
            min_confidence_threshold=0.3, # Default
            state_change_debounce_frames=se_config.get('state_change_threshold', 3)
        )
        
        state_manager = StateManager(state_config, self.fps)
        return state_manager
    
    def _init_visualizer(self):
        """Initialize visualization module"""
        if 'visualization' not in self.config:
             # Fallback if config doesn't have explicit visualization block
             logger.info("No visualization config found, using defaults")
             return PerceptionVisualizer()

        if not self.config['visualization'].get('enabled', True):
            logger.info("Visualization disabled in config")
            return None
        
        vis_config = self.config['visualization']
        
        visualizer = PerceptionVisualizer(
            show_labels=vis_config.get('display', {}).get('show_labels', True),
            show_confidence=vis_config.get('display', {}).get('show_confidence', True),
            show_frequency=vis_config.get('display', {}).get('show_frequency', True),
            box_thickness=vis_config.get('rendering', {}).get('box_thickness', 2),
            font_scale=vis_config.get('rendering', {}).get('font_scale', 0.6),
            font_thickness=vis_config.get('rendering', {}).get('font_thickness', 2)
        )
        return visualizer
    
    def process_frame(self, frame, frame_id: int, timestamp: float) -> dict:
        """Process single frame through complete pipeline."""
        pipeline_start = time.time()
        
        # Stage 1: Detection
        detection_start = time.time()
        detector_output = self.detector.infer(frame, timestamp)
        detection_time = (time.time() - detection_start) * 1000
        
        # Stage 2: Tracking (mock - replace with actual tracker)
        tracking_start = time.time()
        tracked_objects = self._mock_tracking(detector_output)
        tracking_time = (time.time() - tracking_start) * 1000
        
        # Stage 3: State Estimation
        state_start = time.time()
        state_results = {}
        if self.state_manager is not None:
            state_results = self._estimate_states(tracked_objects, timestamp)
        state_time = (time.time() - state_start) * 1000
        
        # Stage 4: Visualization
        annotated_frame = frame.copy()
        if self.visualizer is not None:
            annotated_frame = self._visualize(annotated_frame, tracked_objects, state_results, frame_id)
        
        # Calculate total latency
        total_latency = (time.time() - pipeline_start) * 1000
        
        # Results
        results = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'num_detections': len(detector_output.detections),
            'num_tracked': len(tracked_objects),
            'latency': total_latency,
            'detections': [det.to_dict() for det in detector_output.detections],
            'tracked_objects': tracked_objects
        }
        
        return results, annotated_frame
    
    def _mock_tracking(self, detector_output: DetectorOutput) -> List[dict]:
        """Mock tracking - simple pass-through with spatial hashing for ID"""
        tracked = []
        for i, detection in enumerate(detector_output.detections):
            # Simple mock: use position hash as track_id
            x1, y1, x2, y2 = detection.bbox
            track_id = hash((int((x1+x2)/2/100), int((y1+y2)/2/100))) % 1000
            
            tracked.append({
                'track_id': abs(track_id),
                'bbox': detection.bbox,
                'class_id': detection.class_id,
                'class_name': detection.class_name,
                'confidence': detection.confidence,
                'is_active': detection.confidence > 0.6
            })
        return tracked
    
    def _estimate_states(self, tracked_objects: List[dict], timestamp: float) -> Dict:
        """Run state estimation"""
        state_results = {}
        for obj in tracked_objects:
            det_input = DetectionInput(
                track_id=obj['track_id'],
                light_type=obj['class_name'],
                is_active=obj['is_active'],
                confidence=obj['confidence'],
                timestamp=timestamp
            )
            
            estimate = self.state_manager.update(det_input)
            key = (obj['track_id'], obj['class_name'])
            state_results[key] = estimate
            
            # Update object
            obj['state'] = estimate.state.value
            obj['state_confidence'] = estimate.confidence
            obj['blink_frequency'] = estimate.blink_frequency
        return state_results
    
    def _visualize(self, frame, tracked_objects: List[dict], state_results: Dict, frame_id: int):
        """Apply visualization"""
        # Draw detections
        for obj in tracked_objects:
            key = (obj['track_id'], obj['class_name'])
            if key in state_results:
                self.visualizer.draw_detection(
                    frame,
                    obj['bbox'],
                    state_results[key],
                    obj['class_name'],
                    obj['track_id']
                )
        
        # Info panel
        # (Simplified call for now, assuming visualizer handles defaults)
        # self.visualizer.draw_info_panel(frame, frame_id, ...)
        
        return frame
    
    def run(self, video_source, max_frames=None):
        logger.info(f"\nStarting pipeline with source: {video_source}")
        
        # Handle int source (camera) vs string (file)
        try:
            source = int(video_source)
        except (ValueError, TypeError):
            source = video_source

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {video_source}")
        
        self.start_time = time.time()
        self.frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream")
                    break
                
                timestamp = time.time()
                
                results, annotated_frame = self.process_frame(frame, self.frame_count, timestamp)
                
                # Show
                if self.visualizer is not None:
                    cv2.imshow('Automotive Perception Pipeline', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User requested stop")
                        break
                
                self.frame_count += 1
                if max_frames and self.frame_count >= max_frames:
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def shutdown(self):
        if self.detector:
            self.detector.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Automotive Perception Pipeline')
    parser.add_argument('--config', type=str, default='configs/pipeline_config.yaml', help='Path to configuration')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam, or file path)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    pipeline = AutomotivePerceptionPipeline(config)
    
    if not pipeline.initialize():
        return 1
        
    try:
        pipeline.run(args.source)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return 1
    finally:
        pipeline.shutdown()
        
    return 0

if __name__ == '__main__':
    exit(main())
