"""
Complete Integrated Pipeline Example

Demonstrates the full pipeline with:
1. RTMDet Detection
2. ByteTrack Tracking
3. FSM State Estimation
4. Real-time Visualization

This shows how all components work together.
"""

import cv2
import yaml
import time
import logging
from pathlib import Path
from typing import Dict, List

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


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutomotivePerceptionPipeline:
    """
    Complete automotive perception pipeline.
    
    Pipeline Flow:
        Camera → Detection → Tracking → State Estimation → Visualization
    """
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to pipeline_config.yaml
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
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
        """
        Initialize all pipeline components.
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("=" * 60)
            logger.info("Automotive Perception Pipeline v2.0")
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
            
            logger.info("\n✓ Pipeline initialized successfully")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return False
    
    def _init_detector(self):
        """Initialize detection module"""
        detection_config = self.config.get('detection', {})
        
        # Create detector using factory
        backend = detection_config.get('backend', 'rtmdet')
        detector = DetectorFactory.create(backend, detection_config)
        
        # Initialize detector (loads model)
        if not detector.initialize():
            raise RuntimeError("Detector initialization failed")
        
        # Log detector info
        stats = detector.get_statistics()
        logger.info(f"✓ Detector: RTMDet-{stats['variant'].upper()}")
        logger.info(f"  Device: {stats['device']}")
        logger.info(f"  FP16: {stats['fp16']}")
        logger.info(f"  Classes: {len(detector.get_class_names())}")
        
        return detector
    
    def _init_state_manager(self):
        """Initialize state estimation module"""
        if not self.config['state_estimation']['enabled']:
            logger.info("State estimation disabled in config")
            return None
        
        # Create state estimation config
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
        
        state_manager = StateManager(state_config, self.fps)
        
        logger.info("✓ State estimation enabled")
        logger.info(f"  Window size: {state_config.window_size} frames")
        logger.info(f"  Blink range: {state_config.min_blink_frequency}-{state_config.max_blink_frequency} Hz")
        
        return state_manager
    
    def _init_visualizer(self):
        """Initialize visualization module"""
        if not self.config['visualization']['enabled']:
            logger.info("Visualization disabled in config")
            return None
        
        vis_config = self.config['visualization']
        
        visualizer = PerceptionVisualizer(
            show_labels=vis_config['display']['show_labels'],
            show_confidence=vis_config['display']['show_confidence'],
            show_frequency=vis_config['display']['show_frequency'],
            box_thickness=vis_config['rendering']['box_thickness'],
            font_scale=vis_config['rendering']['font_scale'],
            font_thickness=vis_config['rendering']['font_thickness']
        )
        
        logger.info("✓ Visualization enabled")
        
        return visualizer
    
    def process_frame(self, frame, frame_id: int, timestamp: float) -> dict:
        """
        Process single frame through complete pipeline.
        
        Args:
            frame: Input frame (BGR)
            frame_id: Frame number
            timestamp: Frame timestamp
            
        Returns:
            Complete perception results
        """
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
        if self.visualizer is not None:
            frame = self._visualize(frame, tracked_objects, state_results, frame_id)
        
        # Calculate total latency
        total_latency = (time.time() - pipeline_start) * 1000
        
        # Check latency budget
        if total_latency > self.max_latency_ms:
            logger.warning(f"Latency exceeded: {total_latency:.1f}ms > {self.max_latency_ms}ms")
        
        # Prepare results
        results = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'num_detections': len(detector_output.detections),
            'num_tracked': len(tracked_objects),
            'latency': {
                'detection_ms': detection_time,
                'tracking_ms': tracking_time,
                'state_estimation_ms': state_time,
                'total_ms': total_latency
            },
            'detections': [det.to_dict() for det in detector_output.detections],
            'tracked_objects': tracked_objects
        }
        
        return results, frame
    
    def _mock_tracking(self, detector_output: DetectorOutput) -> List[dict]:
        """
        Mock tracking - replace with actual ByteTrack.
        
        For now, assigns simple stable IDs.
        """
        tracked = []
        
        for i, detection in enumerate(detector_output.detections):
            # Simple mock: use position hash as track_id
            x1, y1, x2, y2 = detection.bbox
            track_id = hash((int((x1+x2)/2/100), int((y1+y2)/2/100))) % 1000
            
            tracked.append({
                'track_id': track_id,
                'bbox': detection.bbox,
                'class_id': detection.class_id,
                'class_name': detection.class_name,
                'confidence': detection.confidence,
                'is_active': detection.confidence > 0.6  # Simple activation
            })
        
        return tracked
    
    def _estimate_states(self, tracked_objects: List[dict], timestamp: float) -> Dict:
        """Run state estimation on tracked objects"""
        state_results = {}
        
        for obj in tracked_objects:
            # Create detection input
            det_input = DetectionInput(
                track_id=obj['track_id'],
                light_type=obj['class_name'],
                is_active=obj['is_active'],
                confidence=obj['confidence'],
                timestamp=timestamp
            )
            
            # Update state
            estimate = self.state_manager.update(det_input)
            
            # Store result
            key = (obj['track_id'], obj['class_name'])
            state_results[key] = estimate
            
            # Update object with state info
            obj['state'] = estimate.state.value
            obj['state_confidence'] = estimate.confidence
            obj['blink_frequency'] = estimate.blink_frequency
        
        return state_results
    
    def _visualize(self, frame, tracked_objects: List[dict], state_results: Dict, frame_id: int):
        """Apply visualization overlays"""
        # Draw each detection
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
        
        # Draw info panel
        if self.config['visualization']['display']['show_info_panel']:
            state_dist = self.state_manager.get_statistics()['state_distribution'] if self.state_manager else None
            
            # Calculate FPS
            fps = self.frame_count / (time.time() - self.start_time) if self.start_time else 0
            
            self.visualizer.draw_info_panel(
                frame,
                frame_id,
                fps,
                0.0,  # Placeholder latency
                len(tracked_objects),
                state_dist
            )
        
        return frame
    
    def run(self, video_source: str = 0, max_frames: int = None):
        """
        Run complete pipeline.
        
        Args:
            video_source: Video file path or camera index
            max_frames: Maximum frames to process (None = infinite)
        """
        logger.info(f"\nStarting pipeline with source: {video_source}")
        
        # Open video source
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {video_source}")
        
        self.start_time = time.time()
        self.frame_count = 0
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream")
                    break
                
                timestamp = time.time()
                
                # Process frame
                results, annotated_frame = self.process_frame(
                    frame, 
                    self.frame_count, 
                    timestamp
                )
                
                # Display
                if self.visualizer is not None:
                    cv2.imshow('Automotive Perception Pipeline', annotated_frame)
                    
                    # Press 'q' to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User requested stop")
                        break
                
                self.frame_count += 1
                
                # Check max frames
                if max_frames and self.frame_count >= max_frames:
                    logger.info(f"Reached max frames: {max_frames}")
                    break
                
                # Log progress
                if self.frame_count % 30 == 0:
                    fps = self.frame_count / (time.time() - self.start_time)
                    logger.info(
                        f"Frame {self.frame_count}: "
                        f"{results['num_detections']} detections, "
                        f"{results['latency']['total_ms']:.1f}ms, "
                        f"{fps:.1f} FPS"
                    )
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Log final statistics
            total_time = time.time() - self.start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            
            logger.info("\n" + "=" * 60)
            logger.info("Pipeline Execution Summary")
            logger.info("=" * 60)
            logger.info(f"Total frames: {self.frame_count}")
            logger.info(f"Total time: {total_time:.2f}s")
            logger.info(f"Average FPS: {avg_fps:.2f}")
            
            if self.detector:
                det_stats = self.detector.get_statistics()
                logger.info(f"Average detection latency: {det_stats['average_latency_ms']:.2f}ms")
    
    def shutdown(self):
        """Shutdown all pipeline components"""
        logger.info("\nShutting down pipeline...")
        
        if self.detector:
            self.detector.shutdown()
        
        logger.info("✓ Pipeline shutdown complete")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automotive Perception Pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pipeline_config_with_detection.yaml',
        help='Path to pipeline configuration'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source (file path or camera index)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum frames to process'
    )
    
    args = parser.parse_args()
    
    # Convert source to int if it's a camera index
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # Create pipeline
    pipeline = AutomotivePerceptionPipeline(args.config)
    
    # Initialize
    if not pipeline.initialize():
        logger.error("Pipeline initialization failed")
        return 1
    
    try:
        # Run pipeline
        pipeline.run(source, args.max_frames)
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return 1
    
    finally:
        # Clean shutdown
        pipeline.shutdown()
    
    return 0


if __name__ == '__main__':
    exit(main())
