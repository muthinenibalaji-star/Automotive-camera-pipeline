"""
RTMDet Detector Implementation

This is the production-facing detector that implements DetectorBase.
It orchestrates the complete detection pipeline:
    Load → Preprocess → Infer → Postprocess → Output

Design Rules:
- Load model ONCE (in initialize)
- Batch size = 1 (real-time)
- No side effects
- No global state
- Thread-safe operations
"""

import time
import logging
import numpy as np
from typing import List
from mmdet.apis import inference_detector

from ..detector_base import DetectorBase, Detection, DetectorOutput, DetectorFactory
from .rtmdet_types import RTMDetConfig
from .rtmdet_loader import create_loader
from .rtmdet_preprocess import create_preprocessor
from .rtmdet_postprocess import create_postprocessor


logger = logging.getLogger(__name__)


class RTMDetDetector(DetectorBase):
    """
    RTMDet detector implementation.
    
    This class implements the DetectorBase contract and provides
    RTMDet-specific detection functionality.
    
    Lifecycle:
        1. __init__() - Create instance with config
        2. initialize() - Load model (once)
        3. infer() - Run detection (loop)
        4. shutdown() - Clean up
    """
    
    def __init__(self, config: dict):
        """
        Initialize RTMDet detector.
        
        Args:
            config: Detector configuration from pipeline_config.yaml
        """
        super().__init__(config)
        
        # Parse RTMDet-specific config
        self.rtmdet_config = RTMDetConfig.from_dict(config)
        
        # Components (initialized in initialize())
        self.loader = None
        self.preprocessor = None
        self.postprocessor = None
        self.model = None
        
        # Performance tracking
        self.total_inferences = 0
        self.total_inference_time = 0.0
    
    def initialize(self) -> bool:
        """
        Initialize detector - load model and prepare for inference.
        
        This is called ONCE before the inference loop.
        
        Returns:
            True if initialization successful
        """
        if self.is_initialized:
            logger.warning("Detector already initialized")
            return True
        
        try:
            logger.info("Initializing RTMDet detector...")
            
            # Create components
            self.loader = create_loader(self.rtmdet_config)
            self.preprocessor = create_preprocessor(self.rtmdet_config)
            self.postprocessor = create_postprocessor(self.rtmdet_config)
            
            # Load model
            if not self.loader.load():
                raise RuntimeError("Model loading failed")
            
            self.model = self.loader.get_model()
            
            # Log model info
            model_info = self.loader.get_model_info()
            logger.info(f"Model variant: RTMDet-{model_info['variant'].upper()}")
            logger.info(f"Parameters: {model_info['num_parameters']:,}")
            logger.info(f"Device: {model_info['device']}")
            logger.info(f"FP16: {model_info['fp16']}")
            
            self.is_initialized = True
            logger.info("✓ RTMDet detector initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Detector initialization failed: {e}")
            raise RuntimeError(f"Detector initialization failed: {e}")
    
    def infer(self, frame: np.ndarray, timestamp: float) -> DetectorOutput:
        """
        Run detection on a single frame.
        
        Args:
            frame: Input frame (HWC, BGR, uint8)
            timestamp: Frame timestamp
            
        Returns:
            DetectorOutput with standardized detections
        """
        if not self.is_initialized:
            raise RuntimeError("Detector not initialized. Call initialize() first.")
        
        # Validate input
        if not self.validate_frame(frame):
            raise ValueError("Invalid frame format")
        
        # Start timing
        start_time = time.time()
        
        try:
            # Preprocess
            preprocess_result = self.preprocessor.preprocess(frame)
            
            # Run inference
            raw_output = inference_detector(self.model, frame)
            
            # Calculate inference time
            inference_time_ms = (time.time() - start_time) * 1000
            
            # Postprocess
            rtmdet_result = self.postprocessor.postprocess(
                raw_output,
                preprocess_result,
                inference_time_ms
            )
            
            # Convert to Detection objects
            detections = self.postprocessor.convert_to_detections(rtmdet_result)
            
            # Update statistics
            self.total_inferences += 1
            self.total_inference_time += inference_time_ms
            
            # Create output
            output = DetectorOutput(
                detections=detections,
                timestamp=timestamp,
                inference_time_ms=inference_time_ms,
                frame_shape=frame.shape
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            # Return empty detections on error
            return DetectorOutput(
                detections=[],
                timestamp=timestamp,
                inference_time_ms=(time.time() - start_time) * 1000,
                frame_shape=frame.shape
            )
    
    def shutdown(self) -> bool:
        """
        Clean up detector resources.
        
        Returns:
            True if shutdown successful
        """
        if not self.is_initialized:
            logger.warning("Detector not initialized")
            return True
        
        try:
            logger.info("Shutting down RTMDet detector...")
            
            # Log final statistics
            if self.total_inferences > 0:
                avg_latency = self.total_inference_time / self.total_inferences
                logger.info(f"Total inferences: {self.total_inferences}")
                logger.info(f"Average latency: {avg_latency:.2f} ms")
            
            # Unload model
            if self.loader is not None:
                self.loader.unload()
            
            # Clear references
            self.model = None
            self.loader = None
            self.preprocessor = None
            self.postprocessor = None
            
            self.is_initialized = False
            logger.info("✓ Detector shut down successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Detector shutdown failed: {e}")
            return False
    
    def get_class_names(self) -> List[str]:
        """
        Get list of detectable class names.
        
        Returns:
            List of class names
        """
        return self.rtmdet_config.class_names
    
    def get_statistics(self) -> dict:
        """
        Get detector performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_latency = 0.0
        if self.total_inferences > 0:
            avg_latency = self.total_inference_time / self.total_inferences
        
        return {
            'total_inferences': self.total_inferences,
            'total_inference_time_ms': self.total_inference_time,
            'average_latency_ms': avg_latency,
            'variant': self.rtmdet_config.variant.value,
            'device': self.rtmdet_config.device,
            'fp16': self.rtmdet_config.fp16,
            'score_threshold': self.rtmdet_config.score_threshold
        }


# Register RTMDet detector with factory
DetectorFactory.register('rtmdet', RTMDetDetector)


def create_rtmdet_detector(config: dict) -> RTMDetDetector:
    """
    Factory function to create RTMDet detector.
    
    Args:
        config: Detector configuration
        
    Returns:
        RTMDetDetector instance
    """
    return RTMDetDetector(config)
