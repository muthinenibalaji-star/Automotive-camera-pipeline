"""
RTMDet Type Definitions

Centralized type definitions for RTMDet detector implementation.
"""

from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
import numpy as np


class RTMDetVariant(Enum):
    """Supported RTMDet model variants"""
    S = "s"  # RTMDet-S (smaller, faster)
    M = "m"  # RTMDet-M (larger, more accurate)
    
    @classmethod
    def from_string(cls, variant: str) -> 'RTMDetVariant':
        """Create variant from config string"""
        variant_lower = variant.lower()
        if variant_lower == 's':
            return cls.S
        elif variant_lower == 'm':
            return cls.M
        else:
            raise ValueError(f"Unknown RTMDet variant: {variant}. Use 's' or 'm'")


class RTMDetConfig:
    """
    Configuration for RTMDet detector.
    
    This is the data structure that controls detector behavior.
    All parameters are config-driven, not code-driven.
    """
    
    def __init__(
        self,
        variant: str = 's',
        config_path: str = None,
        weights_path: str = None,
        device: str = 'cuda',
        fp16: bool = True,
        score_threshold: float = 0.4,
        nms_threshold: float = 0.45,
        max_detections: int = 100,
        input_size: Tuple[int, int] = (1280, 720),
        class_names: List[str] = None
    ):
        """
        Initialize RTMDet configuration.
        
        Args:
            variant: Model variant ('s' or 'm')
            config_path: Path to MMDetection config file
            weights_path: Path to model weights (.pth)
            device: Device to run on ('cuda' or 'cpu')
            fp16: Enable FP16 inference
            score_threshold: Confidence threshold for detections
            nms_threshold: NMS IoU threshold
            max_detections: Maximum detections per frame
            input_size: Model input size (width, height)
            class_names: List of class names
        """
        self.variant = RTMDetVariant.from_string(variant)
        self.config_path = config_path
        self.weights_path = weights_path
        self.device = device
        self.fp16 = fp16 and device == 'cuda'  # FP16 only on CUDA
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.input_size = input_size
        self.class_names = class_names or self._default_light_classes()
    
    @staticmethod
    def _default_light_classes() -> List[str]:
        """Default automotive light classes"""
        return [
            'left_indicator',
            'right_indicator',
            'brake_light',
            'reverse_light',
            'headlight',
            'fog_light',
            'tail_light',
            'hazard_light'
        ]
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'RTMDetConfig':
        """Create config from dictionary (typically from YAML)"""
        model_config = config_dict.get('model', {})
        thresholds = config_dict.get('thresholds', {})
        
        return cls(
            variant=model_config.get('variant', 's'),
            config_path=model_config.get('config_path'),
            weights_path=model_config.get('weights_path'),
            device=config_dict.get('device', 'cuda'),
            fp16=config_dict.get('fp16', True),
            score_threshold=thresholds.get('score', 0.4),
            nms_threshold=thresholds.get('nms', 0.45),
            max_detections=thresholds.get('max_detections', 100),
            input_size=tuple(model_config.get('input_size', [1280, 720])),
            class_names=model_config.get('class_names')
        )
    
    def validate(self) -> bool:
        """Validate configuration"""
        if self.config_path is None:
            raise ValueError("config_path is required")
        
        if self.weights_path is None:
            raise ValueError("weights_path is required")
        
        if self.score_threshold < 0 or self.score_threshold > 1:
            raise ValueError("score_threshold must be in [0, 1]")
        
        if self.nms_threshold < 0 or self.nms_threshold > 1:
            raise ValueError("nms_threshold must be in [0, 1]")
        
        if self.device not in ['cuda', 'cpu']:
            raise ValueError("device must be 'cuda' or 'cpu'")
        
        return True
    
    def __repr__(self) -> str:
        return (
            f"RTMDetConfig("
            f"variant={self.variant.value}, "
            f"device={self.device}, "
            f"fp16={self.fp16}, "
            f"score_thresh={self.score_threshold})"
        )


class RTMDetInferenceResult:
    """
    Raw output from RTMDet model inference.
    
    This is an internal type - not exposed to the pipeline.
    Gets converted to DetectorOutput by postprocessing.
    """
    
    def __init__(
        self,
        bboxes: np.ndarray,  # (N, 4) in xyxy format
        scores: np.ndarray,  # (N,)
        labels: np.ndarray,  # (N,)
        inference_time_ms: float
    ):
        self.bboxes = bboxes
        self.scores = scores
        self.labels = labels
        self.inference_time_ms = inference_time_ms
    
    def __len__(self) -> int:
        return len(self.bboxes)


class PreprocessResult:
    """Result of preprocessing step"""
    
    def __init__(
        self,
        tensor: Any,  # Tensor (framework-specific)
        scale_factor: Tuple[float, float],  # (w_scale, h_scale)
        original_shape: Tuple[int, int, int]  # (H, W, C)
    ):
        self.tensor = tensor
        self.scale_factor = scale_factor
        self.original_shape = original_shape
