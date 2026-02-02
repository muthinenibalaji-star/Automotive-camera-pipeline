"""
Detector Base Contract

This is the ONLY interface the pipeline depends on.
All detector implementations MUST implement this contract.

Design Principles:
- Input agnostic (any frame source)
- Output standardized (stable schema)
- Runtime autonomous (no external state)
- Testable in isolation
- Swappable implementations (RTMDet-S/M, future detectors)
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class Detection:
    """
    Standardized detection output.
    
    This is the LOCKED interface between detection and the rest of the pipeline.
    
    Attributes:
        bbox: Bounding box in xyxy format [x1, y1, x2, y2]
        class_id: Integer class identifier
        class_name: Human-readable class name
        confidence: Detection confidence score [0.0, 1.0]
    """
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'bbox': list(self.bbox),
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': round(self.confidence, 3)
        }
    
    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Convert bbox from xyxy to xywh format"""
        x1, y1, x2, y2 = self.bbox
        return (x1, y1, x2 - x1, y2 - y1)


@dataclass
class DetectorOutput:
    """
    Complete detector output for a single frame.
    
    Attributes:
        detections: List of Detection objects
        timestamp: Frame timestamp
        inference_time_ms: Detection latency in milliseconds
        frame_shape: Original frame shape (H, W, C)
    """
    detections: List[Detection]
    timestamp: float
    inference_time_ms: float
    frame_shape: Tuple[int, int, int]
    
    def __len__(self) -> int:
        return len(self.detections)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization"""
        return {
            'timestamp': self.timestamp,
            'num_detections': len(self.detections),
            'inference_time_ms': round(self.inference_time_ms, 2),
            'detections': [det.to_dict() for det in self.detections]
        }


class DetectorBase(ABC):
    """
    Abstract base class for all detectors.
    
    This defines the contract that all detector implementations must follow.
    The pipeline depends ONLY on this interface.
    
    Lifecycle:
        1. __init__() - Create detector instance
        2. initialize() - Load model, allocate resources
        3. infer() - Run detection (called in loop)
        4. shutdown() - Clean up resources
    
    Design Rules:
        - No global state
        - No side effects
        - Deterministic output format
        - Thread-safe operations
        - No visualization logic
        - No tracking logic
        - No semantic interpretation
    """
    
    def __init__(self, config: dict):
        """
        Initialize detector with configuration.
        
        Args:
            config: Detector configuration dictionary
        """
        self.config = config
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize detector - load model, allocate GPU memory.
        
        This is called ONCE before the inference loop starts.
        
        Returns:
            True if initialization successful, False otherwise
            
        Raises:
            RuntimeError: If initialization fails critically
        """
        pass
    
    @abstractmethod
    def infer(self, frame: np.ndarray, timestamp: float) -> DetectorOutput:
        """
        Run detection on a single frame.
        
        This is called in the main inference loop.
        
        Args:
            frame: Input frame (HWC, BGR, uint8)
            timestamp: Frame timestamp in seconds
            
        Returns:
            DetectorOutput with standardized detections
            
        Raises:
            RuntimeError: If inference fails
            
        Design Constraints:
            - Must be thread-safe
            - Must not modify input frame
            - Must handle any valid frame shape
            - Must return in < 100ms (latency budget)
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """
        Clean up detector resources.
        
        This is called ONCE when the pipeline terminates.
        
        Returns:
            True if shutdown successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_class_names(self) -> List[str]:
        """
        Get list of all detectable class names.
        
        Returns:
            List of class names in order of class_id
        """
        pass
    
    def validate_frame(self, frame: np.ndarray) -> bool:
        """
        Validate input frame format.
        
        Args:
            frame: Input frame to validate
            
        Returns:
            True if frame is valid, False otherwise
        """
        if frame is None:
            return False
        
        if not isinstance(frame, np.ndarray):
            return False
        
        if frame.ndim != 3:
            return False
        
        if frame.shape[2] != 3:
            return False
        
        if frame.dtype != np.uint8:
            return False
        
        return True
    
    def __enter__(self):
        """Context manager entry"""
        if not self.is_initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
        return False


class DetectorFactory:
    """
    Factory for creating detector instances.
    
    Usage:
        detector = DetectorFactory.create('rtmdet', config)
        detector.initialize()
        output = detector.infer(frame, timestamp)
    """
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str, detector_class: type):
        """Register a detector implementation"""
        cls._registry[name] = detector_class
    
    @classmethod
    def create(cls, name: str, config: dict) -> DetectorBase:
        """
        Create detector instance by name.
        
        Args:
            name: Detector backend name ('rtmdet', etc.)
            config: Detector configuration
            
        Returns:
            Detector instance
            
        Raises:
            ValueError: If detector name not registered
        """
        if name not in cls._registry:
            raise ValueError(
                f"Detector '{name}' not registered. "
                f"Available: {list(cls._registry.keys())}"
            )
        
        detector_class = cls._registry[name]
        return detector_class(config)
    
    @classmethod
    def list_detectors(cls) -> List[str]:
        """List all registered detector backends"""
        return list(cls._registry.keys())
