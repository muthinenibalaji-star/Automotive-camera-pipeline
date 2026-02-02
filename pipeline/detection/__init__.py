"""
Detection Module

Provides object detection functionality for the automotive camera pipeline.

Key Components:
- DetectorBase: Abstract interface (contract)
- RTMDetDetector: RTMDet implementation
- DetectorFactory: Factory for creating detectors

Usage:
    from pipeline.detection import DetectorFactory
    
    detector = DetectorFactory.create('rtmdet', config)
    detector.initialize()
    
    output = detector.infer(frame, timestamp)
    
    for detection in output.detections:
        print(f"{detection.class_name}: {detection.confidence:.2f}")
    
    detector.shutdown()
"""

from .detector_base import (
    DetectorBase,
    Detection,
    DetectorOutput,
    DetectorFactory
)

# Import RTMDet to register it with factory
from .rtmdet.rtmdet_detector import RTMDetDetector

__all__ = [
    'DetectorBase',
    'Detection',
    'DetectorOutput',
    'DetectorFactory',
    'RTMDetDetector'
]

__version__ = '1.0.0'
