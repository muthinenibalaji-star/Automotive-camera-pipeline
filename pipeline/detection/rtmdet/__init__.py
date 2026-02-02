"""
RTMDet Implementation Module

MMDetection-based RTMDet detector for automotive light detection.

Components:
- rtmdet_detector: Main detector implementation
- rtmdet_loader: Model loading and configuration
- rtmdet_preprocess: Frame preprocessing
- rtmdet_postprocess: Output postprocessing
- rtmdet_types: Type definitions
"""

from .rtmdet_detector import RTMDetDetector, create_rtmdet_detector
from .rtmdet_types import RTMDetConfig, RTMDetVariant
from .rtmdet_loader import create_loader
from .rtmdet_preprocess import create_preprocessor
from .rtmdet_postprocess import create_postprocessor

__all__ = [
    'RTMDetDetector',
    'create_rtmdet_detector',
    'RTMDetConfig',
    'RTMDetVariant',
    'create_loader',
    'create_preprocessor',
    'create_postprocessor'
]
