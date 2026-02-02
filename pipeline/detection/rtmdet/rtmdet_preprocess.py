"""
RTMDet Preprocessing

Responsibilities:
- Resize frame to model input size
- Normalize pixel values
- Convert to tensor format
- Move to GPU

Design principle: Keep it minimal - RTMDet is robust enough.
"""

import cv2
import torch
import numpy as np
from typing import Tuple
import logging

from .rtmdet_types import RTMDetConfig, PreprocessResult


logger = logging.getLogger(__name__)


class RTMDetPreprocessor:
    """
    Preprocesses frames for RTMDet inference.
    
    Minimal preprocessing philosophy:
    - No fancy augmentations (inference only)
    - No aggressive color correction
    - Trust RTMDet's robustness
    """
    
    def __init__(self, config: RTMDetConfig):
        """
        Initialize preprocessor.
        
        Args:
            config: RTMDet configuration
        """
        self.config = config
        self.target_size = config.input_size  # (width, height)
        self.device = config.device
        self.fp16 = config.fp16
        
        # Normalization parameters (ImageNet standard)
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    
    def preprocess(self, frame: np.ndarray) -> PreprocessResult:
        """
        Preprocess frame for RTMDet inference.
        
        Args:
            frame: Input frame (HWC, BGR, uint8)
            
        Returns:
            PreprocessResult with tensor and metadata
        """
        # Store original shape
        original_shape = frame.shape  # (H, W, C)
        
        # Resize frame
        resized_frame, scale_factor = self._resize_frame(frame)
        
        # Normalize
        normalized = self._normalize(resized_frame)
        
        # Convert to tensor
        tensor = self._to_tensor(normalized)
        
        # Move to device
        tensor = tensor.to(self.device)
        
        # Convert to FP16 if enabled
        if self.fp16:
            tensor = tensor.half()
        
        return PreprocessResult(
            tensor=tensor,
            scale_factor=scale_factor,
            original_shape=original_shape
        )
    
    def _resize_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Resize frame to model input size.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (resized_frame, scale_factor)
        """
        h, w = frame.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate scale factor
        scale_w = target_w / w
        scale_h = target_h / h
        
        # Resize using INTER_LINEAR (good balance of speed and quality)
        resized = cv2.resize(
            frame,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR
        )
        
        return resized, (scale_w, scale_h)
    
    def _normalize(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values using ImageNet statistics.
        
        Args:
            frame: BGR frame (uint8)
            
        Returns:
            Normalized frame (float32)
        """
        # Convert to float32
        frame = frame.astype(np.float32)
        
        # Normalize: (x - mean) / std
        frame = (frame - self.mean) / self.std
        
        return frame
    
    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array to PyTorch tensor.
        
        Args:
            frame: Normalized frame (HWC)
            
        Returns:
            Tensor in NCHW format
        """
        # HWC to CHW
        frame = frame.transpose(2, 0, 1)
        
        # Add batch dimension: CHW -> NCHW
        frame = np.expand_dims(frame, axis=0)
        
        # Convert to tensor
        tensor = torch.from_numpy(frame.copy())
        
        return tensor
    
    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor back to displayable image (for debugging).
        
        Args:
            tensor: Normalized tensor (NCHW or CHW)
            
        Returns:
            BGR image (uint8)
        """
        # Remove batch dim if present
        if tensor.ndim == 4:
            tensor = tensor[0]
        
        # Convert to numpy
        img = tensor.cpu().numpy()
        
        # CHW to HWC
        img = img.transpose(1, 2, 0)
        
        # Denormalize
        img = (img * self.std) + self.mean
        
        # Clip and convert to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img


def create_preprocessor(config: RTMDetConfig) -> RTMDetPreprocessor:
    """
    Factory function to create preprocessor.
    
    Args:
        config: RTMDet configuration
        
    Returns:
        Configured preprocessor instance
    """
    return RTMDetPreprocessor(config)
