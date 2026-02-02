"""
RTMDet Model Loader

Responsibilities:
- Load MMDetection model from config and weights
- Move model to GPU
- Enable FP16 if configured
- Set model to eval mode

Design principle: Model selection is config-driven, not code-driven.
"""

import torch
from mmdet.apis import init_detector
from pathlib import Path
import logging
from typing import Optional

from .rtmdet_types import RTMDetConfig, RTMDetVariant


logger = logging.getLogger(__name__)


class RTMDetLoader:
    """
    Loads and configures RTMDet models.
    
    This class handles the one-time model loading operation.
    Models are loaded once and reused for all inferences.
    """
    
    def __init__(self, config: RTMDetConfig):
        """
        Initialize loader with configuration.
        
        Args:
            config: RTMDet configuration
        """
        self.config = config
        self.model = None
    
    def load(self) -> bool:
        """
        Load model from disk and prepare for inference.
        
        Returns:
            True if loading successful
            
        Raises:
            RuntimeError: If loading fails
        """
        try:
            # Validate config
            self.config.validate()
            
            # Check paths exist
            self._validate_paths()
            
            # Log loading info
            logger.info(f"Loading RTMDet-{self.config.variant.value.upper()}")
            logger.info(f"  Config: {self.config.config_path}")
            logger.info(f"  Weights: {self.config.weights_path}")
            logger.info(f"  Device: {self.config.device}")
            logger.info(f"  FP16: {self.config.fp16}")
            
            # Load model using MMDetection
            self.model = init_detector(
                config=self.config.config_path,
                checkpoint=self.config.weights_path,
                device=self.config.device
            )
            
            # Set to eval mode
            self.model.eval()
            
            # Enable FP16 if configured
            if self.config.fp16:
                self._enable_fp16()
            
            logger.info("✓ Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _validate_paths(self):
        """Validate that model files exist"""
        config_path = Path(self.config.config_path)
        weights_path = Path(self.config.weights_path)
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config.config_path}"
            )
        
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights file not found: {self.config.weights_path}"
            )
    
    def _enable_fp16(self):
        """
        Enable FP16 (half precision) inference.
        
        This provides ~2x speedup on modern GPUs with minimal accuracy loss.
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping FP16")
            return
        
        try:
            # Convert model to FP16
            self.model = self.model.half()
            logger.info("✓ FP16 enabled")
            
        except Exception as e:
            logger.warning(f"Failed to enable FP16: {e}")
            # Continue with FP32
    
    def get_model(self):
        """
        Get the loaded model.
        
        Returns:
            Loaded MMDetection model
            
        Raises:
            RuntimeError: If model not loaded yet
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        return self.model
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        if self.model is None:
            return {'loaded': False}
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'loaded': True,
            'variant': self.config.variant.value,
            'device': self.config.device,
            'fp16': self.config.fp16,
            'num_parameters': num_params,
            'num_trainable_parameters': num_trainable,
            'config_path': self.config.config_path,
            'weights_path': self.config.weights_path
        }
    
    def unload(self):
        """Unload model and free GPU memory"""
        if self.model is not None:
            del self.model
            self.model = None
            
            # Force GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Model unloaded")


def create_loader(config: RTMDetConfig) -> RTMDetLoader:
    """
    Factory function to create model loader.
    
    Args:
        config: RTMDet configuration
        
    Returns:
        Configured RTMDetLoader instance
    """
    return RTMDetLoader(config)
