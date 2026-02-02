"""
RTMDet Postprocessing

Responsibilities:
- Confidence thresholding
- Class filtering (lights only)
- NMS (Non-Maximum Suppression)
- Convert MMDetection output → pipeline format
- Scale bboxes back to original coordinates

CRITICAL: Never pass raw MMDetection outputs upstream.
This is where we enforce a stable output schema.
"""

import numpy as np
import torch
from typing import List, Tuple
import logging

from ..detector_base import Detection
from .rtmdet_types import RTMDetConfig, RTMDetInferenceResult, PreprocessResult


logger = logging.getLogger(__name__)


class RTMDetPostprocessor:
    """
    Postprocesses RTMDet outputs into standardized detections.
    
    This is the critical layer that ensures stable output schema.
    """
    
    def __init__(self, config: RTMDetConfig):
        """
        Initialize postprocessor.
        
        Args:
            config: RTMDet configuration
        """
        self.config = config
        self.score_threshold = config.score_threshold
        self.nms_threshold = config.nms_threshold
        self.max_detections = config.max_detections
        self.class_names = config.class_names
    
    def postprocess(
        self,
        raw_output: any,  # MMDetection output
        preprocess_result: PreprocessResult,
        inference_time_ms: float
    ) -> RTMDetInferenceResult:
        """
        Convert raw MMDetection output to internal format.
        
        Args:
            raw_output: Raw output from MMDetection model
            preprocess_result: Preprocessing metadata
            inference_time_ms: Inference latency
            
        Returns:
            RTMDetInferenceResult with filtered and scaled detections
        """
        # Extract detections from MMDetection output format
        bboxes, scores, labels = self._extract_mmdet_output(raw_output)
        
        # Filter by confidence threshold
        bboxes, scores, labels = self._filter_by_confidence(bboxes, scores, labels)
        
        # Apply NMS
        bboxes, scores, labels = self._apply_nms(bboxes, scores, labels)
        
        # Limit number of detections
        bboxes, scores, labels = self._limit_detections(bboxes, scores, labels)
        
        # Scale bboxes back to original image coordinates
        bboxes = self._scale_bboxes(bboxes, preprocess_result)
        
        return RTMDetInferenceResult(
            bboxes=bboxes,
            scores=scores,
            labels=labels,
            inference_time_ms=inference_time_ms
        )
    
    def convert_to_detections(
        self,
        result: RTMDetInferenceResult
    ) -> List[Detection]:
        """
        Convert RTMDetInferenceResult to standardized Detection objects.
        
        This is the final conversion to the pipeline's output format.
        
        Args:
            result: RTMDet inference result
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        for bbox, score, label in zip(result.bboxes, result.scores, result.labels):
            # Get class name
            class_name = self._get_class_name(label)
            
            # Create Detection object
            detection = Detection(
                bbox=tuple(bbox.tolist()),  # (x1, y1, x2, y2)
                class_id=int(label),
                class_name=class_name,
                confidence=float(score)
            )
            
            detections.append(detection)
        
        return detections
    
    def _extract_mmdet_output(
        self,
        raw_output: any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract bboxes, scores, and labels from MMDetection output.
        
        MMDetection returns results in a specific format that varies by version.
        This function handles the extraction robustly.
        
        Args:
            raw_output: Raw MMDetection output
            
        Returns:
            Tuple of (bboxes, scores, labels)
        """
        try:
            # MMDetection 3.x format: DetDataSample
            if hasattr(raw_output, 'pred_instances'):
                pred_instances = raw_output.pred_instances
                bboxes = pred_instances.bboxes.cpu().numpy()  # (N, 4)
                scores = pred_instances.scores.cpu().numpy()  # (N,)
                labels = pred_instances.labels.cpu().numpy()  # (N,)
            
            # Legacy format: list of arrays
            elif isinstance(raw_output, list):
                # Typically: [bboxes_per_class] where bboxes_per_class is (N, 5) [x1,y1,x2,y2,score]
                all_bboxes = []
                all_scores = []
                all_labels = []
                
                for class_id, class_dets in enumerate(raw_output):
                    if len(class_dets) > 0:
                        all_bboxes.append(class_dets[:, :4])
                        all_scores.append(class_dets[:, 4])
                        all_labels.append(np.full(len(class_dets), class_id))
                
                if len(all_bboxes) > 0:
                    bboxes = np.vstack(all_bboxes)
                    scores = np.concatenate(all_scores)
                    labels = np.concatenate(all_labels)
                else:
                    bboxes = np.empty((0, 4))
                    scores = np.empty(0)
                    labels = np.empty(0, dtype=int)
            
            else:
                # Unknown format
                logger.warning(f"Unknown MMDetection output format: {type(raw_output)}")
                bboxes = np.empty((0, 4))
                scores = np.empty(0)
                labels = np.empty(0, dtype=int)
            
            return bboxes, scores, labels
            
        except Exception as e:
            logger.error(f"Failed to extract MMDetection output: {e}")
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    
    def _filter_by_confidence(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter detections by confidence threshold"""
        if len(scores) == 0:
            return bboxes, scores, labels
        
        mask = scores >= self.score_threshold
        return bboxes[mask], scores[mask], labels[mask]
    
    def _apply_nms(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.
        
        NMS is applied per class to avoid suppressing different light types.
        """
        if len(bboxes) == 0:
            return bboxes, scores, labels
        
        # Convert to torch for torchvision NMS
        bboxes_torch = torch.from_numpy(bboxes).float()
        scores_torch = torch.from_numpy(scores).float()
        labels_torch = torch.from_numpy(labels).long()
        
        # Apply NMS per class
        keep_indices = []
        
        for class_id in torch.unique(labels_torch):
            class_mask = labels_torch == class_id
            class_bboxes = bboxes_torch[class_mask]
            class_scores = scores_torch[class_mask]
            
            # Apply NMS for this class
            keep = torch.ops.torchvision.nms(
                class_bboxes,
                class_scores,
                self.nms_threshold
            )
            
            # Get original indices
            class_indices = torch.where(class_mask)[0]
            keep_indices.extend(class_indices[keep].tolist())
        
        # Sort by score
        keep_indices = sorted(keep_indices, key=lambda i: scores[i], reverse=True)
        
        return bboxes[keep_indices], scores[keep_indices], labels[keep_indices]
    
    def _limit_detections(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Limit number of detections to max_detections"""
        if len(bboxes) <= self.max_detections:
            return bboxes, scores, labels
        
        # Keep top-k by score
        return bboxes[:self.max_detections], scores[:self.max_detections], labels[:self.max_detections]
    
    def _scale_bboxes(
        self,
        bboxes: np.ndarray,
        preprocess_result: PreprocessResult
    ) -> np.ndarray:
        """
        Scale bboxes from model input size back to original image size.
        
        Args:
            bboxes: Bboxes in model input coordinates
            preprocess_result: Contains scale factor
            
        Returns:
            Bboxes in original image coordinates
        """
        if len(bboxes) == 0:
            return bboxes
        
        scale_w, scale_h = preprocess_result.scale_factor
        
        # Scale back
        bboxes[:, [0, 2]] /= scale_w  # x coordinates
        bboxes[:, [1, 3]] /= scale_h  # y coordinates
        
        # Clip to image bounds
        h, w, _ = preprocess_result.original_shape
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, w)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, h)
        
        return bboxes
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID"""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        else:
            return f"unknown_{class_id}"


def create_postprocessor(config: RTMDetConfig) -> RTMDetPostprocessor:
    """
    Factory function to create postprocessor.
    
    Args:
        config: RTMDet configuration
        
    Returns:
        Configured postprocessor instance
    """
    return RTMDetPostprocessor(config)
