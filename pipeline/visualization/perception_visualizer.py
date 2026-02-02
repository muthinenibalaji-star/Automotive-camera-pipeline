"""
Visualization Module - Real-time Debug Overlays

Provides runtime visualization for perception debugging:
- Color-coded bounding boxes per state
- State labels and confidence display
- Non-intrusive overlay rendering
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from ..state_estimation import LightState, StateEstimate


class PerceptionVisualizer:
    """
    Real-time visualization overlay for perception debugging.
    
    Features:
    - Color-coded bounding boxes
    - State and confidence labels
    - Minimal performance impact
    - Configurable display options
    """
    
    # State-specific colors (BGR format for OpenCV)
    STATE_COLORS = {
        LightState.UNKNOWN: (128, 128, 128),  # Gray
        LightState.OFF: (235, 134, 46),       # Blue
        LightState.ON: (114, 59, 162),        # Magenta
        LightState.BLINK: (1, 143, 241)       # Orange
    }
    
    def __init__(
        self,
        show_labels: bool = True,
        show_confidence: bool = True,
        show_frequency: bool = True,
        box_thickness: int = 2,
        font_scale: float = 0.5,
        font_thickness: int = 1
    ):
        self.show_labels = show_labels
        self.show_confidence = show_confidence
        self.show_frequency = show_frequency
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw_detection(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],  # (x, y, w, h)
        state_estimate: StateEstimate,
        light_type: str,
        track_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Draw detection with state-based color coding.
        
        Args:
            frame: Input image (BGR)
            bbox: Bounding box (x, y, width, height)
            state_estimate: State estimation result
            light_type: Type of light (for label)
            track_id: Optional track ID to display
            
        Returns:
            Annotated frame
        """
        x, y, w, h = bbox
        
        # Get color based on state
        color = self.STATE_COLORS.get(state_estimate.state, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, self.box_thickness)
        
        if self.show_labels:
            # Prepare label text
            label_parts = []
            
            # Track ID
            if track_id is not None:
                label_parts.append(f"ID:{track_id}")
            
            # Light type
            label_parts.append(light_type)
            
            # State
            label_parts.append(state_estimate.state.value)
            
            # Confidence
            if self.show_confidence:
                label_parts.append(f"{state_estimate.confidence:.2f}")
            
            # Blink frequency
            if self.show_frequency and state_estimate.blink_frequency is not None:
                label_parts.append(f"{state_estimate.blink_frequency:.1f}Hz")
            
            label = " | ".join(label_parts)
            
            # Calculate label size and position
            (label_w, label_h), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.font_thickness
            )
            
            # Draw label background
            label_y = max(y - 5, label_h + 5)  # Position above bbox
            cv2.rectangle(
                frame,
                (x, label_y - label_h - 5),
                (x + label_w + 5, label_y + 5),
                color,
                -1  # Filled
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x + 2, label_y),
                self.font,
                self.font_scale,
                (255, 255, 255),  # White text
                self.font_thickness,
                cv2.LINE_AA
            )
        
        return frame
    
    def draw_multiple_detections(
        self,
        frame: np.ndarray,
        detections: Dict[Tuple[int, str], Tuple[Tuple[int, int, int, int], StateEstimate]]
    ) -> np.ndarray:
        """
        Draw multiple detections on frame.
        
        Args:
            frame: Input image
            detections: Dict mapping (track_id, light_type) to (bbox, state_estimate)
            
        Returns:
            Annotated frame
        """
        for (track_id, light_type), (bbox, state_estimate) in detections.items():
            frame = self.draw_detection(frame, bbox, state_estimate, light_type, track_id)
        
        return frame
    
    def draw_info_panel(
        self,
        frame: np.ndarray,
        frame_id: int,
        fps: float,
        latency_ms: float,
        num_tracks: int,
        state_distribution: Optional[Dict[str, int]] = None
    ) -> np.ndarray:
        """
        Draw information panel on frame.
        
        Args:
            frame: Input image
            frame_id: Current frame number
            fps: Processing FPS
            latency_ms: End-to-end latency
            num_tracks: Number of active tracks
            state_distribution: Optional state count distribution
            
        Returns:
            Frame with info panel
        """
        panel_height = 120
        panel_color = (40, 40, 40)  # Dark gray
        text_color = (255, 255, 255)  # White
        
        # Draw semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, panel_height), panel_color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw info text
        y_offset = 25
        line_spacing = 20
        
        info_lines = [
            f"Frame: {frame_id}",
            f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms",
            f"Active Tracks: {num_tracks}"
        ]
        
        if state_distribution:
            state_summary = " | ".join([f"{state}: {count}" for state, count in state_distribution.items() if count > 0])
            info_lines.append(f"States: {state_summary}")
        
        for i, line in enumerate(info_lines):
            y = y_offset + i * line_spacing
            cv2.putText(
                frame,
                line,
                (10, y),
                self.font,
                self.font_scale,
                text_color,
                self.font_thickness,
                cv2.LINE_AA
            )
        
        return frame
    
    def create_state_legend(self, width: int = 200, height: int = 150) -> np.ndarray:
        """
        Create a legend showing state colors.
        
        Returns:
            Small image with color legend
        """
        legend = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
        
        y_offset = 20
        box_size = 20
        spacing = 30
        
        # Title
        cv2.putText(
            legend,
            "State Legend",
            (10, 15),
            self.font,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
        
        # Draw each state
        for i, (state, color) in enumerate(self.STATE_COLORS.items()):
            y = y_offset + i * spacing
            
            # Color box
            cv2.rectangle(legend, (10, y), (10 + box_size, y + box_size), color, -1)
            cv2.rectangle(legend, (10, y), (10 + box_size, y + box_size), (0, 0, 0), 1)
            
            # State label
            cv2.putText(
                legend,
                state.value,
                (40, y + 15),
                self.font,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
        
        return legend
