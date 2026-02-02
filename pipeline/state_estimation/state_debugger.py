"""
State Debugger - Visualization and Analysis Tools

Provides debugging capabilities for state estimation:
- Time-series plots of activation signals and FSM states
- State transition analysis
- Offline validation tools
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from .light_state_estimator import LightState, LightStateEstimator


class StateDebugger:
    """
    Debug and visualization tools for state estimation.
    
    Features:
    - Multi-panel plots (activation signal + state timeline)
    - Color-coded state visualization
    - State transition markers
    - Exportable plots for validation reports
    """
    
    # State color mapping for visualization
    STATE_COLORS = {
        LightState.UNKNOWN: '#808080',  # Gray
        LightState.OFF: '#2E86AB',      # Blue
        LightState.ON: '#A23B72',       # Magenta
        LightState.BLINK: '#F18F01'     # Orange
    }
    
    def __init__(self, output_dir: str = "debug_plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_state_timeline(
        self,
        estimator: LightStateEstimator,
        track_id: int,
        light_type: str,
        save: bool = True
    ) -> Optional[Figure]:
        """
        Create comprehensive state timeline visualization.
        
        Plots:
        1. Activation signal (binary detections over time)
        2. State timeline with color coding
        3. Confidence evolution
        
        Args:
            estimator: The state estimator to visualize
            track_id: Track identifier
            light_type: Light type (for labeling)
            save: Whether to save plot to disk
            
        Returns:
            matplotlib Figure if not saved, else None
        """
        if len(estimator.activation_buffer) == 0:
            return None
        
        # Extract data from estimator
        timestamps = np.array(list(estimator.timestamp_buffer))
        activations = np.array(list(estimator.activation_buffer))
        state_history = estimator.get_state_history()
        
        # Normalize timestamps to start at 0
        if len(timestamps) > 0:
            timestamps = timestamps - timestamps[0]
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f'State Timeline - Track {track_id} - {light_type}', fontsize=14, fontweight='bold')
        
        # Plot 1: Activation Signal
        ax1 = axes[0]
        ax1.plot(timestamps, activations, 'o-', markersize=3, linewidth=1, color='#1E88E5')
        ax1.fill_between(timestamps, 0, activations, alpha=0.3, color='#1E88E5')
        ax1.set_ylabel('Activation', fontweight='bold')
        ax1.set_ylim(-0.1, 1.1)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Binary Activation Signal', fontsize=11)
        
        # Plot 2: State Timeline
        ax2 = axes[1]
        self._plot_state_segments(ax2, state_history, timestamps)
        ax2.set_ylabel('State', fontweight='bold')
        ax2.set_yticks([0, 1, 2, 3])
        ax2.set_yticklabels(['UNKNOWN', 'OFF', 'ON', 'BLINK'])
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_title('FSM State Evolution', fontsize=11)
        
        # Plot 3: Confidence
        ax3 = axes[2]
        if len(state_history) > 0:
            state_timestamps = [h[0] - timestamps[0] for h in state_history]
            confidences = [h[2] for h in state_history]
            ax3.plot(state_timestamps, confidences, '-', linewidth=2, color='#4CAF50')
            ax3.fill_between(state_timestamps, 0, confidences, alpha=0.3, color='#4CAF50')
        ax3.set_ylabel('Confidence', fontweight='bold')
        ax3.set_xlabel('Time (seconds)', fontweight='bold')
        ax3.set_ylim(0, 1.1)
        ax3.axhline(y=0.6, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Min Threshold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('State Confidence', fontsize=11)
        
        plt.tight_layout()
        
        if save:
            filename = f"state_timeline_track{track_id}_{light_type}.png"
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return None
        else:
            return fig
    
    def _plot_state_segments(self, ax, state_history: List[Tuple[float, str, float]], timestamps: np.ndarray):
        """Plot colored state segments on timeline"""
        if len(state_history) == 0:
            return
        
        # Map state strings to numeric values
        state_map = {
            'UNKNOWN': 0,
            'OFF': 1,
            'ON': 2,
            'BLINK': 3
        }
        
        # Normalize timestamps
        start_time = timestamps[0] if len(timestamps) > 0 else 0
        
        # Plot state segments
        prev_time = state_history[0][0] - start_time
        prev_state = state_history[0][1]
        
        for i in range(1, len(state_history)):
            curr_time = state_history[i][0] - start_time
            curr_state = state_history[i][1]
            
            if curr_state != prev_state:
                # Plot segment
                state_value = state_map.get(prev_state, 0)
                color = self.STATE_COLORS.get(LightState(prev_state), '#808080')
                ax.plot([prev_time, curr_time], [state_value, state_value], 
                       linewidth=8, color=color, solid_capstyle='butt')
                
                # Mark transition
                ax.axvline(x=curr_time, color='black', linestyle=':', alpha=0.5, linewidth=1)
                
                prev_time = curr_time
                prev_state = curr_state
        
        # Plot final segment
        if len(timestamps) > 0:
            final_time = timestamps[-1] - start_time
            state_value = state_map.get(prev_state, 0)
            color = self.STATE_COLORS.get(LightState(prev_state), '#808080')
            ax.plot([prev_time, final_time], [state_value, state_value],
                   linewidth=8, color=color, solid_capstyle='butt')
    
    def plot_activation_frequency_analysis(
        self,
        estimator: LightStateEstimator,
        track_id: int,
        light_type: str,
        save: bool = True
    ) -> Optional[Figure]:
        """
        Analyze and visualize blink frequency detection.
        
        Shows:
        - Activation signal with detected edges
        - Inter-edge interval histogram
        - Estimated frequency
        """
        if len(estimator.activation_buffer) < 10:
            return None
        
        signal = np.array(list(estimator.activation_buffer))
        timestamps = np.array(list(estimator.timestamp_buffer))
        
        # Detect edges
        edges = estimator._detect_rising_edges(signal)
        edge_indices = np.where(edges)[0]
        
        if len(edge_indices) < 2:
            return None
        
        # Compute intervals
        intervals = np.diff(edge_indices)
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        fig.suptitle(f'Blink Frequency Analysis - Track {track_id} - {light_type}', 
                    fontsize=14, fontweight='bold')
        
        # Plot 1: Signal with edges
        ax1 = axes[0]
        ax1.plot(signal, 'o-', markersize=3, linewidth=1, color='#1E88E5', label='Activation')
        ax1.scatter(edge_indices, signal[edge_indices], color='red', s=100, 
                   marker='^', zorder=5, label='Rising Edges')
        ax1.set_ylabel('Activation', fontweight='bold')
        ax1.set_xlabel('Frame', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Edge Detection', fontsize=11)
        
        # Plot 2: Interval distribution
        ax2 = axes[1]
        ax2.hist(intervals, bins=min(20, len(intervals)), color='#4CAF50', alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.mean(intervals), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(intervals):.1f} frames')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_xlabel('Inter-Edge Interval (frames)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Interval Distribution', fontsize=11)
        
        # Add frequency estimate
        freq = estimator._estimate_blink_frequency()
        if freq is not None:
            fig.text(0.5, 0.02, f'Estimated Frequency: {freq:.2f} Hz', 
                    ha='center', fontsize=12, fontweight='bold', color='#F18F01')
        
        plt.tight_layout()
        
        if save:
            filename = f"frequency_analysis_track{track_id}_{light_type}.png"
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return None
        else:
            return fig
    
    def generate_validation_report(
        self,
        estimators: Dict[Tuple[int, str], LightStateEstimator],
        output_file: str = "validation_report.png"
    ):
        """
        Generate multi-track validation report.
        
        Creates a grid of state timelines for all tracked objects.
        """
        if len(estimators) == 0:
            return
        
        num_tracks = len(estimators)
        cols = min(3, num_tracks)
        rows = (num_tracks + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        if num_tracks == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        fig.suptitle('Multi-Track State Validation Report', fontsize=16, fontweight='bold')
        
        for idx, ((track_id, light_type), estimator) in enumerate(estimators.items()):
            ax = axes[idx]
            
            if len(estimator.activation_buffer) > 0:
                timestamps = np.array(list(estimator.timestamp_buffer))
                timestamps = timestamps - timestamps[0]
                activations = np.array(list(estimator.activation_buffer))
                
                ax.plot(timestamps, activations, 'o-', markersize=2, linewidth=1)
                ax.set_title(f'Track {track_id} - {light_type}', fontsize=10)
                ax.set_xlabel('Time (s)', fontsize=8)
                ax.set_ylabel('Activation', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # Add state annotation
                state_text = f"State: {estimator.current_state.value}\nConf: {estimator.confidence:.2f}"
                ax.text(0.02, 0.98, state_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide unused subplots
        for idx in range(num_tracks, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        filepath = self.output_dir / output_file
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
