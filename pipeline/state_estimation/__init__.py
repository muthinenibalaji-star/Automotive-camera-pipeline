
State Estimation Module

Provides deterministic, FSM-based state estimation for automotive lights.

Key Components:
- LightStateEstimator: Core FSM implementation
- StateManager: Multi-object state tracking
- StateDebugger: Visualization and validation tools

Design Principles:
- Deterministic behavior (no ML in state logic)
- Temporal consistency via sliding windows
- Explainability and traceability
- Production-ready for automotive validation
"""

from .light_state_estimator import (
    LightStateEstimator,
    LightState,
    StateEstimate,
    StateEstimatorConfig
)

from .state_manager import (
    StateManager,
    DetectionInput
)

from .state_debugger import StateDebugger

__all__ = [
    'LightStateEstimator',
    'LightState',
    'StateEstimate',
    'StateEstimatorConfig',
    'StateManager',
    'DetectionInput',
    'StateDebugger'
]

__version__ = '1.0.0'
__author__ = 'Automotive Perception Team'
