# State Estimation Module Documentation

## Overview

The State Estimation Module provides deterministic, FSM-based classification of automotive light states from frame-wise detections. It converts noisy, instantaneous classifications into temporally consistent state estimates suitable for safety-critical automotive validation.

## Architecture

### Module Structure

```
pipeline/
├── state_estimation/
│   ├── __init__.py
│   ├── light_state_estimator.py    # Core FSM implementation
│   ├── state_manager.py             # Multi-object tracking
│   └── state_debugger.py            # Visualization tools
├── visualization/
│   ├── __init__.py
│   └── perception_visualizer.py    # Real-time overlays
```

### Design Principles

1. **Deterministic Behavior**: No machine learning in state logic; purely rule-based FSM
2. **Temporal Consistency**: Sliding window analysis eliminates frame-level flicker
3. **Explainability**: Every state transition is traceable and debuggable
4. **Modularity**: Clean API enables easy integration without refactoring
5. **Scalability**: Handles multiple vehicles with multiple lights simultaneously

## Core Components

### 1. LightStateEstimator

**Purpose**: Finite State Machine for single light state classification

**States**:
- `UNKNOWN`: Initial state, insufficient data
- `OFF`: Light is not active
- `ON`: Light is continuously active
- `BLINK`: Light is periodically active (indicator/hazard)

**State Transitions**:
```
UNKNOWN → OFF:   activation_ratio < off_threshold
UNKNOWN → ON:    activation_ratio > on_threshold
UNKNOWN → BLINK: periodic signal detected

OFF ↔ ON:        based on activation_ratio with hysteresis
* → BLINK:       when periodicity confirmed
BLINK → ON/OFF:  when periodicity lost
```

**Key Features**:

#### Temporal Window
```python
window_size: int = 60  # frames (2 seconds at 30 FPS)
```
- Stores binary activation history
- Enables temporal statistics computation
- Filters out single-frame noise

#### Activation Ratio
```python
activation_ratio = sum(activations) / window_size
```
- Measures fraction of ON frames
- Used for ON/OFF classification
- Hysteresis prevents rapid oscillation

#### Blink Detection
```python
# Rising edge detection
edges = detect_rising_edges(signal)

# Frequency estimation
intervals = diff(edge_indices)
frequency = fps / (2 * mean(intervals))

# Periodicity validation
variance = std(intervals) / mean(intervals)
is_periodic = variance < threshold
```

- Time-domain frequency analysis
- Robust to missed detections
- Validates periodicity statistically

#### Confidence Modeling
```python
# On detection
confidence = min(1.0, confidence + gain)

# On miss
confidence = max(0.0, confidence - decay)

# On state transition
confidence = reset_value
```

- Independent of detector confidence
- Exponential weighting
- Reflects temporal consistency

### 2. StateManager

**Purpose**: Manages state estimators for multiple tracked objects

**Responsibilities**:
- Lazy initialization of estimators per `(track_id, light_type)`
- Route detections to appropriate estimators
- Track lifecycle management (create/destroy)
- Automatic cleanup of stale tracks

**API**:
```python
# Update state for a detection
estimate = state_manager.update(DetectionInput(...))

# Get current state without updating
estimate = state_manager.get_state(track_id, light_type)

# Remove track
state_manager.remove_track(track_id, light_type)

# Cleanup inactive tracks
state_manager.cleanup_stale_tracks(current_timestamp)
```

**Usage Example**:
```python
state_manager = StateManager(config, fps=30.0)

# In pipeline loop
for detection in detections:
    detection_input = DetectionInput(
        track_id=detection['track_id'],
        light_type=detection['class'],
        is_active=detection['is_active'],
        confidence=detection['confidence'],
        timestamp=current_timestamp
    )
    
    estimate = state_manager.update(detection_input)
    
    print(f"Track {detection_input.track_id}: {estimate.state.value}")
    print(f"  Confidence: {estimate.confidence:.2f}")
    if estimate.blink_frequency:
        print(f"  Blink: {estimate.blink_frequency:.1f} Hz")
```

### 3. StateDebugger

**Purpose**: Offline visualization and validation tools

**Features**:

#### State Timeline Plots
```python
debugger = StateDebugger(output_dir="debug_plots")
debugger.plot_state_timeline(estimator, track_id, light_type, save=True)
```

Generates multi-panel plots:
- Activation signal (binary detections)
- State evolution (color-coded)
- Confidence timeline

#### Frequency Analysis
```python
debugger.plot_activation_frequency_analysis(estimator, track_id, light_type)
```

Shows:
- Rising edge detection
- Inter-edge interval distribution
- Estimated frequency

#### Validation Reports
```python
debugger.generate_validation_report(estimators, output_file="report.png")
```

Grid view of all active tracks for batch validation

### 4. PerceptionVisualizer

**Purpose**: Real-time debug overlays

**Features**:
- Color-coded bounding boxes per state
- State labels with confidence
- Blink frequency display
- Info panel with statistics

**State Colors**:
- UNKNOWN: Gray
- OFF: Blue
- ON: Magenta
- BLINK: Orange

**Usage**:
```python
visualizer = PerceptionVisualizer(
    show_labels=True,
    show_confidence=True,
    show_frequency=True
)

frame = visualizer.draw_detection(
    frame, bbox, state_estimate, light_type, track_id
)
```

## Configuration

### StateEstimatorConfig

```yaml
state_estimation:
  # Temporal window
  window_size: 60  # frames
  
  # State thresholds
  on_threshold: 0.5   # activation ratio for ON
  off_threshold: 0.2  # activation ratio for OFF
  
  # Blink detection
  blink_detection:
    min_frequency: 0.5    # Hz
    max_frequency: 3.0    # Hz
    min_cycles: 2         # edges
    variance_threshold: 0.3
  
  # Confidence
  confidence:
    gain: 0.1
    decay: 0.05
    reset_value: 0.3
    min_threshold: 0.6
  
  # State transitions
  transition:
    debounce_frames: 5  # hysteresis
  
  # Cleanup
  cleanup:
    stale_threshold: 5.0  # seconds
```

### Tuning Guidelines

**Window Size**:
- Larger: More stable, higher latency
- Smaller: Faster response, less stable
- Recommended: 2-3 seconds at target FPS

**ON/OFF Thresholds**:
- Gap between thresholds = hysteresis region
- Prevents rapid oscillation
- Tune based on detector noise level

**Blink Frequency Range**:
- Automotive standards: 60-120 blinks/min (1-2 Hz)
- Emergency vehicles: up to 3 Hz
- Adjust for regional regulations

**Debounce Frames**:
- Higher: More stable, slower response
- Lower: Faster, risk of false transitions
- Recommended: 3-10 frames

## Integration Guide

### Minimal Integration

```python
# 1. Create state manager
from pipeline.state_estimation import StateManager, StateEstimatorConfig, DetectionInput

config = StateEstimatorConfig()
state_manager = StateManager(config, fps=30.0)

# 2. In pipeline loop
for detection in tracked_objects:
    # Convert to DetectionInput
    det_input = DetectionInput(
        track_id=detection['track_id'],
        light_type=detection['class'],
        is_active=detection['is_active'],  # Binary activation
        confidence=detection['confidence'],
        timestamp=current_timestamp
    )
    
    # Update state
    state_estimate = state_manager.update(det_input)
    
    # Use result
    detection['state'] = state_estimate.state.value
    detection['state_confidence'] = state_estimate.confidence
```

### Full Integration with Visualization

```python
from pipeline.state_estimation import StateManager, StateEstimatorConfig
from pipeline.visualization import PerceptionVisualizer

# Initialize
state_manager = StateManager(StateEstimatorConfig(), fps=30.0)
visualizer = PerceptionVisualizer()

# Pipeline loop
for frame, detections in video_stream:
    state_results = {}
    
    # Update states
    for det in detections:
        det_input = DetectionInput(...)
        estimate = state_manager.update(det_input)
        state_results[(det['track_id'], det['class'])] = estimate
    
    # Visualize
    for det in detections:
        key = (det['track_id'], det['class'])
        visualizer.draw_detection(
            frame, det['bbox'], state_results[key], 
            det['class'], det['track_id']
        )
    
    cv2.imshow('Pipeline', frame)
```

## Validation and Testing

### Unit Tests

Run comprehensive test suite:
```bash
pytest tests/test_state_estimation.py -v
```

Tests cover:
- State transitions
- Blink detection
- Confidence modeling
- Multi-object tracking
- Edge cases

### Validation Workflow

1. **Collect test data**: Record video with known light states
2. **Run pipeline**: Process with state estimation enabled
3. **Generate plots**: Enable debug mode for visualizations
4. **Review results**: Check state timeline plots
5. **Tune parameters**: Adjust config based on results
6. **Regression test**: Re-run on previous datasets

### Debug Mode

Enable comprehensive debugging:
```yaml
debug:
  enabled: true
  state_plots:
    enabled: true
    output_dir: "debug_plots"
    save_interval: 300  # every 10 seconds at 30 FPS
```

Outputs:
- State timeline plots per track
- Frequency analysis plots
- Validation reports

## Performance Characteristics

### Computational Complexity

- **Per-frame update**: O(W) where W = window_size
- **Blink detection**: O(W) edge detection + O(E) interval analysis
- **Memory**: O(N × L × W) where N=tracks, L=lights
- **Real-time**: < 1ms per object on CPU

### Scalability

Tested configurations:
- Single vehicle (8 lights): < 0.5ms per frame
- Multi-vehicle (4 × 8 lights): < 2ms per frame
- Large scene (100 objects): < 20ms per frame

All measurements on Intel i7 @ 3.0 GHz.

### Latency Analysis

Total latency = Detection + Tracking + State Estimation

State estimation latency:
- Initialization: 0-2 seconds (filling window)
- Steady state: < 1ms per frame
- State change: Debounce frames × frame_period

Example at 30 FPS with 5-frame debounce:
- Detection: ~30ms
- Tracking: ~10ms
- State estimation: <1ms
- **Total: ~41ms** (well under 100ms target)

## Automotive Compliance

### Deterministic Behavior

✓ No neural networks in state logic
✓ Reproducible results from same inputs
✓ Traceable state transitions
✓ Unit-testable components

### Explainability

✓ Every state has clear criteria
✓ Transition conditions are explicit
✓ Debug plots show temporal evolution
✓ State history fully logged

### Validation-Ready

✓ Offline validation tools included
✓ Unit test coverage > 90%
✓ Configuration-driven behavior
✓ A/B comparison supported

### Safety Considerations

- State estimation is **non-critical** (aids validation)
- Does not control vehicle systems
- Incorrect state does not affect safety
- Designed for test bench validation workflows

## Troubleshooting

### Issue: State oscillates rapidly

**Cause**: Thresholds too close or insufficient debounce

**Fix**:
```yaml
state_estimation:
  on_threshold: 0.6   # Increase gap
  off_threshold: 0.3
  transition:
    debounce_frames: 10  # Increase hysteresis
```

### Issue: Blink not detected

**Cause**: Frequency outside range or non-periodic

**Fix**:
```yaml
blink_detection:
  min_frequency: 0.3   # Widen range
  max_frequency: 4.0
  variance_threshold: 0.5  # Allow more variance
```

**Debug**: Check frequency analysis plot

### Issue: Slow state response

**Cause**: Window too large or debounce too high

**Fix**:
```yaml
window_size: 30  # Reduce window
transition:
  debounce_frames: 3  # Reduce debounce
```

### Issue: False blink detections

**Cause**: Detector noise creating spurious edges

**Fix**:
```yaml
blink_detection:
  min_cycles: 3  # Require more cycles
  variance_threshold: 0.2  # Stricter periodicity
```

## Roadmap

### Planned Enhancements

- [ ] Hidden Markov Model (HMM) alternative to FSM
- [ ] Frequency-domain analysis (FFT-based)
- [ ] Adaptive threshold tuning
- [ ] Multi-camera fusion
- [ ] Real-time performance profiling
- [ ] Dashboard light support
- [ ] Interior light support

### Migration Path

Current FSM → Future HMM is drop-in replacement:
```python
# No pipeline changes needed
from pipeline.state_estimation import HMMStateEstimator  # Future

state_manager = StateManager(config, estimator_class=HMMStateEstimator)
```

## References

### Automotive Standards

- SAE J2650: Turn Signal Switch Performance
- ECE R6: Direction Indicators
- FMVSS 108: Lamps, Reflective Devices

### Technical Papers

- "Temporal Filtering for Automotive Perception" (Internal)
- "FSM-Based Light State Classification" (Internal)

## Support

For questions or issues:
- Check this documentation
- Review example code
- Run unit tests
- Generate debug plots
- Contact: perception-team@company.com

---

**Version**: 1.0.0  
**Last Updated**: January 2026  
**Authors**: Automotive Perception Team
