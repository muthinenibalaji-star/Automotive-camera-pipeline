# Automotive Camera Pipeline

Real-time automotive light state detection and classification system using RTMDet and FSM-based state estimation.

## Features

- **Object Detection** — RTMDet-based light detection (brake lights, indicators, etc.)
- **State Estimation** — Deterministic FSM classifies lights as OFF, ON, or BLINK
- **Real-Time Visualization** — Color-coded bounding boxes with state labels
- **Blink Detection** — Frequency analysis for turn signal identification

---

## Quick Start (Windows)

### 1. Clone and Setup
```powershell
# Clone the repository
git clone https://github.com/muthinenibalaji-star/Automotive-camera-pipeline.git
cd Automotive-camera-pipeline

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install MMDetection (required for detection)
pip install -U openmim
mim install mmcv>=2.0.0 mmdet>=3.0.0
```

### 2. Download Model Weights
```powershell
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest models
```

### 3. Run the Pipeline

**With a video file:**
```powershell
python src/main.py --config configs/pipeline_config.yaml --source "path/to/video.mp4"
```

**With webcam:**
```powershell
python src/main.py --config configs/pipeline_config.yaml --source 0
```

> **Tip**: Press `Q` to quit the visualization window.

---

## Project Structure

```
Automotive-camera-pipeline/
├── src/
│   └── main.py                    # Main entry point
├── pipeline/
│   ├── detection/                 # RTMDet detector
│   ├── state_estimation/          # FSM-based state classification
│   └── visualization/             # Real-time overlays
├── configs/
│   └── pipeline_config.yaml       # Main configuration
├── models/                        # Model weights (downloaded)
├── data/
│   ├── input/                     # Place test videos here
│   └── output/                    # Results saved here
└── examples/
    └── integrated_pipeline_example.py
```

---

## Configuration

Edit `configs/pipeline_config.yaml` to customize:

| Setting | Description | Default |
|---------|-------------|---------|
| `detection.device` | `cuda` or `cpu` | `cpu` |
| `detection.thresholds.score` | Confidence threshold | `0.5` |
| `cameras.camera_0.source` | Video source | `0` (webcam) |
| `state_estimation.on_threshold` | ON state threshold | `120` |

---

## Output States

| State | Color | Meaning |
|-------|-------|---------|
| **UNKNOWN** | Gray | Insufficient data |
| **OFF** | Blue | Light is off |
| **ON** | Magenta | Light is on continuously |
| **BLINK** | Orange | Periodic blinking detected |

---

## Troubleshooting

### "No module named 'cv2'"
```powershell
pip install opencv-python
```

### "No module named 'mmdet'"
```powershell
pip install -U openmim
mim install mmcv>=2.0.0 mmdet>=3.0.0
```

### CUDA not available
Set `detection.device: "cpu"` in `configs/pipeline_config.yaml`

### Camera not found
Check your camera is connected, or use a video file with `--source "path/to/video.mp4"`

---

## Training on Custom Data

Want to detect your own objects? See our **[Training Guide](docs/TRAINING_GUIDE.md)** for step-by-step instructions on:
1. Collecting and annotating data
2. Configuring MMDetection
3. Fine-tuning RTMDet on your dataset

---

## Requirements

- Python 3.9+
- Windows 10/11 or Linux
- (Optional) NVIDIA GPU with CUDA for faster inference

---

## License

MIT License - See [LICENSE](LICENSE)
