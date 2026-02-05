# Automotive Camera Pipeline

Real-time vehicle lights detection and state classification using **RTMDet-m** and FSM-based state estimation.

## Features

- **Object Detection** — RTMDet-based light detection (12 vehicle light classes)
- **State Estimation** — Deterministic FSM classifies lights as OFF, ON, or BLINK
- **Real-Time Visualization** — Color-coded bounding boxes with state labels
- **Custom Training** — Full training pipeline for your own dataset

---

## One-Click Setup (Windows)

```powershell
# Clone the repository
git clone https://github.com/your-username/Automotive-camera-pipeline.git
cd Automotive-camera-pipeline

# Run one-click setup (creates venv, installs everything)
setup.bat
```

That's it! The setup script will:
1. Create virtual environment
2. Install all Python dependencies
3. Install MMDetection ecosystem
4. Download RTMDet model weights
5. Create required directories

---

## Quick Start

### Run Inference (Pre-trained Model)

```powershell
# Activate virtual environment
.venv\Scripts\activate

# With webcam
python src\main.py --config configs\pipeline_config.yaml --source 0

# With video file
python src\main.py --config configs\pipeline_config.yaml --source "path\to\video.mp4"
```

> **Tip**: Press `Q` to quit the visualization window.

---

## Train on Your Custom Data

This repository includes a complete **RTMDet-m training pipeline** for 12 vehicle light classes.

### Prepare Dataset

Place your COCO-format dataset in `data/vehicle_lights/`:
```
data/vehicle_lights/
├── annotations/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── train/        # Training images
├── val/          # Validation images
└── test/         # Test images
```

### Validate Dataset

```powershell
# Sanity check (COCO structure, bbox validity, class distribution)
python tools\dataset_sanity_check.py --ann-file data\vehicle_lights\annotations\train.json --img-dir data\vehicle_lights\train\

# Category mapping validation
python tools\category_mapping_validator.py --config configs\vehicle_lights\rtmdet_m_vehicle_lights.py --ann-file data\vehicle_lights\annotations\train.json

# Visualize 200 random samples
python tools\visualize_samples.py --ann-file data\vehicle_lights\annotations\train.json --img-dir data\vehicle_lights\train\ --output-dir visualizations\ --num-samples 200
```

### Start Training

```powershell
python tools\train.py configs\vehicle_lights\rtmdet_m_vehicle_lights.py
```

The training script automatically runs pre-flight validation checks.

### Evaluate Trained Model

```powershell
python tools\test.py configs\vehicle_lights\rtmdet_m_vehicle_lights.py work_dirs\rtmdet_m_vehicle_lights\epoch_300.pth
```

See **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** for detailed training instructions.

---

## Project Structure

```
Automotive-camera-pipeline/
├── setup.bat                      # One-click Windows setup
├── src/main.py                    # Main inference entry point
├── configs/
│   ├── pipeline_config.yaml       # Runtime configuration
│   └── vehicle_lights/            # Training configs
│       └── rtmdet_m_vehicle_lights.py
├── tools/                         # Training & validation tools
│   ├── train.py                   # Training script
│   ├── test.py                    # Evaluation script
│   ├── dataset_sanity_check.py    # Dataset validator
│   ├── category_mapping_validator.py
│   └── visualize_samples.py
├── pipeline/                      # Core modules
│   ├── detection/                 # RTMDet detector
│   ├── state_estimation/          # FSM state classifier
│   └── visualization/             # Overlays
├── data/
│   ├── vehicle_lights/            # Training dataset (COCO format)
│   ├── input/                     # Test videos
│   └── output/                    # Results
├── models/                        # Model weights
├── work_dirs/                     # Training outputs
└── docs/                          # Documentation
```

---

## Vehicle Light Classes (12 Total)

```
front_headlight_left    front_headlight_right
front_indicator_left    front_indicator_right
front_all_weather_left  front_all_weather_right
rear_brake_left         rear_brake_right
rear_indicator_left     rear_indicator_right
rear_tailgate_left      rear_tailgate_right
```

---

## Configuration

Edit `configs/pipeline_config.yaml`:

| Setting | Description | Default |
|---------|-------------|---------|
| `detection.device` | `cuda` or `cpu` | `cpu` |
| `detection.thresholds.score` | Confidence threshold | `0.5` |
| `cameras.camera_0.source` | Video source | `0` (webcam) |

---

## Troubleshooting

### "No module named 'mmdet'"
```powershell
pip install -U openmim
mim install "mmcv>=2.0.0" "mmdet>=3.0.0"
```

### CUDA not available
Set `detection.device: "cpu"` in `configs/pipeline_config.yaml`

### Training sanity check fails
- Ensure your COCO JSON has correct `images`, `annotations`, `categories` keys
- Verify category IDs match the class order (1-12)
- Check image files exist in the correct directories

---

## Requirements

- **Python**: 3.9+
- **OS**: Windows 10/11 or Linux
- **GPU**: NVIDIA with CUDA (recommended for training)

---

## License

MIT License - See [LICENSE](LICENSE)

