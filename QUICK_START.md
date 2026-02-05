# Quick Start Guide

Get up and running in under 5 minutes!

---

## One-Click Setup (Recommended)

**Double-click `setup.bat`** or run in PowerShell:

```powershell
.\setup.bat
```

This will:
1. ✅ Create virtual environment
2. ✅ Install all Python dependencies
3. ✅ Install MMDetection ecosystem
4. ✅ Download RTMDet model weights
5. ✅ Create all required directories

---

## Running the Pipeline

### Activate Environment
```powershell
.venv\Scripts\activate
```

### Inference (Pre-trained Model)
```powershell
# With webcam
python src\main.py --config configs\pipeline_config.yaml --source 0

# With video file
python src\main.py --config configs\pipeline_config.yaml --source "data\input\your_video.mp4"
```

---

## Training on Custom Data

### 1. Prepare Dataset
Place COCO-format dataset in `data/vehicle_lights/`:
```
data/vehicle_lights/
├── annotations/train.json
├── annotations/val.json
├── train/  (images)
└── val/    (images)
```

### 2. Validate Dataset
```powershell
python tools\dataset_sanity_check.py --ann-file data\vehicle_lights\annotations\train.json --img-dir data\vehicle_lights\train\
```

### 3. Start Training
```powershell
python tools\train.py configs\vehicle_lights\rtmdet_m_vehicle_lights.py
```

See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for detailed instructions.

---

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `P` | Pause/Resume |
| `S` | Save screenshot |

---

## Common Issues

### "ModuleNotFoundError: No module named 'mmdet'"
```powershell
pip install -U openmim
mim install "mmcv>=2.0.0" "mmdet>=3.0.0"
```

### "CUDA out of memory"  
Edit `configs/pipeline_config.yaml` and set `detection.device: "cpu"`

---

## Next Steps

- Edit `configs/pipeline_config.yaml` to customize settings
- See [README.md](README.md) for full documentation
- Check [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for training

