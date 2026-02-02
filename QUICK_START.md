# Quick Start Guide

Get up and running in under 5 minutes!

---

## Option 1: Automated Setup (Recommended)

**Right-click `setup.ps1` → "Run with PowerShell"**

This will:
1. Create virtual environment
2. Install all dependencies
3. Download model weights
4. Create data directories

Then run: `.\run.ps1 "data\input\your_video.mp4"`

---

## Option 2: Manual Setup

### Step 1: Create Virtual Environment
```powershell
cd Automotive-camera-pipeline
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
pip install -U openmim
mim install "mmcv>=2.0.0" "mmdet>=3.0.0"
```

### Step 3: Download Model Weights
```powershell
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest models
```

### Step 4: Run the Pipeline
```powershell
python src/main.py --config configs/pipeline_config.yaml --source "path/to/video.mp4"
```

---

## Test Videos

Place test videos in `data/input/` and run:
```powershell
.\run.ps1 "data\input\test_video.mp4"
```

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
Run: `mim install mmdet>=3.0.0`

### "CUDA out of memory"  
Edit `configs/pipeline_config.yaml` and set `detection.device: "cpu"`

### Camera not working
Use a video file instead: `--source "path/to/video.mp4"`

---

## Next Steps

- Edit `configs/pipeline_config.yaml` to customize detection settings
- See [README.md](README.md) for full documentation
- Check `examples/` for integration examples
