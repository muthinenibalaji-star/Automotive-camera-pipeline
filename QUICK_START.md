# 🚀 Quick Reference Guide - Automotive Camera Pipeline

## Team Member Onboarding (5 Minutes)

### First Time Setup
```powershell
# 1. Open PowerShell as Administrator
# 2. Navigate to project folder
cd C:\path\to\automotive-camera-pipeline

# 3. Run setup (one-time only)
.\setup-windows.ps1

# Wait ~10-15 minutes for Docker build
```

### Daily Usage

#### Start Pipeline
```powershell
docker-compose up
```

#### Stop Pipeline
```powershell
# Press Ctrl+C, then:
docker-compose down
```

#### Run in Background
```powershell
docker-compose up -d        # Start
docker-compose logs -f      # View logs
docker-compose down         # Stop
```

## Common Tasks

### Change Camera Source
Edit `configs/pipeline_config.yaml`:
```yaml
cameras:
  camera_0:
    source_type: "usb"    # or "dataset" or "industrial"
    source: 0             # device index or video path
```
Then restart: `docker-compose restart`

### Process Video File
```yaml
# In pipeline_config.yaml
cameras:
  camera_0:
    source_type: "dataset"
    source: "/app/data/input/my_video.mp4"
```
Place video in `data/input/` folder

### Add Model Weights
1. Copy `.pth` file to `models/` folder
2. Update config:
```yaml
detection:
  checkpoint_path: "models/my_model.pth"
```

### View Results
Results saved to: `data/output/results_TIMESTAMP.json`

## Troubleshooting

### "Docker not running"
→ Start Docker Desktop, wait for whale icon

### "GPU not found"
→ Update NVIDIA drivers, restart Docker Desktop

### "Camera not accessible"
→ Close other apps using camera, check device index

### Rebuild Container
```powershell
docker-compose build --no-cache
```

## Development

### Edit Code
Just edit files in `src/` - changes apply immediately
Restart container: `docker-compose restart`

### Run Tests
```powershell
docker-compose run automotive-pipeline pytest tests/
```

### Use Jupyter
```powershell
docker-compose --profile dev up jupyter-dev
# Open: http://localhost:8888
```

## File Locations

| Purpose | Location | Access |
|---------|----------|--------|
| Input videos | `data/input/` | Read |
| Results (JSON) | `data/output/` | Write |
| Model weights | `models/` | Read |
| Logs | `logs/` | Write |
| Config | `configs/` | Edit |
| Source code | `src/` | Edit |

## Configuration Cheat Sheet

### System
```yaml
system:
  num_cameras: 1          # Number of cameras
  target_fps: 30          # Target frame rate
  max_latency_ms: 100     # Alert if exceeded
```

### Detection
```yaml
detection:
  confidence_threshold: 0.5    # Detection confidence
  fp16: true                   # Faster inference
```

### State Estimation
```yaml
state_estimation:
  on_threshold: 120       # Light ON brightness
  off_threshold: 50       # Light OFF brightness
  blink_frequency_hz: [1.0, 2.0]  # Expected blink rate
```

## Performance Tips

✅ Enable FP16: `fp16: true` in config
✅ Reduce resolution if slow
✅ Close other GPU applications
✅ Use SSD for video files
✅ Check GPU usage: `nvidia-smi -l 1`

## Getting Help

1. Check logs: `logs/pipeline.log`
2. View Docker logs: `docker-compose logs`
3. Test GPU: `nvidia-smi`
4. Contact team lead
5. Check full README.md

## Important Commands

```powershell
# Essential
docker-compose up              # Start pipeline
docker-compose down            # Stop pipeline
docker-compose restart         # Restart pipeline
docker-compose logs -f         # View logs

# Development  
docker-compose build           # Rebuild container
docker-compose ps              # Check status
docker-compose exec automotive-pipeline bash  # Enter container

# Cleanup
docker-compose down -v         # Remove volumes
docker system prune            # Clean Docker cache
```

## Success Checklist

- [ ] Docker Desktop running
- [ ] GPU detected (`nvidia-smi` works)
- [ ] Config file edited (`configs/pipeline_config.yaml`)
- [ ] Model weights in `models/` folder
- [ ] Camera accessible or video in `data/input/`
- [ ] Pipeline starts without errors
- [ ] Results appear in `data/output/`

---
**Need more details?** See full `README.md`
