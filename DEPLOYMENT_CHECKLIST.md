# 📦 Team Deployment Checklist

## For System Administrator

### One-Time Infrastructure Setup

- [ ] **Install Docker Desktop** on all team workstations
  - Download: https://www.docker.com/products/docker-desktop
  - Enable WSL2 backend
  - Allocate 12GB+ RAM in Docker settings

- [ ] **Verify NVIDIA Drivers** on GPU workstations
  - Visit: https://www.nvidia.com/Download/index.aspx
  - Install latest drivers for RTX series
  - Restart computers after installation

- [ ] **Install NVIDIA Container Toolkit** (WSL2)
  ```bash
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt-get update && sudo apt-get install -y nvidia-docker2
  ```

- [ ] **Share Project Repository**
  - Upload to shared drive or Git repository
  - Set appropriate permissions

- [ ] **Distribute Model Weights**
  - Copy `.pth` files to shared location
  - Document model versions

## For Each Team Member

### Initial Setup (15-20 minutes)

- [ ] **Copy project folder** to local machine
  - Location: `C:\Projects\automotive-camera-pipeline\`

- [ ] **Open PowerShell as Administrator**
  - `Win + X` → "Windows PowerShell (Admin)"

- [ ] **Navigate to project**
  ```powershell
  cd C:\Projects\automotive-camera-pipeline
  ```

- [ ] **Run setup script**
  ```powershell
  .\setup-windows.ps1
  ```
  - Wait 10-15 minutes for Docker build
  - Green checkmarks = success

- [ ] **Copy model weights**
  - Place `.pth` file in `models/` folder
  - Update `configs/pipeline_config.yaml` if needed

- [ ] **Test GPU access**
  ```powershell
  nvidia-smi
  ```
  - Should show your GPU

- [ ] **Test Docker**
  ```powershell
  docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
  ```
  - Should show GPU inside Docker

### First Run

- [ ] **Configure camera source**
  - Edit `configs/pipeline_config.yaml`
  - Set `source_type` and `source`

- [ ] **Start pipeline**
  ```powershell
  docker-compose up
  ```

- [ ] **Verify output**
  - Check `data/output/` for JSON results
  - Watch console for errors

- [ ] **Stop pipeline**
  - Press `Ctrl+C`
  - Then: `docker-compose down`

## Common Daily Workflow

### Morning Startup
```powershell
cd C:\Projects\automotive-camera-pipeline
docker-compose up -d
```

### Check Logs
```powershell
docker-compose logs -f
```

### Process Video Files
1. Copy video to `data/input/`
2. Update config:
   ```yaml
   cameras:
     camera_0:
       source_type: "dataset"
       source: "/app/data/input/my_video.mp4"
   ```
3. Restart: `docker-compose restart`
4. Results in `data/output/`

### End of Day
```powershell
docker-compose down
```

## Team Collaboration

### Sharing Results
- **Location**: `data/output/`
- **Format**: Timestamped JSON files
- **Backup**: Copy to shared drive regularly

### Code Changes
- **Edit**: Files in `src/` folder
- **Test**: `docker-compose restart`
- **Share**: Commit to Git (if using version control)

### Configuration Changes
- **Edit**: `configs/pipeline_config.yaml`
- **Reload**: `docker-compose restart`
- **Document**: Note changes in team log

## Troubleshooting

### Quick Fixes

**Pipeline won't start:**
```powershell
docker-compose down
docker-compose build --no-cache
docker-compose up
```

**GPU not working:**
```powershell
# 1. Check drivers
nvidia-smi

# 2. Restart Docker Desktop

# 3. Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

**Camera not found:**
- Close other apps (Zoom, Teams, etc.)
- Check USB connection
- Try different USB port
- Update config with correct device index

### Get Help
1. Check `TROUBLESHOOTING.md`
2. Review logs: `docker-compose logs`
3. Check `logs/pipeline.log`
4. Contact team lead with error messages

## Performance Tips

### For RTX 3060/3070
- Resolution: 1280x720
- FPS: 25
- FP16: enabled

### For RTX 3080/3090/4090
- Resolution: 1920x1080
- FPS: 30
- FP16: enabled

### For RTX A5000/A6000
- Multi-camera: 4 cameras
- Resolution: 1920x1080
- FPS: 30
- FP16: enabled

## Safety Reminders

- ✅ Always stop pipeline before shutting down: `docker-compose down`
- ✅ Backup results regularly
- ✅ Keep model weights secure
- ✅ Document any config changes
- ✅ Close other GPU applications before running
- ✅ Monitor GPU temperature (should be < 80°C)

## Success Criteria

You know it's working when:
- ✅ Docker container starts without errors
- ✅ GPU utilization shows in `nvidia-smi`
- ✅ Camera feed is processing
- ✅ JSON results appear in `data/output/`
- ✅ Latency < 100ms
- ✅ Detection confidence > 0.5

## Contact Information

**System Administrator**: [name] - [email]
**Team Lead**: [name] - [email]
**Technical Support**: [email/slack channel]

## Resources

- **Full Documentation**: `README.md`
- **Quick Reference**: `QUICK_START.md`
- **Troubleshooting**: `TROUBLESHOOTING.md`
- **Docker Desktop**: https://www.docker.com/products/docker-desktop
- **NVIDIA Drivers**: https://www.nvidia.com/Download/index.aspx

## Version Information

- **Pipeline Version**: 1.0.0
- **Last Updated**: January 2026
- **Docker Image**: automotive-camera-pipeline:latest
- **Required Docker**: 4.0+
- **Required GPU**: NVIDIA RTX series

---

**Questions?** Check documentation or contact your team lead!
