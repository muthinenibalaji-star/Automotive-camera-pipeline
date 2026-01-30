# 🔧 Troubleshooting Guide - Automotive Camera Pipeline

## Setup Issues

### 1. Docker Desktop Installation Fails

**Symptoms:**
- Installer crashes or hangs
- "Hyper-V not available" error
- "WSL2 installation incomplete" error

**Solutions:**

**Enable WSL2:**
```powershell
# Run as Administrator
wsl --install
wsl --set-default-version 2
```

**Enable Hyper-V (Windows Pro/Enterprise):**
```powershell
# Run as Administrator
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All
```

**Enable Virtual Machine Platform:**
```powershell
# Run as Administrator
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

Then restart your computer.

### 2. Docker Build Fails with Network Errors

**Symptoms:**
```
ERROR: Could not connect to repository
Temporary failure resolving 'archive.ubuntu.com'
```

**Solutions:**

**Configure Docker DNS:**
1. Open Docker Desktop
2. Settings → Docker Engine
3. Add to JSON configuration:
```json
{
  "dns": ["8.8.8.8", "8.8.4.4", "1.1.1.1"]
}
```
4. Apply & Restart

**Check Corporate Proxy:**
If behind corporate firewall:
```json
{
  "proxies": {
    "http-proxy": "http://proxy.company.com:8080",
    "https-proxy": "http://proxy.company.com:8080"
  }
}
```

### 3. GPU Not Detected

**Symptoms:**
```
RuntimeError: CUDA not available
nvidia-smi: command not found
```

**Solutions:**

**Update NVIDIA Drivers:**
1. Visit: https://www.nvidia.com/Download/index.aspx
2. Select your GPU model
3. Download and install latest driver
4. Restart computer

**Install NVIDIA Container Toolkit (WSL2):**
```bash
# Open WSL2 terminal
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Verify GPU Access:**
```powershell
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

Should show your GPU details.

### 4. "Permission Denied" Errors

**Symptoms:**
```
ERROR: Cannot create directory
Permission denied: '/app/data/output'
```

**Solutions:**

**Windows:**
- Run PowerShell as Administrator
- Check folder permissions: Right-click → Properties → Security
- Add "Full Control" for your user

**Docker:**
```powershell
# Reset Docker Desktop
docker system prune -a
# Restart Docker Desktop
```

## Runtime Issues

### 5. Camera Not Accessible

**Symptoms:**
```
ERROR: Cannot open camera device 0
Video capture failed
```

**Solutions:**

**Check Camera Index:**
```powershell
# In WSL2
ls -l /dev/video*
```

Should show: `/dev/video0`, `/dev/video1`, etc.

**Close Other Applications:**
- Close Zoom, Skype, Teams, etc.
- Check Task Manager for processes using camera

**Update docker-compose.yml:**
```yaml
devices:
  - /dev/video0:/dev/video0  # Match your camera index
```

**Try Different USB Port:**
- USB 3.0 ports recommended
- Avoid USB hubs

### 6. Slow Inference / Low FPS

**Symptoms:**
- Processing < 10 FPS
- High latency (>200ms)
- Stuttering video

**Solutions:**

**Enable FP16:**
```yaml
# pipeline_config.yaml
detection:
  fp16: true
```

**Reduce Resolution:**
```yaml
cameras:
  camera_0:
    width: 1280  # Instead of 1920
    height: 720  # Instead of 1080
```

**Check GPU Usage:**
```powershell
nvidia-smi -l 1
```
- GPU Usage should be 50-90%
- If low, check CPU bottleneck

**Close Background Apps:**
- Close Chrome, games, other GPU apps
- Check Task Manager GPU usage

**Batch Processing:**
```yaml
performance:
  batch_size: 2  # Process 2 frames at once
```

### 7. Container Keeps Restarting

**Symptoms:**
```
docker-compose ps
# Shows: Restarting (1) 2 seconds ago
```

**Solutions:**

**Check Logs:**
```powershell
docker-compose logs automotive-pipeline
```

**Common Causes:**
- Missing model weights → Add to `models/` folder
- Invalid config → Validate YAML syntax
- CUDA out of memory → Reduce batch_size or resolution

**Disable Auto-Restart:**
```yaml
# docker-compose.yml
restart: "no"  # Temporarily disable
```

### 8. JSON Output Not Generated

**Symptoms:**
- `data/output/` folder empty
- No results files

**Solutions:**

**Check Permissions:**
```powershell
# Ensure folder exists and is writable
New-Item -ItemType Directory -Path "data/output" -Force
```

**Check Config:**
```yaml
output:
  output_directory: "data/output"  # Correct path
  format: "json"
```

**Check Logs:**
```powershell
docker-compose logs | Select-String "output"
```

**Manual Test:**
```powershell
docker-compose exec automotive-pipeline bash
cd /app/data/output
touch test.txt  # Should succeed
```

### 9. Memory Issues

**Symptoms:**
```
CUDA out of memory
Killed
Docker container stopped unexpectedly
```

**Solutions:**

**Increase Docker Memory:**
1. Docker Desktop → Settings → Resources
2. Memory: Set to 12GB+ (16GB recommended)
3. Apply & Restart

**Reduce Model Size:**
```yaml
detection:
  fp16: true  # Uses less memory
```

**Reduce Batch Size:**
```yaml
performance:
  batch_size: 1
```

**Monitor Memory:**
```powershell
nvidia-smi -l 1
# Watch "Memory-Usage" column
```

### 10. "Module Not Found" Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'mmdet'
ImportError: cannot import name 'ByteTrack'
```

**Solutions:**

**Rebuild Container:**
```powershell
docker-compose build --no-cache
```

**Check Requirements:**
```powershell
docker-compose exec automotive-pipeline pip list | grep mmdet
```

**Manual Install:**
```powershell
docker-compose exec automotive-pipeline bash
pip install mmdet --upgrade
```

## Configuration Issues

### 11. YAML Syntax Errors

**Symptoms:**
```
yaml.scanner.ScannerError: while scanning for the next token
```

**Solutions:**

**Use YAML Validator:**
- https://www.yamllint.com/
- Copy/paste your config file

**Common Mistakes:**
```yaml
# ❌ Wrong - inconsistent indentation
system:
  num_cameras: 1
   target_fps: 30

# ✅ Correct
system:
  num_cameras: 1
  target_fps: 30

# ❌ Wrong - missing space after colon
detection:
  fp16:true

# ✅ Correct
detection:
  fp16: true
```

### 12. Model Weights Issues

**Symptoms:**
```
FileNotFoundError: models/rtmdet_weights.pth
RuntimeError: Error loading checkpoint
```

**Solutions:**

**Check File Exists:**
```powershell
dir models\
# Should show .pth file
```

**Verify Path in Config:**
```yaml
detection:
  checkpoint_path: "models/rtmdet_weights.pth"  # Match actual filename
```

**Check File Integrity:**
```powershell
docker-compose exec automotive-pipeline python -c "import torch; torch.load('/app/models/rtmdet_weights.pth')"
```

## Network Issues

### 13. Cannot Access Jupyter/TensorBoard

**Symptoms:**
- `localhost:8888` not loading
- Connection refused

**Solutions:**

**Check Container Running:**
```powershell
docker-compose ps
# jupyter-dev should be "Up"
```

**Check Port Binding:**
```powershell
netstat -an | Select-String "8888"
# Should show LISTENING
```

**Try 127.0.0.1:**
- Instead of `localhost:8888`
- Try `127.0.0.1:8888`

**Windows Firewall:**
- Allow Docker Desktop through firewall
- Settings → Firewall → Allow app

### 14. Industrial Camera Connection

**Symptoms:**
```
ERROR: Cannot connect to 192.168.1.100
Connection timeout
```

**Solutions:**

**Check Network:**
```powershell
ping 192.168.1.100
```

**Configure docker-compose.yml:**
```yaml
network_mode: "host"  # Required for camera access
```

**Check Camera Settings:**
- Verify IP address
- Check firewall rules
- Ensure same subnet

## Performance Optimization

### 15. Optimize for Your Hardware

**NVIDIA RTX 3060/3070:**
```yaml
system:
  target_fps: 25
cameras:
  camera_0:
    width: 1280
    height: 720
detection:
  fp16: true
  confidence_threshold: 0.6
```

**NVIDIA RTX 3080/3090/4090:**
```yaml
system:
  target_fps: 30
cameras:
  camera_0:
    width: 1920
    height: 1080
detection:
  fp16: true
  confidence_threshold: 0.5
```

**NVIDIA RTX A5000/A6000:**
```yaml
system:
  num_cameras: 4
  target_fps: 30
detection:
  fp16: true
performance:
  batch_size: 2
```

## Advanced Debugging

### Enable Verbose Logging

```yaml
system:
  log_level: DEBUG
```

### Profile Performance

```yaml
performance:
  enable_profiling: true
```

### Enter Container Shell

```powershell
docker-compose exec automotive-pipeline bash

# Inside container:
python -c "import torch; print(torch.cuda.is_available())"
python -c "import mmdet; print(mmdet.__version__)"
ls -la /app/models/
```

### Check System Resources

```powershell
# GPU
nvidia-smi

# Docker stats
docker stats

# Disk space
docker system df
```

## Getting Additional Help

### Collect Debug Information

```powershell
# Create debug report
docker-compose logs > debug.log
nvidia-smi > gpu_info.txt
docker-compose config > compose_config.yml
docker version > docker_version.txt
```

Send these files to your team lead.

### Useful Commands

```powershell
# Full system reset
docker-compose down -v
docker system prune -a
docker volume prune

# Check Docker Desktop status
docker info

# Validate compose file
docker-compose config

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Emergency Recovery

### Complete Cleanup

```powershell
# 1. Stop everything
docker-compose down -v

# 2. Remove all containers
docker rm -f $(docker ps -aq)

# 3. Remove all images
docker rmi -f $(docker images -aq)

# 4. Clean system
docker system prune -a --volumes

# 5. Restart Docker Desktop

# 6. Rebuild
docker-compose build --no-cache
```

---

**Still Having Issues?**
- Check `logs/pipeline.log` for detailed error messages
- Review Docker logs: `docker-compose logs`
- Contact team lead with debug information
- Check project README.md for updates
