# Automotive Camera Pipeline - Windows Setup Script
# Run as Administrator

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Automotive Camera Pipeline Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: Please run this script as Administrator" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    pause
    exit 1
}

# Function to check if a command exists
function Test-Command {
    param($command)
    $null = Get-Command $command -ErrorAction SilentlyContinue
    return $?
}

# 1. Check Docker Desktop
Write-Host "[1/6] Checking Docker Desktop..." -ForegroundColor Yellow
if (-not (Test-Command docker)) {
    Write-Host "Docker Desktop not found!" -ForegroundColor Red
    Write-Host "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    Write-Host "After installation, enable WSL2 backend and restart your computer." -ForegroundColor Yellow
    pause
    exit 1
}
Write-Host "Docker Desktop found!" -ForegroundColor Green

# 2. Check Docker is running
Write-Host "[2/6] Checking if Docker is running..." -ForegroundColor Yellow
docker ps > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and wait for it to fully initialize." -ForegroundColor Yellow
    pause
    exit 1
}
Write-Host "Docker is running!" -ForegroundColor Green

# 3. Check NVIDIA GPU and drivers
Write-Host "[3/6] Checking NVIDIA GPU..." -ForegroundColor Yellow
nvidia-smi > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: nvidia-smi not found or GPU not detected" -ForegroundColor Yellow
    Write-Host "If you have an NVIDIA GPU, please install the latest drivers:" -ForegroundColor Yellow
    Write-Host "https://www.nvidia.com/Download/index.aspx" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Continuing without GPU support..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
} else {
    Write-Host "NVIDIA GPU detected!" -ForegroundColor Green
    nvidia-smi --query-gpu=name --format=csv,noheader
}

# 4. Create project structure
Write-Host "[4/6] Creating project structure..." -ForegroundColor Yellow
$directories = @(
    "src",
    "configs",
    "models",
    "data/input",
    "data/output",
    "data/datasets",
    "logs",
    "tests",
    "notebooks",
    "docs"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created: $dir" -ForegroundColor Gray
    }
}
Write-Host "Project structure created!" -ForegroundColor Green

# 5. Create configuration files
Write-Host "[5/6] Creating default configuration..." -ForegroundColor Yellow

# Create default pipeline config
$configContent = @"
# Automotive Camera Pipeline Configuration

# System Settings
system:
  num_cameras: 1
  target_fps: 30
  max_latency_ms: 100
  gpu_device: 0
  log_level: INFO

# Camera Settings
cameras:
  camera_0:
    source_type: "usb"  # Options: usb, dataset, industrial
    source: 0           # Device index or path
    width: 1920
    height: 1080
    fps: 30
    
# Detection Model
detection:
  model_type: "rtmdet"
  config_path: "configs/rtmdet_config.py"
  checkpoint_path: "models/rtmdet_weights.pth"
  confidence_threshold: 0.5
  nms_threshold: 0.45
  device: "cuda"
  fp16: true
  
  # Light classes
  classes:
    - "left_indicator"
    - "right_indicator"
    - "brake_light"
    - "reverse_light"
    - "headlight"
    - "fog_light"

# Tracking
tracking:
  method: "bytetrack"
  track_thresh: 0.5
  track_buffer: 30
  match_thresh: 0.8
  min_box_area: 100

# State Estimation
state_estimation:
  # Intensity thresholds
  on_threshold: 120
  off_threshold: 50
  
  # Blinking detection
  blink_frequency_hz: [1.0, 2.0]  # Min, Max
  blink_window_frames: 60
  min_blink_cycles: 2
  
  # Temporal filtering
  temporal_window: 5
  state_change_threshold: 3

# Output
output:
  format: "json"
  include_timestamps: true
  output_directory: "data/output"
  log_frames: false
  save_visualizations: true

# Performance
performance:
  batch_size: 1
  num_workers: 2
  prefetch_frames: 3
"@

$configPath = "configs/pipeline_config.yaml"
if (-not (Test-Path $configPath)) {
    $configContent | Out-File -FilePath $configPath -Encoding UTF8
    Write-Host "Created: $configPath" -ForegroundColor Gray
}

Write-Host "Configuration created!" -ForegroundColor Green

# 6. Build Docker image
Write-Host "[6/6] Building Docker image..." -ForegroundColor Yellow
Write-Host "This may take 10-15 minutes on first run..." -ForegroundColor Gray

docker-compose build
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker build failed!" -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Quick Start Commands:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Start pipeline:        docker-compose up" -ForegroundColor White
Write-Host "  Start in background:   docker-compose up -d" -ForegroundColor White
Write-Host "  Stop pipeline:         docker-compose down" -ForegroundColor White
Write-Host "  View logs:             docker-compose logs -f" -ForegroundColor White
Write-Host "  Rebuild:               docker-compose build" -ForegroundColor White
Write-Host ""
Write-Host "Development Tools:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Jupyter Notebook:      docker-compose --profile dev up jupyter-dev" -ForegroundColor White
Write-Host "  TensorBoard:           docker-compose --profile monitoring up tensorboard" -ForegroundColor White
Write-Host ""
Write-Host "Access URLs:" -ForegroundColor Cyan
Write-Host "  Jupyter:               http://localhost:8888" -ForegroundColor White
Write-Host "  TensorBoard:           http://localhost:6006" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Place your model weights in ./models/" -ForegroundColor White
Write-Host "  2. Configure cameras in ./configs/pipeline_config.yaml" -ForegroundColor White
Write-Host "  3. Place input data in ./data/input/" -ForegroundColor White
Write-Host "  4. Run: docker-compose up" -ForegroundColor White
Write-Host ""
pause
