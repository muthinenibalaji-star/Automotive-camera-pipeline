# Automotive Camera Pipeline - Windows Setup Script
# Run this script ONCE to set up the project

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Automotive Camera Pipeline Setup" -ForegroundColor Cyan  
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "[1/5] Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found! Please install Python 3.9+ from python.org" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "[2/5] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "  Virtual environment already exists, skipping..." -ForegroundColor Gray
} else {
    python -m venv .venv
    Write-Host "  Created .venv" -ForegroundColor Green
}

# Activate and install dependencies
Write-Host "[3/5] Installing dependencies..." -ForegroundColor Yellow
. .venv\Scripts\Activate.ps1
pip install --upgrade pip | Out-Null
pip install -r requirements.txt

# Install MMDetection
Write-Host "[4/5] Installing MMDetection ecosystem..." -ForegroundColor Yellow
pip install -U openmim | Out-Null
mim install "mmcv>=2.0.0" "mmdet>=3.0.0"

# Download model weights
Write-Host "[5/5] Downloading model weights..." -ForegroundColor Yellow
if (Test-Path "models\rtmdet_tiny_8xb32-300e_coco*.pth") {
    Write-Host "  Model weights already exist, skipping..." -ForegroundColor Gray
} else {
    mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest models
    Write-Host "  Downloaded RTMDet-Tiny weights" -ForegroundColor Green
}

# Create data directories
Write-Host ""
Write-Host "Creating data directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data\input" | Out-Null
New-Item -ItemType Directory -Force -Path "data\output" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "To run the pipeline:" -ForegroundColor Cyan
Write-Host "  1. Place a test video in data\input\" -ForegroundColor White
Write-Host "  2. Run: .\run.ps1 `"data\input\your_video.mp4`"" -ForegroundColor White
Write-Host ""
Write-Host "Or double-click run.bat to use your webcam" -ForegroundColor White
Write-Host ""

Read-Host "Press Enter to exit"
