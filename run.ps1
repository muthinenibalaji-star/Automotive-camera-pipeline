# Automotive Camera Pipeline - Quick Run Script
# Usage: .\run.ps1 [video_path]

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "[ERROR] Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup first:"
    Write-Host "  python -m venv .venv"
    Write-Host "  .venv\Scripts\pip install -r requirements.txt"
    Read-Host "Press Enter to exit"
    exit 1
}

. .venv\Scripts\Activate.ps1

if ($args.Count -eq 0) {
    Write-Host "Running with webcam (source 0)..." -ForegroundColor Cyan
    python src\main.py --config configs\pipeline_config.yaml --source 0
} else {
    Write-Host "Running with video: $($args[0])" -ForegroundColor Cyan
    python src\main.py --config configs\pipeline_config.yaml --source $args[0]
}

Read-Host "Press Enter to exit"
