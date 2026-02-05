@echo off
REM ========================================================
REM Automotive Camera Pipeline - One-Click Setup (Windows)
REM ========================================================
REM Run this script after cloning the repository.
REM It will set up everything you need to start training.
REM ========================================================

echo.
echo ========================================================
echo    Automotive Camera Pipeline - Setup
echo ========================================================
echo.

cd /d "%~dp0"

REM Check Python
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.9+ from python.org
    pause
    exit /b 1
)
python --version
echo.

REM Create virtual environment
echo [2/6] Creating virtual environment...
if exist ".venv" (
    echo   Virtual environment already exists, skipping...
) else (
    python -m venv .venv
    echo   Created .venv
)
echo.

REM Activate virtual environment
echo [3/6] Activating virtual environment...
call .venv\Scripts\activate.bat
echo   Activated!
echo.

REM Upgrade pip
echo [4/6] Upgrading pip and installing dependencies...
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements
    pause
    exit /b 1
)
echo.

REM Install MMDetection
echo [5/6] Installing MMDetection ecosystem...
pip install -U openmim >nul 2>&1
mim install "mmcv>=2.0.0" "mmdet>=3.0.0"
if errorlevel 1 (
    echo [ERROR] Failed to install MMDetection
    pause
    exit /b 1
)
echo.

REM Download model weights
echo [6/6] Downloading RTMDet model weights...
if exist "models\rtmdet_tiny*.pth" (
    echo   Model weights already exist, skipping...
) else (
    mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest models
)
echo.

REM Create required directories
echo Creating directory structure...
if not exist "data\vehicle_lights\train" mkdir "data\vehicle_lights\train"
if not exist "data\vehicle_lights\val" mkdir "data\vehicle_lights\val"
if not exist "data\vehicle_lights\test" mkdir "data\vehicle_lights\test"
if not exist "data\vehicle_lights\annotations" mkdir "data\vehicle_lights\annotations"
if not exist "data\input" mkdir "data\input"
if not exist "data\output" mkdir "data\output"
if not exist "logs" mkdir "logs"
if not exist "work_dirs" mkdir "work_dirs"
if not exist "visualizations" mkdir "visualizations"
echo   Done!
echo.

REM Final message
echo ========================================================
echo   Setup Complete!
echo ========================================================
echo.
echo NEXT STEPS:
echo.
echo 1. For INFERENCE (using pre-trained model):
echo    python src\main.py --config configs\pipeline_config.yaml --source 0
echo.
echo 2. For TRAINING (on your custom data):
echo    a. Place your COCO dataset in data\vehicle_lights\
echo    b. Run validation:
echo       python tools\dataset_sanity_check.py --ann-file data\vehicle_lights\annotations\train.json --img-dir data\vehicle_lights\train\
echo    c. Start training:
echo       python tools\train.py configs\vehicle_lights\rtmdet_m_vehicle_lights.py
echo.
echo See docs\TRAINING_GUIDE.md for detailed instructions.
echo ========================================================
echo.

pause
