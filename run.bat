@echo off
REM Automotive Camera Pipeline - Quick Run Script
REM Usage: run.bat [video_path]

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup first:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

if "%~1"=="" (
    echo Running with webcam (source 0)...
    python src\main.py --config configs\pipeline_config.yaml --source 0
) else (
    echo Running with video: %~1
    python src\main.py --config configs\pipeline_config.yaml --source "%~1"
)

pause
