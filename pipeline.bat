@echo off
REM Automotive Camera Pipeline - Command Helper
REM Quick access to common Docker commands

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="start" goto start
if "%1"=="stop" goto stop
if "%1"=="restart" goto restart
if "%1"=="logs" goto logs
if "%1"=="build" goto build
if "%1"=="clean" goto clean
if "%1"=="test" goto test
if "%1"=="jupyter" goto jupyter
if "%1"=="shell" goto shell
if "%1"=="status" goto status
goto help

:help
echo.
echo Automotive Camera Pipeline - Command Helper
echo ============================================
echo.
echo Usage: pipeline [command]
echo.
echo Commands:
echo   start      - Start the pipeline
echo   stop       - Stop the pipeline
echo   restart    - Restart the pipeline
echo   logs       - View pipeline logs
echo   build      - Rebuild Docker image
echo   clean      - Clean Docker resources
echo   test       - Run tests
echo   jupyter    - Start Jupyter notebook
echo   shell      - Open shell in container
echo   status     - Check pipeline status
echo   help       - Show this help
echo.
goto end

:start
echo Starting pipeline...
docker-compose up -d
echo Pipeline started! Use 'pipeline logs' to view output
goto end

:stop
echo Stopping pipeline...
docker-compose down
echo Pipeline stopped!
goto end

:restart
echo Restarting pipeline...
docker-compose restart
echo Pipeline restarted!
goto end

:logs
echo Showing logs (Ctrl+C to exit)...
docker-compose logs -f automotive-pipeline
goto end

:build
echo Building Docker image...
docker-compose build
echo Build complete!
goto end

:clean
echo Cleaning Docker resources...
docker-compose down -v
docker system prune -f
echo Cleanup complete!
goto end

:test
echo Running tests...
docker-compose run --rm automotive-pipeline pytest tests/ -v
goto end

:jupyter
echo Starting Jupyter notebook...
docker-compose --profile dev up jupyter-dev
goto end

:shell
echo Opening shell in container...
docker-compose exec automotive-pipeline bash
goto end

:status
echo Checking pipeline status...
docker-compose ps
echo.
echo GPU Status:
nvidia-smi
goto end

:end
