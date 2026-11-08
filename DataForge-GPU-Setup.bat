@echo off
setlocal EnableExtensions
title DataForge GPU Setup

echo ==========================================
echo       DataForge GPU - Windows Setup
echo ==========================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo Python was not found.
    echo Please install Python 3.10 or newer:
    echo https://www.python.org/downloads/
    echo IMPORTANT: Enable "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
    echo Python 3.10 or newer is required.
    echo Please upgrade Python and try again.
    pause
    exit /b 1
)

if not exist "%~dp0requirements.txt" (
    echo requirements.txt was not found.
    echo Extract the complete DataForge-GPU repository before running setup.
    pause
    exit /b 1
)

if not exist "%~dp0app.py" (
    echo app.py was not found.
    echo Extract the complete DataForge-GPU repository before running setup.
    pause
    exit /b 1
)

echo [1/3] Updating pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo Could not update pip. Continuing to dependency installation...
)

echo.
echo [2/3] Installing DataForge GPU dependencies...
python -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo.
    echo Dependency installation failed. Check your internet connection
    echo and Python installation, then run this setup again.
    pause
    exit /b 1
)

echo.
echo [3/3] Setup complete. Starting DataForge GPU...
echo The app will run locally at http://127.0.0.1:8000
echo Keep this window open while using the application.
echo.

start "" http://127.0.0.1:8000
cd /d "%~dp0"
python app.py

echo.
echo DataForge GPU has stopped.
pause
