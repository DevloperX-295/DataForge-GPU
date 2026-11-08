@echo off
setlocal
title DataForge GPU - AI Training Data Debugger

python --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python 3.10+ is required. Install it from https://www.python.org/downloads/
  echo During installation, enable "Add Python to PATH".
  pause
  exit /b 1
)

for /f "tokens=2" %%V in ('python -c "import sys; print(sys.version_info.major, sys.version_info.minor)"') do set PY_MINOR=%%V
if %PY_MINOR% LSS 10 (
  echo [ERROR] Python 3.10 or newer is required.
  pause
  exit /b 1
)

echo [1/2] Installing/verifying dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Dependency installation failed.
  pause
  exit /b 1
)

echo [2/2] Starting DataForge GPU at http://127.0.0.1:8000
start "" cmd /c "timeout /t 2 ^>nul ^& start http://127.0.0.1:8000"
python app.py
pause
