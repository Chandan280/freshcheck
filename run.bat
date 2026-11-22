@echo off
REM FreshCheck one-click launcher for Windows: double-click this file.
REM Uses the Python server (same app and UI). For the C++ server use WSL: ./run.sh
cd /d "%~dp0"
where python >nul 2>nul || (echo Python is not installed. Get it from https://www.python.org/downloads/ ^(tick "Add to PATH"^) & pause & exit /b 1)

python find_models.py || (pause & exit /b 1)

if not exist .venv (
  echo Setting up, one time only...
  python -m venv .venv
)
.venv\Scripts\python -m pip install -q -r app_fallback\requirements.txt || (pause & exit /b 1)

echo.
echo FreshCheck is starting at http://localhost:7860  - close this window to stop.
start "" cmd /c "timeout /t 4 >nul & start http://localhost:7860"
.venv\Scripts\python app_fallback\server.py
pause
