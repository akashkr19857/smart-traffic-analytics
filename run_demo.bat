@echo off
REM Smart Traffic Analytics one-click demo (Windows + NVIDIA GPU). API :8400, dashboard :3400.
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo [1/3] Public videos + datasets (first run only)...
if not exist "data\raw\vehicles.mp4" "%PY%" data\download_data.py
if not exist "data\scenarios\incidents.mp4" "%PY%" data\make_scenarios.py
echo [2/3] Starting API + camera pipelines on http://localhost:8400 ...
start "Traffic API" /D "%~dp0" "%PY%" -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8400
echo [3/3] Starting dashboard on http://localhost:3400 ...
cd frontend
if not exist node_modules call npm install
if not exist .next\BUILD_ID call npm run build
start "Traffic dashboard" /D "%~dp0frontend" cmd /k npm run start
timeout /t 8 >nul
start http://localhost:3400
endlocal
