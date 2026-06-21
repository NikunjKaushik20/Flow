@echo off
echo ================================================
echo GRIDLOCK API SERVER
echo ================================================
echo.
echo Starting backend on http://localhost:8000
echo.
echo Press Ctrl+C to stop
echo ================================================
echo.
cd /d "%~dp0"
python api_server.py
