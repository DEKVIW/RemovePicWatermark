@echo off
setlocal
cd /d "%~dp0"
echo Launching...
".venv\Scripts\python.exe" -u "run_gui.py"
echo.
echo Exit code %ERRORLEVEL%
pause