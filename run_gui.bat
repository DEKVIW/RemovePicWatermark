@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================
echo  一览清图  GUI 启动
echo  Dir: %CD%
echo ============================================
echo.

set "PY=%CD%\.venv\Scripts\python.exe"
set "SCRIPT=%CD%\run_gui.py"
set "LOG=%CD%\gui_launch.log"

if not exist "%PY%" (
    echo [ERROR] Python not found:
    echo   %PY%
    echo.
    echo Create venv and install deps first.
    echo.
    pause
    exit /b 1
)

if not exist "%SCRIPT%" (
    echo [ERROR] run_gui.py not found:
    echo   %SCRIPT%
    echo.
    pause
    exit /b 1
)

echo Using:
echo   %PY%
echo   %SCRIPT%
echo.
echo Starting GUI...
echo If window does not appear, open gui_launch.log
echo.

"%PY%" -u "%SCRIPT%" 1>"%LOG%" 2>&1
set "ERR=%ERRORLEVEL%"

if not "%ERR%"=="0" (
    echo.
    echo [ERROR] Exit code %ERR%
    echo ---- log ----
    if exist "%LOG%" type "%LOG%"
    echo --------------
    echo.
    pause
    exit /b %ERR%
)

endlocal
exit /b 0