@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title Alien Garden - Interactive Showcase
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

rem Prefer a project-local virtual environment when it exists.
set "PYTHON="
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON="%~dp0.venv\Scripts\python.exe"
if defined PYTHON goto :python_ready

rem Prefer the Windows Python launcher, then fall back to python on PATH.
py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 set "PYTHON=py -3"
if defined PYTHON goto :python_ready

python -c "import sys" >nul 2>nul
if not errorlevel 1 set "PYTHON=python"
if defined PYTHON goto :python_ready

echo [ERROR] Python 3.10+ was not found.
echo Install it from https://www.python.org/downloads/ and enable "Add Python to PATH".
echo Then double-click run.bat again.
echo.
pause
exit /b 1

:python_ready
%PYTHON% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.10 or newer is required.
    %PYTHON% --version
    echo.
    pause
    exit /b 1
)

rem Verify runtime dependencies; install them automatically on first run.
%PYTHON% -c "from PyQt5 import QtWidgets, QtMultimedia; import cv2, numpy, pyzbar, PIL" >nul 2>nul
if errorlevel 1 (
    echo [SETUP] Missing dependencies. Installing from requirements.txt ...
    echo.
    %PYTHON% -m pip install -r requirements.txt
    if errorlevel 1 goto :install_failed
    %PYTHON% -c "from PyQt5 import QtWidgets, QtMultimedia; import cv2, numpy, pyzbar, PIL" >nul 2>nul
    if errorlevel 1 goto :install_failed
)

echo ==================================================
echo  Alien Garden Interactive Showcase
echo  Fullscreen standby video + camera QR / pose check
echo  SPACE / ENTER: capture    C: preview on/off
echo  F11: window/fullscreen    ESC: quit
echo ==================================================
echo.
%PYTHON% main.py %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] main.py exited with code %EXIT_CODE%.
    echo Check the messages above, then press any key to close.
    pause >nul
)
endlocal & exit /b %EXIT_CODE%

:install_failed
echo.
echo [ERROR] Dependency installation failed.
echo Try running this command manually:
echo     %PYTHON% -m pip install -r requirements.txt
echo If pyzbar reports a missing DLL, install the Microsoft Visual C++ Redistributable.
echo.
pause
endlocal & exit /b 1
