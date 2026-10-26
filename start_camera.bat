@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set CAMERA=%1
if "%CAMERA%"=="" set CAMERA=0
title Alien Garden - Camera Scanner
echo ============================================
echo  Camera + QR scanner  (q / ESC: quit, s: save)
echo  Camera index: %CAMERA%
echo ============================================
echo.
python scanner.py --camera %CAMERA%
if errorlevel 1 (
    echo.
    echo [ERROR] Camera failed to start.
    echo Close other apps that may use the camera, or try another index:
    echo     start_camera.bat 1
    echo.
    pause
)
