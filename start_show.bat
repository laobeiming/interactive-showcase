@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
title Alien Garden - Stage 2 Linkage
echo ==================================================
echo  Stage 2 demo: standby + camera scan + scene link
echo  C: scan page / standby    ESC: quit    F11: fullscreen
echo ==================================================
echo.
python main.py --windowed --no-video --start-camera
if errorlevel 1 (
    echo.
    echo [ERROR] Program exited with an error.
    echo.
    pause
)
