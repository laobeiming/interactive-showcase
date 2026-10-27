rem Equivalent to opening Shell:startup in Windows Explorer.
@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "PROJECT_DIR=%~dp0"
set "RUN_BAT=%~dp0run.bat"
set "SHORTCUT_NAME=AlienGardenShowcase.lnk"
if not defined ALIEN_GARDEN_STARTUP_DIR set "ALIEN_GARDEN_STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "STARTUP_DIR=%ALIEN_GARDEN_STARTUP_DIR%"

if /I "%~1"=="--remove" goto remove
if /I "%~1"=="/remove" goto remove
if /I "%~1"=="--check" goto check
if /I "%~1"=="--help" goto help
if /I "%~1"=="/?" goto help

if not exist "%RUN_BAT%" (
    echo [ERROR] run.bat was not found:
    echo         %RUN_BAT%
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$shell = New-Object -ComObject WScript.Shell; if (-not (Test-Path -LiteralPath $env:STARTUP_DIR)) { New-Item -ItemType Directory -Path $env:STARTUP_DIR -Force | Out-Null }; $link = $shell.CreateShortcut((Join-Path $env:STARTUP_DIR $env:SHORTCUT_NAME)); $link.TargetPath = $env:RUN_BAT; $link.WorkingDirectory = $env:PROJECT_DIR; $link.WindowStyle = 7; $link.Description = 'Alien Garden Interactive Showcase'; $link.Save()"
if errorlevel 1 goto failed

if not exist "%STARTUP_DIR%\%SHORTCUT_NAME%" goto failed

echo.
echo [OK] Windows autostart shortcut created.
echo      Shortcut : %STARTUP_DIR%\%SHORTCUT_NAME%
echo      Target   : %RUN_BAT%
echo.
echo The showcase will start automatically after the next Windows sign-in.
echo Run setup_autostart.bat --remove to undo this change.
echo.
pause
exit /b 0

:remove
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Remove-Item -LiteralPath (Join-Path $env:STARTUP_DIR $env:SHORTCUT_NAME) -Force -ErrorAction SilentlyContinue"
echo.
echo [OK] Autostart shortcut removed (if it existed).
echo.
pause
exit /b 0

:check
if exist "%STARTUP_DIR%\%SHORTCUT_NAME%" (
    echo [OK] Autostart is enabled:
    echo      %STARTUP_DIR%\%SHORTCUT_NAME%
    exit /b 0
)
echo [INFO] Autostart is not enabled yet.
exit /b 1

:failed
echo.
echo [ERROR] Failed to create the Windows autostart shortcut.
echo Try running this script as the same Windows user that will operate the showcase.
echo.
pause
exit /b 1

:help
echo Usage:
echo   setup_autostart.bat            Add run.bat to the current user's Startup folder
echo   setup_autostart.bat --check    Check whether autostart is enabled
echo   setup_autostart.bat --remove   Remove the autostart shortcut
echo.
exit /b 0
