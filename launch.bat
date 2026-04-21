@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM FILE: launch.bat
REM VERSION: 1.2.0
REM START_MODULE_CONTRACT
REM   PURPOSE: Provide a double-clickable Windows BAT entrypoint that launches the shared cross-platform Python launcher flow.
REM   SCOPE: Repository-root bootstrap, Python launcher presence checks, delegation into launch.py through py/python, and error-path pause handling for double-click operator visibility.
REM   DEPENDS: M-ROOT-LAUNCHER
REM   LINKS: M-WINDOWS-LAUNCHER-BAT
REM   ROLE: SCRIPT
REM   MAP_MODE: LOCALS
REM END_MODULE_CONTRACT
REM
REM START_MODULE_MAP
REM   PROJECT_ROOT - Repository root resolved from the BAT file location.
REM   PYTHON_LAUNCHER - Root-level shared Python launcher delegated to by this clickable entrypoint.
REM END_MODULE_MAP
REM
REM START_CHANGE_SUMMARY
REM   LAST_CHANGE: [v1.2.0 - Rewired the clickable BAT entrypoint to delegate through the shared root-level Python launcher while preserving error-path pause handling]
REM END_CHANGE_SUMMARY

set "PROJECT_ROOT=%~dp0"
set "PYTHON_LAUNCHER=%PROJECT_ROOT%launch.py"

if not exist "%PYTHON_LAUNCHER%" (
    echo [launch.bat] Expected shared Python launcher was not found: "%PYTHON_LAUNCHER%"
    pause
    exit /b 1
)

where py >nul 2>nul
if not errorlevel 1 (
    py -3.11 "%PYTHON_LAUNCHER%"
    set "EXIT_CODE=%ERRORLEVEL%"
    goto after_launch
)

where python >nul 2>nul
if not errorlevel 1 (
    python "%PYTHON_LAUNCHER%"
    set "EXIT_CODE=%ERRORLEVEL%"
    goto after_launch
)

echo [launch.bat] Neither 'py' nor 'python' was found in PATH.
pause
exit /b 1

:after_launch
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [launch.bat] Launcher exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%
