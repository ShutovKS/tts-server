@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM FILE: launch.bat
REM VERSION: 1.1.0
REM START_MODULE_CONTRACT
REM   PURPOSE: Provide a double-clickable Windows BAT entrypoint that launches the guided interactive runtime flow through the existing Windows CMD compatibility wrapper.
REM   SCOPE: Repository-root bootstrap, script presence checks, delegation into scripts\launch-windows.cmd for the supported interactive Windows launch path, and error-path pause handling for double-click operator visibility.
REM   DEPENDS: M-WINDOWS-LAUNCHER-CMD
REM   LINKS: M-WINDOWS-LAUNCHER-BAT
REM   ROLE: SCRIPT
REM   MAP_MODE: LOCALS
REM END_MODULE_CONTRACT
REM
REM START_MODULE_MAP
REM   PROJECT_ROOT - Repository root resolved from the BAT file location.
REM   CMD_WRAPPER - Canonical Windows CMD compatibility launcher path delegated to by this clickable entrypoint.
REM END_MODULE_MAP
REM
REM START_CHANGE_SUMMARY
REM   LAST_CHANGE: [v1.1.0 - Added an error-path pause so double-click launches keep the console open long enough for operators to read failures]
REM END_CHANGE_SUMMARY

set "PROJECT_ROOT=%~dp0"
set "CMD_WRAPPER=%PROJECT_ROOT%scripts\launch-windows.cmd"

if not exist "%CMD_WRAPPER%" (
    echo [launch.bat] Expected Windows launcher wrapper was not found: "%CMD_WRAPPER%"
    pause
    exit /b 1
)

call "%CMD_WRAPPER%"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [launch.bat] Launcher exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%
