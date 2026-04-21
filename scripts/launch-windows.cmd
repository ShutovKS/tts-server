@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM FILE: scripts/launch-windows.cmd
REM VERSION: 1.0.0
REM START_MODULE_CONTRACT
REM   PURPOSE: Provide a Windows CMD launcher entrypoint that bypasses PowerShell script-signing policy by running the existing PowerShell orchestrator as an inline command.
REM   SCOPE: Windows-only bootstrap, project-root resolution, PowerShell availability checks, and delegation into scripts/launch-windows.ps1 through a signed-file-independent command string.
REM   DEPENDS: M-WINDOWS-LAUNCHER
REM   LINKS: M-WINDOWS-LAUNCHER-CMD
REM   ROLE: SCRIPT
REM   MAP_MODE: LOCALS
REM END_MODULE_CONTRACT
REM
REM START_MODULE_MAP
REM   SCRIPT_DIR - Directory containing this CMD wrapper.
REM   PROJECT_ROOT - Repository root resolved relative to the wrapper location.
REM   PS_SCRIPT - Canonical interactive PowerShell launcher path.
REM END_MODULE_MAP
REM
REM START_CHANGE_SUMMARY
REM   LAST_CHANGE: [v1.0.0 - Added a CMD wrapper that launches the existing PowerShell orchestration through -Command so Windows hosts with MachinePolicy AllSigned still have a supported interactive entrypoint]
REM END_CHANGE_SUMMARY

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"
set "PS_SCRIPT=%SCRIPT_DIR%launch-windows.ps1"
set "TTS_LAUNCH_PROJECT_ROOT=%PROJECT_ROOT%"

where powershell.exe >nul 2>nul
if errorlevel 1 (
    echo [launch-windows.cmd] powershell.exe was not found in PATH.
    exit /b 1
)

if not exist "%PS_SCRIPT%" (
    echo [launch-windows.cmd] Expected launcher script was not found: "%PS_SCRIPT%"
    exit /b 1
)

powershell.exe -NoLogo -NoProfile -Command "& { Set-Location -LiteralPath '%PROJECT_ROOT%'; . ([ScriptBlock]::Create((Get-Content -LiteralPath '%PS_SCRIPT%' -Raw))) }"
set "EXIT_CODE=%ERRORLEVEL%"
exit /b %EXIT_CODE%
