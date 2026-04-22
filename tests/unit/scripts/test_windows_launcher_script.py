# FILE: tests/unit/scripts/test_windows_launcher_script.py
# VERSION: 1.4.2
# START_MODULE_CONTRACT
#   PURPOSE: Validate the Windows PowerShell and CMD launcher entrypoints stay aligned with the documented profile-aware Windows launch flow.
#   SCOPE: launcher-script presence, PowerShell orchestration markers, curated model menu anchors, CMD delegation shape, and bounded secret-handling command references
#   DEPENDS: M-WINDOWS-LAUNCHER, M-WINDOWS-LAUNCHER-CMD
#   LINKS: V-M-WINDOWS-LAUNCHER, V-M-WINDOWS-LAUNCHER-CMD
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   SCRIPT_PATH - Canonical path to the interactive Windows PowerShell launcher script.
#   CMD_SCRIPT_PATH - Canonical path to the Windows CMD wrapper.
#   test_windows_launcher_script_exists_with_grace_contract - Verifies the PowerShell launcher file exists and retains top-level GRACE contract anchors.
#   test_windows_launcher_script_reuses_profile_aware_launcher_and_family_download_paths - Verifies the PowerShell launcher delegates env setup and execution to launcher commands while keeping HF and Piper download flows explicit.
#   test_windows_launcher_script_manages_http_server_pid_restart - Verifies the PowerShell launcher carries PID-file lifecycle helpers for restarting launcher-managed HTTP server processes and prompting on foreign listeners.
#   test_windows_cmd_launcher_wraps_powershell_script_without_file_execution - Verifies the CMD wrapper executes the PowerShell launcher via an inline command string instead of PowerShell -File script execution.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.4.2 - Added regression coverage for launcher-managed HTTP server PID files so reruns can restart owned processes and prompt on foreign listeners]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "launch-windows.ps1"
CMD_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "launch-windows.cmd"


def test_windows_launcher_script_exists_with_grace_contract():
    contents = SCRIPT_PATH.read_text(encoding="utf-8")

    assert SCRIPT_PATH.exists()
    assert "# START_MODULE_CONTRACT" in contents
    assert "#   PURPOSE: Provide an interactive Windows PowerShell launcher" in contents
    assert "$SCRIPT:MODEL_OPTIONS" in contents
    assert "TTS_LAUNCH_PROJECT_ROOT" in contents


def test_windows_launcher_script_reuses_profile_aware_launcher_and_family_download_paths():
    contents = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "py -3.11 -m launcher --project-root" in contents
    assert "create-env" in contents
    assert "check-env" in contents
    assert "exec --family" in contents
    assert "expected_python_path" in contents
    assert "/health/live" in contents
    assert "127.0.0.1" in contents
    assert "Selected family:" in contents
    assert "Runtime capability bindings:" in contents
    assert "TTS_ACTIVE_FAMILY" in contents
    assert "TTS_DEFAULT_CUSTOM_MODEL" in contents
    assert "TTS_DEFAULT_DESIGN_MODEL" in contents
    assert "TTS_DEFAULT_CLONE_MODEL" in contents
    assert "snapshot_download" in contents
    assert "piper.download_voices" in contents
    assert "RepoId = 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'" in contents
    assert "RepoId = 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign'" in contents
    assert "RepoId = 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'" in contents
    assert "RepoId = 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'" in contents
    assert "RepoId = 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'" in contents
    assert "RepoId = 'k2-fsa/OmniVoice'" in contents
    assert "HF_TOKEN" in contents
    assert "Qwen Custom 1.7B" in contents
    assert "OmniVoice" in contents
    assert "Piper en_US lessac medium" in contents


def test_windows_launcher_script_manages_http_server_pid_restart():
    contents = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "Get-HttpServerPidFilePath" in contents
    assert "Read-HttpServerPidFile" in contents
    assert "Clear-HttpServerPidFile" in contents
    assert "Get-TcpOwningProcessId" in contents
    assert "Stop-HttpServerProcess" in contents
    assert "Ensure-HttpServerLaunchTarget" in contents
    assert ".state/launcher/http-server.pid" in contents
    assert "Launcher-managed HTTP server is already running. [R]estart / [K]eep existing / [C]hange port" in contents
    assert "Stopping existing launcher-managed HTTP server" in contents
    assert "Port is occupied by a non-launcher process. [S]top and restart / [K]eep existing / [C]hange port" in contents


def test_windows_cmd_launcher_wraps_powershell_script_without_file_execution():
    contents = CMD_SCRIPT_PATH.read_text(encoding="utf-8")

    assert CMD_SCRIPT_PATH.exists()
    assert "START_MODULE_CONTRACT" in contents
    assert "bypasses PowerShell script-signing policy" in contents
    assert "launch-windows.ps1" in contents
    assert "TTS_LAUNCH_PROJECT_ROOT" in contents
    assert "powershell.exe -NoLogo -NoProfile -Command" in contents
    assert "Get-Content -LiteralPath" in contents
    assert "[ScriptBlock]::Create" in contents
    assert "-File" not in contents
