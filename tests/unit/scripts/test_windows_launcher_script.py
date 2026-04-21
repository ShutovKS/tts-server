# FILE: tests/unit/scripts/test_windows_launcher_script.py
# VERSION: 1.3.0
# START_MODULE_CONTRACT
#   PURPOSE: Validate the Windows PowerShell, CMD, and clickable BAT launcher entrypoints stay aligned with the documented profile-aware Windows launch flow.
#   SCOPE: launcher-script presence, PowerShell orchestration markers, curated model menu anchors, CMD and BAT delegation shape, bounded secret-handling command references, and BAT error-path pause behavior
#   DEPENDS: M-WINDOWS-LAUNCHER, M-WINDOWS-LAUNCHER-CMD, M-WINDOWS-LAUNCHER-BAT
#   LINKS: V-M-WINDOWS-LAUNCHER, V-M-WINDOWS-LAUNCHER-CMD, V-M-WINDOWS-LAUNCHER-BAT
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   SCRIPT_PATH - Canonical path to the interactive Windows PowerShell launcher script.
#   CMD_SCRIPT_PATH - Canonical path to the Windows CMD wrapper.
#   BAT_SCRIPT_PATH - Canonical path to the clickable repository-root Windows BAT entrypoint.
#   test_windows_launcher_script_exists_with_grace_contract - Verifies the PowerShell launcher file exists and retains top-level GRACE contract anchors.
#   test_windows_launcher_script_reuses_profile_aware_launcher_and_family_download_paths - Verifies the PowerShell launcher delegates env setup and execution to launcher commands while keeping HF and Piper download flows explicit.
#   test_windows_cmd_launcher_wraps_powershell_script_without_file_execution - Verifies the CMD wrapper executes the PowerShell launcher via an inline command string instead of PowerShell -File script execution.
#   test_windows_bat_launcher_delegates_to_cmd_wrapper - Verifies the clickable BAT entrypoint delegates to the existing CMD compatibility wrapper.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.3.0 - Added deterministic coverage for the BAT error-path pause so double-click failures remain visible to operators]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "launch-windows.ps1"
CMD_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "launch-windows.cmd"
BAT_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "launch.bat"


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
    assert "HF_TOKEN" in contents
    assert "Qwen Custom 1.7B" in contents
    assert "OmniVoice" in contents
    assert "Piper en_US lessac medium" in contents


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


def test_windows_bat_launcher_delegates_to_cmd_wrapper():
    contents = BAT_SCRIPT_PATH.read_text(encoding="utf-8")

    assert BAT_SCRIPT_PATH.exists()
    assert "START_MODULE_CONTRACT" in contents
    assert "double-clickable Windows BAT entrypoint" in contents
    assert "scripts\\launch-windows.cmd" in contents
    assert "call \"%CMD_WRAPPER%\"" in contents
    assert "pause" in contents
    assert "Launcher exited with code %EXIT_CODE%." in contents
