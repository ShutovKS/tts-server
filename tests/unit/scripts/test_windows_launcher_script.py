# FILE: tests/unit/scripts/test_windows_launcher_script.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Validate the Windows launcher script stays aligned with the profile-aware launcher flow and documented family download strategies.
#   SCOPE: script presence, launcher orchestration markers, curated model menu anchors, and bounded secret-handling command references
#   DEPENDS: M-WINDOWS-LAUNCHER
#   LINKS: V-M-WINDOWS-LAUNCHER
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   SCRIPT_PATH - Canonical path to the interactive Windows launcher script.
#   test_windows_launcher_script_exists_with_grace_contract - Verifies the script file exists and retains top-level GRACE contract anchors.
#   test_windows_launcher_script_reuses_profile_aware_launcher_and_family_download_paths - Verifies the script delegates env setup and execution to launcher commands while keeping HF and Piper download flows explicit.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added deterministic coverage for the new interactive Windows launcher script surface]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "launch-windows.ps1"


def test_windows_launcher_script_exists_with_grace_contract():
    contents = SCRIPT_PATH.read_text(encoding="utf-8")

    assert SCRIPT_PATH.exists()
    assert "# START_MODULE_CONTRACT" in contents
    assert "#   PURPOSE: Provide an interactive Windows PowerShell launcher" in contents
    assert "$SCRIPT:MODEL_OPTIONS" in contents


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
    assert "Selected model folder:" in contents
    assert "snapshot_download" in contents
    assert "piper.download_voices" in contents
    assert "HF_TOKEN" in contents
    assert "Qwen Custom 1.7B" in contents
    assert "OmniVoice" in contents
    assert "Piper en_US lessac medium" in contents
