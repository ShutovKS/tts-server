# FILE: tests/unit/scripts/test_root_launchers.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Validate the root-level cross-platform launcher entrypoints stay aligned with the shared launcher dispatch flow.
#   SCOPE: root Python launcher subprocess delegation, Unix shell wrapper delegation shape, and Windows BAT wrapper delegation markers
#   DEPENDS: M-ROOT-LAUNCHER, M-ROOT-LAUNCHER-SH, M-WINDOWS-LAUNCHER-BAT
#   LINKS: V-M-ROOT-LAUNCHER, V-M-ROOT-LAUNCHER-SH, V-M-WINDOWS-LAUNCHER-BAT
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   PROJECT_ROOT - Repository root used to resolve the root launcher entrypoints.
#   test_root_python_launcher_delegates_to_launcher_module - Verifies launch.py executes the shared launcher dispatch command from the repository root.
#   test_root_unix_shell_wrapper_delegates_to_launch_py - Verifies launch.sh stays a thin wrapper over launch.py.
#   test_root_windows_bat_wrapper_delegates_to_launch_py - Verifies launch.bat stays a thin wrapper over launch.py while preserving pause-on-error markers.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added deterministic coverage for the new root-level cross-platform launcher entrypoints]
# END_CHANGE_SUMMARY

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LAUNCH_PY_PATH = PROJECT_ROOT / "launch.py"
LAUNCH_SH_PATH = PROJECT_ROOT / "launch.sh"
LAUNCH_BAT_PATH = PROJECT_ROOT / "launch.bat"
pytestmark = pytest.mark.unit


def _load_launch_module():
    spec = importlib.util.spec_from_file_location("root_launch", LAUNCH_PY_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("Unable to load launch.py module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_root_python_launcher_delegates_to_launcher_module(monkeypatch: pytest.MonkeyPatch):
    launch_module = _load_launch_module()
    recorded: dict[str, object] = {}

    def fake_run(command: list[str], *, cwd: str, check: bool) -> subprocess.CompletedProcess[str]:
        recorded["command"] = command
        recorded["cwd"] = cwd
        recorded["check"] = check
        return subprocess.CompletedProcess(command, 9)

    monkeypatch.setattr(launch_module.subprocess, "run", fake_run)
    monkeypatch.setattr(launch_module.sys, "executable", sys.executable)

    exit_code = launch_module.main()

    assert exit_code == 9
    assert recorded["command"] == [
        sys.executable,
        "-m",
        "launcher",
        "--project-root",
        str(PROJECT_ROOT),
        "launch",
    ]
    assert recorded["cwd"] == str(PROJECT_ROOT)
    assert recorded["check"] is False


def test_root_unix_shell_wrapper_delegates_to_launch_py():
    contents = LAUNCH_SH_PATH.read_text(encoding="utf-8")

    assert LAUNCH_SH_PATH.exists()
    assert "START_MODULE_CONTRACT" in contents
    assert "root-level Unix shell wrapper" in contents
    assert 'exec python3 "$SCRIPT_DIR/launch.py"' in contents


def test_root_windows_bat_wrapper_delegates_to_launch_py():
    contents = LAUNCH_BAT_PATH.read_text(encoding="utf-8")

    assert LAUNCH_BAT_PATH.exists()
    assert "START_MODULE_CONTRACT" in contents
    assert "shared cross-platform Python launcher flow" in contents
    assert "launch.py" in contents
    assert 'py -3.11 "%PYTHON_LAUNCHER%"' in contents
    assert 'python "%PYTHON_LAUNCHER%"' in contents
    assert "Neither 'py' nor 'python' was found in PATH." in contents
    assert "pause" in contents
