# FILE: tests/unit/scripts/test_launcher_launch.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Validate the universal launcher dispatch command that routes to the existing platform-specific interactive wrappers.
#   SCOPE: deterministic wrapper command selection, environment shaping, subprocess delegation, and unsupported-platform failure payloads
#   DEPENDS: M-LAUNCHER
#   LINKS: V-M-LAUNCHER
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   PROJECT_ROOT - Repository root used to assert wrapper paths and launch cwd.
#   test_interactive_launcher_command_selects_windows_cmd_wrapper - Verifies Windows dispatch prefers the CMD compatibility wrapper.
#   test_interactive_launcher_command_selects_macos_wrapper - Verifies macOS dispatch chooses the guided shell wrapper.
#   test_interactive_launcher_command_selects_linux_wrapper - Verifies Linux dispatch chooses the guided shell wrapper.
#   test_interactive_launcher_env_sets_project_root_fallback - Verifies the universal launch flow preserves the wrapper project-root fallback contract.
#   test_launcher_launch_delegates_to_platform_wrapper - Verifies the launch subcommand invokes the selected wrapper with inherited interactivity and propagated exit code.
#   test_launcher_launch_reports_unsupported_platform - Verifies unsupported hosts fail with a structured JSON payload rather than a traceback.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added unit coverage for the universal launcher dispatch subcommand and its platform-specific wrapper routing]
# END_CHANGE_SUMMARY

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

launcher_main = importlib.import_module("launcher.main")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.unit


def test_interactive_launcher_command_selects_windows_cmd_wrapper():
    command = launcher_main._interactive_launcher_command(PROJECT_ROOT, platform_system="Windows")

    assert command == ["cmd.exe", "/c", str(PROJECT_ROOT / "scripts" / "launch-windows.cmd")]


def test_interactive_launcher_command_selects_macos_wrapper():
    command = launcher_main._interactive_launcher_command(PROJECT_ROOT, platform_system="Darwin")

    assert command == ["bash", str(PROJECT_ROOT / "scripts" / "launch-macos.sh")]


def test_interactive_launcher_command_selects_linux_wrapper():
    command = launcher_main._interactive_launcher_command(PROJECT_ROOT, platform_system="Linux")

    assert command == ["bash", str(PROJECT_ROOT / "scripts" / "launch-linux.sh")]


def test_interactive_launcher_env_sets_project_root_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(launcher_main.os, "environ", {"PATH": "existing-path"})

    env = launcher_main._interactive_launcher_env(PROJECT_ROOT)

    assert env["PATH"] == "existing-path"
    assert env["TTS_LAUNCH_PROJECT_ROOT"] == str(PROJECT_ROOT)


def test_launcher_launch_delegates_to_platform_wrapper(
    monkeypatch: pytest.MonkeyPatch,
):
    recorded: dict[str, object] = {}

    def fake_run(
        command: list[str], *, cwd: str, env: dict[str, str], check: bool
    ) -> subprocess.CompletedProcess[str]:
        recorded["command"] = command
        recorded["cwd"] = cwd
        recorded["env"] = env
        recorded["check"] = check
        return subprocess.CompletedProcess(command, 17)

    monkeypatch.setattr(launcher_main.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher_main.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["launcher", "--project-root", str(PROJECT_ROOT), "launch"])

    exit_code = launcher_main.main()

    assert exit_code == 17
    assert recorded["command"] == ["bash", str(PROJECT_ROOT / "scripts" / "launch-linux.sh")]
    assert recorded["cwd"] == str(PROJECT_ROOT)
    assert recorded["check"] is False
    assert recorded["env"]["TTS_LAUNCH_PROJECT_ROOT"] == str(PROJECT_ROOT)


def test_launcher_launch_reports_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(launcher_main.platform, "system", lambda: "FreeBSD")
    monkeypatch.setattr(sys, "argv", ["launcher", "--project-root", str(PROJECT_ROOT), "launch"])

    exit_code = launcher_main.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload["launch"] == {
        "project_root": str(PROJECT_ROOT),
        "platform": "freebsd",
        "error": "unsupported_platform",
        "message": "Unsupported interactive launcher platform: freebsd",
    }
