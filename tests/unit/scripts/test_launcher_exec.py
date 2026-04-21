# FILE: tests/unit/scripts/test_launcher_exec.py
# VERSION: 1.0.1
# START_MODULE_CONTRACT
#   PURPOSE: Validate the launcher exec command for resolved family entrypoint execution planning.
#   SCOPE: deterministic dry-run command payloads and missing-python command wiring behavior
#   DEPENDS: M-LAUNCHER
#   LINKS: V-M-LAUNCHER
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _make_deterministic_host_profile - Build a stable host profile for dry-run launcher command assertions
#   _expected_python_path - Compute the platform-aware dedicated interpreter path for a family environment
#   _run_exec_dry_run - Execute launcher exec --dry-run in-process with deterministic host probing and interpreter existence
#   test_launcher_exec_dry_run_outputs_platform_aware_qwen_command - Verifies dry-run exec returns the exact qwen server command payload
#   test_launcher_exec_reports_missing_python_when_env_absent - Verifies subprocess CLI wiring reports the exact missing-interpreter payload for telegram execution
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.1 - Shifted exec dry-run evidence to deterministic in-process command assertions while preserving subprocess coverage for missing-interpreter wiring]
# END_CHANGE_SUMMARY

from __future__ import annotations

import json
import importlib
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from profiles.schema import HostProfile

launcher_main = importlib.import_module("launcher.main")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.unit


# START_CONTRACT: _make_deterministic_host_profile
#   PURPOSE: Build a stable host profile used to keep launcher exec dry-run assertions independent from local optional runtimes.
#   INPUTS: none
#   OUTPUTS: { HostProfile - normalized host profile with predictable compatibility characteristics }
#   SIDE_EFFECTS: none
#   LINKS: V-M-LAUNCHER, M-PROFILE-SCHEMA
# END_CONTRACT: _make_deterministic_host_profile
def _make_deterministic_host_profile() -> HostProfile:
    system_name = platform.system() or "Windows"
    architecture = platform.machine() or "amd64"
    return HostProfile(
        key=f"{system_name.lower()}-{architecture.lower()}",
        platform_system=system_name,
        architecture=architecture,
        python_version="3.11.9",
        ffmpeg_available=True,
        docker_available=True,
        torch_runtime_available=True,
        cuda_available=False,
        onnx_providers=("CPUExecutionProvider",),
    )


# START_CONTRACT: _expected_python_path
#   PURPOSE: Compute the platform-aware dedicated interpreter path expected in launcher exec payloads.
#   INPUTS: { env_name: str - isolated family environment name, project_root: Path - repository root used to build the env path }
#   OUTPUTS: { str - expected interpreter path for the current platform }
#   SIDE_EFFECTS: none
#   LINKS: V-M-LAUNCHER
# END_CONTRACT: _expected_python_path
def _expected_python_path(env_name: str, project_root: Path = PROJECT_ROOT) -> str:
    env_root = project_root / ".envs" / env_name
    if platform.system().lower() == "windows":
        return str(env_root / "Scripts" / "python.exe")
    return str(env_root / "bin" / "python")


# START_CONTRACT: _run_exec_dry_run
#   PURPOSE: Execute launcher exec in dry-run mode with deterministic resolver and interpreter-existence behavior.
#   INPUTS: { monkeypatch: pytest.MonkeyPatch - pytest monkeypatch fixture, capsys: pytest.CaptureFixture[str] - stdout capture fixture, family: str - requested runtime family, module: str - requested launcher module, python_exists: bool - whether the resolved interpreter should be reported as present }
#   OUTPUTS: { dict[str, object] - parsed launcher JSON payload }
#   SIDE_EFFECTS: Temporarily overrides launcher resolver behavior, interpreter existence checks, and process argv for the current test
#   LINKS: V-M-LAUNCHER
# END_CONTRACT: _run_exec_dry_run
def _run_exec_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    family: str,
    module: str,
    python_exists: bool,
) -> dict[str, object]:
    expected_python_path = _expected_python_path(family)
    monkeypatch.setattr(
        launcher_main.ProfileResolver,
        "resolve_host",
        lambda self: _make_deterministic_host_profile(),
    )
    monkeypatch.setattr(
        launcher_main.ProfileResolver,
        "_family_env_runtime_ready",
        lambda self, family_profile: False,
    )
    monkeypatch.setattr(
        launcher_main.Path,
        "exists",
        lambda path: str(path) == expected_python_path if python_exists else False,
    )
    monkeypatch.setattr(
        launcher_main.os,
        "environ",
        {
            "TTS_ACTIVE_FAMILY": family,
            "TTS_DEFAULT_CUSTOM_MODEL": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "TTS_DEFAULT_DESIGN_MODEL": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
            "TTS_DEFAULT_CLONE_MODEL": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["launcher", "exec", "--family", family, "--module", module, "--dry-run"],
    )

    exit_code = launcher_main.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    return json.loads(captured.out)


def test_launcher_exec_dry_run_outputs_platform_aware_qwen_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    payload = _run_exec_dry_run(
        monkeypatch,
        capsys,
        family="qwen",
        module="server",
        python_exists=True,
    )

    assert payload["exec"] == {
        "family": "qwen",
        "module": "server",
        "command": [_expected_python_path("qwen"), "-m", "server"],
        "dry_run": True,
        "python_exists": True,
        "runtime_bindings": {
            "bindings": {
                "family": "qwen",
                "custom_model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                "design_model": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
                "clone_model": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
            },
            "capability_status": {
                "custom": {
                    "bound": True,
                    "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                    "env_var": "TTS_DEFAULT_CUSTOM_MODEL",
                },
                "design": {
                    "bound": True,
                    "model": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
                    "env_var": "TTS_DEFAULT_DESIGN_MODEL",
                },
                "clone": {
                    "bound": True,
                    "model": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
                    "env_var": "TTS_DEFAULT_CLONE_MODEL",
                },
            },
        },
    }


def test_launcher_exec_reports_missing_python_when_env_absent():
    with tempfile.TemporaryDirectory(prefix="tts-launcher-exec-") as temp_dir:
        isolated_root = Path(temp_dir) / PROJECT_ROOT.name
        shutil.copytree(
            PROJECT_ROOT,
            isolated_root,
            ignore=shutil.ignore_patterns(
                ".git", ".envs", ".venv*", "__pycache__", ".pytest_cache"
            ),
        )

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "launcher",
                "--project-root",
                str(isolated_root),
                "exec",
                "--family",
                "omnivoice",
                "--module",
                "telegram",
            ],
            cwd=isolated_root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 1
        payload = json.loads(completed.stdout)
        assert payload["exec"]["family"] == "omnivoice"
        assert payload["exec"]["module"] == "telegram"
        assert Path(payload["exec"]["command"][0]).resolve() == Path(
            _expected_python_path("omnivoice", isolated_root)
        ).resolve()
        assert payload["exec"]["command"][1:] == ["-m", "telegram_bot"]
        assert payload["exec"]["dry_run"] is False
        assert payload["exec"]["python_exists"] is False
        assert payload["exec"]["runtime_bindings"] == {
            "bindings": {
                "family": "omnivoice",
                "custom_model": "OmniVoice-Custom",
                "design_model": "OmniVoice-Design",
                "clone_model": "OmniVoice-Clone",
            },
            "capability_status": {
                "custom": {"bound": True, "model": "OmniVoice-Custom", "env_var": "TTS_DEFAULT_CUSTOM_MODEL"},
                "design": {"bound": True, "model": "OmniVoice-Design", "env_var": "TTS_DEFAULT_DESIGN_MODEL"},
                "clone": {"bound": True, "model": "OmniVoice-Clone", "env_var": "TTS_DEFAULT_CLONE_MODEL"},
            },
        }
        assert payload["exec"]["error"] == "expected_python_missing"
