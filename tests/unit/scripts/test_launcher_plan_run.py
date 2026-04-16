# FILE: tests/unit/scripts/test_launcher_plan_run.py
# VERSION: 1.0.1
# START_MODULE_CONTRACT
#   PURPOSE: Validate the launcher plan-run command for family/module runtime planning.
#   SCOPE: deterministic launcher plan payloads for resolved family/module contours
#   DEPENDS: M-LAUNCHER
#   LINKS: V-M-LAUNCHER
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _make_deterministic_host_profile - Build a stable host profile for launcher payload assertions without depending on local optional runtimes
#   _expected_python_path - Compute the platform-aware dedicated interpreter path for a family environment
#   _expected_platform_pack_name - Resolve the platform overlay pack expected for the deterministic host profile
#   _expected_pack_files - Build the exact dependency pack file list expected in launcher payloads
#   _expected_preview_lines - Build the exact requirements preview lines expected in launcher payloads
#   _run_plan_run - Execute the launcher plan-run flow in-process with deterministic host probing
#   test_launcher_plan_run_outputs_deterministic_qwen_server_launch_plan - Verifies qwen server planning includes the exact interpreter, backend, and dependency pack payload
#   test_launcher_plan_run_outputs_deterministic_piper_cli_launch_plan - Verifies piper CLI planning includes the exact interpreter, backend, and dependency pack payload
#   test_launcher_plan_run_outputs_deterministic_omnivoice_cli_launch_plan - Verifies omnivoice CLI planning includes the exact interpreter, backend, and dependency pack payload
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.1 - Replaced launcher subprocess-only checks with deterministic in-process payload assertions and explicit unit selection]
# END_CHANGE_SUMMARY

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import pytest

import launcher.main as launcher_main
from profiles.schema import HostProfile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.unit


# START_CONTRACT: _make_deterministic_host_profile
#   PURPOSE: Build a stable host profile used to keep launcher payload assertions independent from local optional runtime packages.
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
#   PURPOSE: Compute the platform-aware dedicated interpreter path expected in launcher payloads.
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


# START_CONTRACT: _expected_platform_pack_name
#   PURPOSE: Resolve the platform overlay pack expected for the deterministic host profile.
#   INPUTS: { host_profile: HostProfile - normalized host profile used by the test resolver }
#   OUTPUTS: { str - expected platform overlay dependency pack name }
#   SIDE_EFFECTS: none
#   LINKS: V-M-LAUNCHER
# END_CONTRACT: _expected_platform_pack_name
def _expected_platform_pack_name(host_profile: HostProfile) -> str:
    if host_profile.platform_system.lower() == "darwin" and host_profile.architecture.lower() in {"arm64", "aarch64"}:
        return "apple-silicon"
    return "cpu"


# START_CONTRACT: _expected_pack_files
#   PURPOSE: Build the exact dependency pack file list expected in plan-run payloads.
#   INPUTS: { family: str - requested runtime family, module: str - requested launcher module, host_profile: HostProfile - normalized host profile used by the resolver }
#   OUTPUTS: { list[str] - ordered dependency pack file paths }
#   SIDE_EFFECTS: none
#   LINKS: V-M-LAUNCHER
# END_CONTRACT: _expected_pack_files
def _expected_pack_files(family: str, module: str, host_profile: HostProfile) -> list[str]:
    return [
        str(PROJECT_ROOT / "profiles" / "packs" / "base" / "common.txt"),
        str(PROJECT_ROOT / "profiles" / "packs" / "platform" / f"{host_profile.platform_system.lower()}.txt"),
        str(PROJECT_ROOT / "profiles" / "packs" / "platform" / f"{_expected_platform_pack_name(host_profile)}.txt"),
        str(PROJECT_ROOT / "profiles" / "packs" / "module" / f"{module}.txt"),
        str(PROJECT_ROOT / "profiles" / "packs" / "family" / f"{family}.txt"),
    ]


# START_CONTRACT: _expected_preview_lines
#   PURPOSE: Build the exact compiled-requirements preview lines expected in plan-run payloads.
#   INPUTS: { family: str - requested runtime family, module: str - requested launcher module, host_profile: HostProfile - normalized host profile used by the resolver }
#   OUTPUTS: { list[str] - ordered preview lines rendered by launcher planning }
#   SIDE_EFFECTS: none
#   LINKS: V-M-LAUNCHER
# END_CONTRACT: _expected_preview_lines
def _expected_preview_lines(family: str, module: str, host_profile: HostProfile) -> list[str]:
    pack_files = _expected_pack_files(family, module, host_profile)
    return [
        "# generated by launcher",
        f"# family: {family}",
        f"# module: {module}",
        f"# platform: {host_profile.platform_system.lower()}/{host_profile.architecture.lower()}",
        "",
        *(f"-r {Path(pack_file).resolve().as_posix()}" for pack_file in pack_files),
    ]


# START_CONTRACT: _run_plan_run
#   PURPOSE: Execute launcher plan-run in-process with deterministic host and environment probing.
#   INPUTS: { monkeypatch: pytest.MonkeyPatch - pytest monkeypatch fixture, capsys: pytest.CaptureFixture[str] - stdout capture fixture, family: str - requested runtime family, module: str - requested launcher module }
#   OUTPUTS: { dict[str, object] - parsed launcher JSON payload }
#   SIDE_EFFECTS: Temporarily overrides launcher resolver behavior and process argv for the current test
#   LINKS: V-M-LAUNCHER
# END_CONTRACT: _run_plan_run
def _run_plan_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    family: str,
    module: str,
) -> dict[str, object]:
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
        sys,
        "argv",
        ["launcher", "plan-run", "--family", family, "--module", module],
    )

    exit_code = launcher_main.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    return json.loads(captured.out)


def test_launcher_plan_run_outputs_deterministic_qwen_server_launch_plan(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    host_profile = _make_deterministic_host_profile()
    payload = _run_plan_run(monkeypatch, capsys, family="qwen", module="server")

    assert payload["launch_plan"] == {
        "family": "qwen",
        "module": "server",
        "required_env_name": "qwen",
        "expected_python_path": _expected_python_path("qwen"),
        "dependency_plan": {
            "pack_refs": {
                "base": ["common"],
                "platform": [host_profile.platform_system.lower(), _expected_platform_pack_name(host_profile)],
                "module": ["server"],
                "family": ["qwen"],
            },
            "pack_files": _expected_pack_files("qwen", "server", host_profile),
            "preview_lines": _expected_preview_lines("qwen", "server", host_profile),
        },
        "entrypoint": "python -m server",
        "selected_backend": "mlx" if host_profile.platform_system.lower() == "darwin" else "torch",
        "backend_candidates": ["mlx", "qwen_fast", "torch"],
        "compatible": True,
        "reasons": [],
    }


def test_launcher_plan_run_outputs_deterministic_piper_cli_launch_plan(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    host_profile = _make_deterministic_host_profile()
    payload = _run_plan_run(monkeypatch, capsys, family="piper", module="cli")

    assert payload["launch_plan"] == {
        "family": "piper",
        "module": "cli",
        "required_env_name": "piper",
        "expected_python_path": _expected_python_path("piper"),
        "dependency_plan": {
            "pack_refs": {
                "base": ["common"],
                "platform": [host_profile.platform_system.lower(), _expected_platform_pack_name(host_profile)],
                "module": ["cli"],
                "family": ["piper"],
            },
            "pack_files": _expected_pack_files("piper", "cli", host_profile),
            "preview_lines": _expected_preview_lines("piper", "cli", host_profile),
        },
        "entrypoint": "python -m cli",
        "selected_backend": "onnx",
        "backend_candidates": ["onnx"],
        "compatible": True,
        "reasons": [],
    }


def test_launcher_plan_run_outputs_deterministic_omnivoice_cli_launch_plan(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    host_profile = _make_deterministic_host_profile()
    payload = _run_plan_run(monkeypatch, capsys, family="omnivoice", module="cli")

    assert payload["launch_plan"] == {
        "family": "omnivoice",
        "module": "cli",
        "required_env_name": "omnivoice",
        "expected_python_path": _expected_python_path("omnivoice"),
        "dependency_plan": {
            "pack_refs": {
                "base": ["common"],
                "platform": [host_profile.platform_system.lower(), _expected_platform_pack_name(host_profile)],
                "module": ["cli"],
                "family": ["omnivoice"],
            },
            "pack_files": _expected_pack_files("omnivoice", "cli", host_profile),
            "preview_lines": _expected_preview_lines("omnivoice", "cli", host_profile),
        },
        "entrypoint": "python -m cli",
        "selected_backend": "torch",
        "backend_candidates": ["torch"],
        "compatible": True,
        "reasons": [],
    }
