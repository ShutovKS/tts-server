# FILE: tests/unit/scripts/test_launcher_create_env.py
# VERSION: 1.1.3
# START_MODULE_CONTRACT
#   PURPOSE: Validate the launcher create-env command for isolated family environment planning and failure reporting.
#   SCOPE: dry-run step output, platform-aware interpreter/env paths, requirements preview formatting, and deterministic apply failure payloads
#   DEPENDS: M-LAUNCHER
#   LINKS: V-M-LAUNCHER
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _preview_has_suffix - Normalize requirements preview paths before suffix assertions
#   _run_create_env - Execute launcher create-env in-process and capture its JSON payload deterministically
#   test_launcher_create_env_outputs_qwen_steps_without_apply - Verifies qwen dry-run output exposes the launcher's current bootstrap argv and pip install steps
#   test_launcher_create_env_outputs_omnivoice_steps_without_apply - Verifies omnivoice dry-run output resolves the family-specific env root and dependency pack preview without claiming host-shell portability
#   test_launcher_create_env_preview_uses_pip_safe_posix_paths_on_windows - Verifies preview requirements use pip-safe POSIX include paths on Windows while remaining stable elsewhere
#   test_launcher_create_env_dry_run_avoids_runtime_bootstrap_imports - Verifies create-env dry-run can emit JSON even when runtime-heavy imports such as numpy and core.bootstrap are blocked.
#   test_launcher_create_env_apply_reports_error_json_when_install_fails - Verifies apply mode reports a deterministic install failure envelope and removes the compiled requirements file
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.3 - Updated create-env assertions for host-aware venv creation commands while preserving runtime-import isolation coverage]
# END_CHANGE_SUMMARY

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from launcher.main import main


pytestmark = pytest.mark.unit


PROJECT_ROOT = Path(__file__).resolve().parents[3]


# START_CONTRACT: _preview_has_suffix
#   PURPOSE: Check whether the previewed requirements file includes a specific dependency pack path suffix independent of separator style.
#   INPUTS: { payload: dict - launcher JSON payload, suffix: str - normalized Windows-style suffix to locate }
#   OUTPUTS: { bool - true when a matching preview include line is present }
#   SIDE_EFFECTS: none
#   LINKS: M-LAUNCHER
# END_CONTRACT: _preview_has_suffix
def _preview_has_suffix(payload: dict, suffix: str) -> bool:
    return any(
        line.replace("/", "\\").endswith(suffix)
        for line in payload["create_env"]["compiled_requirements"]["preview_lines"]
        if line.startswith("-r ")
    )


# START_CONTRACT: _run_create_env
#   PURPOSE: Execute launcher create-env in-process so tests can assert structured output without ambient shell state.
#   INPUTS: { args: tuple[str, ...] - create-env CLI arguments following the subcommand }
#   OUTPUTS: { tuple[int, dict] - launcher exit code and parsed JSON payload }
#   SIDE_EFFECTS: Mutates process argv for the duration of the call and captures stdout through pytest fixtures
#   LINKS: M-LAUNCHER
# END_CONTRACT: _run_create_env
def _run_create_env(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, dict]:
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "launcher",
            "--project-root",
            str(PROJECT_ROOT),
            "create-env",
            *args,
        ]
        exit_code = main()
    finally:
        sys.argv = original_argv
    captured = capsys.readouterr()
    return exit_code, json.loads(captured.out)


def test_launcher_create_env_outputs_qwen_steps_without_apply(capsys: pytest.CaptureFixture[str]):
    exit_code, payload = _run_create_env(capsys, "--family", "qwen", "--module", "server")
    steps = payload["create_env"]["steps"]

    assert exit_code == 0
    assert payload["create_env"]["apply"] is False
    assert payload["create_env"]["family"] == "qwen"
    if platform.system().lower() == "windows":
        assert steps[0] == ["py", "-3.11", "-m", "venv", payload["create_env"]["expected_env_root"]]
    else:
        assert steps[0] == [sys.executable, "-m", "venv", payload["create_env"]["expected_env_root"]]
    assert steps[1][1:] == ["-m", "pip", "install", "--upgrade", "pip"]
    assert steps[2][-2:] == ["-r", payload["create_env"]["compiled_requirements_path"]]
    assert _preview_has_suffix(payload, "profiles\\packs\\family\\qwen.txt")


def test_launcher_create_env_outputs_omnivoice_steps_without_apply(capsys: pytest.CaptureFixture[str]):
    exit_code, payload = _run_create_env(capsys, "--family", "omnivoice", "--module", "telegram")
    steps = payload["create_env"]["steps"]
    expected_env_fragment = str(Path(".envs") / "omnivoice")

    assert exit_code == 0
    assert payload["create_env"]["family"] == "omnivoice"
    assert expected_env_fragment in steps[0][-1]
    if platform.system().lower() == "windows":
        assert payload["create_env"]["expected_python_path"].endswith(str(Path(".envs") / "omnivoice" / "Scripts" / "python.exe"))
    else:
        assert payload["create_env"]["expected_python_path"].endswith(str(Path(".envs") / "omnivoice" / "bin" / "python"))
    assert _preview_has_suffix(payload, "profiles\\packs\\family\\omnivoice.txt")


def test_launcher_create_env_preview_uses_pip_safe_posix_paths_on_windows(capsys: pytest.CaptureFixture[str]):
    exit_code, payload = _run_create_env(capsys, "--family", "qwen", "--module", "server")
    preview_lines = [
        line for line in payload["create_env"]["compiled_requirements"]["preview_lines"] if line.startswith("-r ")
    ]

    assert exit_code == 0
    if platform.system().lower() == "windows":
        assert all("\\" not in line[3:] for line in preview_lines)
        assert any(line[3:].startswith("C:/") for line in preview_lines)
    else:
        assert all(line[3:].startswith("/") for line in preview_lines)


def test_launcher_create_env_dry_run_avoids_runtime_bootstrap_imports(tmp_path: Path):
    sitecustomize_path = tmp_path / "sitecustomize.py"
    sitecustomize_path.write_text(
        "import builtins\n"
        "import os\n"
        "\n"
        "_real_import = builtins.__import__\n"
        "_blocked = tuple(filter(None, os.environ.get('TTS_BLOCKED_IMPORTS', '').split(',')))\n"
        "\n"
        "def _guard(name, globals=None, locals=None, fromlist=(), level=0):\n"
        "    for blocked in _blocked:\n"
        "        if name == blocked or name.startswith(blocked + '.'):\n"
        "            raise RuntimeError(f'blocked import: {name}')\n"
        "    return _real_import(name, globals, locals, fromlist, level)\n"
        "\n"
        "builtins.__import__ = _guard\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{tmp_path}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(tmp_path)
    )
    env["TTS_BLOCKED_IMPORTS"] = "numpy,torch,core.bootstrap,core.backends,core.backends.torch_backend"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "launcher",
            "--project-root",
            str(PROJECT_ROOT),
            "create-env",
            "--family",
            "qwen",
            "--module",
            "server",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["create_env"]["family"] == "qwen"
    assert payload["create_env"]["apply"] is False


def test_launcher_create_env_apply_reports_error_json_when_install_fails(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    compiled_requirements_path = tmp_path / "compiled-requirements.txt"
    compiled_requirements_path.write_text("-r /tmp/family.txt\n", encoding="utf-8")
    removed_paths: list[Path] = []
    real_path_exists = Path.exists
    real_path_unlink = Path.unlink
    install_command: list[str] | None = None

    def fake_exists(path: Path) -> bool:
        path_str = str(path)
        if path_str == str(compiled_requirements_path):
            return not removed_paths
        if path_str.endswith(str(Path(".envs") / "qwen" / "Scripts" / "python.exe")):
            return True
        if path_str.endswith(str(Path(".envs") / "qwen" / "bin" / "python")):
            return True
        return real_path_exists(path)

    def fake_unlink(path: Path, missing_ok: bool = False) -> None:
        if path == compiled_requirements_path:
            removed_paths.append(path)
            return
        real_path_unlink(path, missing_ok=missing_ok)

    def fake_run(cmd: list[str], check: bool, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal install_command
        if len(cmd) >= 4 and cmd[1:3] == ["-m", "venv"]:
            raise AssertionError("bootstrap step should be skipped when expected python already exists")
        if len(cmd) >= 3 and cmd[1] == "-c":
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout='{"torch": false, "qwen_tts": false}\n',
                stderr="",
            )
        if cmd[-2:] == ["--upgrade", "pip"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        install_command = cmd
        raise subprocess.CalledProcessError(
            returncode=23,
            cmd=cmd,
            output="install stdout\n",
            stderr="install stderr\n",
        )

    monkeypatch.setattr("launcher.main._write_compiled_requirements_file", lambda resolved: str(compiled_requirements_path))
    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(Path, "unlink", fake_unlink)
    monkeypatch.setattr(subprocess, "run", fake_run)

    exit_code, payload = _run_create_env(capsys, "--family", "qwen", "--module", "server", "--apply")

    assert exit_code == 1
    assert install_command is not None
    assert payload["create_env"]["apply"] is True
    assert payload["create_env"]["created"] is True
    assert payload["create_env"]["error"] == {
        "step": "install_compiled_requirements",
        "command": install_command,
        "returncode": 23,
        "stdout": "install stdout",
        "stderr": "install stderr",
    }
    assert removed_paths == [compiled_requirements_path]
