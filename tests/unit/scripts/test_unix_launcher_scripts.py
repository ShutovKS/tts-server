# FILE: tests/unit/scripts/test_unix_launcher_scripts.py
# VERSION: 1.1.6
# START_MODULE_CONTRACT
#   PURPOSE: Validate the macOS and Linux launcher entrypoints stay aligned with the documented profile-aware guided launch flow.
#   SCOPE: launcher-script presence, shell orchestration markers, curated family/model menu anchors, runtime capability binding markers, platform-specific dependency guidance, and bounded secret-handling command references
#   DEPENDS: M-MACOS-LAUNCHER, M-LINUX-LAUNCHER
#   LINKS: V-M-MACOS-LAUNCHER, V-M-LINUX-LAUNCHER
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   MACOS_SCRIPT_PATH - Canonical path to the interactive macOS launcher script.
#   LINUX_SCRIPT_PATH - Canonical path to the interactive Linux launcher script.
#   test_macos_launcher_script_exists_with_grace_contract - Verifies the macOS launcher file exists and retains top-level GRACE contract anchors.
#   test_macos_launcher_script_reuses_profile_aware_launcher_and_brew_guidance - Verifies the macOS launcher delegates env setup and execution to launcher commands while keeping Homebrew guidance bounded and opt-in.
#   test_unix_launcher_menu_rendering_uses_stderr_while_returning_selection_on_stdout - Verifies the Unix launcher menu helpers print UI to stderr while preserving stdout for selected records.
#   test_unix_launcher_multi_select_returns_newline_terminated_records - Verifies Unix multi-select helpers newline-terminate selected records so process-substitution readers do not drop the last choice.
#   test_unix_launcher_server_start_checks_port_before_health_probe - Verifies Unix launcher server startup preflights the target port and does not ignore failed health probes.
#   test_unix_launcher_server_restart_uses_pid_file_management - Verifies Unix launchers carry repo-local PID lifecycle helpers for restarting launcher-managed HTTP server processes.
#   test_linux_launcher_script_exists_with_grace_contract - Verifies the Linux launcher file exists and retains top-level GRACE contract anchors.
#   test_linux_launcher_script_reuses_profile_aware_launcher_and_manual_package_guidance - Verifies the Linux launcher delegates env setup and execution to launcher commands while keeping package-manager guidance manual-only.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.6 - Strengthened Unix launcher regression coverage so restart flows track and stop the real HTTP listener PID instead of only the launcher wrapper]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


MACOS_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "launch-macos.sh"
LINUX_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "launch-linux.sh"


def test_macos_launcher_script_exists_with_grace_contract():
    contents = MACOS_SCRIPT_PATH.read_text(encoding="utf-8")

    assert MACOS_SCRIPT_PATH.exists()
    assert "# START_MODULE_CONTRACT" in contents
    assert "#   PURPOSE: Provide an interactive macOS launcher" in contents
    assert "FAMILY_OPTIONS_DATA" in contents
    assert "MODEL_OPTIONS_DATA" in contents
    assert "assert_macos_preflight" in contents
    assert "brew install" in contents


def test_macos_launcher_script_reuses_profile_aware_launcher_and_brew_guidance():
    contents = MACOS_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "python3.11 -m launcher --project-root" in contents
    assert "create-env" in contents
    assert "check-env" in contents
    assert "exec --family" in contents
    assert "expected_python_path" in contents
    assert "/health/live" in contents
    assert "Select family to prepare" in contents
    assert "select_multiple_menu_options" in contents
    assert "TTS_ACTIVE_FAMILY" in contents
    assert "TTS_DEFAULT_CUSTOM_MODEL" in contents
    assert "TTS_DEFAULT_DESIGN_MODEL" in contents
    assert "TTS_DEFAULT_CLONE_MODEL" in contents
    assert "Runtime capability bindings:" in contents
    assert "Automatically preparing the only model" in contents
    assert "snapshot_download" in contents
    assert "piper.download_voices" in contents
    assert "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" in contents
    assert "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" in contents
    assert "Qwen/Qwen3-TTS-12Hz-1.7B-Base" in contents
    assert "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" in contents
    assert "Qwen/Qwen3-TTS-12Hz-0.6B-Base" in contents
    assert "k2-fsa/OmniVoice" in contents
    assert "HF_TOKEN" in contents
    assert "TTS_TELEGRAM_BOT_TOKEN" in contents
    assert "Run brew install now? [y/N]:" in contents
    assert "Homebrew was not found" in contents
    assert "Qwen Custom 1.7B" in contents
    assert "OmniVoice" in contents
    assert "Piper en_US lessac medium" in contents


def test_linux_launcher_script_exists_with_grace_contract():
    contents = LINUX_SCRIPT_PATH.read_text(encoding="utf-8")

    assert LINUX_SCRIPT_PATH.exists()
    assert "# START_MODULE_CONTRACT" in contents
    assert "#   PURPOSE: Provide an interactive Linux launcher" in contents
    assert "FAMILY_OPTIONS_DATA" in contents
    assert "MODEL_OPTIONS_DATA" in contents
    assert "detect_package_manager" in contents
    assert "print_linux_install_guidance" in contents


def test_linux_launcher_script_reuses_profile_aware_launcher_and_manual_package_guidance():
    contents = LINUX_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "python3.11 -m launcher --project-root" in contents
    assert "create-env" in contents
    assert "check-env" in contents
    assert "exec --family" in contents
    assert "expected_python_path" in contents
    assert "/health/live" in contents
    assert "Select family to prepare" in contents
    assert "select_multiple_menu_options" in contents
    assert "TTS_ACTIVE_FAMILY" in contents
    assert "TTS_DEFAULT_CUSTOM_MODEL" in contents
    assert "TTS_DEFAULT_DESIGN_MODEL" in contents
    assert "TTS_DEFAULT_CLONE_MODEL" in contents
    assert "Runtime capability bindings:" in contents
    assert "Automatically preparing the only model" in contents
    assert "snapshot_download" in contents
    assert "piper.download_voices" in contents
    assert "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" in contents
    assert "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" in contents
    assert "Qwen/Qwen3-TTS-12Hz-1.7B-Base" in contents
    assert "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" in contents
    assert "Qwen/Qwen3-TTS-12Hz-0.6B-Base" in contents
    assert "k2-fsa/OmniVoice" in contents
    assert "HF_TOKEN" in contents
    assert "TTS_TELEGRAM_BOT_TOKEN" in contents
    assert "This script does not install system packages automatically." in contents
    assert "sudo apt-get install -y python3.11 python3.11-venv ffmpeg" in contents
    assert "sudo dnf install -y python3.11 ffmpeg" in contents
    assert "sudo yum install -y python3.11 ffmpeg" in contents
    assert "sudo pacman -S --needed python ffmpeg" in contents
    assert "sudo zypper install -y python311 ffmpeg" in contents


def test_unix_launcher_menu_rendering_uses_stderr_while_returning_selection_on_stdout():
    linux_contents = LINUX_SCRIPT_PATH.read_text(encoding="utf-8")
    macos_contents = MACOS_SCRIPT_PATH.read_text(encoding="utf-8")

    for contents in (linux_contents, macos_contents):
        assert "printf '\\n%s\\n' \"$prompt\" >&2" in contents
        assert "printf '[%d] %s\\n' \"$count\" \"$label\" >&2" in contents
        assert "printf '%s' \"$line\"" in contents


def test_unix_launcher_multi_select_returns_newline_terminated_records():
    linux_contents = LINUX_SCRIPT_PATH.read_text(encoding="utf-8")
    macos_contents = MACOS_SCRIPT_PATH.read_text(encoding="utf-8")

    for contents in (linux_contents, macos_contents):
        assert "printf '%s\\n' \"$selection_output\"" in contents


def test_unix_launcher_server_start_checks_port_before_health_probe():
    linux_contents = LINUX_SCRIPT_PATH.read_text(encoding="utf-8")
    macos_contents = MACOS_SCRIPT_PATH.read_text(encoding="utf-8")

    for contents in (linux_contents, macos_contents):
        assert "assert_http_server_port_available()" in contents
        assert "ensure_http_server_launch_target \"$project_root\" \"${TTS_HOST:-0.0.0.0}\" \"${TTS_PORT:-8000}\"" in contents
        assert "wait_http_health_check \"${TTS_HOST:-0.0.0.0}\" \"${TTS_PORT:-8000}\"" in contents
        assert 'wait_http_health_check "${TTS_HOST:-0.0.0.0}" "${TTS_PORT:-8000}" || true' not in contents


def test_unix_launcher_server_restart_uses_pid_file_management():
    linux_contents = LINUX_SCRIPT_PATH.read_text(encoding="utf-8")
    macos_contents = MACOS_SCRIPT_PATH.read_text(encoding="utf-8")

    for contents in (linux_contents, macos_contents):
        assert "http_server_pid_file_path()" in contents
        assert "load_http_server_pid_file()" in contents
        assert "stop_http_server_pid()" in contents
        assert "stop_http_server_listener_pid()" in contents
        assert "resolve_http_server_port_owner_pid()" in contents
        assert "ensure_http_server_launch_target()" in contents
        assert ".state/launcher/http-server.pid" in contents
        assert "Launcher-managed HTTP server is already running. [R]estart / [K]eep existing / [C]hange port:" in contents
        assert "Stopping existing launcher-managed HTTP server" in contents
        assert "Stopping existing launcher-managed HTTP listener" in contents
        assert 'stop_http_server_listener_pid "$bind_host" "$bind_port" "$project_root" "$HTTP_SERVER_PID"' in contents
        assert "Port is occupied by a non-launcher process. [S]top and restart / [K]eep existing / [C]hange port:" in contents
        assert 'listener_pid="$(resolve_http_server_port_owner_pid "${TTS_HOST:-0.0.0.0}" "${TTS_PORT:-8000}" 2>/dev/null || true)"' in contents
        assert 'if [[ -n "$listener_pid" ]]; then' in contents
        assert 'exec --family "$family" --module "$module" >/dev/null 2>&1 &' in contents
