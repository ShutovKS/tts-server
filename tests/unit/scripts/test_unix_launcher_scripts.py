# FILE: tests/unit/scripts/test_unix_launcher_scripts.py
# VERSION: 1.1.0
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
#   test_linux_launcher_script_exists_with_grace_contract - Verifies the Linux launcher file exists and retains top-level GRACE contract anchors.
#   test_linux_launcher_script_reuses_profile_aware_launcher_and_manual_package_guidance - Verifies the Linux launcher delegates env setup and execution to launcher commands while keeping package-manager guidance manual-only.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Expanded Unix launcher coverage to assert Windows-flow parity for family-first selection, multi-model preparation, and runtime capability bindings]
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
    assert "HF_TOKEN" in contents
    assert "TTS_TELEGRAM_BOT_TOKEN" in contents
    assert "This script does not install system packages automatically." in contents
    assert "sudo apt-get install -y python3.11 python3.11-venv ffmpeg" in contents
    assert "sudo dnf install -y python3.11 ffmpeg" in contents
    assert "sudo yum install -y python3.11 ffmpeg" in contents
    assert "sudo pacman -S --needed python ffmpeg" in contents
    assert "sudo zypper install -y python311 ffmpeg" in contents
