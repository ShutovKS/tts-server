from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cli import main as cli_main
from cli.runtime import CliRuntime, run_cli
from cli.runtime_config import CliSettings


pytestmark = pytest.mark.unit


def test_cli_main_delegates_to_runtime(monkeypatch: pytest.MonkeyPatch):
    called = {"value": False}

    def fake_run_cli() -> None:
        called["value"] = True

    monkeypatch.setattr(cli_main, "run_cli", fake_run_cli)
    cli_main.run_cli()

    assert called["value"] is True


def test_cli_module_entrypoint_delegates_to_runtime():
    content = Path("cli/__main__.py").read_text(encoding="utf-8")
    assert "from cli.runtime import run_cli" in content
    assert "run_cli()" in content
    assert "TTSService" not in content
    assert "ModelRegistry" not in content


def test_cli_runtime_uses_core_runtime_only(tmp_path: Path):
    settings = CliSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
    )

    runtime = CliRuntime(settings)

    assert runtime.registry.__class__.__module__ == "core.services.model_registry"
    assert runtime.service.__class__.__module__ == "core.application.tts_app_service"
    assert runtime.service.tts_service.registry is runtime.registry


def test_run_cli_exits_cleanly_on_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch):
    runtime = MagicMock()
    runtime.run.side_effect = KeyboardInterrupt()

    monkeypatch.setattr("cli.runtime.CliRuntime", lambda: runtime)

    run_cli()

    runtime.run.assert_called_once_with()
