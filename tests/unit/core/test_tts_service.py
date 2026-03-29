from __future__ import annotations

import logging
from pathlib import Path

import pytest

from core.config import CoreSettings
from core.contracts.commands import VoiceCloneCommand
from core.models.catalog import MODEL_SPECS
from core.services.tts_service import TTSService
from tests.support.api_fakes import extract_json_logs, make_wav_bytes


pytestmark = pytest.mark.unit


class StubRegistry:
    def get_model(self, model_name=None, mode=None):
        spec = next(spec for spec in MODEL_SPECS.values() if spec.mode == (mode or "clone"))
        return spec, object()


class LoggingRegistry(StubRegistry):
    def __init__(self):
        self.calls = 0

    def get_model(self, model_name=None, mode=None):
        self.calls += 1
        return super().get_model(model_name=model_name, mode=mode)


def _make_core_settings(tmp_path: Path) -> CoreSettings:
    settings = CoreSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
    )
    settings.ensure_directories()
    return settings


def test_synthesize_clone_passes_ref_audio_as_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _make_core_settings(tmp_path)
    ref_audio_path = tmp_path / "reference.wav"
    ref_audio_path.write_bytes(make_wav_bytes())
    service = TTSService(registry=StubRegistry(), settings=settings)
    captured_kwargs = {}

    def fake_generate_audio(**kwargs):
        captured_kwargs.update(kwargs)
        output_dir = Path(kwargs["output_path"])
        (output_dir / "audio_0001.wav").write_bytes(make_wav_bytes())

    monkeypatch.setattr("core.services.tts_service.generate_audio", fake_generate_audio)

    result = service.synthesize_clone(
        VoiceCloneCommand(
            text="Clone this",
            ref_audio_path=ref_audio_path,
            ref_text="Clone this",
        )
    )

    assert result.mode == "clone"
    assert isinstance(captured_kwargs["ref_audio"], str)
    assert captured_kwargs["ref_audio"].endswith("reference.wav")


def test_tts_service_emits_structured_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    settings = _make_core_settings(tmp_path)
    registry = LoggingRegistry()
    service = TTSService(registry=registry, settings=settings)
    ref_audio_path = tmp_path / "reference.wav"
    ref_audio_path.write_bytes(make_wav_bytes())

    def fake_generate_audio(**kwargs):
        output_dir = Path(kwargs["output_path"])
        (output_dir / "audio_0001.wav").write_bytes(make_wav_bytes())

    monkeypatch.setattr("core.services.tts_service.generate_audio", fake_generate_audio)
    caplog.set_level(logging.INFO)

    result = service.synthesize_clone(
        VoiceCloneCommand(
            text="Clone this",
            ref_audio_path=ref_audio_path,
            ref_text="Clone this",
        )
    )

    assert result.mode == "clone"
    started_logs = extract_json_logs(caplog, "tts.clone.started")
    completed_logs = extract_json_logs(caplog, "tts.generation.completed")
    assert registry.calls == 1
    assert any(item["mode"] == "clone" and item["text_length"] == len("Clone this") for item in started_logs)
    assert any(item["mode"] == "clone" and item["model"] == result.model for item in completed_logs)
