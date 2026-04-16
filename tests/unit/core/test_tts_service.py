# FILE: tests/unit/core/test_tts_service.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for the core TTS service orchestration and logging.
#   SCOPE: Clone synthesis, language normalization, structured log emission
#   DEPENDS: M-CORE
#   LINKS: V-M-CORE
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   StubRegistry - Minimal registry stub returning model specs by mode
#   LoggingRegistry - Registry stub that counts model resolution calls
#   _make_core_settings - Build isolated core settings for unit tests
#   test_synthesize_clone_passes_ref_audio_as_string - Verifies clone requests pass file paths as strings to generation
#   test_synthesize_clone_passes_explicit_language - Verifies clone requests normalize explicit language values
#   test_tts_service_emits_structured_logs - Verifies structured synthesis logs include mode and language context
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT and MODULE_MAP]
# END_CHANGE_SUMMARY

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from core.config import CoreSettings
from core.contracts.commands import VoiceCloneCommand, VoiceDesignCommand
from core.models.catalog import MODEL_SPECS
from core.services.tts_service import TTSService
from tests.support.api_fakes import extract_json_logs, make_wav_bytes


pytestmark = pytest.mark.unit


class StubRegistry:
    @property
    def backend(self):
        return type("BackendStub", (), {"key": "torch", "label": "PyTorch + Transformers"})()

    def get_model_spec(self, model_name=None, mode=None):
        if model_name is not None:
            return next(
                spec
                for spec in MODEL_SPECS.values()
                if model_name in {spec.api_name, spec.folder, spec.key, spec.model_id}
            )
        return next(spec for spec in MODEL_SPECS.values() if spec.mode == (mode or "clone"))

    def get_model(self, model_name=None, mode=None):
        spec = self.get_model_spec(model_name=model_name, mode=mode)
        return spec, type("HandleStub", (), {"backend_key": "torch", "spec": spec})()

    def backend_for_spec(self, spec):
        return self.backend

    def backend_route_for_spec(self, spec):
        return {
            "route_reason": "registry_model_resolution",
            "execution_backend": self.backend.key,
        }


class LoggingRegistry(StubRegistry):
    def __init__(self):
        self.calls = 0

    def get_model(self, model_name=None, mode=None):
        self.calls += 1
        return super().get_model(model_name=model_name, mode=mode)


class FamilyAwareRegistry(StubRegistry):
    def get_model(self, model_name=None, mode=None):
        spec = self.get_model_spec(model_name=model_name, mode=mode)
        return spec, type("HandleStub", (), {"backend_key": "torch", "spec": spec})()


def _make_core_settings(tmp_path: Path) -> CoreSettings:
    settings = CoreSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
    )
    settings.ensure_directories()
    return settings


def test_synthesize_clone_passes_ref_audio_as_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _make_core_settings(tmp_path)
    ref_audio_path = tmp_path / "reference.wav"
    ref_audio_path.write_bytes(make_wav_bytes())
    service = TTSService(registry=StubRegistry(), settings=settings)  # type: ignore[arg-type]
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
    assert captured_kwargs["language"] == "auto"


def test_synthesize_clone_passes_explicit_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _make_core_settings(tmp_path)
    ref_audio_path = tmp_path / "reference.wav"
    ref_audio_path.write_bytes(make_wav_bytes())
    service = TTSService(registry=StubRegistry(), settings=settings)  # type: ignore[arg-type]
    captured_kwargs = {}

    def fake_generate_audio(**kwargs):
        captured_kwargs.update(kwargs)
        output_dir = Path(kwargs["output_path"])
        (output_dir / "audio_0001.wav").write_bytes(make_wav_bytes())

    monkeypatch.setattr("core.services.tts_service.generate_audio", fake_generate_audio)

    service.synthesize_clone(
        VoiceCloneCommand(
            text="Clone this",
            ref_audio_path=ref_audio_path,
            ref_text="Clone this",
            language="Ru ",
        )
    )

    assert captured_kwargs["language"] == "ru"


def test_tts_service_emits_structured_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    settings = _make_core_settings(tmp_path)
    registry = LoggingRegistry()
    service = TTSService(registry=registry, settings=settings)  # type: ignore[arg-type]
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
    started_logs = extract_json_logs(
        caplog, "[TTSService][synthesize_clone][SYNTHESIZE_CLONE]"
    )
    completed_logs = extract_json_logs(
        caplog, "[TTSService][_run_generation][BLOCK_PERSIST_OUTPUT]"
    )
    assert registry.calls == 1
    assert any(
        item["mode"] == "clone"
        and item["text_length"] == len("Clone this")
        and item["language"] == "auto"
        for item in started_logs
    )
    assert any(
        item["mode"] == "clone"
        and item["model"] == result.model
        and item["language"] == "auto"
        for item in completed_logs
    )


def test_tts_service_routes_omnivoice_family_payload_to_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _make_core_settings(tmp_path)
    service = TTSService(registry=FamilyAwareRegistry(), settings=settings)  # type: ignore[arg-type]
    captured_kwargs = {}

    def fake_generate_audio(**kwargs):
        captured_kwargs.update(kwargs)
        output_dir = Path(kwargs["output_path"])
        (output_dir / "audio_0001.wav").write_bytes(make_wav_bytes())

    monkeypatch.setattr("core.services.tts_service.generate_audio", fake_generate_audio)

    result = service.synthesize_design(
        VoiceDesignCommand(
            text="Hello",
            model="omnivoice-design-1",
            voice_description="Warm bilingual narrator",
        )
    )

    assert result.mode == "design"
    assert captured_kwargs["mode"] == "design"
    assert captured_kwargs["instruct"] == "Warm bilingual narrator"
    assert captured_kwargs["language"] == "auto"
