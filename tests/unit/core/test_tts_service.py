# FILE: tests/unit/core/test_tts_service.py
# VERSION: 1.1.0
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
#   _StubBackend - Stable backend stub with a configurable execute() callback used in place of the removed generate_audio shim
#   StubRegistry - Minimal registry stub returning model specs by mode and backed by a stable _StubBackend instance
#   LoggingRegistry - Registry stub that counts model resolution calls
#   _make_core_settings - Build isolated core settings for unit tests
#   _capture_execute - Build an execute callback that records the ExecutionRequest fields the tests assert on
#   test_synthesize_clone_passes_ref_audio_as_string - Verifies clone requests pass file paths as strings to generation
#   test_synthesize_clone_passes_explicit_language - Verifies clone requests normalize explicit language values
#   test_synthesize_clone_preserves_missing_ref_text - Verifies clone requests preserve None ref_text in the kwargs
#   test_tts_service_emits_structured_logs - Verifies structured synthesis logs include mode and language context
#   test_tts_service_routes_omnivoice_family_payload_to_backend - Verifies the OmniVoice family routes its payload through the backend execution contract
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Phase 3.11: replaced generate_audio monkeypatches with a stable _StubBackend whose execute() callback is overridden per test, mirroring the new SynthesisCoordinator -> backend.execute(ExecutionRequest) flow]
# END_CHANGE_SUMMARY

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from core.backends.base import ExecutionRequest
from core.config import CoreSettings
from core.contracts.commands import VoiceCloneCommand, VoiceDesignCommand
from core.models.catalog import MODEL_SPECS
from core.services.tts_service import TTSService
from tests.support.api_fakes import extract_json_logs, make_wav_bytes

pytestmark = pytest.mark.unit


class _StubBackend:
    key = "torch"
    label = "PyTorch + Transformers"

    def __init__(self) -> None:
        self.execute = lambda request: None  # type: ignore[assignment]


class StubRegistry:
    def __init__(self) -> None:
        self._backend = _StubBackend()

    @property
    def backend(self):
        return self._backend

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
        super().__init__()
        self.calls = 0

    def get_model(self, model_name=None, mode=None):
        self.calls += 1
        return super().get_model(model_name=model_name, mode=mode)


class FamilyAwareRegistry(StubRegistry):
    def get_model(self, model_name=None, mode=None):
        spec = self.get_model_spec(model_name=model_name, mode=mode)
        return spec, type("HandleStub", (), {"backend_key": "torch", "spec": spec})()


def _capture_execute(captured: dict, *, write_audio: bool = True):
    def _execute(request: ExecutionRequest) -> None:
        captured.update(request.generation_kwargs)
        captured["text"] = request.text
        captured["language"] = request.language
        captured["output_path"] = str(request.output_dir)
        captured["mode"] = request.execution_mode
        captured["handle"] = request.handle
        if write_audio:
            (Path(request.output_dir) / "audio_0001.wav").write_bytes(make_wav_bytes())

    return _execute


def _make_core_settings(tmp_path: Path) -> CoreSettings:
    qwen_clone_model = next(
        spec.model_id
        for spec in MODEL_SPECS.values()
        if spec.family == "Qwen3-TTS" and spec.mode == "clone"
    )
    qwen_design_model = next(
        spec.model_id
        for spec in MODEL_SPECS.values()
        if spec.family == "Qwen3-TTS" and spec.mode == "design"
    )
    settings = CoreSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        active_family="qwen",
        default_design_model=qwen_design_model,
        default_clone_model=qwen_clone_model,
    )
    settings.ensure_directories()
    return settings


def test_synthesize_clone_passes_ref_audio_as_string(tmp_path: Path):
    settings = _make_core_settings(tmp_path)
    ref_audio_path = tmp_path / "reference.wav"
    ref_audio_path.write_bytes(make_wav_bytes())
    registry = StubRegistry()
    service = TTSService(registry=registry, settings=settings)  # type: ignore[arg-type]
    captured_kwargs: dict = {}
    registry.backend.execute = _capture_execute(captured_kwargs)

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


def test_synthesize_clone_passes_explicit_language(tmp_path: Path):
    settings = _make_core_settings(tmp_path)
    ref_audio_path = tmp_path / "reference.wav"
    ref_audio_path.write_bytes(make_wav_bytes())
    registry = StubRegistry()
    service = TTSService(registry=registry, settings=settings)  # type: ignore[arg-type]
    captured_kwargs: dict = {}
    registry.backend.execute = _capture_execute(captured_kwargs)

    service.synthesize_clone(
        VoiceCloneCommand(
            text="Clone this",
            ref_audio_path=ref_audio_path,
            ref_text="Clone this",
            language="Ru ",
        )
    )

    assert captured_kwargs["language"] == "ru"


def test_synthesize_clone_preserves_missing_ref_text(tmp_path: Path):
    settings = _make_core_settings(tmp_path)
    ref_audio_path = tmp_path / "reference.wav"
    ref_audio_path.write_bytes(make_wav_bytes())
    registry = StubRegistry()
    service = TTSService(registry=registry, settings=settings)  # type: ignore[arg-type]
    captured_kwargs: dict = {}
    registry.backend.execute = _capture_execute(captured_kwargs)

    service.synthesize_clone(
        VoiceCloneCommand(
            text="Clone this",
            ref_audio_path=ref_audio_path,
            ref_text=None,
        )
    )

    assert "ref_text" in captured_kwargs
    assert captured_kwargs["ref_text"] is None


def test_tts_service_emits_structured_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    settings = _make_core_settings(tmp_path)
    registry = LoggingRegistry()
    service = TTSService(registry=registry, settings=settings)  # type: ignore[arg-type]
    ref_audio_path = tmp_path / "reference.wav"
    ref_audio_path.write_bytes(make_wav_bytes())
    captured_kwargs: dict = {}
    registry.backend.execute = _capture_execute(captured_kwargs)
    caplog.set_level(logging.INFO)

    result = service.synthesize_clone(
        VoiceCloneCommand(
            text="Clone this",
            ref_audio_path=ref_audio_path,
            ref_text="Clone this",
        )
    )

    assert result.mode == "clone"
    started_logs = extract_json_logs(caplog, "[TTSService][synthesize_clone][SYNTHESIZE_CLONE]")
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
        item["mode"] == "clone" and item["model"] == result.model and item["language"] == "auto"
        for item in completed_logs
    )


def test_tts_service_routes_omnivoice_family_payload_to_backend(tmp_path: Path):
    settings = _make_core_settings(tmp_path)
    registry = FamilyAwareRegistry()
    service = TTSService(registry=registry, settings=settings)  # type: ignore[arg-type]
    captured_kwargs: dict = {}
    registry.backend.execute = _capture_execute(captured_kwargs)

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
