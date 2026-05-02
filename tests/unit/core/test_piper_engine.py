# FILE: tests/unit/core/test_piper_engine.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Verify the first production Piper TTSEngine and its guarded TTSService route preserve current ONNX/Piper behavior under deterministic fake-runtime conditions.
#   SCOPE: Piper engine load/synthesize behavior, missing-artifact error mapping, and explicit TTSService engine-route gating
#   DEPENDS: M-ENGINE-CONTRACTS, M-BACKENDS, M-TTS-SERVICE
#   LINKS: V-M-ENGINE-PIPER, V-M-BACKENDS-V2
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _write_piper_artifacts - Helper that creates manifest-compatible Piper artifacts for deterministic tests
#   _EngineAwareRegistry - Minimal runtime registry exposing an ONNX backend for Piper engine-route tests
#   test_piper_onnx_engine_loads_and_synthesizes_wav_bytes - Verifies production Piper engine load/synthesize parity under a fake Piper runtime
#   test_piper_onnx_engine_missing_artifacts_raise_controlled_model_load_error - Verifies missing model.onnx/model.onnx.json artifacts surface a controlled error
#   test_tts_service_routes_piper_through_engine_when_explicitly_enabled - Verifies the guarded runtime flag routes Piper custom synthesis through the engine registry
#   test_tts_service_keeps_legacy_backend_path_when_piper_engine_is_disabled - Verifies legacy backend execution remains the fallback path when the flag is off
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Task 10: added deterministic coverage for the first production Piper engine and its guarded TTSService route]
# END_CHANGE_SUMMARY

from __future__ import annotations

import io
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.backends.base import ExecutionRequest, LoadedModelHandle, TTSBackend
from core.config import CoreSettings
from core.contracts import BackendRouteInfo
from core.contracts.commands import CustomVoiceCommand
from core.engines import SynthesisJob
from core.engines.piper import PiperOnnxEngine
from core.errors import ModelLoadError
from core.models.catalog import MODEL_SPECS, ModelSpec
from core.services.tts_service import TTSService

pytestmark = pytest.mark.unit


def _write_piper_artifacts(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.onnx").write_bytes(b"onnx")
    (model_dir / "model.onnx.json").write_text(
        json.dumps(
            {
                "audio": {"sample_rate": 22050},
                "espeak": {"voice": "en-us"},
                "phoneme_type": "espeak",
                "num_symbols": 10,
                "num_speakers": 1,
                "phoneme_id_map": {"_": [0], "^": [1], "$": [2], "a": [3]},
            }
        ),
        encoding="utf-8",
    )


class _FallbackOnnxBackend(TTSBackend):
    key = "onnx"
    label = "ONNX Runtime"

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.execute_calls = 0

    def capabilities(self):  # pragma: no cover
        raise NotImplementedError

    def is_available(self) -> bool:
        return True

    def supports_platform(self) -> bool:
        return True

    def resolve_model_path(self, folder_name: str) -> Path | None:
        path = self.models_dir / folder_name
        return path if path.exists() else None

    def load_model(self, spec: ModelSpec) -> LoadedModelHandle:
        return LoadedModelHandle(
            spec=spec,
            runtime_model=object(),
            resolved_path=self.resolve_model_path(spec.folder),
            backend_key=self.key,
        )

    def inspect_model(self, spec: ModelSpec) -> dict:
        return {}

    def readiness_diagnostics(self):  # pragma: no cover
        raise NotImplementedError

    def cache_diagnostics(self) -> dict:
        return {}

    def metrics_summary(self) -> dict:
        return {}

    def preload_models(self, specs):  # pragma: no cover
        return {}

    def execute(self, request: ExecutionRequest) -> None:
        self.execute_calls += 1
        (Path(request.output_dir) / "audio_0001.wav").write_bytes(b"legacy-backend-audio")


class _EngineAwareRegistry:
    def __init__(self, models_dir: Path) -> None:
        self._backend = _FallbackOnnxBackend(models_dir)

    @property
    def backend(self) -> TTSBackend:
        return self._backend

    def get_model_spec(self, model_name: str | None = None, mode: str | None = None) -> ModelSpec:
        if model_name is not None:
            return next(
                spec
                for spec in MODEL_SPECS.values()
                if model_name in {spec.api_name, spec.folder, spec.key, spec.model_id}
            )
        return next(spec for spec in MODEL_SPECS.values() if spec.mode == (mode or "custom"))

    def get_model(
        self, model_name: str | None = None, mode: str | None = None
    ) -> tuple[ModelSpec, LoadedModelHandle]:
        spec = self.get_model_spec(model_name=model_name, mode=mode)
        return spec, self._backend.load_model(spec)

    def backend_for_spec(self, spec: ModelSpec) -> TTSBackend:
        return self._backend

    def backend_route_for_spec(self, spec: ModelSpec) -> BackendRouteInfo:
        return {"route_reason": "registry_model_resolution", "execution_backend": self._backend.key}


def _make_settings(tmp_path: Path, *, piper_engine_enabled: bool) -> CoreSettings:
    settings = CoreSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        upload_staging_dir=tmp_path / ".uploads",
        active_family="piper",
        default_custom_model=MODEL_SPECS["piper-1"].model_id,
        piper_engine_enabled=piper_engine_enabled,
    )
    settings.ensure_directories()
    return settings


def test_piper_onnx_engine_loads_and_synthesizes_wav_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = MODEL_SPECS["piper-1"]
    model_dir = tmp_path / spec.folder
    _write_piper_artifacts(model_dir)
    engine = PiperOnnxEngine()

    class _FakeVoice:
        def synthesize_wav(self, text: str, wav_file) -> None:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00\x00\x01\x00")

    monkeypatch.setattr(
        "core.engines.piper.PiperVoice",
        SimpleNamespace(load=lambda model_path, config_path, use_cuda=False: _FakeVoice()),
    )

    handle = engine.load_model(spec=spec, backend_key="onnx", model_path=model_dir)
    audio = engine.synthesize(
        handle,
        job=SynthesisJob(
            capability="preset_speaker_tts",
            execution_mode="custom",
            text="Hello Piper",
            language="en",
            output_dir=tmp_path,
            payload={"voice": spec.model_id},
        ),
    )

    assert handle.engine_key == "piper-onnx"
    assert handle.backend_key == "onnx"
    assert audio.audio_format == "wav"
    assert audio.sample_rate == 22050
    with wave.open(io.BytesIO(audio.waveform), "rb") as wav_file:
        assert wav_file.getframerate() == 22050
        assert wav_file.readframes(2) == b"\x00\x00\x01\x00"


def test_piper_onnx_engine_missing_artifacts_raise_controlled_model_load_error(tmp_path: Path) -> None:
    spec = MODEL_SPECS["piper-1"]
    model_dir = tmp_path / spec.folder
    model_dir.mkdir(parents=True, exist_ok=True)
    engine = PiperOnnxEngine()

    with pytest.raises(ModelLoadError, match="Missing required Piper model artifacts") as exc_info:
        engine.load_model(spec=spec, backend_key="onnx", model_path=model_dir)

    details = exc_info.value.context.to_dict()
    assert details["missing_artifacts"] == ["model.onnx", "model.onnx.json"]
    assert details["required_artifacts"] == ["model.onnx", "model.onnx.json"]
    assert details["engine"] == "piper-onnx"


def test_tts_service_routes_piper_through_engine_when_explicitly_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _make_settings(tmp_path, piper_engine_enabled=True)
    registry = _EngineAwareRegistry(settings.models_dir)
    spec = MODEL_SPECS["piper-1"]
    _write_piper_artifacts(settings.models_dir / spec.folder)

    class _FakeVoice:
        def synthesize_wav(self, text: str, wav_file) -> None:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x11\x22\x33\x44")

    monkeypatch.setattr(
        "core.engines.piper.PiperVoice",
        SimpleNamespace(load=lambda model_path, config_path, use_cuda=False: _FakeVoice()),
    )

    service = TTSService(registry=registry, settings=settings)  # type: ignore[arg-type]
    result = service.synthesize_custom(
        CustomVoiceCommand(text="Hello Piper", model=spec.model_id, speaker="ignored")
    )
    backend = registry.backend

    assert result.backend == "onnx"
    assert result.model == spec.model_id
    assert result.audio.bytes_data.startswith(b"RIFF")
    assert isinstance(backend, _FallbackOnnxBackend)
    assert backend.execute_calls == 0


def test_tts_service_keeps_legacy_backend_path_when_piper_engine_is_disabled(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, piper_engine_enabled=False)
    registry = _EngineAwareRegistry(settings.models_dir)
    service = TTSService(registry=registry, settings=settings)  # type: ignore[arg-type]
    spec = MODEL_SPECS["piper-1"]

    result = service.synthesize_custom(
        CustomVoiceCommand(text="Hello Piper", model=spec.model_id, speaker="ignored")
    )
    backend = registry.backend

    assert result.backend == "onnx"
    assert result.audio.bytes_data == b"legacy-backend-audio"
    assert isinstance(backend, _FallbackOnnxBackend)
    assert backend.execute_calls == 1
