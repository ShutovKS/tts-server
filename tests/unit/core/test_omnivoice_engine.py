# FILE: tests/unit/core/test_omnivoice_engine.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Verify the production OmniVoice TTSEngine and its generic TTSService route preserve current Torch/OmniVoice behavior under deterministic fake-runtime conditions.
#   SCOPE: OmniVoice engine load/synthesize parity, missing-runtime error mapping, sample-rate validation, and explicit TTSService engine-route coverage for custom, design, and clone
#   DEPENDS: M-ENGINE-CONTRACTS, M-BACKENDS, M-TTS-SERVICE, M-OMNIVOICE-FAMILY
#   LINKS: V-M-ENGINE-OMNIVOICE, V-M-TTS-SERVICE
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _EngineAwareRegistry - Minimal runtime registry exposing a Torch backend for OmniVoice engine-route tests
#   _make_settings - Build OmniVoice-focused CoreSettings for deterministic engine-route tests
#   test_omnivoice_torch_engine_loads_and_synthesizes_design_audio - Verifies OmniVoice engine design-mode parity under a fake omnivoice runtime
#   test_omnivoice_torch_engine_missing_runtime_raises_controlled_model_load_error - Verifies missing omnivoice runtime surfaces a controlled error
#   test_omnivoice_torch_engine_missing_sample_rate_raises_controlled_error - Verifies missing audio tokenizer sample-rate metadata surfaces a controlled generation error
#   test_tts_service_routes_omnivoice_custom_design_and_clone_through_engine - Verifies custom, design, and clone synthesis use the engine seam instead of backend.execute fallback
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Task 16: added deterministic coverage for the OmniVoice Torch engine and its generic TTSService route]
# END_CHANGE_SUMMARY

from __future__ import annotations

import io
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.backends.base import ExecutionRequest, LoadedModelHandle, TTSBackend
from core.config import CoreSettings
from core.contracts import BackendRouteInfo
from core.contracts.commands import CustomVoiceCommand, VoiceCloneCommand, VoiceDesignCommand
from core.engines import OmniVoiceTorchEngine, SynthesisJob
from core.errors import ModelLoadError, TTSGenerationError
from core.models.catalog import MODEL_SPECS, ModelSpec
from core.services.tts_service import TTSService
from tests.support.api_fakes import make_wav_bytes

pytestmark = pytest.mark.unit


class _FallbackTorchBackend(TTSBackend):
    key = "torch"
    label = "PyTorch + Transformers"

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
        self._backend = _FallbackTorchBackend(models_dir)

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
        return next(spec for spec in MODEL_SPECS.values() if spec.mode == (mode or "design"))

    def get_model(
        self, model_name: str | None = None, mode: str | None = None
    ) -> tuple[ModelSpec, LoadedModelHandle]:
        spec = self.get_model_spec(model_name=model_name, mode=mode)
        return spec, self._backend.load_model(spec)

    def backend_for_spec(self, spec: ModelSpec) -> TTSBackend:
        return self._backend

    def backend_route_for_spec(self, spec: ModelSpec) -> BackendRouteInfo:
        return {"route_reason": "registry_model_resolution", "execution_backend": self._backend.key}


def _make_settings(tmp_path: Path) -> CoreSettings:
    omnivoice_custom_model = next(
        spec.model_id for spec in MODEL_SPECS.values() if spec.family_key == "omnivoice" and spec.mode == "custom"
    )
    omnivoice_design_model = next(
        spec.model_id for spec in MODEL_SPECS.values() if spec.family_key == "omnivoice" and spec.mode == "design"
    )
    omnivoice_clone_model = next(
        spec.model_id for spec in MODEL_SPECS.values() if spec.family_key == "omnivoice" and spec.mode == "clone"
    )
    settings = CoreSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        upload_staging_dir=tmp_path / ".uploads",
        active_family="omnivoice",
        default_custom_model=omnivoice_custom_model,
        default_design_model=omnivoice_design_model,
        default_clone_model=omnivoice_clone_model,
    )
    settings.ensure_directories()
    return settings


def _omnivoice_design_spec() -> ModelSpec:
    return next(spec for spec in MODEL_SPECS.values() if spec.family_key == "omnivoice" and spec.mode == "design")


def test_omnivoice_torch_engine_loads_and_synthesizes_design_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _omnivoice_design_spec()
    model_dir = tmp_path / spec.folder
    model_dir.mkdir(parents=True, exist_ok=True)
    engine = OmniVoiceTorchEngine()

    class _FakeOmniVoiceRuntime:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path
            self.audio_tokenizer = SimpleNamespace(config=SimpleNamespace(sample_rate=24000))

        def generate(self, **kwargs):
            assert kwargs == {"text": "Hello OmniVoice", "language": "en", "instruct": "Warm narrator"}
            return [[0.0] * 8]

    monkeypatch.setattr("core.engines.omnivoice.load_omnivoice_model_cls", lambda: _FakeOmniVoiceRuntime)
    monkeypatch.setattr("core.engines.omnivoice.torch", object())
    monkeypatch.setattr(
        "soundfile.write",
        lambda target, data, sample_rate, format=None: _write_fake_wave(target, sample_rate=sample_rate),
    )

    handle = engine.load_model(spec=spec, backend_key="torch", model_path=model_dir)
    audio = engine.synthesize(
        handle,
        job=SynthesisJob(
            capability="voice_description_tts",
            execution_mode="design",
            text="Hello OmniVoice",
            language="en",
            output_dir=tmp_path,
            payload={"instruct": "Warm narrator"},
        ),
    )

    assert handle.engine_key == "omnivoice-torch"
    assert handle.backend_key == "torch"
    assert audio.audio_format == "wav"
    assert audio.sample_rate == 24000
    with wave.open(io.BytesIO(audio.waveform), "rb") as wav_file:
        assert wav_file.getframerate() == 24000


def test_omnivoice_torch_engine_missing_runtime_raises_controlled_model_load_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _omnivoice_design_spec()
    model_dir = tmp_path / spec.folder
    model_dir.mkdir(parents=True, exist_ok=True)
    engine = OmniVoiceTorchEngine()

    monkeypatch.setattr("core.engines.omnivoice.load_omnivoice_model_cls", lambda: None)
    monkeypatch.setattr("core.engines.omnivoice.OMNIVOICE_IMPORT_ERROR", ImportError("omnivoice missing"))
    monkeypatch.setattr("core.engines.omnivoice.torch", object())

    with pytest.raises(ModelLoadError, match="omnivoice missing") as exc_info:
        engine.load_model(spec=spec, backend_key="torch", model_path=model_dir)

    details = exc_info.value.context.to_dict()
    assert details["runtime_dependency"] == "omnivoice.OmniVoice"
    assert details["engine"] == "omnivoice-torch"
    assert details["backend"] == "torch"


def test_omnivoice_torch_engine_missing_sample_rate_raises_controlled_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _omnivoice_design_spec()
    model_dir = tmp_path / spec.folder
    model_dir.mkdir(parents=True, exist_ok=True)
    engine = OmniVoiceTorchEngine()

    class _FakeOmniVoiceRuntime:
        def __init__(self, model_path: str) -> None:
            self.audio_tokenizer = SimpleNamespace(config=SimpleNamespace(sample_rate=None))

        def generate(self, **kwargs):
            return [[0.0] * 8]

    monkeypatch.setattr("core.engines.omnivoice.load_omnivoice_model_cls", lambda: _FakeOmniVoiceRuntime)
    monkeypatch.setattr("core.engines.omnivoice.torch", object())

    handle = engine.load_model(spec=spec, backend_key="torch", model_path=model_dir)
    with pytest.raises(TTSGenerationError, match="sample rate") as exc_info:
        engine.synthesize(
            handle,
            job=SynthesisJob(
                capability="voice_description_tts",
                execution_mode="design",
                text="Hello OmniVoice",
                language="en",
                output_dir=tmp_path,
                payload={"instruct": "Warm narrator"},
            ),
        )

    details = exc_info.value.context.to_dict()
    assert details["failure_kind"] == "missing_sample_rate"
    assert details["family"] == "omnivoice"


def test_tts_service_routes_omnivoice_custom_design_and_clone_through_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _make_settings(tmp_path)
    registry = _EngineAwareRegistry(settings.models_dir)
    custom_spec = next(spec for spec in MODEL_SPECS.values() if spec.family_key == "omnivoice" and spec.mode == "custom")
    design_spec = next(spec for spec in MODEL_SPECS.values() if spec.family_key == "omnivoice" and spec.mode == "design")
    clone_spec = next(spec for spec in MODEL_SPECS.values() if spec.family_key == "omnivoice" and spec.mode == "clone")
    for spec in (custom_spec, design_spec, clone_spec):
        (settings.models_dir / spec.folder).mkdir(parents=True, exist_ok=True)
    ref_audio_path = tmp_path / "reference.wav"
    ref_audio_path.write_bytes(make_wav_bytes())

    class _FakeOmniVoiceRuntime:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path
            self.audio_tokenizer = SimpleNamespace(config=SimpleNamespace(sample_rate=24000))

        def generate(self, **kwargs):
            if kwargs["text"] == "Clone this":
                assert kwargs["ref_audio"].endswith("reference.wav")
                assert kwargs["ref_text"] == "Clone this"
            elif kwargs["text"] == "Hello OmniVoice":
                assert kwargs["instruct"] == "Warm bilingual narrator"
            elif kwargs["text"] == "Hello custom":
                assert kwargs["speed"] == 1.25
                assert kwargs["instruct"] == "Friendly"
            return [[0.0] * 48000]

    monkeypatch.setattr("core.engines.omnivoice.load_omnivoice_model_cls", lambda: _FakeOmniVoiceRuntime)
    monkeypatch.setattr("core.engines.omnivoice.torch", object())
    monkeypatch.setattr(
        "soundfile.write",
        lambda target, data, sample_rate, format=None: _write_fake_wave(target, sample_rate=sample_rate),
    )

    service = TTSService(registry=registry, settings=settings)  # type: ignore[arg-type]
    clone_result = service.synthesize_clone(
        VoiceCloneCommand(text="Clone this", ref_audio_path=ref_audio_path, ref_text="Clone this")
    )
    design_result = service.synthesize_design(
        VoiceDesignCommand(
            text="Hello OmniVoice",
            model=design_spec.model_id,
            voice_description="Warm bilingual narrator",
        )
    )
    custom_result = service.synthesize_custom(
        CustomVoiceCommand(
            text="Hello custom",
            model=custom_spec.model_id,
            speaker="ignored",
            instruct="Friendly",
            speed=1.25,
        )
    )

    backend = registry.backend
    assert clone_result.backend == "torch"
    assert design_result.backend == "torch"
    assert custom_result.backend == "torch"
    assert clone_result.audio.bytes_data.startswith(b"RIFF")
    assert design_result.audio.bytes_data.startswith(b"RIFF")
    assert custom_result.audio.bytes_data.startswith(b"RIFF")
    assert isinstance(backend, _FallbackTorchBackend)
    assert backend.execute_calls == 0


def _write_fake_wave(target, *, sample_rate: int) -> None:
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00\x01\x00")
    payload = wav_buffer.getvalue()
    if hasattr(target, "write"):
        target.write(payload)
        return
    Path(target).write_bytes(payload)
