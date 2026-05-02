# FILE: tests/unit/core/test_tts_service.py
# VERSION: 1.6.1
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for the core TTS service orchestration and logging.
#   SCOPE: Clone synthesis, family-adapter discovery wiring, duplicate-key validation, language normalization, structured log emission, guarded engine-route fallback behavior, generic Qwen3 engine routing, and scheduler gateway execution coverage
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
#   _DiscoveredAdapter - Test-local adapter class used to assert discovery filtering and explicit discovery-driven service wiring
#   _DuplicateAdapterOne - Test-local adapter class used to assert duplicate key validation
#   _DuplicateAdapterTwo - Second test-local adapter class sharing a duplicate key
#   test_default_discovery_filters_test_local_family_adapters - Verifies normal TTSService construction ignores test-local family adapters leaked into the subclass registry
#   test_tts_service_builds_family_adapters_from_discovery - Verifies TTSService wiring consumes discover_family_adapter_classes() output instead of hardcoded constructors
#   test_build_family_adapter_map_rejects_duplicate_keys - Verifies duplicate discovered family keys raise ValueError with the duplicate key in the message
#   test_synthesize_clone_passes_ref_audio_as_string - Verifies clone requests pass file paths as strings to generation
#   test_synthesize_clone_passes_explicit_language - Verifies clone requests normalize explicit language values
#   test_synthesize_clone_preserves_missing_ref_text - Verifies clone requests preserve None ref_text in the kwargs
#   test_tts_service_emits_structured_logs - Verifies structured synthesis logs include mode and language context
#   test_tts_service_routes_omnivoice_family_payload_to_backend - Verifies the OmniVoice family routes its payload through the backend execution contract
#   test_build_engine_registry_returns_none_when_flag_is_disabled - Verifies engine wiring stays absent unless the explicit runtime flag is enabled
#   test_tts_service_routes_legacy_backend_execution_through_scheduler_gateway - Verifies the legacy backend lane executes through EngineScheduler instead of direct InferenceGuard acquire/release
#   test_tts_service_routes_qwen3_custom_through_engine_scheduler_gateway - Verifies the Qwen3 custom engine lane executes through EngineScheduler without backend.execute fallback
#   test_tts_service_routes_qwen3_engine_execution_through_scheduler_gateway - Verifies the Qwen3 engine lane executes through EngineScheduler for design and clone synthesis without backend.execute fallback
#   test_tts_service_routes_piper_engine_execution_through_scheduler_gateway - Verifies the Piper engine lane executes through EngineScheduler when the explicit engine path is enabled
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.6.1 - Task 15 review-fix: added deterministic Qwen3 custom engine-route coverage and kept scheduler-gated behavior assertions intact]
# END_CHANGE_SUMMARY

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from core.backends.base import ExecutionRequest, LoadedModelHandle, TTSBackend
from core.config import CoreSettings
from core.contracts import BackendRouteInfo
from core.contracts.commands import CustomVoiceCommand, VoiceCloneCommand, VoiceDesignCommand
from core.engines import Qwen3TorchEngine
from core.contracts.synthesis import ExecutionPlan
from core.discovery import discover_family_adapter_classes
from core.engines import EngineScheduler
from core.model_families.base import FamilyPreparedExecution, ModelFamilyAdapter
from core.models.catalog import MODEL_SPECS, ModelSpec
from core.services.tts_service import TTSService, _build_engine_registry, _build_family_adapter_map
from tests.support.api_fakes import extract_json_logs, make_wav_bytes

pytestmark = pytest.mark.unit


class _StubBackend(TTSBackend):
    key = "torch"
    label = "PyTorch + Transformers"

    def __init__(self) -> None:
        self.execute = lambda request: None  # type: ignore[assignment]

    def execute(self, request: ExecutionRequest) -> None:
        return None

    def capabilities(self):  # pragma: no cover - not exercised in these tests
        raise NotImplementedError

    def is_available(self) -> bool:
        return True

    def supports_platform(self) -> bool:
        return True

    def resolve_model_path(self, folder_name: str) -> Path | None:  # pragma: no cover
        return None

    def load_model(self, spec):  # pragma: no cover
        raise NotImplementedError

    def inspect_model(self, spec) -> dict[str, Any]:  # pragma: no cover
        return {}

    def readiness_diagnostics(self):  # pragma: no cover
        raise NotImplementedError

    def cache_diagnostics(self) -> dict[str, Any]:  # pragma: no cover
        return {}

    def metrics_summary(self) -> dict[str, Any]:  # pragma: no cover
        return {}

    def preload_models(self, specs):  # pragma: no cover
        return {}


class StubRegistry:
    def __init__(self) -> None:
        self._backend = _StubBackend()

    @property
    def backend(self) -> TTSBackend:
        return self._backend

    def get_model_spec(self, model_name: str | None = None, mode: str | None = None):
        if model_name is not None:
            return next(
                spec
                for spec in MODEL_SPECS.values()
                if model_name in {spec.api_name, spec.folder, spec.key, spec.model_id}
            )
        return next(spec for spec in MODEL_SPECS.values() if spec.mode == (mode or "clone"))

    def get_model(
        self, model_name: str | None = None, mode: str | None = None
    ) -> tuple[ModelSpec, LoadedModelHandle]:
        spec = self.get_model_spec(model_name=model_name, mode=mode)
        return spec, LoadedModelHandle(
            spec=spec,
            runtime_model=object(),
            resolved_path=None,
            backend_key="torch",
        )

    def backend_for_spec(self, spec) -> TTSBackend:
        return self.backend

    def backend_route_for_spec(self, spec) -> BackendRouteInfo:
        return {
            "route_reason": "registry_model_resolution",
            "execution_backend": self.backend.key,
        }


class LoggingRegistry(StubRegistry):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def get_model(self, model_name: str | None = None, mode: str | None = None):
        self.calls += 1
        return super().get_model(model_name=model_name, mode=mode)


class FamilyAwareRegistry(StubRegistry):
    pass


class _DiscoveredAdapter(ModelFamilyAdapter):
    key = "_discovered_family"
    label = "Discovered family"

    def capabilities(self) -> tuple[str, ...]:
        return ("preset_speaker_tts",)

    def supports_plan(self, plan: ExecutionPlan) -> bool:
        return plan.family_key == self.key

    def prepare_execution(self, plan: ExecutionPlan) -> FamilyPreparedExecution:  # pragma: no cover
        return FamilyPreparedExecution(
            execution_mode=plan.execution_mode,
            generation_kwargs={"language": plan.request.language},
        )


class _DuplicateAdapterOne(_DiscoveredAdapter):
    key = "duplicate-family"
    label = "Duplicate adapter one"


class _DuplicateAdapterTwo(_DiscoveredAdapter):
    key = "duplicate-family"
    label = "Duplicate adapter two"


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
    qwen_custom_model = next(
        spec.model_id
        for spec in MODEL_SPECS.values()
        if spec.family == "Qwen3-TTS" and spec.mode == "custom"
    )
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
        default_custom_model=qwen_custom_model,
        default_design_model=qwen_design_model,
        default_clone_model=qwen_clone_model,
    )
    settings.ensure_directories()
    return settings


def test_tts_service_builds_family_adapters_from_discovery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    settings = _make_core_settings(tmp_path)
    registry = StubRegistry()

    monkeypatch.setattr(
        "core.services.tts_service.discover_family_adapter_classes",
        lambda: (_DiscoveredAdapter,),
    )

    service = TTSService(registry=registry, settings=settings)  # type: ignore[arg-type]

    assert tuple(service._family_adapters) == ("_discovered_family",)
    assert isinstance(service._family_adapters["_discovered_family"], _DiscoveredAdapter)


def test_build_family_adapter_map_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate-family"):
        _build_family_adapter_map((_DuplicateAdapterOne, _DuplicateAdapterTwo))


def test_default_discovery_filters_test_local_family_adapters(tmp_path: Path) -> None:
    settings = _make_core_settings(tmp_path)
    registry = StubRegistry()

    discovered_keys = {cls.key for cls in discover_family_adapter_classes(include_entry_points=False)}
    service = TTSService(registry=registry, settings=settings)  # type: ignore[arg-type]

    assert "duplicate-family" not in discovered_keys
    assert "_discovered_family" not in discovered_keys
    assert set(service._family_adapters) == {"qwen3_tts", "omnivoice", "piper"}


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


def test_tts_service_uses_result_cache_to_short_circuit_repeat_requests(tmp_path: Path):
    from core.contracts.commands import CustomVoiceCommand
    from core.services.result_cache import FileSystemResultCache

    settings = _make_core_settings(tmp_path)
    registry = StubRegistry()
    cache = FileSystemResultCache(tmp_path / ".cache" / "results")
    service = TTSService(  # type: ignore[arg-type]
        registry=registry,
        settings=settings,
        result_cache=cache,
    )
    captured_kwargs: dict = {}
    call_counter = {"n": 0}

    def _execute(request: ExecutionRequest) -> None:
        call_counter["n"] += 1
        _capture_execute(captured_kwargs)(request)

    registry.backend.execute = _execute

    qwen_custom_model = next(
        spec.model_id
        for spec in MODEL_SPECS.values()
        if spec.family == "Qwen3-TTS" and spec.mode == "custom"
    )
    command = CustomVoiceCommand(
        text="Cache me",
        model=qwen_custom_model,
        speaker="Vivian",
        instruct="Normal tone",
        speed=1.0,
    )

    first = service.synthesize_custom(command)
    second = service.synthesize_custom(command)

    assert call_counter["n"] == 1
    assert first.audio.bytes_data == second.audio.bytes_data
    assert second.model == first.model
    assert second.backend == first.backend


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


def test_build_engine_registry_returns_none_when_flag_is_disabled(tmp_path: Path) -> None:
    settings = _make_core_settings(tmp_path)

    registry = _build_engine_registry(settings)

    assert registry is not None
    assert registry.keys() == ("qwen3-torch",)


def test_tts_service_routes_qwen3_custom_through_engine_scheduler_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    registry = StubRegistry()
    settings = _make_core_settings(tmp_path)
    scheduler = EngineScheduler()
    scheduler_calls: list[dict[str, object]] = []
    original_submit = scheduler.submit_engine_task
    execute_calls = {"count": 0}

    def fail_execute(request: ExecutionRequest) -> None:
        execute_calls["count"] += 1
        raise AssertionError("backend.execute should not be used for Qwen3 custom engine routing")

    registry.backend.execute = fail_execute

    def record_submit(**kwargs):
        scheduler_calls.append(
            {
                "engine_key": kwargs["engine_key"],
                "device_key": kwargs.get("device_key"),
            }
        )
        return original_submit(**kwargs)

    scheduler.submit_engine_task = record_submit  # type: ignore[method-assign]

    class _FakeQwenRuntime:
        def generate_custom_voice(
            self,
            *,
            text: str,
            language: str,
            speaker: str,
            instruct: str,
            speed: float,
        ):
            assert text == "Hello"
            assert language == "auto"
            assert speaker == "Ryan"
            assert instruct == "Friendly"
            assert speed == 1.15
            return ([[0.0] * 8], 24000)

    monkeypatch.setattr(
        "core.engines.qwen3.load_qwen_tts_model_cls",
        lambda: type(
            "_FakeQwenModelClass",
            (),
            {"from_pretrained": staticmethod(lambda model_path, device_map, dtype: _FakeQwenRuntime())},
        ),
    )
    monkeypatch.setattr("core.engines.qwen3.torch", object())
    monkeypatch.setattr(
        "soundfile.write",
        lambda target, data, sample_rate, format=None: _write_fake_wave(target, sample_rate=sample_rate),
    )

    service = TTSService(  # type: ignore[arg-type]
        registry=registry,
        settings=settings,
        scheduler=scheduler,
    )
    custom_spec = next(
        spec for spec in MODEL_SPECS.values() if spec.family == "Qwen3-TTS" and spec.mode == "custom"
    )
    (settings.models_dir / custom_spec.folder).mkdir(parents=True, exist_ok=True)
    registry.backend.resolve_model_path = lambda folder_name: settings.models_dir / folder_name  # type: ignore[method-assign]

    result = service.synthesize_custom(
        CustomVoiceCommand(text="Hello", speaker="Ryan", instruct="Friendly", speed=1.15)
    )

    assert result.backend == "torch"
    assert result.audio.bytes_data.startswith(b"RIFF")
    assert execute_calls["count"] == 0
    assert scheduler_calls == [{"engine_key": "qwen3-torch", "device_key": None}]


def test_tts_service_routes_qwen3_engine_execution_through_scheduler_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    registry = StubRegistry()
    settings = _make_core_settings(tmp_path)
    scheduler = EngineScheduler()
    scheduler_calls: list[dict[str, object]] = []
    original_submit = scheduler.submit_engine_task
    registry.backend.execute = _capture_execute({})

    def record_submit(**kwargs):
        scheduler_calls.append(
            {
                "engine_key": kwargs["engine_key"],
                "device_key": kwargs.get("device_key"),
            }
        )
        return original_submit(**kwargs)

    scheduler.submit_engine_task = record_submit  # type: ignore[method-assign]

    class _FakeQwenRuntime:
        def generate_voice_design(self, *, text: str, language: str, instruct: str):
            return ([[0.0, 0.0, 0.0, 0.0]], 24000)

        def generate_voice_clone(self, *, text: str, language: str, ref_audio: str, ref_text: str | None):
            return ([[0.0] * 48000], 24000)

    monkeypatch.setattr(
        "core.engines.qwen3.load_qwen_tts_model_cls",
        lambda: type(
            "_FakeQwenModelClass",
            (),
            {"from_pretrained": staticmethod(lambda model_path, device_map, dtype: _FakeQwenRuntime())},
        ),
    )
    monkeypatch.setattr("core.engines.qwen3.torch", object())

    service = TTSService(  # type: ignore[arg-type]
        registry=registry,
        settings=settings,
        scheduler=scheduler,
    )
    design_spec = next(
        spec for spec in MODEL_SPECS.values() if spec.family == "Qwen3-TTS" and spec.mode == "design"
    )
    clone_spec = next(
        spec for spec in MODEL_SPECS.values() if spec.family == "Qwen3-TTS" and spec.mode == "clone"
    )
    (settings.models_dir / design_spec.folder).mkdir(parents=True, exist_ok=True)
    (settings.models_dir / clone_spec.folder).mkdir(parents=True, exist_ok=True)
    registry.backend.resolve_model_path = lambda folder_name: settings.models_dir / folder_name  # type: ignore[method-assign]
    ref_audio_path = tmp_path / "reference.wav"
    ref_audio_path.write_bytes(make_wav_bytes())

    design_result = service.synthesize_design(
        VoiceDesignCommand(text="Hello", voice_description="Warm narrator")
    )
    clone_result = service.synthesize_clone(
        VoiceCloneCommand(text="Clone this", ref_audio_path=ref_audio_path, ref_text="Clone this")
    )

    assert design_result.backend == "torch"
    assert clone_result.backend == "torch"
    assert scheduler_calls == [
        {"engine_key": "qwen3-torch", "device_key": None},
        {"engine_key": "qwen3-torch", "device_key": None},
    ]


def _write_fake_wave(target, *, sample_rate: int) -> None:
    import io
    import wave

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


def test_tts_service_routes_legacy_backend_execution_through_scheduler_gateway(tmp_path: Path):
    settings = _make_core_settings(tmp_path)
    registry = StubRegistry()
    scheduler = EngineScheduler()
    service = TTSService(  # type: ignore[arg-type]
        registry=registry,
        settings=settings,
        scheduler=scheduler,
    )
    captured_kwargs: dict = {}
    scheduler_calls: list[dict[str, object]] = []
    original_submit = scheduler.submit_engine_task

    def record_submit(**kwargs):
        scheduler_calls.append(
            {
                "engine_key": kwargs["engine_key"],
                "device_key": kwargs.get("device_key"),
            }
        )
        return original_submit(**kwargs)

    scheduler.submit_engine_task = record_submit  # type: ignore[method-assign]
    registry.backend.execute = _capture_execute(captured_kwargs)

    result = service.synthesize_design(
        VoiceDesignCommand(text="Hello", model="omnivoice-design-1", voice_description="Warm")
    )

    assert result.mode == "design"
    assert captured_kwargs["mode"] == "design"
    assert scheduler_calls == [{"engine_key": "tts-service-compat", "device_key": None}]


def test_tts_service_routes_piper_engine_execution_through_scheduler_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from types import SimpleNamespace

    from tests.unit.core.test_piper_engine import _EngineAwareRegistry, _make_settings, _write_piper_artifacts

    settings = _make_settings(tmp_path, piper_engine_enabled=True)
    registry = _EngineAwareRegistry(settings.models_dir)
    spec = MODEL_SPECS["piper-1"]
    _write_piper_artifacts(settings.models_dir / spec.folder)
    scheduler = EngineScheduler()
    scheduler_calls: list[dict[str, object]] = []
    original_submit = scheduler.submit_engine_task

    def record_submit(**kwargs):
        scheduler_calls.append(
            {
                "engine_key": kwargs["engine_key"],
                "device_key": kwargs.get("device_key"),
            }
        )
        return original_submit(**kwargs)

    scheduler.submit_engine_task = record_submit  # type: ignore[method-assign]

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

    service = TTSService(  # type: ignore[arg-type]
        registry=registry,
        settings=settings,
        scheduler=scheduler,
    )

    result = service.synthesize_custom(
        CustomVoiceCommand(text="Hello Piper", model=spec.model_id, speaker="ignored")
    )

    assert result.backend == "onnx"
    assert result.audio.bytes_data.startswith(b"RIFF")
    assert scheduler_calls == [{"engine_key": "piper-onnx", "device_key": None}]
