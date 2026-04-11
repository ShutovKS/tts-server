# FILE: tests/unit/core/test_backend_registry.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for backend registry selection and model resolution.
#   SCOPE: Backend discovery, capability checks, model metadata inspection
#   DEPENDS: M-CORE
#   LINKS: V-M-CORE
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   StubBackend - Minimal backend stub used for registry selection and capability tests
#   NoAffinityBackend - Backend stub with no manifest affinity match for rejection tests
#   test_backend_registry_prefers_explicit_backend - Verifies explicit backend selection wins over autoselection
#   test_backend_registry_raises_for_unknown_backend - Verifies unknown backend ids are rejected
#   test_backend_registry_rejects_unsupported_mode - Verifies unsupported synthesis modes raise capability errors
#   test_backend_registry_lists_selected_backend_metadata - Verifies list_backends reports selection state
#   test_backend_registry_resolves_model_spec_for_mode - Verifies mode-based model resolution
#   test_backend_registry_prefers_highest_rollout_preference_for_mode - Verifies rollout preference ordering for model selection
#   test_backend_registry_raises_model_not_available_for_unknown_model_identifier - Verifies unknown model ids surface not-available errors
#   test_backend_registry_raises_model_not_available_when_mode_has_no_local_artifacts - Verifies missing local artifacts surface not-available errors
#   test_backend_registry_rejects_model_when_backend_is_not_in_affinity - Verifies backend affinity is enforced
#   test_model_registry_applies_listed_preload_policy - Verifies listed preload policy preloads requested models
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT and MODULE_MAP]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path

import pytest

from core.backends.base import ExecutionRequest, LoadedModelHandle, TTSBackend
from core.backends.capabilities import BackendCapabilitySet, BackendDiagnostics
from core.backends.registry import BackendRegistry
from core.errors import (
    BackendCapabilityError,
    BackendNotAvailableError,
    ModelNotAvailableError,
)
from core.models.catalog import MODEL_SPECS
from core.services.model_registry import ModelRegistry


pytestmark = pytest.mark.unit


class StubBackend(TTSBackend):
    def __init__(
        self,
        *,
        key: str,
        available: bool,
        platform_supported: bool,
        ready: bool | None = None,
        supports_custom: bool = True,
        supports_design: bool | None = None,
        supports_clone: bool = True,
        diagnostics_reason: str | None = None,
        missing_folders: set[str] | None = None,
    ):
        self.key = key
        self.label = key.upper()
        self._available = available
        self._platform_supported = platform_supported
        self._ready = ready
        self._supports_custom = supports_custom
        self._supports_design = (
            self.key != "clone-only" if supports_design is None else supports_design
        )
        self._supports_clone = supports_clone
        self._diagnostics_reason = diagnostics_reason
        self._missing_folders = missing_folders or set()
        self.preloaded_specs: list[str] = []

    def capabilities(self) -> BackendCapabilitySet:
        return BackendCapabilitySet(
            supports_custom=self._supports_custom,
            supports_design=self._supports_design,
            supports_clone=self._supports_clone,
            platforms=("darwin", "linux", "windows"),
        )

    def is_available(self) -> bool:
        return self._available

    def supports_platform(self) -> bool:
        return self._platform_supported

    def resolve_model_path(self, folder_name: str) -> Path | None:
        if folder_name in self._missing_folders:
            return None
        return Path(".models") / folder_name

    def load_model(self, spec):
        return LoadedModelHandle(
            spec=spec,
            runtime_model=object(),
            resolved_path=Path(".models") / spec.folder,
            backend_key=self.key,
        )

    def inspect_model(self, spec):
        cached = spec.api_name in self.preloaded_specs
        return {
            "key": spec.key,
            "id": spec.api_name,
            "name": spec.public_name,
            "mode": spec.mode,
            "folder": spec.folder,
            "backend": self.key,
            "configured": True,
            "available": True,
            "loadable": True,
            "runtime_ready": self._available,
            "cached": cached,
            "resolved_path": str(Path(".models") / spec.folder),
            "runtime_path": str(Path(".models") / spec.folder),
            "cache": {
                "loaded": cached,
                "cache_key": spec.folder,
                "backend": self.key,
                "normalized_runtime": False,
                "runtime_path": str(Path(".models") / spec.folder),
                "eviction_policy": "not_configured",
            },
            "missing_artifacts": [],
            "required_artifacts": ["config.json"],
            "capabilities": self.capabilities().to_dict(),
        }

    def readiness_diagnostics(self) -> BackendDiagnostics:
        ready = (
            self._available and self._platform_supported
            if self._ready is None
            else self._ready
        )
        return BackendDiagnostics(
            backend_key=self.key,
            backend_label=self.label,
            available=self._available,
            ready=ready,
            reason=self._diagnostics_reason,
            details={},
        )

    def cache_diagnostics(self) -> dict[str, object]:
        return {
            "cached_model_count": len(self.preloaded_specs),
            "cached_model_ids": list(self.preloaded_specs),
            "cache_policy": {
                "cache_scope": "process_local",
                "eviction": "not_configured",
            },
            "loaded_models": [
                {
                    "cache_key": model_id,
                    "model_id": model_id,
                    "backend": self.key,
                    "loaded": True,
                    "resolved_path": str(Path(".models") / model_id),
                    "runtime_path": str(Path(".models") / model_id),
                    "normalized_runtime": False,
                }
                for model_id in self.preloaded_specs
            ],
        }

    def preload_models(self, specs) -> dict[str, object]:
        loaded_model_ids = [spec.api_name for spec in specs]
        self.preloaded_specs = loaded_model_ids
        return {
            "requested": len(specs),
            "attempted": len(specs),
            "loaded": len(specs),
            "failed": 0,
            "loaded_model_ids": loaded_model_ids,
            "failed_model_ids": [],
            "errors": [],
        }

    def synthesize_custom(
        self,
        handle,
        *,
        text: str,
        output_dir: Path,
        language: str,
        speaker: str,
        instruct: str,
        speed: float,
    ) -> None:
        return None

    def synthesize_design(
        self,
        handle,
        *,
        text: str,
        output_dir: Path,
        language: str,
        voice_description: str,
    ) -> None:
        return None

    def synthesize_clone(
        self,
        handle,
        *,
        text: str,
        output_dir: Path,
        language: str,
        ref_audio_path: Path,
        ref_text: str | None,
    ) -> None:
        return None


class NoAffinityBackend(StubBackend):
    def __init__(self):
        super().__init__(key="custom-backend", available=True, platform_supported=True)


def test_backend_registry_prefers_explicit_backend():
    registry = BackendRegistry(
        [
            StubBackend(key="mlx", available=True, platform_supported=True),
            StubBackend(key="torch", available=True, platform_supported=True),
        ],
        requested_backend="torch",
        autoselect=True,
    )

    assert registry.selected_backend.key == "torch"
    assert registry.selection.selection_reason == "explicit_config"


def test_backend_registry_raises_for_unknown_backend():
    with pytest.raises(BackendNotAvailableError):
        BackendRegistry(
            [StubBackend(key="mlx", available=True, platform_supported=True)],
            requested_backend="unknown",
            autoselect=True,
        )


def test_backend_registry_rejects_unsupported_mode():
    registry = BackendRegistry(
        [StubBackend(key="clone-only", available=True, platform_supported=True)],
        autoselect=True,
    )

    with pytest.raises(BackendCapabilityError):
        registry.ensure_mode_supported("design")


def test_backend_registry_lists_selected_backend_metadata():
    registry = BackendRegistry(
        [
            StubBackend(key="mlx", available=False, platform_supported=True),
            StubBackend(key="torch", available=True, platform_supported=True),
        ],
        requested_backend="torch",
        autoselect=True,
    )

    payload = registry.list_backends()

    assert any(item["key"] == "torch" and item["selected"] is True for item in payload)
    assert any(item["key"] == "mlx" and item["selected"] is False for item in payload)
    assert all("host_reason" in item for item in payload)
    assert all("selection_score" in item for item in payload)


def test_backend_registry_resolves_model_spec_for_mode():
    registry = BackendRegistry(
        [StubBackend(key="torch", available=True, platform_supported=True)],
        requested_backend="torch",
        autoselect=True,
    )

    spec = registry.get_model_spec(mode="custom")

    assert spec == MODEL_SPECS["1"]


def test_backend_registry_prefers_highest_rollout_preference_for_mode():
    registry = BackendRegistry(
        [StubBackend(key="torch", available=True, platform_supported=True)],
        requested_backend="torch",
        autoselect=True,
    )

    spec = registry.get_model_spec(mode="clone")

    assert spec == MODEL_SPECS["3"]


def test_backend_registry_raises_model_not_available_for_unknown_model_identifier():
    registry = BackendRegistry(
        [StubBackend(key="torch", available=True, platform_supported=True)],
        requested_backend="torch",
        autoselect=True,
    )

    with pytest.raises(ModelNotAvailableError) as exc_info:
        registry.get_model_spec(model_name="unknown-model")

    assert exc_info.value.model_name == "unknown-model"
    assert exc_info.value.context.to_dict() == {
        "reason": "Requested model is not available: unknown-model",
        "model": "unknown-model",
        "backend": "torch",
    }


def test_backend_registry_raises_model_not_available_when_mode_has_no_local_artifacts():
    registry = BackendRegistry(
        [
            StubBackend(
                key="torch",
                available=True,
                platform_supported=True,
                missing_folders={
                    spec.folder
                    for spec in MODEL_SPECS.values()
                    if spec.mode == "custom"
                },
            )
        ],
        requested_backend="torch",
        autoselect=True,
    )

    with pytest.raises(ModelNotAvailableError) as exc_info:
        registry.get_model_spec(mode="custom")

    assert exc_info.value.model_name is None
    assert exc_info.value.context.to_dict() == {
        "reason": "No local model is available for mode: custom",
        "mode": "custom",
        "backend": "torch",
    }


def test_backend_registry_rejects_model_when_backend_is_not_in_affinity():
    registry = BackendRegistry(
        [NoAffinityBackend()], requested_backend="custom-backend", autoselect=True
    )

    with pytest.raises(BackendCapabilityError) as exc_info:
        registry.get_model_spec(model_name=MODEL_SPECS["1"].api_name)

    assert exc_info.value.context.to_dict() == {
        "reason": "Backend 'custom-backend' is not enabled for model 'Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit'",
        "backend": "custom-backend",
        "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "mode": "custom",
    }


def test_model_registry_applies_listed_preload_policy():
    backend = StubBackend(key="torch", available=True, platform_supported=True)
    backend_registry = BackendRegistry(
        [backend], requested_backend="torch", autoselect=True
    )

    registry = ModelRegistry(
        backend_registry=backend_registry,
        preload_policy="listed",
        preload_model_ids=(MODEL_SPECS["1"].api_name,),
    )

    report = registry.readiness_report()

    assert backend.preloaded_specs == [MODEL_SPECS["1"].api_name]
    assert report["loaded_models"] == 1
    assert report["preload"]["policy"] == "listed"
    assert report["preload"]["loaded_model_ids"] == [MODEL_SPECS["1"].api_name]
    assert report["cache_diagnostics"]["cached_model_count"] == 1
    assert report["items"][0]["preload"]["status"] == "loaded"


def test_model_registry_exposes_family_aware_descriptors():
    backend = StubBackend(key="torch", available=True, platform_supported=True)
    backend_registry = BackendRegistry(
        [backend], requested_backend="torch", autoselect=True
    )

    registry = ModelRegistry(backend_registry=backend_registry)

    descriptors = registry.model_descriptors

    assert descriptors[0].family_key == "qwen3_tts"
    assert descriptors[0].model_id == descriptors[0].folder
    assert descriptors[0].supported_capabilities


def test_model_registry_exposes_split_registry_runtime_state():
    backend = StubBackend(key="torch", available=True, platform_supported=True)
    backend_registry = BackendRegistry(
        [backend], requested_backend="torch", autoselect=True
    )

    registry = ModelRegistry(
        backend_registry=backend_registry,
        preload_policy="listed",
        preload_model_ids=(MODEL_SPECS["1"].api_name,),
    )

    descriptor = registry.model_descriptors[0]
    runtime_state = registry.runtime_models.descriptor_runtime_state(descriptor)

    assert runtime_state["declared"] is True
    assert runtime_state["installed"] is True
    assert runtime_state["loadable"] is True
    assert runtime_state["runtime_ready"] is True
    assert runtime_state["loaded"] is True
    assert runtime_state["preload_status"] == "loaded"


def test_backend_execute_bridges_runtime_request_to_legacy_mode_methods():
    backend = StubBackend(key="torch", available=True, platform_supported=True)
    spec = MODEL_SPECS["1"]
    handle = LoadedModelHandle(
        spec=spec,
        runtime_model=object(),
        resolved_path=Path(".models") / spec.folder,
        backend_key="torch",
    )
    captured: dict[str, object] = {}

    def fake_synthesize_custom(handle, **kwargs):
        captured.update(kwargs)

    backend.synthesize_custom = fake_synthesize_custom  # type: ignore[method-assign]

    backend.execute(
        ExecutionRequest(
            handle=handle,
            text="Hello",
            output_dir=Path("/tmp"),
            language="auto",
            legacy_mode="custom",
            generation_kwargs={
                "voice": "Ryan",
                "instruct": "Friendly",
                "speed": 1.0,
            },
        )
    )

    assert captured == {
        "text": "Hello",
        "output_dir": Path("/tmp"),
        "language": "auto",
        "speaker": "Ryan",
        "instruct": "Friendly",
        "speed": 1.0,
    }


def test_model_registry_routes_piper_model_to_compatible_backend():
    backend_registry = BackendRegistry(
        [
            StubBackend(key="mlx", available=True, platform_supported=True),
            StubBackend(key="onnx", available=True, platform_supported=True),
        ],
        autoselect=True,
    )
    registry = ModelRegistry(backend_registry=backend_registry)

    spec, handle = registry.get_model(model_name=MODEL_SPECS["piper-1"].model_id)

    assert backend_registry.selected_backend.key == "mlx"
    assert spec == MODEL_SPECS["piper-1"]
    assert handle.backend_key == "onnx"


def test_model_registry_lists_per_model_backend_for_second_family():
    backend_registry = BackendRegistry(
        [
            StubBackend(key="mlx", available=True, platform_supported=True),
            StubBackend(key="onnx", available=True, platform_supported=True),
        ],
        autoselect=True,
    )
    registry = ModelRegistry(backend_registry=backend_registry)

    models = registry.list_models()
    piper_item = next(
        item for item in models if item["id"] == "Piper-en_US-lessac-medium"
    )

    assert piper_item["backend"] == "onnx"
    assert piper_item["family_key"] == "piper"
    assert piper_item["selected_backend"] == "mlx"
    assert piper_item["execution_backend"] == "onnx"
    assert piper_item["route"]["routing_mode"] == "per_model_backend_fallback"
    assert (
        piper_item["route"]["route_reason"]
        == "selected_backend_incompatible_with_model"
    )


def test_backend_registry_explains_per_model_backend_route_for_piper():
    backend_registry = BackendRegistry(
        [
            StubBackend(key="mlx", available=True, platform_supported=True),
            StubBackend(key="onnx", available=True, platform_supported=True),
        ],
        autoselect=True,
    )

    route = backend_registry.explain_backend_route_for_spec(MODEL_SPECS["piper-1"])

    assert route["selected_backend"] == "mlx"
    assert route["execution_backend"] == "onnx"
    assert route["routing_mode"] == "per_model_backend_fallback"
    assert route["route_reason"] == "selected_backend_incompatible_with_model"


def test_model_registry_readiness_report_exposes_mixed_backend_routing_summary():
    backend_registry = BackendRegistry(
        [
            StubBackend(key="mlx", available=True, platform_supported=True),
            StubBackend(key="onnx", available=True, platform_supported=True),
        ],
        autoselect=True,
    )
    registry = ModelRegistry(backend_registry=backend_registry)

    report = registry.readiness_report()

    assert report["routing"]["mixed_backend_routing"] is True
    assert report["routing"]["per_model_backend_overrides"] >= 1
    assert report["host"]["platform_system"] in {"darwin", "linux", "windows"}


def test_backend_registry_exposes_host_snapshot_for_selection_context():
    registry = BackendRegistry(
        [StubBackend(key="torch", available=True, platform_supported=True)],
        requested_backend="torch",
        autoselect=True,
    )

    snapshot = registry.host_snapshot

    assert snapshot.platform_system in {"darwin", "linux", "windows"}
    assert isinstance(snapshot.ffmpeg_available, bool)


def test_backend_registry_prefers_qwen_fast_for_custom_model_when_ready():
    backend_registry = BackendRegistry(
        [
            StubBackend(
                key="qwen_fast",
                available=True,
                platform_supported=True,
                supports_design=False,
                supports_clone=False,
            ),
            StubBackend(key="torch", available=True, platform_supported=True),
        ],
        requested_backend="qwen_fast",
        autoselect=True,
    )

    route = backend_registry.explain_backend_route_for_spec(MODEL_SPECS["1"])

    assert route["selected_backend"] == "qwen_fast"
    assert route["execution_backend"] == "qwen_fast"
    assert route["route_reason"] == "selected_backend_supports_model"


def test_backend_registry_falls_back_to_torch_when_qwen_fast_not_ready():
    backend_registry = BackendRegistry(
        [
            StubBackend(
                key="qwen_fast",
                available=True,
                platform_supported=True,
                ready=False,
                supports_design=False,
                supports_clone=False,
                diagnostics_reason="cuda_required",
            ),
            StubBackend(key="torch", available=True, platform_supported=True),
        ],
        requested_backend="qwen_fast",
        autoselect=True,
    )

    route = backend_registry.explain_backend_route_for_spec(MODEL_SPECS["1"])

    assert route["selected_backend"] == "qwen_fast"
    assert route["execution_backend"] == "torch"
    assert route["routing_mode"] == "per_model_backend_fallback"
    assert route["route_reason"] == "selected_backend_unavailable_for_model_route"
    qwen_fast_candidate = next(
        item for item in route["candidates"] if item["key"] == "qwen_fast"
    )
    assert qwen_fast_candidate["ready"] is False
    assert qwen_fast_candidate["route_reason"] == "cuda_required"
