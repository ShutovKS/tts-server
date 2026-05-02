# FILE: tests/unit/core/test_engine_bridge.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for the temporary engine compatibility bridge that exposes legacy family/backend lanes through EngineRegistry metadata without changing synthesis behavior.
#   SCOPE: Legacy bridge ledger inventory, EngineRegistry coexistence, metadata-only non-execution guarantees, and planner/service path invariance evidence.
#   DEPENDS: M-ENGINE-BRIDGE, M-ENGINE-REGISTRY, M-TTS-SERVICE, M-SYNTHESIS-PLANNER
#   LINKS: V-M-ENGINE-BRIDGE
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _BridgeRegistryStub - Minimal runtime registry stub used to prove planner/service execution remains on the legacy path.
#   _make_core_settings - Build isolated core settings for bridge coexistence tests.
#   test_engine_compatibility_bridge_exposes_legacy_family_backend_records - Verifies deterministic temporary bridge ledger coverage for current family/backend lanes.
#   test_engine_compatibility_bridge_registers_temporary_legacy_registry_entries - Verifies bridge inventory can populate EngineRegistry with temporary legacy keys and aliases.
#   test_engine_compatibility_bridge_coexists_with_planner_and_service_execution - Verifies planner output and TTSService execution stay on the current legacy runtime path while the bridge exists.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Task 8 compatibility bridge: added deterministic unit coverage for legacy engine inventory, temporary registry registration, and unchanged planner/service execution behavior]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from core.backends.base import ExecutionRequest, LoadedModelHandle, TTSBackend
from core.config import CoreSettings
from core.contracts import BackendRouteInfo
from core.contracts.commands import CustomVoiceCommand
from core.contracts.synthesis import SynthesisRequest
from core.engines.compatibility import (
    EngineCompatibilityBridge,
    build_legacy_engine_registry,
)
from core.models.catalog import MODEL_SPECS, ModelSpec
from core.services.tts_service import TTSService, _build_family_adapter_map
from tests.support.api_fakes import make_wav_bytes

pytestmark = pytest.mark.unit


class _StubBackend(TTSBackend):
    key = "torch"
    label = "PyTorch + Transformers"

    def __init__(self) -> None:
        self.execute_calls = 0
        self.last_request: ExecutionRequest | None = None

    def execute(self, request: ExecutionRequest) -> None:
        self.execute_calls += 1
        self.last_request = request
        (Path(request.output_dir) / "audio_0001.wav").write_bytes(make_wav_bytes())

    def capabilities(self):  # pragma: no cover
        raise NotImplementedError

    def is_available(self) -> bool:
        return True

    def supports_platform(self) -> bool:
        return True

    def resolve_model_path(self, folder_name: str) -> Path | None:  # pragma: no cover
        return None

    def load_model(self, spec):  # pragma: no cover
        raise NotImplementedError

    def inspect_model(self, spec):  # pragma: no cover
        return {}

    def readiness_diagnostics(self):  # pragma: no cover
        raise NotImplementedError

    def cache_diagnostics(self):  # pragma: no cover
        return {}

    def metrics_summary(self):  # pragma: no cover
        return {}

    def preload_models(self, specs):  # pragma: no cover
        return {}


class _BridgeRegistryStub:
    def __init__(self) -> None:
        self._backend = _StubBackend()

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
        return spec, LoadedModelHandle(
            spec=spec,
            runtime_model=object(),
            resolved_path=None,
            backend_key="torch",
        )

    def backend_for_spec(self, spec: ModelSpec) -> TTSBackend:
        return self._backend

    def backend_route_for_spec(self, spec: ModelSpec) -> BackendRouteInfo:
        return {
            "route_reason": "registry_model_resolution",
            "execution_backend": self._backend.key,
        }


def _make_core_settings(tmp_path: Path) -> CoreSettings:
    qwen_custom_model = next(
        spec.model_id
        for spec in MODEL_SPECS.values()
        if spec.family == "Qwen3-TTS" and spec.mode == "custom"
    )
    settings = CoreSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        active_family="qwen",
        default_custom_model=qwen_custom_model,
    )
    settings.ensure_directories()
    return settings


def test_engine_compatibility_bridge_exposes_legacy_family_backend_records() -> None:
    bridge = EngineCompatibilityBridge(family_adapters=tuple(_build_family_adapter_map().values()))

    records = bridge.legacy_records()
    keys = {record.engine_key for record in records}

    assert keys == {
        "legacy-omnivoice-torch",
        "legacy-piper-onnx",
        "legacy-qwen3_tts-mlx",
        "legacy-qwen3_tts-qwen_fast",
        "legacy-qwen3_tts-torch",
    }
    qwen_fast_record = next(record for record in records if record.engine_key == "legacy-qwen3_tts-qwen_fast")
    assert qwen_fast_record.family_key == "qwen3_tts"
    assert qwen_fast_record.backend_key == "qwen_fast"
    assert "preset_speaker_tts" in qwen_fast_record.capabilities
    assert qwen_fast_record.model_ids
    assert "TEMPORARY compatibility bridge" in qwen_fast_record.deletion_criteria


def test_engine_compatibility_bridge_registers_temporary_legacy_registry_entries() -> None:
    registry = build_legacy_engine_registry(
        family_adapters=tuple(_build_family_adapter_map().values())
    )

    assert registry.keys() == (
        "legacy-omnivoice-torch",
        "legacy-piper-onnx",
        "legacy-qwen3_tts-mlx",
        "legacy-qwen3_tts-qwen_fast",
        "legacy-qwen3_tts-torch",
    )
    legacy_qwen = registry.resolve_engine(
        engine_key="legacy-qwen3_tts-torch",
        capability="preset_speaker_tts",
        family="qwen3_tts",
        backend_key="torch",
    )
    legacy_alias = registry.get("legacy-qwen3_tts-torch-bridge")
    availability_reason = legacy_qwen.availability().reason

    assert legacy_qwen.key == "legacy-qwen3_tts-torch"
    assert availability_reason is not None
    assert "TEMPORARY compatibility bridge" in availability_reason
    assert legacy_alias is legacy_qwen


def test_engine_compatibility_bridge_coexists_with_planner_and_service_execution(
    tmp_path: Path,
) -> None:
    settings = _make_core_settings(tmp_path)
    registry = _BridgeRegistryStub()
    bridge_registry = build_legacy_engine_registry(
        family_adapters=tuple(_build_family_adapter_map().values())
    )
    service = TTSService(registry=registry, settings=settings)  # type: ignore[arg-type]

    plan = service.planner.plan(
        SynthesisRequest.from_command(CustomVoiceCommand(text="Hello", speaker="Ryan"))
    )
    result = service.synthesize_custom(CustomVoiceCommand(text="Hello", speaker="Ryan"))
    backend = cast(_StubBackend, registry.backend)

    assert plan.backend_key == "torch"
    assert plan.family_key == "qwen3_tts"
    assert backend.execute_calls == 1
    assert backend.last_request is not None
    assert backend.last_request.execution_mode == "custom"
    assert result.backend == "torch"
    assert bridge_registry.get("legacy-qwen3_tts-torch") is not None
    assert bridge_registry.resolve_engine(
        capability="preset_speaker_tts",
        family="qwen3_tts",
        backend_key="torch",
    ).key == "legacy-qwen3_tts-torch"
