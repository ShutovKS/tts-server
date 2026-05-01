# FILE: tests/unit/core/test_family_plugin.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for the unified ModelFamilyPlugin extension contract and FamilyPluginRegistry introduced in Phase 2.5.
#   SCOPE: plugin lifecycle (register / lookup / capability and backend filters), default validate_artifacts behavior, FamilyExecutionRequest / FamilyExecutionResult DTO shape
#   DEPENDS: M-MODEL-FAMILY
#   LINKS: V-M-MODEL-FAMILY
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _StubPlugin - Minimal ModelFamilyPlugin subclass used to exercise the registry.
#   build_plugin - Helper that constructs _StubPlugin with deterministic identifiers.
#   test_registry_register_and_lookup - Verifies register / get / keys behavior.
#   test_registry_filters - Verifies for_capability / for_backend filtering.
#   test_registry_rejects_duplicate_keys - Verifies duplicate registration is rejected.
#   test_registry_rejects_empty_key_plugin - Verifies plugins must declare a non-empty key.
#   test_default_validate_artifacts_uses_spec_rules - Verifies the default validate_artifacts delegates to ModelSpec.
#   test_default_validate_artifacts_handles_missing_path - Verifies the default validate_artifacts handles None paths.
#   test_family_execution_request_is_frozen - Verifies the request DTO is immutable.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Phase 2.5 plugin contract coverage]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from core.model_families import (
    FamilyExecutionRequest,
    FamilyExecutionResult,
    FamilyPluginRegistry,
    ModelFamilyPlugin,
)
from core.models.catalog import MODEL_SPECS

pytestmark = pytest.mark.unit


class _StubPlugin(ModelFamilyPlugin):
    def __init__(
        self,
        *,
        key: str,
        capabilities: tuple[str, ...],
        supported_backends: tuple[str, ...],
        available: bool = True,
        import_error: Exception | None = None,
    ) -> None:
        self.key = key
        self.label = f"stub:{key}"
        self.capabilities = capabilities
        self.supported_backends = supported_backends
        self._available = available
        self._import_error = import_error

    def is_available(self) -> bool:
        return self._available

    def import_error(self) -> Exception | None:
        return self._import_error

    def load_model(
        self,
        *,
        spec: Any,
        backend_key: str,
        model_path: Path,
    ) -> Any:
        return {"spec": spec, "backend": backend_key, "path": str(model_path)}

    def synthesize(
        self,
        model: Any,
        request: FamilyExecutionRequest,
        *,
        backend_key: str,
    ) -> FamilyExecutionResult:
        return FamilyExecutionResult(
            waveforms=[b"audio:" + request.text.encode()], sample_rate=24000
        )


def build_plugin(
    *,
    key: str,
    capabilities: tuple[str, ...] = ("preset_speaker_tts",),
    supported_backends: tuple[str, ...] = ("torch",),
) -> _StubPlugin:
    return _StubPlugin(
        key=key,
        capabilities=capabilities,
        supported_backends=supported_backends,
    )


def test_registry_register_and_lookup() -> None:
    registry = FamilyPluginRegistry()
    plugin = build_plugin(key="alpha")

    registry.register(plugin)

    assert registry.get("alpha") is plugin
    assert registry.get("missing") is None
    assert registry.keys() == ("alpha",)
    assert registry.plugins() == (plugin,)


def test_registry_filters() -> None:
    qwen_like = build_plugin(
        key="qwen3_tts",
        capabilities=(
            "preset_speaker_tts",
            "voice_description_tts",
            "reference_voice_clone",
        ),
        supported_backends=("torch", "qwen_fast"),
    )
    piper_like = build_plugin(
        key="piper",
        capabilities=("preset_speaker_tts",),
        supported_backends=("onnx",),
    )

    registry = FamilyPluginRegistry(plugins=(qwen_like, piper_like))

    assert registry.for_capability("preset_speaker_tts") == (piper_like, qwen_like)
    assert registry.for_capability("voice_description_tts") == (qwen_like,)
    assert registry.for_capability("emotion_transfer") == ()
    assert registry.for_backend("torch") == (qwen_like,)
    assert registry.for_backend("onnx") == (piper_like,)


def test_registry_rejects_duplicate_keys() -> None:
    registry = FamilyPluginRegistry()
    registry.register(build_plugin(key="dupe"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(build_plugin(key="dupe"))


def test_registry_rejects_empty_key_plugin() -> None:
    plugin = build_plugin(key="dummy")
    plugin.key = ""

    registry = FamilyPluginRegistry()
    with pytest.raises(ValueError, match="non-empty"):
        registry.register(plugin)


def test_default_validate_artifacts_uses_spec_rules(tmp_path: Path) -> None:
    plugin = build_plugin(key="qwen3_tts", supported_backends=("torch",))
    spec = next(iter(MODEL_SPECS.values()))

    result = plugin.validate_artifacts(spec=spec, backend_key="torch", model_path=tmp_path)

    assert "loadable" in result
    assert "required_artifacts" in result
    assert "missing_artifacts" in result


def test_default_validate_artifacts_handles_missing_path() -> None:
    plugin = build_plugin(key="qwen3_tts", supported_backends=("torch",))
    spec = next(iter(MODEL_SPECS.values()))

    result = plugin.validate_artifacts(spec=spec, backend_key="torch", model_path=None)

    assert result["loadable"] is False
    assert "model_directory" in result["missing_artifacts"]


def test_family_execution_request_is_frozen(tmp_path: Path) -> None:
    request = FamilyExecutionRequest(
        capability="preset_speaker_tts",
        execution_mode="custom",
        text="hello",
        language="en",
        output_dir=tmp_path,
        payload={"voice": "preset", "speed": 1.0},
    )

    with pytest.raises(FrozenInstanceError):
        request.text = "mutated"  # type: ignore[misc]
