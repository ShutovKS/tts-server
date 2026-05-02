# FILE: tests/unit/core/test_engine_registry.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for the engine registry and optional engine discovery loader.
#   SCOPE: explicit registration, disabled config skipping, deterministic capability/language selection, duplicate key failures, and optional entry-point load isolation
#   DEPENDS: M-ENGINE-REGISTRY
#   LINKS: V-M-ENGINE-REGISTRY
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _StubEngine - Minimal TTSEngine implementation used to exercise registry behavior.
#   _AbstractStubEngine - Abstract TTSEngine subclass used to assert invalid entry-point filtering.
#   test_registry_register_and_get_by_key_or_alias - Verifies direct registration and alias lookup.
#   test_registry_resolves_explicit_key_before_capability_matching - Verifies explicit engine key lookup wins when constraints are satisfied.
#   test_registry_resolves_by_capability_language_priority_and_order - Verifies capability/language matching uses deterministic priority and registration order.
#   test_registry_skips_disabled_configs_deterministically - Verifies disabled engine configs are not registered.
#   test_registry_rejects_duplicate_engine_keys - Verifies duplicate keys raise deterministic errors naming the key.
#   test_load_engine_registry_skips_failed_optional_entry_point_loads - Verifies optional entry-point failures are logged and isolated in non-fail-fast mode.
#   test_load_engine_registry_rejects_invalid_entry_point_objects_in_non_fail_fast_mode - Verifies invalid entry-point objects/classes are logged and skipped.
#   test_load_engine_registry_fail_fast_raises_entry_point_errors - Verifies fail-fast mode raises instead of warning-and-skip behavior.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added focused unit coverage for engine registry registration, deterministic resolution, disabled-config filtering, and optional entry-point loader isolation]
# END_CHANGE_SUMMARY

from __future__ import annotations

from abc import abstractmethod
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from core.engines import (
    AudioBuffer,
    EngineAvailability,
    EngineCapabilities,
    EngineRegistry,
    EngineRegistryError,
    ModelHandle,
    SynthesisJob,
    TTSEngine,
    load_engine_registry,
    parse_engine_settings,
)
from core.engines.registry import ENGINE_ENTRY_POINT_GROUP
from core.models.catalog import MODEL_SPECS

pytestmark = pytest.mark.unit


# START_BLOCK_TEST_STUBS
class _StubEngine(TTSEngine):
    def __init__(
        self,
        *,
        key: str,
        families: tuple[str, ...] = ("qwen3_tts",),
        backends: tuple[str, ...] = ("torch",),
        capabilities: tuple[str, ...] = ("preset_speaker_tts",),
        languages: tuple[str, ...] = (),
        available: bool = True,
        enabled: bool = True,
    ) -> None:
        self.key = key
        self.label = f"stub:{key}"
        self.aliases = (f"{key}-alias",)
        self.languages = languages
        self._families = families
        self._backends = backends
        self._capabilities = capabilities
        self._available = available
        self._enabled = enabled

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            families=self._families,
            backends=self._backends,
            capabilities=self._capabilities,
        )

    def availability(self) -> EngineAvailability:
        return EngineAvailability(
            engine_key=self.key,
            is_available=self._available,
            is_enabled=self._enabled,
        )

    def load_model(
        self,
        *,
        spec,
        backend_key: str,
        model_path: Path | None,
    ) -> ModelHandle:
        return ModelHandle(
            spec=spec,
            runtime_model=object(),
            resolved_path=model_path,
            engine_key=self.key,
            backend_key=backend_key,
            family_key=self._families[0] if self._families else "",
        )

    def synthesize(self, handle: ModelHandle, job: SynthesisJob) -> AudioBuffer:
        return AudioBuffer(waveform=b"audio", sample_rate=24000)


class _AbstractStubEngine(TTSEngine):
    key = "abstract-entry"
    label = "abstract-entry"

    @abstractmethod
    def capabilities(self) -> EngineCapabilities: ...

    @abstractmethod
    def availability(self) -> EngineAvailability: ...

    @abstractmethod
    def load_model(self, *, spec, backend_key: str, model_path: Path | None) -> ModelHandle: ...

    @abstractmethod
    def synthesize(self, handle: ModelHandle, job: SynthesisJob) -> AudioBuffer: ...


class _EntryPointEngine(_StubEngine):
    def __init__(self) -> None:
        super().__init__(key="entry-engine")


# END_BLOCK_TEST_STUBS


def test_registry_register_and_get_by_key_or_alias() -> None:
    engine = _StubEngine(key="qwen-torch")
    registry = EngineRegistry()

    registry.register(engine)

    assert registry.get("qwen-torch") is engine
    assert registry.get("qwen-torch-alias") is engine
    assert registry.keys() == ("qwen-torch",)


def test_registry_resolves_explicit_key_before_capability_matching() -> None:
    first = _StubEngine(key="alpha", capabilities=("preset_speaker_tts",))
    second = _StubEngine(key="beta", capabilities=("preset_speaker_tts",))
    registry = EngineRegistry()
    registry.register(first)
    registry.register(second)

    resolved = registry.resolve_engine(engine_key="beta", capability="preset_speaker_tts")

    assert resolved is second


def test_registry_resolves_by_capability_language_priority_and_order() -> None:
    settings = parse_engine_settings(
        {
            "engines": [
                {
                    "kind": "torch",
                    "name": "generic",
                    "family": "qwen3_tts",
                    "capabilities": ["preset_speaker_tts"],
                    "priority": 50,
                },
                {
                    "kind": "torch",
                    "name": "english",
                    "family": "qwen3_tts",
                    "capabilities": ["preset_speaker_tts"],
                    "priority": 20,
                    "params": {"languages": ["en", "en-US"]},
                },
                {
                    "kind": "torch",
                    "name": "english-fallback",
                    "family": "qwen3_tts",
                    "capabilities": ["preset_speaker_tts"],
                    "priority": 20,
                    "params": {"languages": ["en"]},
                },
            ]
        }
    )

    registry = load_engine_registry(
        explicit_engines=(
            _StubEngine(key="generic"),
            _StubEngine(key="english"),
            _StubEngine(key="english-fallback"),
        ),
        settings=settings,
        include_entry_points=False,
    )

    assert (
        registry.resolve_engine(capability="preset_speaker_tts", language="en").key
        == "english"
    )
    assert (
        registry.resolve_engine(capability="preset_speaker_tts", language="fr").key
        == "generic"
    )


def test_registry_skips_disabled_configs_deterministically() -> None:
    settings = parse_engine_settings(
        {
            "engines": [
                {
                    "kind": "disabled",
                    "name": "legacy-engine",
                    "aliases": ["legacy-engine-alias"],
                    "reason": "disabled on this host",
                },
                {
                    "kind": "onnx",
                    "name": "piper-live",
                    "family": "piper",
                    "capabilities": ["preset_speaker_tts"],
                    "priority": 10,
                },
            ]
        }
    )

    registry = load_engine_registry(
        explicit_engines=(
            _StubEngine(key="legacy-engine"),
            _StubEngine(key="piper-live", families=("piper",), backends=("onnx",)),
        ),
        settings=settings,
        include_entry_points=False,
    )

    assert registry.keys() == ("piper-live",)
    assert registry.get("legacy-engine") is None


def test_registry_rejects_duplicate_engine_keys() -> None:
    registry = EngineRegistry()
    registry.register(_StubEngine(key="dupe"))

    with pytest.raises(EngineRegistryError, match="duplicate engine key 'dupe'"):
        registry.register(_StubEngine(key="dupe"))


def test_load_engine_registry_skips_failed_optional_entry_point_loads(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ok_entry = EntryPoint(
        name="entry-engine",
        value="tests.unit.core.test_engine_registry:_EntryPointEngine",
        group=ENGINE_ENTRY_POINT_GROUP,
    )
    bad_entry = EntryPoint(
        name="broken-engine",
        value="tests.unit.core.test_engine_registry:_BrokenEngine",
        group=ENGINE_ENTRY_POINT_GROUP,
    )
    load_results: list[Any] = [_EntryPointEngine, RuntimeError("boom")]

    def loader() -> list[EntryPoint]:
        return [ok_entry, bad_entry]

    def fake_load(_self) -> Any:
        result = load_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    with caplog.at_level("WARNING"):
        with patch.object(EntryPoint, "load", fake_load):
            registry = load_engine_registry(
                include_entry_points=True,
                entry_points_loader=loader,
                fail_fast=False,
            )

    assert registry.keys() == ("entry-engine",)
    assert "Skipping optional engine entry point after load failure" in caplog.text
    assert "broken-engine" in caplog.text


def test_load_engine_registry_rejects_invalid_entry_point_objects_in_non_fail_fast_mode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    not_class_entry = EntryPoint(
        name="not-class",
        value="x:y",
        group=ENGINE_ENTRY_POINT_GROUP,
    )
    wrong_base_entry = EntryPoint(
        name="wrong-base",
        value="x:y",
        group=ENGINE_ENTRY_POINT_GROUP,
    )
    abstract_entry = EntryPoint(
        name="abstract",
        value="x:y",
        group=ENGINE_ENTRY_POINT_GROUP,
    )
    load_results: list[Any] = [42, dict, _AbstractStubEngine]

    def loader() -> list[EntryPoint]:
        return [not_class_entry, wrong_base_entry, abstract_entry]

    def fake_load(_self) -> Any:
        return load_results.pop(0)

    with caplog.at_level("WARNING"):
        with patch.object(EntryPoint, "load", fake_load):
            registry = load_engine_registry(
                include_entry_points=True,
                entry_points_loader=loader,
                fail_fast=False,
            )

    assert registry.keys() == ()
    assert "not-class" in caplog.text
    assert "wrong-base" in caplog.text
    assert "abstract" in caplog.text


def test_load_engine_registry_fail_fast_raises_entry_point_errors() -> None:
    broken_entry = EntryPoint(
        name="broken-engine",
        value="x:y",
        group=ENGINE_ENTRY_POINT_GROUP,
    )

    def loader() -> list[EntryPoint]:
        return [broken_entry]

    with patch.object(EntryPoint, "load", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            load_engine_registry(
                include_entry_points=True,
                entry_points_loader=loader,
                fail_fast=True,
            )


def test_registry_resolve_rejects_unmatched_explicit_constraints() -> None:
    engine = _StubEngine(key="qwen-en", languages=("en",))
    registry = EngineRegistry()
    registry.register(engine)

    with pytest.raises(EngineRegistryError, match="does not satisfy the requested constraints"):
        registry.resolve_engine(engine_key="qwen-en", language="fr")


def test_registry_resolve_ignores_unavailable_engines_by_default() -> None:
    unavailable = _StubEngine(key="offline", available=False)
    available = _StubEngine(key="online")
    registry = EngineRegistry()
    registry.register(unavailable)
    registry.register(available)

    assert registry.resolve_engine(capability="preset_speaker_tts").key == "online"


def test_registry_stub_engine_contract_stays_compatible() -> None:
    spec = next(iter(MODEL_SPECS.values()))
    engine = _StubEngine(key="contract-check")

    handle = engine.load_model(spec=spec, backend_key="torch", model_path=Path(".models/demo"))
    result = engine.synthesize(
        handle,
        SynthesisJob(
            capability="preset_speaker_tts",
            execution_mode="custom",
            text="hello",
            language="en",
            output_dir=Path(".outputs"),
        ),
    )

    assert handle.engine_key == "contract-check"
    assert result.sample_rate == 24000
