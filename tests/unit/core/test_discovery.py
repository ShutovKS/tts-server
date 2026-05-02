# FILE: tests/unit/core/test_discovery.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for the auto-discovery helpers introduced in Phase 2.6.
#   SCOPE: subclass-based discovery for TTSBackend / ModelFamilyAdapter / ModelFamilyPlugin, entry-point discovery via injected loader, dedupe + sort behavior, abstract-class filtering, error-tolerant entry-point loading
#   DEPENDS: M-DISCOVERY
#   LINKS: V-M-DISCOVERY
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _StubBackend - In-test concrete TTSBackend used to assert backend discovery.
#   _StubAdapter - In-test concrete ModelFamilyAdapter used to assert adapter discovery.
#   _StubPlugin - In-test concrete ModelFamilyPlugin used to assert plugin discovery.
#   _AbstractStubPlugin - In-test abstract plugin used to assert abstract-class filtering.
#   _ExternalBackend - In-test concrete subclass used to assert entry-point discovery.
#   test_subclass_discovery_filters_test_local_family_adapters_by_default - Verifies normal family-adapter discovery excludes test-local tests.* subclasses.
#   test_subclass_discovery_can_include_test_local_family_adapters_when_requested - Verifies tests can still opt into explicit visibility of local adapter subclasses.
#   test_subclass_discovery_finds_concrete_subclasses - Verifies in-process discovery picks up registered subclasses.
#   test_subclass_discovery_skips_abstract_subclasses - Verifies abstract subclasses are filtered out.
#   test_entry_point_loader_loads_external_classes - Verifies entry-point discovery via an injected loader.
#   test_entry_point_loader_skips_invalid_entries - Verifies non-class / wrong-base / abstract entries are skipped.
#   test_entry_point_loader_invalid_callable_raises - Verifies non-callable loaders raise TypeError.
#   test_dedup_and_sort_is_stable - Verifies fully qualified names dedupe duplicates and ordering is deterministic.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Task 3 regression fix: family-adapter discovery tests now cover the default tests.* filter and the explicit include_test_classes override]
# END_CHANGE_SUMMARY

from __future__ import annotations

from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from core.backends.base import LoadedModelHandle, TTSBackend
from core.backends.capabilities import BackendCapabilitySet, BackendDiagnostics
from core.contracts.synthesis import ExecutionPlan
from core.discovery import (
    BACKEND_ENTRY_POINT_GROUP,
    FAMILY_PLUGIN_ENTRY_POINT_GROUP,
    discover_backend_classes,
    discover_family_adapter_classes,
    discover_family_plugin_classes,
)
from core.model_families.base import FamilyPreparedExecution, ModelFamilyAdapter
from core.model_families.plugin import (
    FamilyExecutionRequest,
    FamilyExecutionResult,
    ModelFamilyPlugin,
)
from core.models.manifest import ModelSpec

pytestmark = pytest.mark.unit


# START_BLOCK_TEST_STUBS
class _StubBackend(TTSBackend):
    key = "_stub_backend"
    label = "Stub backend"

    def execute(self, request):  # pragma: no cover - executed only when discovered
        raise NotImplementedError

    def capabilities(self) -> BackendCapabilitySet:
        return BackendCapabilitySet(
            supports_custom=True,
            supports_design=False,
            supports_clone=False,
            supports_streaming=False,
            supports_local_models=True,
            supports_voice_prompt_cache=False,
            supports_reference_transcription=False,
            preferred_formats=("wav",),
            platforms=("linux",),
        )

    def is_available(self) -> bool:
        return True

    def supports_platform(self) -> bool:
        return True

    def resolve_model_path(self, folder_name: str) -> Path | None:  # pragma: no cover
        return None

    def load_model(self, spec: ModelSpec) -> LoadedModelHandle:  # pragma: no cover
        raise NotImplementedError

    def inspect_model(self, spec: ModelSpec) -> dict[str, Any]:  # pragma: no cover
        return {}

    def readiness_diagnostics(self) -> BackendDiagnostics:  # pragma: no cover
        return BackendDiagnostics(
            backend_key=self.key,
            backend_label=self.label,
            available=True,
            ready=True,
            reason=None,
            details={},
        )

    def cache_diagnostics(self) -> dict[str, Any]:  # pragma: no cover
        return {"cached_model_count": 0, "cached_model_ids": [], "loaded_models": []}

    def metrics_summary(self) -> dict[str, Any]:  # pragma: no cover
        return {}

    def preload_models(self, specs):  # pragma: no cover
        return {
            "requested": 0,
            "attempted": 0,
            "loaded": 0,
            "failed": 0,
            "loaded_model_ids": [],
            "failed_model_ids": [],
            "errors": [],
        }


class _StubAdapter(ModelFamilyAdapter):
    key = "_stub_family"
    label = "Stub family"

    def capabilities(self) -> tuple[str, ...]:
        return ("preset_speaker_tts",)

    def supports_plan(self, plan: ExecutionPlan) -> bool:  # pragma: no cover
        return False

    def prepare_execution(self, plan: ExecutionPlan) -> FamilyPreparedExecution:  # pragma: no cover
        raise NotImplementedError


class _StubPlugin(ModelFamilyPlugin):
    key = "_stub_plugin"
    label = "Stub plugin"
    capabilities = ("preset_speaker_tts",)
    supported_backends = ("_stub_backend",)

    def is_available(self) -> bool:
        return True

    def import_error(self) -> Exception | None:
        return None

    def load_model(self, *, spec, backend_key, model_path):  # pragma: no cover
        return None

    def synthesize(
        self,
        model: Any,
        request: FamilyExecutionRequest,
        *,
        backend_key: str,
    ) -> FamilyExecutionResult:  # pragma: no cover
        return FamilyExecutionResult(waveforms=[], sample_rate=24000)


class _AbstractStubPlugin(ModelFamilyPlugin):
    """Intentionally abstract: missing concrete implementations of the abstract methods."""

    key = "_stub_abstract_plugin"
    label = "Abstract stub plugin"
    capabilities = ()
    supported_backends = ()


# END_BLOCK_TEST_STUBS


def test_subclass_discovery_finds_concrete_subclasses() -> None:
    backends = discover_backend_classes(include_entry_points=False)
    adapters = discover_family_adapter_classes(
        include_entry_points=False,
        include_test_classes=True,
    )
    plugins = discover_family_plugin_classes(include_entry_points=False)

    assert _StubBackend in backends
    assert _StubAdapter in adapters
    assert _StubPlugin in plugins


def test_subclass_discovery_filters_test_local_family_adapters_by_default() -> None:
    adapters = discover_family_adapter_classes(include_entry_points=False)

    assert _StubAdapter not in adapters


def test_subclass_discovery_can_include_test_local_family_adapters_when_requested() -> None:
    adapters = discover_family_adapter_classes(
        include_entry_points=False,
        include_test_classes=True,
    )

    assert _StubAdapter in adapters


def test_subclass_discovery_skips_abstract_subclasses() -> None:
    plugins = discover_family_plugin_classes(include_entry_points=False)

    assert _AbstractStubPlugin not in plugins


class _ExternalBackend(_StubBackend):
    """Concrete subclass declared at module scope so subclass discovery picks it up
    once and entry-point discovery can also resolve to it deterministically."""

    key = "_external_backend"
    label = "External backend"


def test_entry_point_loader_loads_external_classes() -> None:
    ep = EntryPoint(
        name="external",
        value="tests.unit.core.test_discovery:_ExternalBackend",
        group=BACKEND_ENTRY_POINT_GROUP,
    )

    def loader() -> list[EntryPoint]:
        return [ep]

    with patch.object(EntryPoint, "load", lambda _self: _ExternalBackend):
        discovered = discover_backend_classes(include_entry_points=True, entry_points_loader=loader)

    assert any(cls is _ExternalBackend for cls in discovered)


def test_entry_point_loader_skips_invalid_entries() -> None:
    ep_not_class = EntryPoint(
        name="bad_value",
        value="x:y",
        group=FAMILY_PLUGIN_ENTRY_POINT_GROUP,
    )
    ep_wrong_base = EntryPoint(
        name="bad_base",
        value="x:y",
        group=FAMILY_PLUGIN_ENTRY_POINT_GROUP,
    )
    ep_abstract = EntryPoint(
        name="bad_abstract",
        value="x:y",
        group=FAMILY_PLUGIN_ENTRY_POINT_GROUP,
    )

    load_results: list[Any] = [42, dict, _AbstractStubPlugin]

    def fake_load(_self) -> Any:
        return load_results.pop(0)

    def loader() -> list[EntryPoint]:
        return [ep_not_class, ep_wrong_base, ep_abstract]

    with patch.object(EntryPoint, "load", fake_load):
        plugins = discover_family_plugin_classes(
            include_entry_points=True,
            entry_points_loader=loader,
        )

    qnames = {f"{cls.__module__}.{cls.__qualname__}" for cls in plugins}
    assert "builtins.dict" not in qnames
    assert "tests.unit.core.test_discovery._AbstractStubPlugin" not in qnames


def test_entry_point_loader_invalid_callable_raises() -> None:
    with pytest.raises(TypeError):
        discover_family_adapter_classes(
            include_entry_points=True,
            entry_points_loader="not-callable",  # type: ignore[arg-type]
        )


def test_dedup_and_sort_is_stable() -> None:
    eps = [
        EntryPoint(name="dup1", value="x:y", group=BACKEND_ENTRY_POINT_GROUP),
        EntryPoint(name="dup2", value="x:y", group=BACKEND_ENTRY_POINT_GROUP),
    ]

    def loader() -> list[EntryPoint]:
        return eps

    with patch.object(EntryPoint, "load", lambda _self: _StubBackend):
        result = discover_backend_classes(include_entry_points=True, entry_points_loader=loader)

    occurrences = sum(1 for cls in result if cls is _StubBackend)
    assert occurrences == 1

    sorted_keys = [getattr(cls, "key", "") for cls in result]
    assert sorted_keys == sorted(sorted_keys)


def test_builtin_backend_import_seed_is_failure_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.discovery import BUILTIN_BACKEND_MODULES, import_builtin_backend_modules

    imported: list[str] = []

    def fake_import_module(module_name: str):
        imported.append(module_name)
        if module_name.endswith("onnx_backend"):
            raise RuntimeError("onnx unavailable")
        return object()

    monkeypatch.setattr("core.discovery.import_module", fake_import_module)

    import_builtin_backend_modules()

    assert imported == list(BUILTIN_BACKEND_MODULES)
