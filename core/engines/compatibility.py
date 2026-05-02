# FILE: core/engines/compatibility.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Expose a temporary compatibility bridge that registers the current legacy family-adapter and backend lanes as engine-like metadata without changing synthesis execution behavior.
#   SCOPE: Legacy engine inventory records, temporary bridge-only TTSEngine shim, registry registration helpers, and coexistence-oriented metadata assembly.
#   DEPENDS: M-ENGINE-CONTRACTS, M-ENGINE-REGISTRY, M-MODEL-FAMILY, M-MODELS, M-BACKENDS
#   LINKS: M-ENGINE-BRIDGE, M-ENGINE-REGISTRY
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LegacyEngineRecord - Immutable bridge ledger entry describing one temporary legacy family/backend lane.
#   EngineCompatibilityBridge - Temporary compatibility bridge that inventories and registers legacy lanes into EngineRegistry.
#   build_legacy_engine_registry - Convenience helper returning an EngineRegistry populated with temporary legacy bridge entries.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Task 8 compatibility bridge: introduced a temporary legacy-to-engine metadata bridge that inventories existing family/backends and registers non-executable TTSEngine shims for coexistence during engine migration]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.backends.torch_backend import built_in_torch_family_strategies
from core.engines.contracts import (
    AudioBuffer,
    EngineAvailability,
    EngineCapabilities,
    ModelHandle,
    SynthesisJob,
    TTSEngine,
)
from core.engines.registry import EngineRegistry
from core.model_families.base import ModelFamilyAdapter
from core.models.catalog import MODEL_SPECS, ModelSpec


# START_CONTRACT: LegacyEngineRecord
#   PURPOSE: Describe one temporary compatibility-bridge engine record backed by the current legacy family-adapter and backend seams.
#   INPUTS: { engine_key: str - Temporary legacy bridge engine key, family_key: str - Legacy family adapter key, backend_key: str - Legacy backend lane key, engine_label: str - Human-readable bridge label, aliases: tuple[str, ...] - Deterministic aliases for registry lookup, capabilities: tuple[str, ...] - Supported synthesis capabilities, model_ids: tuple[str, ...] - Current manifest model identifiers routed through this legacy lane, deletion_criteria: str - Explicit criteria for deleting the temporary bridge after migration }
#   OUTPUTS: { instance - Immutable compatibility ledger entry }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-BRIDGE
# END_CONTRACT: LegacyEngineRecord
@dataclass(frozen=True)
class LegacyEngineRecord:
    engine_key: str
    family_key: str
    backend_key: str
    engine_label: str
    aliases: tuple[str, ...]
    capabilities: tuple[str, ...]
    model_ids: tuple[str, ...]
    deletion_criteria: str


class _LegacyCompatibilityEngine(TTSEngine):
    """TEMPORARY compatibility bridge shim.

    Deletion ledger: remove this shim after the engine migration no longer needs
    legacy family-adapter/backend metadata to coexist with EngineRegistry.
    """

    def __init__(self, record: LegacyEngineRecord) -> None:
        self.key = record.engine_key
        self.label = record.engine_label
        self.aliases = record.aliases
        self._record = record

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            families=(self._record.family_key,),
            backends=(self._record.backend_key,),
            capabilities=self._record.capabilities,
        )

    def availability(self) -> EngineAvailability:
        return EngineAvailability(
            engine_key=self.key,
            is_available=True,
            is_enabled=True,
            reason=(
                "TEMPORARY compatibility bridge only; delete after runtime engine "
                "migration replaces legacy family-adapter/backend execution seams"
            ),
        )

    def load_model(
        self,
        *,
        spec: ModelSpec,
        backend_key: str,
        model_path: Path | None,
    ) -> ModelHandle:
        raise NotImplementedError(
            "TEMPORARY compatibility bridge engines are metadata-only and must be deleted "
            "once runtime engine execution replaces the legacy path"
        )

    def synthesize(self, handle: ModelHandle, job: SynthesisJob) -> AudioBuffer:
        raise NotImplementedError(
            "TEMPORARY compatibility bridge engines are metadata-only and must be deleted "
            "once runtime engine execution replaces the legacy path"
        )


# START_CONTRACT: EngineCompatibilityBridge
#   PURPOSE: Publish the current legacy family-adapter/backend lanes as temporary engine-like metadata while keeping the planner and synthesis service on the existing runtime path.
#   INPUTS: { family_adapters: tuple[ModelFamilyAdapter, ...] - Active legacy family adapters to expose, model_specs: tuple[ModelSpec, ...] - Current manifest model specs used to infer backend lanes }
#   OUTPUTS: { instance - Temporary compatibility bridge for EngineRegistry coexistence }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-BRIDGE, M-ENGINE-REGISTRY
# END_CONTRACT: EngineCompatibilityBridge
class EngineCompatibilityBridge:
    TEMPORARY_DELETION_CRITERIA = (
        "Delete this TEMPORARY compatibility bridge after the engine migration moves "
        "planner/service synthesis execution off the legacy family-adapter and backend seams."
    )

    def __init__(
        self,
        *,
        family_adapters: tuple[ModelFamilyAdapter, ...],
        model_specs: tuple[ModelSpec, ...] = tuple(MODEL_SPECS.values()),
    ) -> None:
        self._family_adapters = tuple(family_adapters)
        self._model_specs = tuple(model_specs)

    # START_CONTRACT: legacy_records
    #   PURPOSE: Inventory the current legacy family/backend lanes as deterministic bridge ledger entries.
    #   INPUTS: {}
    #   OUTPUTS: { tuple[LegacyEngineRecord, ...] - Temporary registry-ready metadata records }
    #   SIDE_EFFECTS: none
    #   LINKS: M-ENGINE-BRIDGE
    # END_CONTRACT: legacy_records
    def legacy_records(self) -> tuple[LegacyEngineRecord, ...]:
        records: list[LegacyEngineRecord] = []
        torch_strategy_families = {
            strategy.family_key for strategy in built_in_torch_family_strategies()
        }

        # START_BLOCK_BUILD_LEGACY_LEDGER
        for adapter in self._family_adapters:
            family_key = adapter.key
            family_specs = tuple(
                spec for spec in self._model_specs if spec.family_key == family_key
            )
            if not family_specs:
                continue

            backend_keys = {
                backend_key
                for spec in family_specs
                for backend_key in spec.backend_support
            }
            if family_key in torch_strategy_families:
                backend_keys.add("torch")

            for backend_key in sorted(backend_keys):
                models_for_backend = tuple(
                    sorted(
                        spec.model_id
                        for spec in family_specs
                        if backend_key in spec.backend_support
                    )
                )
                if not models_for_backend and backend_key != "torch":
                    continue
                records.append(
                    LegacyEngineRecord(
                        engine_key=f"legacy-{family_key}-{backend_key}",
                        family_key=family_key,
                        backend_key=backend_key,
                        engine_label=f"Legacy {adapter.label} via {backend_key}",
                        aliases=(
                            f"legacy-{family_key}-{backend_key}",
                            f"legacy-{family_key}-{backend_key}-bridge",
                        ),
                        capabilities=tuple(adapter.capabilities()),
                        model_ids=models_for_backend,
                        deletion_criteria=self.TEMPORARY_DELETION_CRITERIA,
                    )
                )
        # END_BLOCK_BUILD_LEGACY_LEDGER

        return tuple(
            sorted(records, key=lambda record: (record.family_key, record.backend_key, record.engine_key))
        )

    # START_CONTRACT: register_into
    #   PURPOSE: Register all temporary legacy bridge records into an existing EngineRegistry.
    #   INPUTS: { registry: EngineRegistry - Registry that should expose the temporary legacy lanes }
    #   OUTPUTS: { tuple[TTSEngine, ...] - Registered bridge-only engine instances }
    #   SIDE_EFFECTS: Mutates the provided EngineRegistry
    #   LINKS: M-ENGINE-BRIDGE, M-ENGINE-REGISTRY
    # END_CONTRACT: register_into
    def register_into(self, registry: EngineRegistry) -> tuple[TTSEngine, ...]:
        registered: list[TTSEngine] = []
        for record in self.legacy_records():
            engine = _LegacyCompatibilityEngine(record)
            registry.register(engine, source="legacy_compatibility_bridge")
            registered.append(engine)
        return tuple(registered)


# START_CONTRACT: build_legacy_engine_registry
#   PURPOSE: Build an EngineRegistry populated only with the temporary legacy compatibility bridge entries.
#   INPUTS: { family_adapters: tuple[ModelFamilyAdapter, ...] - Active legacy family adapters to expose, model_specs: tuple[ModelSpec, ...] - Current manifest model specs used to infer backend lanes }
#   OUTPUTS: { EngineRegistry - Registry containing temporary legacy bridge metadata entries }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-BRIDGE, M-ENGINE-REGISTRY
# END_CONTRACT: build_legacy_engine_registry
def build_legacy_engine_registry(
    *,
    family_adapters: tuple[ModelFamilyAdapter, ...],
    model_specs: tuple[ModelSpec, ...] = tuple(MODEL_SPECS.values()),
) -> EngineRegistry:
    registry = EngineRegistry()
    EngineCompatibilityBridge(
        family_adapters=family_adapters,
        model_specs=model_specs,
    ).register_into(registry)
    return registry


__all__ = [
    "EngineCompatibilityBridge",
    "LegacyEngineRecord",
    "build_legacy_engine_registry",
]
