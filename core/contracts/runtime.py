# FILE: core/contracts/runtime.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define typed runtime seam protocols for model resolution, backend routing, and execution-facing registry access.
#   SCOPE: Protocol contracts and route payload types used by planner and synthesis service layers
#   DEPENDS: M-BACKENDS, M-CONTRACTS, M-MODELS
#   LINKS: M-CONTRACTS, M-SYNTHESIS-PLANNER, M-TTS-COORDINATOR
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   BackendRouteInfo - Typed backend route payload exposed by runtime registries
#   RuntimePlanningRegistry - Planning-facing registry protocol for model and backend resolution
#   RuntimeExecutionRegistry - Execution-facing registry protocol extending planning access with loaded-handle resolution
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added typed runtime seam protocols for planner and coordinator layers]
# END_CHANGE_SUMMARY

from __future__ import annotations

from typing import Protocol, TypedDict

from core.backends.base import LoadedModelHandle, TTSBackend
from core.models.catalog import ModelSpec


class BackendRouteInfo(TypedDict, total=False):
    route_reason: str
    execution_backend: str


class RuntimePlanningRegistry(Protocol):
    @property
    def backend(self) -> TTSBackend: ...

    def get_model_spec(
        self, model_name: str | None = None, mode: str | None = None
    ) -> ModelSpec: ...

    def backend_for_spec(self, spec: ModelSpec) -> TTSBackend: ...

    def backend_route_for_spec(self, spec: ModelSpec) -> BackendRouteInfo: ...


class RuntimeExecutionRegistry(RuntimePlanningRegistry, Protocol):
    def get_model(
        self, model_name: str | None = None, mode: str | None = None
    ) -> tuple[ModelSpec, LoadedModelHandle]: ...


__all__ = [
    "BackendRouteInfo",
    "RuntimeExecutionRegistry",
    "RuntimePlanningRegistry",
]
