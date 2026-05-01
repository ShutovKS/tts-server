# FILE: core/backends/base.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Define the direct execution contract for TTS inference backends and the from_settings factory hook used by the bootstrap auto-discovery wiring.
#   SCOPE: TTSBackend abstract class with from_settings factory, LoadedModelHandle dataclass, ExecutionRequest dataclass
#   DEPENDS: M-ERRORS
#   LINKS: M-BACKENDS
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   TTSBackend - Abstract backend interface with first-class execute contract and a from_settings factory used by bootstrap auto-discovery
#   LoadedModelHandle - Handle to a loaded model with metadata
#   ExecutionRequest - Runtime-oriented execution request carrying family-prepared inputs
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Phase 2.8: added TTSBackend.from_settings classmethod default factory used by bootstrap auto-discovery]
# END_CHANGE_SUMMARY

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.backends.capabilities import BackendCapabilitySet, BackendDiagnostics
from core.models.catalog import ModelSpec

if TYPE_CHECKING:
    from core.config import CoreSettings
    from core.metrics import OperationalMetricsRegistry


# START_CONTRACT: LoadedModelHandle
#   PURPOSE: Carry a loaded runtime model together with its resolved spec and backend metadata.
#   INPUTS: { spec: ModelSpec - Resolved model specification, runtime_model: Any - Loaded backend-specific runtime object, resolved_path: Path | None - Filesystem path used for loading, backend_key: str - Backend key that produced the handle }
#   OUTPUTS: { instance - Immutable loaded model handle }
#   SIDE_EFFECTS: none
#   LINKS: M-BACKENDS
# END_CONTRACT: LoadedModelHandle
@dataclass(frozen=True)
class LoadedModelHandle:
    spec: ModelSpec
    runtime_model: Any
    resolved_path: Path | None
    backend_key: str


@dataclass(frozen=True)
class ExecutionRequest:
    handle: LoadedModelHandle
    text: str
    output_dir: Path
    language: str
    execution_mode: str
    generation_kwargs: dict[str, Any]


# START_CONTRACT: TTSBackend
#   PURPOSE: Define the abstract interface that all concrete TTS inference backends must implement.
#   INPUTS: {}
#   OUTPUTS: { instance - Abstract backend contract for model loading, readiness, and synthesis }
#   SIDE_EFFECTS: none
#   LINKS: M-BACKENDS
# END_CONTRACT: TTSBackend
class TTSBackend(ABC):
    key: str
    label: str

    @abstractmethod
    # START_CONTRACT: capabilities
    #   PURPOSE: Describe the feature capabilities supported by the backend implementation.
    #   INPUTS: {}
    #   OUTPUTS: { BackendCapabilitySet - Capability flags for the backend }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: capabilities
    def capabilities(self) -> BackendCapabilitySet:
        raise NotImplementedError

    @abstractmethod
    # START_CONTRACT: is_available
    #   PURPOSE: Report whether the backend runtime dependencies are available in the current environment.
    #   INPUTS: {}
    #   OUTPUTS: { bool - True when runtime dependencies are available }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: is_available
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    # START_CONTRACT: supports_platform
    #   PURPOSE: Report whether the backend supports the current operating platform.
    #   INPUTS: {}
    #   OUTPUTS: { bool - True when the backend is compatible with the active platform }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: supports_platform
    def supports_platform(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    # START_CONTRACT: resolve_model_path
    #   PURPOSE: Resolve the on-disk path for a configured model folder.
    #   INPUTS: { folder_name: str - Model directory name from the manifest }
    #   OUTPUTS: { Path | None - Resolved model path when present }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: resolve_model_path
    def resolve_model_path(self, folder_name: str) -> Path | None:
        raise NotImplementedError

    @abstractmethod
    # START_CONTRACT: load_model
    #   PURPOSE: Load a model runtime for the provided specification and return a reusable handle.
    #   INPUTS: { spec: ModelSpec - Model specification to load }
    #   OUTPUTS: { LoadedModelHandle - Loaded backend model handle }
    #   SIDE_EFFECTS: May allocate backend runtime resources and cache loaded models
    #   LINKS: M-BACKENDS
    # END_CONTRACT: load_model
    def load_model(self, spec: ModelSpec) -> LoadedModelHandle:
        raise NotImplementedError

    @abstractmethod
    # START_CONTRACT: inspect_model
    #   PURPOSE: Inspect model availability, artifact completeness, and runtime readiness for the backend.
    #   INPUTS: { spec: ModelSpec - Model specification to inspect }
    #   OUTPUTS: { dict[str, Any] - Structured inspection result for the model }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: inspect_model
    def inspect_model(self, spec: ModelSpec) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    # START_CONTRACT: readiness_diagnostics
    #   PURPOSE: Report backend availability and readiness diagnostics for selection and health checks.
    #   INPUTS: {}
    #   OUTPUTS: { BackendDiagnostics - Structured backend readiness diagnostics }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: readiness_diagnostics
    def readiness_diagnostics(self) -> BackendDiagnostics:
        raise NotImplementedError

    # START_CONTRACT: cache_diagnostics
    #   PURPOSE: Report cache state details for loaded model runtimes.
    #   INPUTS: {}
    #   OUTPUTS: { dict[str, Any] - Structured cache diagnostics }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: cache_diagnostics
    def cache_diagnostics(self) -> dict[str, Any]:
        return {
            "cached_model_count": 0,
            "cached_model_ids": [],
            "cache_policy": {
                "eviction": "not_configured",
            },
            "loaded_models": [],
        }

    # START_CONTRACT: metrics_summary
    #   PURPOSE: Summarize backend-specific model loading and cache metrics.
    #   INPUTS: {}
    #   OUTPUTS: { dict[str, Any] - Backend metrics summary }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: metrics_summary
    def metrics_summary(self) -> dict[str, Any]:
        return {
            "cache": {
                "hit": {},
                "miss": {},
            },
            "load": {
                "failures": {},
                "duration_ms": {},
            },
        }

    # START_CONTRACT: preload_models
    #   PURPOSE: Preload a collection of model specifications into backend runtime memory.
    #   INPUTS: { specs: tuple[ModelSpec, ...] - Model specifications to preload }
    #   OUTPUTS: { dict[str, Any] - Structured preload outcome summary }
    #   SIDE_EFFECTS: May allocate and cache backend runtime resources
    #   LINKS: M-BACKENDS
    # END_CONTRACT: preload_models
    def preload_models(self, specs: tuple[ModelSpec, ...]) -> dict[str, Any]:
        return {
            "requested": 0,
            "attempted": 0,
            "loaded": 0,
            "failed": 0,
            "loaded_model_ids": [],
            "failed_model_ids": [],
            "errors": [],
        }

    @abstractmethod
    # START_CONTRACT: execute
    #   PURPOSE: Execute a family-prepared generation request directly through the backend runtime contract.
    #   INPUTS: { request: ExecutionRequest - Runtime execution request carrying handle, text, output directory, execution mode, and prepared generation kwargs }
    #   OUTPUTS: { None - Writes generated audio into the request output directory }
    #   SIDE_EFFECTS: Performs backend inference and writes audio artifacts to disk
    #   LINKS: M-BACKENDS
    # END_CONTRACT: execute
    def execute(self, request: ExecutionRequest) -> None:
        raise NotImplementedError

    # START_CONTRACT: from_settings
    #   PURPOSE: Build a backend instance from CoreSettings using a uniform factory signature so bootstrap auto-discovery can instantiate any TTSBackend subclass without per-backend wiring.
    #   INPUTS: { settings: CoreSettings - Runtime settings, metrics: OperationalMetricsRegistry | None - Optional shared metrics facade }
    #   OUTPUTS: { TTSBackend - Constructed backend instance ready to be registered }
    #   SIDE_EFFECTS: Default implementation calls cls(settings.models_dir, metrics=metrics); subclasses with non-default constructor signatures override to map their own settings.
    #   LINKS: M-BACKENDS, M-BOOTSTRAP
    # END_CONTRACT: from_settings
    @classmethod
    def from_settings(
        cls,
        settings: CoreSettings,
        *,
        metrics: OperationalMetricsRegistry | None = None,
    ) -> TTSBackend:
        return cls(settings.models_dir, metrics=metrics)


__all__ = [
    "ExecutionRequest",
    "LoadedModelHandle",
    "TTSBackend",
]
