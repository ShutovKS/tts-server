# FILE: core/backends/registry.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Select and manage TTS inference backends based on configuration and availability.
#   SCOPE: BackendRegistry class with backend selection, model spec resolution
#   DEPENDS: M-CONFIG, M-ERRORS, M-OBSERVABILITY
#   LINKS: M-BACKENDS
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   BackendSelection - Selected backend with request and fallback metadata
#   BackendRegistry - Backend selection and model spec resolution
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.1 - Added explainable per-model backend route reporting for mixed-family readiness]
# END_CHANGE_SUMMARY

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Iterable, Sequence

from core.backends.base import TTSBackend
from core.errors import (
    BackendCapabilityError,
    BackendNotAvailableError,
    ModelNotAvailableError,
)
from core.models.catalog import ModelSpec, get_model_manifest
from core.planning.capability_resolver import CapabilityResolver
from core.planning.host_probe import HostProbe


@dataclass(frozen=True)
class BackendSelection:
    backend: TTSBackend
    requested_backend: str | None
    auto_selected: bool
    selection_reason: str


# START_CONTRACT: BackendRegistry
#   PURPOSE: Select the active inference backend and resolve supported model specifications against it.
#   INPUTS: { backends: Sequence[TTSBackend] - Registered backend implementations, requested_backend: str | None - Optional explicit backend override, autoselect: bool - Whether backend selection may fall back automatically, model_manifest_path: object - Optional manifest path override }
#   OUTPUTS: { instance - Backend registry with a resolved backend selection }
#   SIDE_EFFECTS: Loads the model manifest and selects an active backend during initialization
#   LINKS: M-BACKENDS
# END_CONTRACT: BackendRegistry
class BackendRegistry:
    def __init__(
        self,
        backends: Sequence[TTSBackend],
        *,
        requested_backend: str | None = None,
        autoselect: bool = True,
        model_manifest_path=None,
    ):
        if not backends:
            raise ValueError("At least one backend must be registered")
        self._backends = {backend.key: backend for backend in backends}
        self._requested_backend = requested_backend
        self._autoselect = autoselect
        self._model_manifest = (
            get_model_manifest(model_manifest_path)
            if model_manifest_path is not None
            else get_model_manifest()
        )
        self._host_probe = HostProbe()
        self._host_snapshot = self._host_probe.probe()
        self._capability_resolver = CapabilityResolver()
        self._selection = self._select_backend()

    @property
    def selected_backend(self) -> TTSBackend:
        return self._selection.backend

    @property
    def registered_backends(self) -> tuple[TTSBackend, ...]:
        return tuple(self._backends.values())

    @property
    def selection(self) -> BackendSelection:
        return self._selection

    @property
    def model_specs(self) -> tuple[ModelSpec, ...]:
        return tuple(self._model_manifest.enabled_models())

    # START_CONTRACT: list_backends
    #   PURPOSE: List registered backends together with selection state, capabilities, and readiness diagnostics.
    #   INPUTS: {}
    #   OUTPUTS: { list[dict[str, object]] - Structured backend descriptors }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: list_backends
    def list_backends(self) -> list[dict[str, object]]:
        selected_key = self.selected_backend.key
        ranked = {
            candidate.backend_key: candidate
            for candidate in self._capability_resolver.rank_backends(
                backends=tuple(self._backends.values()), host=self._host_snapshot
            )
        }
        return [
            {
                "key": backend.key,
                "label": backend.label,
                "selected": backend.key == selected_key,
                "platform_supported": backend.supports_platform(),
                "available": backend.is_available(),
                "capabilities": backend.capabilities().to_dict(),
                "diagnostics": backend.readiness_diagnostics().to_dict(),
                "host_reason": ranked[backend.key].reason,
                "selection_score": ranked[backend.key].score,
            }
            for backend in self._backends.values()
        ]

    # START_CONTRACT: get_model_spec
    #   PURPOSE: Resolve a manifest model specification that is compatible with the selected backend.
    #   INPUTS: { model_name: str | None - Requested model identifier, mode: str | None - Requested synthesis mode fallback }
    #   OUTPUTS: { ModelSpec - Selected compatible model specification }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: get_model_spec
    def get_model_spec(
        self, model_name: str | None = None, mode: str | None = None
    ) -> ModelSpec:
        # START_BLOCK_RESOLVE_MODEL_BY_NAME
        if model_name:
            for spec in self.model_specs:
                if model_name in {spec.api_name, spec.folder, spec.key}:
                    if spec.supports_backend(self.selected_backend.key):
                        self.ensure_model_supported(spec)
                    else:
                        compatible = [
                            backend
                            for backend in self._backends.values()
                            if spec.supports_backend(backend.key)
                            and backend.capabilities().supports_mode(spec.mode)
                        ]
                        if not compatible:
                            self.ensure_model_supported(spec)
                    return spec
            raise ModelNotAvailableError(
                model_name=model_name,
                details={"model": model_name, "backend": self.selected_backend.key},
            )
        # END_BLOCK_RESOLVE_MODEL_BY_NAME

        # START_BLOCK_RESOLVE_MODEL_BY_MODE
        if mode:
            self.ensure_mode_supported(mode)
            matching_specs = [
                spec for spec in self.model_specs if spec.mode == mode and spec.enabled
            ]
            matching_specs.sort(
                key=lambda spec: spec.rollout.default_preference, reverse=True
            )
            for spec in matching_specs:
                try:
                    backend = self.resolve_backend_for_spec(spec)
                except BackendCapabilityError:
                    continue
                if backend.resolve_model_path(spec.folder):
                    return spec
            if matching_specs:
                raise ModelNotAvailableError(
                    reason=f"No local model is available for mode: {mode}",
                    details={"mode": mode, "backend": self.selected_backend.key},
                )
            raise ModelNotAvailableError(
                reason=f"Requested mode is not available: {mode}",
                details={"mode": mode, "backend": self.selected_backend.key},
            )
        # END_BLOCK_RESOLVE_MODEL_BY_MODE

        # START_BLOCK_REJECT_EMPTY_MODEL_REQUEST
        raise ModelNotAvailableError(
            reason="No model or mode was specified",
            details={"backend": self.selected_backend.key},
        )
        # END_BLOCK_REJECT_EMPTY_MODEL_REQUEST

    # START_CONTRACT: ensure_model_supported
    #   PURPOSE: Validate that the selected backend supports the provided model specification.
    #   INPUTS: { spec: ModelSpec - Model specification to validate }
    #   OUTPUTS: { None - Completes when the model is supported }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: ensure_model_supported
    def ensure_model_supported(self, spec: ModelSpec) -> None:
        self.ensure_mode_supported(spec.mode)
        if spec.supports_backend(self.selected_backend.key):
            return
        raise BackendCapabilityError(
            f"Backend '{self.selected_backend.key}' is not enabled for model '{spec.api_name}'",
            details={
                "backend": self.selected_backend.key,
                "model": spec.api_name,
                "mode": spec.mode,
            },
        )

    # START_CONTRACT: ensure_mode_supported
    #   PURPOSE: Validate that the selected backend supports the requested synthesis mode.
    #   INPUTS: { mode: str - Requested synthesis mode }
    #   OUTPUTS: { None - Completes when the mode is supported }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: ensure_mode_supported
    def ensure_mode_supported(self, mode: str) -> None:
        if self.selected_backend.capabilities().supports_mode(mode):
            return
        raise BackendCapabilityError(
            f"Backend '{self.selected_backend.key}' does not support mode '{mode}'",
            details={"backend": self.selected_backend.key, "mode": mode},
        )

    def resolve_backend_for_spec(self, spec: ModelSpec) -> TTSBackend:
        route = self.explain_backend_route_for_spec(spec)
        execution_backend = route["execution_backend"]
        if execution_backend is None:
            raise BackendCapabilityError(
                f"No registered backend can run model '{spec.api_name}'",
                details={"model": spec.api_name, "mode": spec.mode},
            )
        return self._backends[str(execution_backend)]

    def explain_backend_route_for_spec(self, spec: ModelSpec) -> dict[str, object]:
        selected_backend = self.selected_backend
        ranked_all = {
            candidate.backend_key: candidate
            for candidate in self._capability_resolver.rank_backends(
                backends=tuple(self._backends.values()), host=self._host_snapshot
            )
        }
        compatible: list[TTSBackend] = []
        candidates: list[dict[str, object]] = []
        for backend in self._backends.values():
            model_backend_supported = spec.supports_backend(backend.key)
            mode_supported = backend.capabilities().supports_mode(spec.mode)
            platform_supported = backend.supports_platform()
            available = backend.is_available()
            diagnostics = backend.readiness_diagnostics().to_dict()
            ready = bool(diagnostics.get("ready", platform_supported and available))
            rank = ranked_all[backend.key]
            if not model_backend_supported:
                route_reason = "model_backend_affinity_mismatch"
            elif not mode_supported:
                route_reason = "model_mode_not_supported_by_backend"
            elif not platform_supported:
                route_reason = str(diagnostics.get("reason") or "platform_unsupported")
            elif not available:
                route_reason = str(
                    diagnostics.get("reason") or "runtime_dependency_missing"
                )
            elif not ready:
                route_reason = str(diagnostics.get("reason") or "runtime_not_ready")
            else:
                route_reason = "route_candidate_accepted"
            if model_backend_supported and mode_supported:
                compatible.append(backend)
            candidates.append(
                {
                    "key": backend.key,
                    "label": backend.label,
                    "selected": backend.key == selected_backend.key,
                    "compatible_with_model": model_backend_supported,
                    "supports_mode": mode_supported,
                    "platform_supported": platform_supported,
                    "available": available,
                    "ready": ready,
                    "host_reason": rank.reason,
                    "selection_score": rank.score,
                    "route_reason": route_reason,
                    "diagnostics": diagnostics,
                }
            )

        selected_backend_compatible = selected_backend in compatible
        selected_backend_diagnostics = (
            selected_backend.readiness_diagnostics().to_dict()
        )
        selected_backend_ready = selected_backend_compatible and bool(
            selected_backend_diagnostics.get(
                "ready",
                selected_backend.supports_platform()
                and selected_backend.is_available(),
            )
        )
        execution_backend: TTSBackend | None = None
        routing_mode = "unresolved"
        route_reason = "no_registered_backend_supports_model"
        if compatible:
            if selected_backend_ready:
                execution_backend = selected_backend
                routing_mode = "selected_backend"
                route_reason = "selected_backend_supports_model"
            else:
                ranked_compatible = self._capability_resolver.rank_backends(
                    backends=tuple(compatible), host=self._host_snapshot
                )
                execution_backend = self._backends[ranked_compatible[0].backend_key]
                routing_mode = "per_model_backend_fallback"
                if not ranked_compatible[0].accepted:
                    route_reason = (
                        "no_ready_backend_for_model_using_best_effort_fallback"
                    )
                elif selected_backend_compatible:
                    route_reason = "selected_backend_unavailable_for_model_route"
                else:
                    route_reason = "selected_backend_incompatible_with_model"

        return {
            "selected_backend": selected_backend.key,
            "selected_backend_label": selected_backend.label,
            "selected_backend_compatible_with_model": selected_backend_compatible,
            "selected_backend_ready_for_model": selected_backend_ready,
            "execution_backend": None
            if execution_backend is None
            else execution_backend.key,
            "execution_backend_label": None
            if execution_backend is None
            else execution_backend.label,
            "routing_mode": routing_mode,
            "route_reason": route_reason,
            "candidates": candidates,
        }

    def _select_backend(self) -> BackendSelection:
        # START_BLOCK_HANDLE_EXPLICIT_BACKEND_SELECTION
        if self._requested_backend:
            backend = self._backends.get(self._requested_backend)
            if backend is None:
                raise BackendNotAvailableError(
                    f"Configured backend is unknown: {self._requested_backend}",
                    details={
                        "backend": self._requested_backend,
                        "known_backends": sorted(self._backends),
                    },
                )
            return BackendSelection(
                backend=backend,
                requested_backend=self._requested_backend,
                auto_selected=False,
                selection_reason="explicit_config",
            )
        # END_BLOCK_HANDLE_EXPLICIT_BACKEND_SELECTION

        # START_BLOCK_HANDLE_FIXED_BACKEND_ORDER
        if not self._autoselect:
            first = next(iter(self._backends.values()))
            return BackendSelection(
                backend=first,
                requested_backend=None,
                auto_selected=False,
                selection_reason="first_registered",
            )
        # END_BLOCK_HANDLE_FIXED_BACKEND_ORDER

        # START_BLOCK_AUTOSELECT_READY_BACKEND
        candidates = self._capability_resolver.rank_backends(
            backends=tuple(self._backends.values()), host=self._host_snapshot
        )
        ordered = [self._backends[candidate.backend_key] for candidate in candidates]
        for candidate in candidates:
            if candidate.accepted:
                return BackendSelection(
                    backend=self._backends[candidate.backend_key],
                    requested_backend=None,
                    auto_selected=True,
                    selection_reason=candidate.reason,
                )
        # END_BLOCK_AUTOSELECT_READY_BACKEND
        # START_BLOCK_AUTOSELECT_PLATFORM_BACKEND
        for candidate in candidates:
            backend = self._backends[candidate.backend_key]
            if backend.supports_platform():
                return BackendSelection(
                    backend=backend,
                    requested_backend=None,
                    auto_selected=True,
                    selection_reason=candidate.reason,
                )
        # END_BLOCK_AUTOSELECT_PLATFORM_BACKEND
        # START_BLOCK_FALLBACK_BACKEND_SELECTION
        first = ordered[0]
        return BackendSelection(
            backend=first,
            requested_backend=None,
            auto_selected=True,
            selection_reason="fallback_first_backend",
        )
        # END_BLOCK_FALLBACK_BACKEND_SELECTION

    @staticmethod
    def _prefer_platform_backend(
        backends: Iterable[TTSBackend],
    ) -> Iterable[TTSBackend]:
        current = platform.system().lower()
        if current == "darwin":
            preferred = ["mlx", "torch"]
        else:
            preferred = ["torch", "mlx"]
        ranked = sorted(
            backends,
            key=lambda backend: preferred.index(backend.key)
            if backend.key in preferred
            else len(preferred),
        )
        return ranked

    @property
    def host_snapshot(self):
        return self._host_snapshot


__all__ = [
    "BackendSelection",
    "BackendRegistry",
]
