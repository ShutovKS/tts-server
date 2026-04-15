# FILE: core/registry/runtime_models.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Own runtime preload state and cache-facing readiness information independent from metadata and artifact lookup.
#   SCOPE: RuntimeModelRegistry preload-state wrapper and descriptor-facing readiness helpers
#   DEPENDS: M-ARTIFACT-REGISTRY, M-MODELS
#   LINKS: M-RUNTIME-MODEL-REGISTRY
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   RuntimeModelRegistry - Runtime-facing preload and descriptor-readiness helper
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.3.0 - Runtime state now depends only on artifact state and the public preload-report contract]
# END_CHANGE_SUMMARY

from __future__ import annotations

from typing import Any, Callable

from core.models.manifest import ModelDescriptor
from core.registry.artifacts import ArtifactRegistry


class RuntimeModelRegistry:
    def __init__(
        self,
        artifact_registry: ArtifactRegistry,
        preload_report_provider: Callable[[], dict[str, Any]],
    ):
        self.artifact_registry = artifact_registry
        self._preload_report_provider = preload_report_provider

    def preload_report(self) -> dict[str, object]:
        return dict(self._preload_report_provider())

    def descriptor_runtime_state(
        self, descriptor: ModelDescriptor
    ) -> dict[str, object]:
        artifact_state = self.artifact_registry.descriptor_state(descriptor)
        preload = self.preload_report()
        loaded_ids = set(preload.get("loaded_model_ids", []))
        failed_ids = set(preload.get("failed_model_ids", []))
        if descriptor.model_id in loaded_ids:
            preload_status = "loaded"
        elif descriptor.model_id in failed_ids:
            preload_status = "failed"
        elif descriptor.model_id in preload.get("requested_model_ids", []):
            preload_status = "requested"
        else:
            preload_status = "not_requested"
        return {
            **artifact_state,
            "loaded": descriptor.model_id in loaded_ids,
            "preload_status": preload_status,
        }

    def descriptor_readiness(
        self,
        descriptor: ModelDescriptor,
        *,
        route: dict[str, Any],
        artifact_state: dict[str, object] | None = None,
    ) -> dict[str, object]:
        runtime_state = (
            self.descriptor_runtime_state(descriptor)
            if artifact_state is None
            else dict(artifact_state)
        )
        preload_status = str(runtime_state["preload_status"])
        preload = self.preload_report()
        return {
            **runtime_state,
            "id": descriptor.model_id,
            "family": descriptor.family,
            "family_key": descriptor.family_key,
            "capabilities_supported": list(descriptor.supported_capabilities),
            "backend_support": list(descriptor.backend_support),
            "selected_backend": route["selected_backend"],
            "selected_backend_label": route["selected_backend_label"],
            "execution_backend": route["execution_backend"],
            "execution_backend_label": route["execution_backend_label"],
            "route": route,
            "availability_reason": route["route_reason"],
            "preload": {
                "policy": preload["policy"],
                "requested": preload_status in {"requested", "loaded", "failed"},
                "status": preload_status,
            },
        }


__all__ = ["RuntimeModelRegistry"]
