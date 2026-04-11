# FILE: core/registry/runtime_models.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Own runtime-loaded model handles, preload state, and cache-facing readiness information.
#   SCOPE: RuntimeModelRegistry wrapper around current model loading and preload policy behavior
#   DEPENDS: M-BACKENDS, M-ARTIFACT-REGISTRY, M-MODELS
#   LINKS: M-RUNTIME-MODEL-REGISTRY
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   RuntimeModelRegistry - Runtime-facing wrapper for model loading, preload state, and cache/readiness surfaces
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added runtime model registry wrapper as a split runtime registry surface]
# END_CHANGE_SUMMARY

from __future__ import annotations

from core.backends.base import LoadedModelHandle
from core.models.manifest import ModelDescriptor, ModelSpec
from core.registry.artifacts import ArtifactRegistry


class RuntimeModelRegistry:
    def __init__(self, artifact_registry: ArtifactRegistry, legacy_registry):
        self.artifact_registry = artifact_registry
        self.legacy_registry = legacy_registry

    def get_model(
        self, model_name: str | None = None, mode: str | None = None
    ) -> tuple[ModelSpec, LoadedModelHandle]:
        return self.legacy_registry.get_model(model_name=model_name, mode=mode)

    def preload_report(self) -> dict[str, object]:
        return dict(self.legacy_registry._preload_report)

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


__all__ = ["RuntimeModelRegistry"]
