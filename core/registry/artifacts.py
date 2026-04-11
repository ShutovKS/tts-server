# FILE: core/registry/artifacts.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Resolve installed/local artifact state independently from declared catalog metadata.
#   SCOPE: ArtifactRegistry class with artifact-state inspection helpers
#   DEPENDS: M-BACKENDS, M-MODEL-CATALOG
#   LINKS: M-ARTIFACT-REGISTRY
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   ArtifactRegistry - Artifact-state registry layered over backend inspection and catalog metadata
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added artifact registry to separate installed state from declared catalog metadata]
# END_CHANGE_SUMMARY

from __future__ import annotations

from typing import Any

from core.backends import BackendRegistry
from core.models.manifest import ModelDescriptor, ModelSpec
from core.registry.model_catalog import ModelCatalogRegistry


class ArtifactRegistry:
    def __init__(
        self,
        catalog: ModelCatalogRegistry,
        backend_registry: BackendRegistry,
    ):
        self.catalog = catalog
        self.backend_registry = backend_registry

    def inspect(self, spec: ModelSpec) -> dict[str, Any]:
        return self.backend_registry.selected_backend.inspect_model(spec)

    def descriptor_state(self, descriptor: ModelDescriptor) -> dict[str, Any]:
        spec = next(
            spec for spec in self.catalog.model_specs if spec.key == descriptor.key
        )
        status = self.inspect(spec)
        return {
            "model_id": descriptor.model_id,
            "family_key": descriptor.family_key,
            "declared": True,
            "installed": status["available"],
            "loadable": status["loadable"],
            "runtime_ready": status["runtime_ready"],
            "resolved_path": status["resolved_path"],
            "missing_artifacts": list(status["missing_artifacts"]),
            "required_artifacts": list(status["required_artifacts"]),
        }


__all__ = ["ArtifactRegistry"]
