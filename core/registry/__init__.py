# FILE: core/registry/__init__.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Re-export split registry surfaces for catalog, artifacts, and runtime-loaded models.
#   SCOPE: barrel re-exports for registry layer
#   DEPENDS: M-MODEL-CATALOG, M-ARTIFACT-REGISTRY, M-RUNTIME-MODEL-REGISTRY
#   LINKS: M-MODEL-CATALOG, M-ARTIFACT-REGISTRY, M-RUNTIME-MODEL-REGISTRY
#   ROLE: BARREL
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   Registry surface - Re-export catalog, artifact, and runtime model registries
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added split registry barrel exports]
# END_CHANGE_SUMMARY

from core.registry.artifacts import ArtifactRegistry
from core.registry.model_catalog import ModelCatalogRegistry
from core.registry.runtime_models import RuntimeModelRegistry

__all__ = ["ArtifactRegistry", "ModelCatalogRegistry", "RuntimeModelRegistry"]
