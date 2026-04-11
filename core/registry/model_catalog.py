# FILE: core/registry/model_catalog.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Expose family-aware model catalog metadata independent of artifact and runtime loading state.
#   SCOPE: ModelCatalogRegistry class with descriptor listing and model lookup helpers
#   DEPENDS: M-MODELS
#   LINKS: M-MODEL-CATALOG
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   ModelCatalogRegistry - Family-aware catalog metadata registry derived from manifest-backed specs
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added family-aware catalog registry as the first split registry surface]
# END_CHANGE_SUMMARY

from __future__ import annotations

from core.models.manifest import ModelDescriptor, ModelSpec


class ModelCatalogRegistry:
    def __init__(self, model_specs: tuple[ModelSpec, ...]):
        self._model_specs = tuple(model_specs)
        self._descriptors = tuple(
            ModelDescriptor.from_model_spec(spec) for spec in self._model_specs
        )

    @property
    def model_specs(self) -> tuple[ModelSpec, ...]:
        return self._model_specs

    @property
    def descriptors(self) -> tuple[ModelDescriptor, ...]:
        return self._descriptors

    def get_descriptor(self, model_id: str) -> ModelDescriptor:
        for descriptor in self._descriptors:
            if model_id in {descriptor.model_id, descriptor.folder, descriptor.key}:
                return descriptor
        raise KeyError(model_id)


__all__ = ["ModelCatalogRegistry"]
