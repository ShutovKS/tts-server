from __future__ import annotations

from typing import Any, Optional

from core.backends import BackendRegistry
from core.backends.base import LoadedModelHandle
from core.errors import ModelNotAvailableError
from core.models.catalog import MODEL_SPECS, ModelSpec
from core.observability import get_logger, log_event, operation_scope


LOGGER = get_logger(__name__)


class ModelRegistry:
    def __init__(self, backend_registry: BackendRegistry):
        self.backend_registry = backend_registry

    @property
    def backend(self):
        return self.backend_registry.selected_backend

    def list_models(self) -> list[dict[str, Any]]:
        models = []
        backend = self.backend
        capabilities = backend.capabilities().to_dict()
        for spec in MODEL_SPECS.values():
            status = self.inspect_model(spec)
            models.append(
                {
                    "key": spec.key,
                    "id": spec.api_name,
                    "name": spec.public_name,
                    "mode": spec.mode,
                    "folder": spec.folder,
                    "available": status["available"],
                    "backend": backend.key,
                    "capabilities": capabilities,
                }
            )
        return models

    def get_model_spec(self, model_name: Optional[str] = None, mode: Optional[str] = None) -> ModelSpec:
        return self.backend_registry.get_model_spec(model_name=model_name, mode=mode)

    def get_model(self, model_name: Optional[str] = None, mode: Optional[str] = None) -> tuple[ModelSpec, LoadedModelHandle]:
        with operation_scope("core.model_registry.get_model"):
            spec = self.get_model_spec(model_name=model_name, mode=mode)
            log_event(
                LOGGER,
                level=20,
                event="model_registry.load_requested",
                message="Loading model handle through backend registry",
                model=spec.api_name,
                mode=spec.mode,
                backend=self.backend.key,
            )
            handle = self.backend.load_model(spec)
            log_event(
                LOGGER,
                level=20,
                event="model_registry.load_completed",
                message="Model handle loaded through backend registry",
                model=spec.api_name,
                mode=spec.mode,
                backend=self.backend.key,
                model_path=str(handle.resolved_path) if handle.resolved_path else None,
            )
            return spec, handle

    def resolve_model_path(self, folder_name: str):
        return self.backend.resolve_model_path(folder_name)

    def inspect_model(self, spec: ModelSpec) -> dict[str, Any]:
        return self.backend.inspect_model(spec)

    def readiness_report(self) -> dict[str, Any]:
        items = [self.inspect_model(spec) for spec in MODEL_SPECS.values()]
        available_count = sum(1 for item in items if item["available"])
        loadable_count = sum(1 for item in items if item["loadable"])
        runtime_ready_count = sum(1 for item in items if item["runtime_ready"])
        backend_diagnostics = self.backend.readiness_diagnostics().to_dict()
        report = {
            "configured_models": len(items),
            "available_models": available_count,
            "loadable_models": loadable_count,
            "runtime_ready_models": runtime_ready_count,
            "selected_backend": self.backend.key,
            "selected_backend_label": self.backend.label,
            "backend_selection": {
                "requested_backend": self.backend_registry.selection.requested_backend,
                "auto_selected": self.backend_registry.selection.auto_selected,
                "selection_reason": self.backend_registry.selection.selection_reason,
            },
            "backend_capabilities": self.backend.capabilities().to_dict(),
            "backend_diagnostics": backend_diagnostics,
            "available_backends": self.backend_registry.list_backends(),
            "items": items,
        }
        report["registry_ready"] = runtime_ready_count > 0 and backend_diagnostics["ready"]
        return report

    def is_ready(self) -> tuple[bool, dict[str, Any]]:
        report = self.readiness_report()
        return report["registry_ready"], report
