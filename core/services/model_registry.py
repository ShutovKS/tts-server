# FILE: core/services/model_registry.py
# VERSION: 1.3.1
# START_MODULE_CONTRACT
#   PURPOSE: Discover, validate, preload, and serve model handles through the selected backend.
#   SCOPE: ModelRegistry facade for model spec resolution, per-model inspection, readiness reporting, backend-backed loading, and preload orchestration
#   DEPENDS: M-ARTIFACT-REGISTRY, M-BACKENDS, M-ERRORS, M-METRICS, M-MODEL-CATALOG, M-MODELS, M-OBSERVABILITY, M-PROFILE-RESOLVER, M-RUNTIME-MODEL-REGISTRY
#   LINKS: M-MODEL-REGISTRY
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LOGGER - Module logger for model registry events
#   SUPPORTED_PRELOAD_POLICIES - Allowed model preload policy values
#   ModelRegistry - High-level model discovery and loading facade
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.3.1 - Refreshed GRACE module dependencies and documented the per-model inspection surface]
# END_CHANGE_SUMMARY

from __future__ import annotations

from typing import Any, Optional

from core.backends import BackendRegistry
from core.backends.base import LoadedModelHandle
from core.errors import BackendCapabilityError
from core.errors import ModelLoadError, ModelNotAvailableError
from core.metrics import OperationalMetricsRegistry
from core.models.catalog import ModelSpec
from core.models.manifest import ModelDescriptor
from core.observability import get_logger, log_event, operation_scope
from core.registry import ArtifactRegistry, ModelCatalogRegistry, RuntimeModelRegistry


LOGGER = get_logger(__name__)
SUPPORTED_PRELOAD_POLICIES = {"none", "all", "listed"}


# START_CONTRACT: ModelRegistry
#   PURPOSE: Provide high-level model discovery, readiness reporting, and backend-backed model loading.
#   INPUTS: { backend_registry: BackendRegistry - Backend registry that selects the active backend and model manifest, preload_policy: str - Startup model preload strategy, preload_model_ids: tuple[str, ...] - Explicit model identifiers for listed preload policy, metrics: OperationalMetricsRegistry | None - Optional metrics facade for readiness summaries }
#   OUTPUTS: { instance - Model registry coordinating backend-backed model access }
#   SIDE_EFFECTS: May preload model runtimes during initialization
#   LINKS: M-MODEL-REGISTRY
# END_CONTRACT: ModelRegistry
class ModelRegistry:
    def __init__(
        self,
        backend_registry: BackendRegistry,
        *,
        preload_policy: str = "none",
        preload_model_ids: tuple[str, ...] = (),
        metrics: OperationalMetricsRegistry | None = None,
    ):
        self.backend_registry = backend_registry
        self._metrics = metrics or OperationalMetricsRegistry()
        normalized_policy = (preload_policy or "none").strip().lower()
        self._preload_policy = (
            normalized_policy
            if normalized_policy in SUPPORTED_PRELOAD_POLICIES
            else "none"
        )
        self._preload_model_ids = tuple(preload_model_ids)
        self.catalog = ModelCatalogRegistry(self.backend_registry.model_specs)
        self.artifacts = ArtifactRegistry(self.catalog, self.backend_registry)
        self.runtime_models = RuntimeModelRegistry(self.artifacts, self.preload_report)
        self._preload_report = self._build_preload_report(status="not_started")
        self._apply_preload_policy()

    @property
    def backend(self):
        return self.backend_registry.selected_backend

    def backend_for_spec(self, spec: ModelSpec):
        return self.backend_registry.resolve_backend_for_spec(spec)

    def backend_route_for_spec(self, spec: ModelSpec) -> dict[str, Any]:
        return self.backend_registry.explain_backend_route_for_spec(spec)

    @property
    def model_specs(self) -> tuple[ModelSpec, ...]:
        return self.catalog.model_specs

    @property
    def model_descriptors(self) -> tuple[ModelDescriptor, ...]:
        return self.catalog.descriptors

    # START_CONTRACT: list_models
    #   PURPOSE: List configured models together with backend capability and availability status.
    #   INPUTS: {}
    #   OUTPUTS: { list[dict[str, Any]] - Per-model availability and capability records }
    #   SIDE_EFFECTS: none
    #   LINKS: M-MODEL-REGISTRY
    # END_CONTRACT: list_models
    def list_models(self) -> list[dict[str, Any]]:
        return [self.inspect_model(spec) for spec in self.model_specs]

    # START_CONTRACT: get_model_spec
    #   PURPOSE: Resolve a model specification by model identifier or synthesis mode.
    #   INPUTS: { model_name: Optional[str] - Requested model identifier, mode: Optional[str] - Requested synthesis mode fallback }
    #   OUTPUTS: { ModelSpec - Matching enabled model specification }
    #   SIDE_EFFECTS: none
    #   LINKS: M-MODEL-REGISTRY
    # END_CONTRACT: get_model_spec
    def get_model_spec(
        self, model_name: Optional[str] = None, mode: Optional[str] = None
    ) -> ModelSpec:
        return self.backend_registry.get_model_spec(model_name=model_name, mode=mode)

    # START_CONTRACT: get_model
    #   PURPOSE: Resolve and load a model handle for the active backend.
    #   INPUTS: { model_name: Optional[str] - Requested model identifier, mode: Optional[str] - Requested synthesis mode fallback }
    #   OUTPUTS: { tuple[ModelSpec, LoadedModelHandle] - Resolved model spec and loaded backend handle }
    #   SIDE_EFFECTS: Loads model runtimes through the selected backend and emits structured logs
    #   LINKS: M-MODEL-REGISTRY
    # END_CONTRACT: get_model
    def get_model(
        self, model_name: Optional[str] = None, mode: Optional[str] = None
    ) -> tuple[ModelSpec, LoadedModelHandle]:
        with operation_scope("core.model_registry.get_model"):
            spec = self.get_model_spec(model_name=model_name, mode=mode)
            log_event(
                LOGGER,
                level=20,
                event="[ModelRegistry][get_model][GET_MODEL]",
                message="Loading model handle through backend registry",
                model=spec.api_name,
                mode=spec.mode,
                backend=self.backend_for_spec(spec).key,
            )
            backend = self.backend_for_spec(spec)
            handle = backend.load_model(spec)
            log_event(
                LOGGER,
                level=20,
                event="[ModelRegistry][get_model][GET_MODEL]",
                message="Model handle loaded through backend registry",
                model=spec.api_name,
                mode=spec.mode,
                backend=backend.key,
                model_path=str(handle.resolved_path) if handle.resolved_path else None,
            )
            return spec, handle

    # START_CONTRACT: preload_report
    #   PURPOSE: Expose the current preload report through a stable public registry contract.
    #   INPUTS: {}
    #   OUTPUTS: { dict[str, Any] - Copy of the preload policy execution report }
    #   SIDE_EFFECTS: none
    #   LINKS: M-MODEL-REGISTRY, M-RUNTIME-MODEL-REGISTRY
    # END_CONTRACT: preload_report
    def preload_report(self) -> dict[str, Any]:
        return self._build_preload_report(
            status=self._preload_report["status"],
            requested_model_ids=list(self._preload_report["requested_model_ids"]),
            resolved_model_ids=list(self._preload_report["resolved_model_ids"]),
            loaded_model_ids=list(self._preload_report["loaded_model_ids"]),
            failed_model_ids=list(self._preload_report["failed_model_ids"]),
            errors=list(self._preload_report["errors"]),
            policy_reason=self._preload_report["policy_reason"],
        )

    def resolve_model_path(self, folder_name: str):
        return self.artifacts.resolve_model_path(folder_name)

    # START_CONTRACT: inspect_model
    #   PURPOSE: Build a per-model inspection record that merges routing, artifact, cache, and family-profile state.
    #   INPUTS: { spec: ModelSpec - Model specification to inspect }
    #   OUTPUTS: { dict[str, Any] - Structured inspection payload used by model listing and readiness reports }
    #   SIDE_EFFECTS: none
    #   LINKS: M-MODEL-REGISTRY, M-ARTIFACT-REGISTRY, M-RUNTIME-MODEL-REGISTRY
    # END_CONTRACT: inspect_model
    def inspect_model(self, spec: ModelSpec) -> dict[str, Any]:
        route = self.backend_route_for_spec(spec)
        execution_backend_key = route["execution_backend"]
        descriptor = self.catalog.get_descriptor(spec.model_id)
        family_profile = self._resolve_family_profile(descriptor.family_key)
        try:
            self.backend_for_spec(spec)
        except BackendCapabilityError:
            item = self.runtime_models.descriptor_readiness(
                descriptor,
                route=route,
                artifact_state={
                    "model_id": descriptor.model_id,
                    "family_key": descriptor.family_key,
                    "declared": True,
                    "installed": False,
                    "loadable": False,
                    "runtime_ready": False,
                    "resolved_path": None,
                    "missing_artifacts": [],
                    "required_artifacts": [],
                    "loaded": False,
                    "preload_status": "not_requested",
                },
            )
            item.update(
                {
                    "key": spec.key,
                    "name": spec.public_name,
                    "mode": spec.mode,
                    "folder": spec.folder,
                    "backend": execution_backend_key,
                    "profile": family_profile,
                    "configured": True,
                    "available": False,
                    "cached": False,
                    "runtime_path": None,
                    "cache": {
                        "loaded": False,
                        "cache_key": spec.folder,
                        "backend": None,
                        "normalized_runtime": False,
                        "runtime_path": None,
                        "eviction_policy": "not_configured",
                    },
                    "capabilities": {},
                }
            )
        else:
            item = self.artifacts.inspect(spec)
            item.update(
                self.runtime_models.descriptor_readiness(
                    descriptor,
                    route=route,
                    artifact_state={
                        "model_id": descriptor.model_id,
                        "family_key": descriptor.family_key,
                        "declared": True,
                        "installed": bool(item["available"]),
                        "loadable": bool(item["loadable"]),
                        "runtime_ready": bool(item["runtime_ready"]),
                        "resolved_path": item["resolved_path"],
                        "missing_artifacts": list(item["missing_artifacts"]),
                        "required_artifacts": list(item["required_artifacts"]),
                        "loaded": bool(item.get("cached")),
                        "preload_status": "loaded"
                        if spec.model_id in self.preload_report()["loaded_model_ids"]
                        else "failed"
                        if spec.model_id in self.preload_report()["failed_model_ids"]
                        else "requested"
                        if spec.model_id in self.preload_report()["requested_model_ids"]
                        else "not_requested",
                    },
                )
            )
            item.update(
                {
                    "key": spec.key,
                    "name": spec.public_name,
                    "mode": spec.mode,
                    "folder": spec.folder,
                    "backend": execution_backend_key,
                    "profile": family_profile,
                }
            )
        return item

    def _resolve_family_profile(self, family_key: str) -> dict[str, Any] | None:
        from profiles.resolver import ProfileResolver

        profile_key = {
            "qwen3_tts": "qwen",
            "piper": "piper",
            "omnivoice": "omnivoice",
        }.get(family_key)
        if profile_key is None:
            return None
        try:
            return ProfileResolver().get_family_profile(profile_key).to_dict()
        except ValueError:
            return None

    # START_CONTRACT: readiness_report
    #   PURPOSE: Build a full readiness report for configured models, backend selection, cache state, and preload status.
    #   INPUTS: {}
    #   OUTPUTS: { dict[str, Any] - Registry readiness report for operational consumers }
    #   SIDE_EFFECTS: none
    #   LINKS: M-MODEL-REGISTRY
    # END_CONTRACT: readiness_report
    def readiness_report(self) -> dict[str, Any]:
        # START_BLOCK_COLLECT_MODEL_STATUS
        items = [self.inspect_model(spec) for spec in self.model_specs]
        available_count = sum(1 for item in items if item["available"])
        loadable_count = sum(1 for item in items if item["loadable"])
        runtime_ready_count = sum(1 for item in items if item["runtime_ready"])
        loaded_count = sum(1 for item in items if item.get("cached"))
        per_model_backend_overrides = sum(
            1
            for item in items
            if item.get("execution_backend")
            and item.get("execution_backend") != self.backend.key
        )
        degraded_routes = sum(
            1
            for item in items
            if item.get("route", {}).get("route_reason") == "no_ready_backend_for_model"
        )
        family_summary: dict[str, dict[str, Any]] = {}
        for item in items:
            family_key = str(item.get("family_key") or "unknown")
            summary = family_summary.setdefault(
                family_key,
                {
                    "family": item.get("family"),
                    "configured_models": 0,
                    "available_models": 0,
                    "runtime_ready_models": 0,
                },
            )
            summary["configured_models"] += 1
            if item.get("available"):
                summary["available_models"] += 1
            if item.get("runtime_ready"):
                summary["runtime_ready_models"] += 1
        backend_diagnostics = self.backend.readiness_diagnostics().to_dict()
        cache_diagnostics = self.backend.cache_diagnostics()
        # END_BLOCK_COLLECT_MODEL_STATUS
        # START_BLOCK_BUILD_READINESS_REPORT
        report = {
            "configured_models": len(items),
            "available_models": available_count,
            "loadable_models": loadable_count,
            "runtime_ready_models": runtime_ready_count,
            "loaded_models": loaded_count,
            "selected_backend": self.backend.key,
            "selected_backend_label": self.backend.label,
            "backend_selection": {
                "requested_backend": self.backend_registry.selection.requested_backend,
                "auto_selected": self.backend_registry.selection.auto_selected,
                "selection_reason": self.backend_registry.selection.selection_reason,
            },
            "backend_capabilities": self.backend.capabilities().to_dict(),
            "backend_diagnostics": backend_diagnostics,
            "cache_diagnostics": cache_diagnostics,
            "host": self.backend_registry.host_snapshot.to_dict(),
            "routing": {
                "mixed_backend_routing": per_model_backend_overrides > 0,
                "per_model_backend_overrides": per_model_backend_overrides,
                "degraded_routes": degraded_routes,
            },
            "family_summary": family_summary,
            "family_profiles": {
                family_key: self._resolve_family_profile(family_key)
                for family_key in sorted(family_summary)
            },
            "metrics": self._build_metrics_summary(),
            "preload": self.preload_report(),
            "available_backends": self.backend_registry.list_backends(),
            "items": items,
            "manifest": {
                "version": self.backend_registry._model_manifest.version,
                "metadata": dict(self.backend_registry._model_manifest.metadata),
                "descriptor_count": len(self.model_descriptors),
            },
        }
        report["registry_ready"] = (
            runtime_ready_count > 0 and backend_diagnostics["ready"]
        )
        return report
        # END_BLOCK_BUILD_READINESS_REPORT

    # START_CONTRACT: is_ready
    #   PURPOSE: Evaluate whether the registry has at least one runtime-ready model and return the supporting report.
    #   INPUTS: {}
    #   OUTPUTS: { tuple[bool, dict[str, Any]] - Readiness boolean and full readiness report }
    #   SIDE_EFFECTS: none
    #   LINKS: M-MODEL-REGISTRY
    # END_CONTRACT: is_ready
    def is_ready(self) -> tuple[bool, dict[str, Any]]:
        report = self.readiness_report()
        return report["registry_ready"], report

    def _apply_preload_policy(self) -> None:
        # START_BLOCK_RESOLVE_PRELOAD_SPECS
        specs = self._resolve_preload_specs()
        requested_ids = self._requested_preload_model_ids()
        if not specs:
            policy_reason = (
                "disabled" if self._preload_policy == "none" else "no_matching_models"
            )
            self._preload_report = self._build_preload_report(
                status="skipped",
                requested_model_ids=requested_ids,
                resolved_model_ids=[],
                policy_reason=policy_reason,
            )
            return
        # END_BLOCK_RESOLVE_PRELOAD_SPECS

        # START_BLOCK_EXECUTE_PRELOAD
        preload_groups: dict[str, list[ModelSpec]] = {}
        for spec in specs:
            try:
                preload_groups.setdefault(self.backend_for_spec(spec).key, []).append(
                    spec
                )
            except BackendCapabilityError:
                continue
        loaded_model_ids: list[str] = []
        failed_model_ids: list[str] = []
        errors: list[dict[str, Any]] = []
        for backend_key, backend_specs in preload_groups.items():
            backend = self.backend_registry._backends[backend_key]
            preload_result = backend.preload_models(tuple(backend_specs))
            loaded_model_ids.extend(preload_result["loaded_model_ids"])
            failed_model_ids.extend(preload_result["failed_model_ids"])
            errors.extend(preload_result["errors"])
        status = "completed"
        if failed_model_ids and loaded_model_ids:
            status = "completed_with_errors"
        elif failed_model_ids and not loaded_model_ids:
            status = "failed"
        elif not loaded_model_ids and not failed_model_ids:
            status = "skipped"
        self._preload_report = self._build_preload_report(
            status=status,
            requested_model_ids=requested_ids,
            resolved_model_ids=[spec.api_name for spec in specs],
            loaded_model_ids=loaded_model_ids,
            failed_model_ids=failed_model_ids,
            errors=errors,
            policy_reason="configured",
        )
        # END_BLOCK_EXECUTE_PRELOAD

    def _resolve_preload_specs(self) -> list[ModelSpec]:
        if self._preload_policy == "none":
            return []
        if self._preload_policy == "all":
            return [spec for spec in self.model_specs if self._is_preloadable(spec)]
        if self._preload_policy == "listed":
            specs: list[ModelSpec] = []
            for model_id in self._preload_model_ids:
                try:
                    spec = self.get_model_spec(model_name=model_id)
                except (ModelNotAvailableError, ModelLoadError):
                    continue
                if self._is_preloadable(spec) and spec not in specs:
                    specs.append(spec)
            return specs
        return []

    def _requested_preload_model_ids(self) -> list[str]:
        if self._preload_policy == "all":
            return [spec.model_id for spec in self.model_specs]
        return list(self._preload_model_ids)

    def _is_preloadable(self, spec: ModelSpec) -> bool:
        status = self.inspect_model(spec)
        return bool(status["runtime_ready"])

    def _build_metrics_summary(self) -> dict[str, Any]:
        return {
            "selected_backend": self.backend.metrics_summary(),
            "operational": self._metrics.readiness_summary(),
        }

    def _build_preload_report(
        self,
        *,
        status: str,
        requested_model_ids: list[str] | None = None,
        resolved_model_ids: list[str] | None = None,
        loaded_model_ids: list[str] | None = None,
        failed_model_ids: list[str] | None = None,
        errors: list[dict[str, Any]] | None = None,
        policy_reason: str | None = None,
    ) -> dict[str, Any]:
        requested_model_ids = requested_model_ids or []
        resolved_model_ids = resolved_model_ids or []
        loaded_model_ids = loaded_model_ids or []
        failed_model_ids = failed_model_ids or []
        errors = errors or []
        return {
            "policy": self._preload_policy,
            "configured_model_ids": list(self._preload_model_ids),
            "requested_model_ids": requested_model_ids,
            "resolved_model_ids": resolved_model_ids,
            "status": status,
            "policy_reason": policy_reason,
            "attempted": len(resolved_model_ids),
            "loaded": len(loaded_model_ids),
            "failed": len(failed_model_ids),
            "loaded_model_ids": loaded_model_ids,
            "failed_model_ids": failed_model_ids,
            "errors": errors,
        }


__all__ = [
    "LOGGER",
    "SUPPORTED_PRELOAD_POLICIES",
    "ModelRegistry",
]
