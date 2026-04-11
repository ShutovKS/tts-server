# FILE: core/planning/planner.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Resolve normalized synthesis requests into execution plans using the current registry-backed compatibility bridge.
#   SCOPE: SynthesisPlanner class with request and command planning helpers
#   DEPENDS: M-EXECUTION-PLAN, M-MODEL-REGISTRY, M-OBSERVABILITY, M-MODELS
#   LINKS: M-SYNTHESIS-PLANNER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LOGGER - Module logger for planning events
#   SynthesisPlanner - Planner that resolves normalized requests into current execution plans
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added initial synthesis planner as a compatibility bridge over the current registry-backed runtime]
# END_CHANGE_SUMMARY

from __future__ import annotations

from core.contracts.commands import GenerationCommand
from core.contracts.synthesis import (
    ExecutionPlan,
    SynthesisRequest,
    normalize_family_key,
)
from core.errors import ModelCapabilityError
from core.models.catalog import MODEL_SPECS
from core.observability import get_logger, log_event, operation_scope
from core.services.model_registry import ModelRegistry


LOGGER = get_logger(__name__)


class SynthesisPlanner:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def plan(self, request: SynthesisRequest) -> ExecutionPlan:
        with operation_scope("core.synthesis_planner.plan"):
            if hasattr(self.registry, "get_model_spec"):
                spec = self.registry.get_model_spec(
                    model_name=request.requested_model,
                    mode=request.legacy_mode,
                )
            else:
                spec = self._resolve_spec_from_catalog(request)
            if request.capability not in spec.supported_capabilities:
                raise ModelCapabilityError(
                    model_id=spec.model_id,
                    capability=request.capability,
                    supported_capabilities=spec.supported_capabilities,
                    family=spec.family,
                )
            family_label = str(spec.metadata.get("family", "Qwen3-TTS"))
            resolved_backend = self._resolve_backend_for_spec(spec)
            plan = ExecutionPlan(
                request=request,
                model_spec=spec,
                backend_key=getattr(resolved_backend, "key", "unknown"),
                backend_label=getattr(
                    resolved_backend,
                    "label",
                    "Compatibility backend",
                ),
                family_key=normalize_family_key(family_label),
                family_label=family_label,
                selection_reason=self._selection_reason_for_spec(
                    spec, resolved_backend
                ),
                legacy_mode=request.legacy_mode,
            )
            log_event(
                LOGGER,
                level=20,
                event="[SynthesisPlanner][plan][PLAN_REQUEST]",
                message="Synthesis request resolved into execution plan",
                capability=request.capability,
                requested_model=request.requested_model,
                resolved_model=spec.api_name,
                legacy_mode=plan.legacy_mode,
                backend=plan.backend_key,
                family=plan.family_key,
            )
            return plan

    def _resolve_backend_for_spec(self, spec):
        resolver = getattr(self.registry, "backend_for_spec", None)
        if callable(resolver):
            return resolver(spec)
        return getattr(self.registry, "backend", None)

    def _selection_reason_for_spec(self, spec, backend) -> str:
        route_resolver = getattr(self.registry, "backend_route_for_spec", None)
        if callable(route_resolver):
            route = route_resolver(spec)
            route_reason = route.get("route_reason")
            if isinstance(route_reason, str) and route_reason:
                return route_reason
        return "registry_model_resolution"

    def _resolve_spec_from_catalog(self, request: SynthesisRequest):
        if request.requested_model is not None:
            for spec in MODEL_SPECS.values():
                if request.requested_model in {spec.api_name, spec.folder, spec.key}:
                    return spec
        for spec in MODEL_SPECS.values():
            if spec.mode == request.legacy_mode:
                return spec
        raise ValueError(
            f"Unable to resolve model spec for legacy mode '{request.legacy_mode}'"
        )

    def plan_command(self, command: GenerationCommand) -> ExecutionPlan:
        return self.plan(SynthesisRequest.from_command(command))


__all__ = ["LOGGER", "SynthesisPlanner"]
