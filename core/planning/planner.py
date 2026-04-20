# FILE: core/planning/planner.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Resolve normalized synthesis requests into execution plans using the registry contract as the single planning source of truth.
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
#   LAST_CHANGE: [v1.3.0 - Bound planner to an explicit runtime planning protocol instead of a concrete registry implementation]
# END_CHANGE_SUMMARY

from __future__ import annotations

from core.config import CoreSettings
from core.contracts import RuntimePlanningRegistry
from core.contracts.commands import GenerationCommand
from core.contracts.synthesis import (
    ExecutionPlan,
    SynthesisRequest,
    normalize_family_key,
)
from core.errors import ModelCapabilityError, RuntimeCapabilityNotConfiguredError
from core.observability import get_logger, log_event, operation_scope


LOGGER = get_logger(__name__)


class SynthesisPlanner:
    def __init__(self, registry: RuntimePlanningRegistry, settings: CoreSettings):
        self.registry = registry
        self.settings = settings

    def plan(self, request: SynthesisRequest) -> ExecutionPlan:
        with operation_scope("core.synthesis_planner.plan"):
            resolved_model_name = request.requested_model or self.settings.resolve_runtime_model_binding(
                request.execution_mode
            )
            if resolved_model_name is None:
                raise RuntimeCapabilityNotConfiguredError(
                    capability=request.capability,
                    execution_mode=request.execution_mode,
                    family=self.settings.active_family,
                )
            spec = self.registry.get_model_spec(
                model_name=resolved_model_name,
                mode=request.execution_mode,
            )
            if request.capability not in spec.supported_capabilities:
                raise ModelCapabilityError(
                    model_id=spec.model_id,
                    capability=request.capability,
                    supported_capabilities=spec.supported_capabilities,
                    family=spec.family,
                )
            family_label = str(spec.metadata.get("family", "Qwen3-TTS"))
            resolved_backend = self.registry.backend_for_spec(spec)
            plan = ExecutionPlan(
                request=request,
                model_spec=spec,
                backend_key=resolved_backend.key,
                backend_label=resolved_backend.label,
                family_key=normalize_family_key(family_label),
                family_label=family_label,
                selection_reason=self._selection_reason_for_spec(spec),
                execution_mode=request.execution_mode,
            )
            log_event(
                LOGGER,
                level=20,
                event="[SynthesisPlanner][plan][PLAN_REQUEST]",
                message="Synthesis request resolved into execution plan",
                capability=request.capability,
                requested_model=request.requested_model,
                runtime_bound_model=resolved_model_name if request.requested_model is None else None,
                resolved_model=spec.api_name,
                execution_mode=plan.execution_mode,
                backend=plan.backend_key,
                family=plan.family_key,
            )
            return plan

    def _selection_reason_for_spec(self, spec) -> str:
        route = self.registry.backend_route_for_spec(spec)
        route_reason = route.get("route_reason")
        if isinstance(route_reason, str) and route_reason:
            return route_reason
        return "registry_model_resolution"

    def plan_command(self, command: GenerationCommand) -> ExecutionPlan:
        return self.plan(SynthesisRequest.from_command(command))


__all__ = ["LOGGER", "SynthesisPlanner"]
