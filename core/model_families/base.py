# FILE: core/model_families/base.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define the base model-family adapter contract used to separate family semantics from transport adapters and backend runtimes.
#   SCOPE: FamilyPreparedExecution dataclass, ModelFamilyAdapter abstract base class
#   DEPENDS: M-EXECUTION-PLAN, M-MODELS
#   LINKS: M-MODEL-FAMILY
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   FamilyPreparedExecution - Family-prepared legacy mode and generation kwargs for backend execution
#   ModelFamilyAdapter - Base contract for family capability matching and execution preparation
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added base family adapter contract for the migration architecture]
# END_CHANGE_SUMMARY

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.contracts.synthesis import ExecutionPlan, SynthesisCapability


@dataclass(frozen=True)
class FamilyPreparedExecution:
    legacy_mode: str
    generation_kwargs: dict[str, object]


class ModelFamilyAdapter(ABC):
    key: str
    label: str

    @abstractmethod
    def capabilities(self) -> tuple[SynthesisCapability, ...]:
        raise NotImplementedError

    @abstractmethod
    def supports_plan(self, plan: ExecutionPlan) -> bool:
        raise NotImplementedError

    @abstractmethod
    def prepare_execution(self, plan: ExecutionPlan) -> FamilyPreparedExecution:
        raise NotImplementedError


__all__ = ["FamilyPreparedExecution", "ModelFamilyAdapter"]
