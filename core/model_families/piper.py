# FILE: core/model_families/piper.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide the Piper model-family adapter for direct text-to-waveform synthesis through ONNX-based local voices.
#   SCOPE: PiperFamilyAdapter capability matching and execution preparation
#   DEPENDS: M-MODEL-FAMILY, M-EXECUTION-PLAN, M-ERRORS
#   LINKS: M-MODEL-FAMILY
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   PiperFamilyAdapter - Family adapter for Piper local voice synthesis through ONNX runtime
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added Piper family adapter for the second model family lane]
# END_CHANGE_SUMMARY

from __future__ import annotations

from core.contracts.synthesis import ExecutionPlan
from core.errors import TTSGenerationError
from core.model_families.base import FamilyPreparedExecution, ModelFamilyAdapter


class PiperFamilyAdapter(ModelFamilyAdapter):
    key = "piper"
    label = "Piper"

    def capabilities(self) -> tuple[str, ...]:
        return ("preset_speaker_tts",)

    def supports_plan(self, plan: ExecutionPlan) -> bool:
        return (
            plan.family_key == self.key
            and plan.request.capability in self.capabilities()
        )

    def prepare_execution(self, plan: ExecutionPlan) -> FamilyPreparedExecution:
        if not self.supports_plan(plan):
            raise TTSGenerationError(
                "Execution plan is not supported by the Piper family adapter",
                details={
                    "family": plan.family_key,
                    "capability": plan.request.capability,
                    "model": plan.model_spec.model_id,
                },
            )

        return FamilyPreparedExecution(
            legacy_mode="custom",
            generation_kwargs={
                "language": plan.request.language,
                "voice": plan.model_spec.model_id,
                "instruct": "",
                "speed": 1.0,
                "piper_model": True,
            },
        )


__all__ = ["PiperFamilyAdapter"]
