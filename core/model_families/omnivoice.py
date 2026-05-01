# FILE: core/model_families/omnivoice.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide OmniVoice family semantics for custom, design, and clone synthesis through the shared planner/runtime pipeline.
#   SCOPE: OmniVoiceFamilyAdapter capability matching and execution preparation
#   DEPENDS: M-MODEL-FAMILY, M-EXECUTION-PLAN, M-ERRORS
#   LINKS: M-MODEL-FAMILY
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   OmniVoiceFamilyAdapter - Family adapter for OmniVoice synthesis semantics through the Torch backend
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.1 - Preserved custom-mode speed control in prepared generation kwargs so the shared backend execution contract remains satisfied]
# END_CHANGE_SUMMARY

from __future__ import annotations

from core.contracts.synthesis import (
    ExecutionPlan,
    PresetSpeakerPayload,
    VoiceClonePayload,
    VoiceDesignPayload,
)
from core.errors import TTSGenerationError
from core.model_families.base import FamilyPreparedExecution, ModelFamilyAdapter


class OmniVoiceFamilyAdapter(ModelFamilyAdapter):
    key = "omnivoice"
    label = "OmniVoice"

    def capabilities(self) -> tuple[str, ...]:
        return (
            "preset_speaker_tts",
            "voice_description_tts",
            "reference_voice_clone",
        )

    def supports_plan(self, plan: ExecutionPlan) -> bool:
        return plan.family_key == self.key and plan.request.capability in self.capabilities()

    def prepare_execution(self, plan: ExecutionPlan) -> FamilyPreparedExecution:
        if not self.supports_plan(plan):
            raise TTSGenerationError(
                "Execution plan is not supported by the OmniVoice family adapter",
                details={
                    "family": plan.family_key,
                    "capability": plan.request.capability,
                    "model": plan.model_spec.model_id,
                },
            )

        payload = plan.request.payload
        generation_kwargs: dict[str, object] = {
            "language": plan.request.language,
            "family": self.key,
        }

        if isinstance(payload, PresetSpeakerPayload):
            generation_kwargs.update(
                {
                    "voice": payload.speaker,
                    "instruct": payload.instruct,
                    "speed": payload.speed,
                }
            )
        elif isinstance(payload, VoiceDesignPayload):
            generation_kwargs.update(
                {
                    "instruct": payload.voice_description,
                }
            )
        elif isinstance(payload, VoiceClonePayload):
            generation_kwargs.update(
                {
                    "ref_audio": str(payload.ref_audio_path)
                    if payload.ref_audio_path is not None
                    else None,
                    "ref_text": payload.ref_text,
                }
            )
        else:  # pragma: no cover
            raise TTSGenerationError(
                "Unsupported OmniVoice payload shape",
                details={
                    "family": self.key,
                    "capability": plan.request.capability,
                },
            )

        return FamilyPreparedExecution(
            execution_mode=plan.execution_mode,
            generation_kwargs=generation_kwargs,
        )


__all__ = ["OmniVoiceFamilyAdapter"]
