# FILE: core/model_families/voxcpm.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide VoxCPM family semantics for custom, design, and clone synthesis through the shared planner/runtime pipeline.
#   SCOPE: VoxCPMFamilyAdapter capability matching and execution preparation
#   DEPENDS: M-MODEL-FAMILY, M-EXECUTION-PLAN, M-ERRORS
#   LINKS: M-MODEL-FAMILY
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   VoxCPMFamilyAdapter - Family adapter for VoxCPM synthesis semantics through the Torch backend
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added VoxCPM family adapter for planner-driven execution]
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


class VoxCPMFamilyAdapter(ModelFamilyAdapter):
    key = "voxcpm"
    label = "VoxCPM"

    def capabilities(self) -> tuple[str, ...]:
        return (
            "preset_speaker_tts",
            "voice_description_tts",
            "reference_voice_clone",
        )

    def supports_plan(self, plan: ExecutionPlan) -> bool:
        return (
            plan.family_key == self.key
            and plan.request.capability in self.capabilities()
        )

    def prepare_execution(self, plan: ExecutionPlan) -> FamilyPreparedExecution:
        if not self.supports_plan(plan):
            raise TTSGenerationError(
                "Execution plan is not supported by the VoxCPM family adapter",
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
                "Unsupported VoxCPM payload shape",
                details={
                    "family": self.key,
                    "capability": plan.request.capability,
                },
            )

        return FamilyPreparedExecution(
            legacy_mode=plan.legacy_mode,
            generation_kwargs=generation_kwargs,
        )


__all__ = ["VoxCPMFamilyAdapter"]
