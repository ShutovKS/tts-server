# FILE: core/model_families/qwen3.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Wrap current Qwen3 custom, design, and clone semantics as the first explicit model-family adapter.
#   SCOPE: Qwen3FamilyAdapter class, family metadata helpers for speakers and voice design examples
#   DEPENDS: M-MODEL-FAMILY, M-EXECUTION-PLAN, M-CONTRACTS, M-ERRORS
#   LINKS: M-QWEN3-FAMILY
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   SPEAKER_MAP - Qwen3-supported speakers grouped by language
#   EMOTION_EXAMPLES - Qwen3 voice design prompt examples
#   Qwen3FamilyAdapter - Family adapter that preserves current Qwen3 execution semantics
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added explicit Qwen3 family adapter to preserve current semantics during migration]
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
from core.models.catalog import EMOTION_EXAMPLES, SPEAKER_MAP


class Qwen3FamilyAdapter(ModelFamilyAdapter):
    key = "qwen3_tts"
    label = "Qwen3-TTS"

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
                "Execution plan is not supported by the Qwen3 family adapter",
                details={
                    "family": plan.family_key,
                    "capability": plan.request.capability,
                    "model": plan.model_spec.api_name,
                },
            )

        payload = plan.request.payload
        generation_kwargs: dict[str, object] = {
            "language": plan.request.language,
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
                    "ref_text": payload.ref_text or ".",
                }
            )
        else:  # pragma: no cover - defensive guard for future payload variants
            raise TTSGenerationError(
                "Unsupported Qwen3 payload shape",
                details={
                    "family": self.key,
                    "capability": plan.request.capability,
                },
            )

        return FamilyPreparedExecution(
            legacy_mode=plan.legacy_mode,
            generation_kwargs=generation_kwargs,
        )


__all__ = ["EMOTION_EXAMPLES", "Qwen3FamilyAdapter", "SPEAKER_MAP"]
