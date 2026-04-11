# FILE: core/contracts/synthesis.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define normalized synthesis capability, request, and execution-plan contracts for planner-driven runtime orchestration.
#   SCOPE: Capability types, normalized request payloads, execution plan dataclasses, compatibility helpers
#   DEPENDS: M-CONTRACTS, M-MODELS
#   LINKS: M-EXECUTION-PLAN
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   SynthesisCapability - Normalized capability identifier for synthesis planning
#   PresetSpeakerPayload - Payload for preset-speaker synthesis requests
#   VoiceDesignPayload - Payload for voice-description synthesis requests
#   VoiceClonePayload - Payload for reference-audio clone synthesis requests
#   SynthesisPayload - Union payload covering all normalized synthesis request variants
#   SynthesisRequest - Normalized synthesis request consumed by planner and family adapters
#   ExecutionPlan - Planner output describing family, model, backend, and selection rationale
#   capability_to_legacy_mode - Map normalized capability to current legacy mode identifiers
#   legacy_mode_to_capability - Map current legacy modes to normalized capability identifiers
#   normalize_family_key - Normalize family labels from manifest metadata into stable family keys
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.1 - Added missing SynthesisPayload export to keep GRACE module-map integrity clean]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from core.contracts.commands import (
    CustomVoiceCommand,
    GenerationCommand,
    VoiceCloneCommand,
    VoiceDesignCommand,
)
from core.models.manifest import ModelSpec


SynthesisCapability = Literal[
    "preset_speaker_tts",
    "voice_description_tts",
    "reference_voice_clone",
]

_CAPABILITY_TO_LEGACY_MODE: dict[SynthesisCapability, str] = {
    "preset_speaker_tts": "custom",
    "voice_description_tts": "design",
    "reference_voice_clone": "clone",
}

_LEGACY_MODE_TO_CAPABILITY: dict[str, SynthesisCapability] = {
    value: key for key, value in _CAPABILITY_TO_LEGACY_MODE.items()
}


@dataclass(frozen=True)
class PresetSpeakerPayload:
    speaker: str
    instruct: str
    speed: float


@dataclass(frozen=True)
class VoiceDesignPayload:
    voice_description: str


@dataclass(frozen=True)
class VoiceClonePayload:
    ref_audio_path: Path | None
    ref_text: str | None


SynthesisPayload = PresetSpeakerPayload | VoiceDesignPayload | VoiceClonePayload


def capability_to_legacy_mode(capability: SynthesisCapability) -> str:
    return _CAPABILITY_TO_LEGACY_MODE[capability]


def legacy_mode_to_capability(mode: str) -> SynthesisCapability:
    try:
        return _LEGACY_MODE_TO_CAPABILITY[mode]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unsupported legacy mode: {mode}") from exc


def normalize_family_key(family_label: str | None) -> str:
    raw = (family_label or "qwen3_tts").strip().lower()
    normalized = "".join(character if character.isalnum() else "_" for character in raw)
    collapsed = "_".join(part for part in normalized.split("_") if part)
    return collapsed or "qwen3_tts"


@dataclass(frozen=True)
class SynthesisRequest:
    capability: SynthesisCapability
    text: str
    payload: SynthesisPayload
    requested_model: str | None = None
    save_output: bool = False
    language: str = "auto"
    source_command: str = "unknown"

    def __post_init__(self) -> None:
        normalized_language = self.language.strip().lower()
        if not normalized_language:
            raise ValueError("Language must not be empty")
        object.__setattr__(self, "language", normalized_language)

    @classmethod
    def from_command(cls, command: GenerationCommand) -> "SynthesisRequest":
        if isinstance(command, CustomVoiceCommand):
            return cls(
                capability="preset_speaker_tts",
                text=command.text,
                payload=PresetSpeakerPayload(
                    speaker=command.speaker,
                    instruct=command.instruct,
                    speed=command.speed,
                ),
                requested_model=command.model,
                save_output=command.save_output,
                language=command.language,
                source_command=command.__class__.__name__,
            )

        if isinstance(command, VoiceDesignCommand):
            return cls(
                capability="voice_description_tts",
                text=command.text,
                payload=VoiceDesignPayload(
                    voice_description=command.voice_description,
                ),
                requested_model=command.model,
                save_output=command.save_output,
                language=command.language,
                source_command=command.__class__.__name__,
            )

        if isinstance(command, VoiceCloneCommand):
            return cls(
                capability="reference_voice_clone",
                text=command.text,
                payload=VoiceClonePayload(
                    ref_audio_path=command.ref_audio_path,
                    ref_text=command.ref_text,
                ),
                requested_model=command.model,
                save_output=command.save_output,
                language=command.language,
                source_command=command.__class__.__name__,
            )

        raise TypeError(
            f"Unsupported command type for synthesis normalization: {command.__class__.__name__}"
        )

    @property
    def legacy_mode(self) -> str:
        return capability_to_legacy_mode(self.capability)


@dataclass(frozen=True)
class ExecutionPlan:
    request: SynthesisRequest
    model_spec: ModelSpec
    backend_key: str
    backend_label: str
    family_key: str
    family_label: str
    selection_reason: str
    legacy_mode: str


__all__ = [
    "ExecutionPlan",
    "PresetSpeakerPayload",
    "SynthesisCapability",
    "SynthesisPayload",
    "SynthesisRequest",
    "VoiceClonePayload",
    "VoiceDesignPayload",
    "capability_to_legacy_mode",
    "legacy_mode_to_capability",
    "normalize_family_key",
]
