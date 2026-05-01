# FILE: core/contracts/synthesis.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Define normalized synthesis capability, request, and execution-plan contracts for planner-driven runtime orchestration; capabilities are now declared as runtime data via core.contracts.capabilities.DEFAULT_CAPABILITY_REGISTRY instead of as a closed Literal type, so new families can introduce new capabilities without editing this module.
#   SCOPE: Capability type alias, normalized request payloads, execution plan dataclass, and capability/execution-mode helpers that defer to the default capability registry.
#   DEPENDS: M-CONTRACTS, M-MODELS, M-CAPABILITIES
#   LINKS: M-EXECUTION-PLAN, M-CAPABILITIES
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   SynthesisCapability - Open string alias for synthesis capability identifiers (kept for backwards compatibility; concrete validation is done via the default capability registry).
#   PresetSpeakerPayload - Payload for preset-speaker synthesis requests
#   VoiceDesignPayload - Payload for voice-description synthesis requests
#   VoiceClonePayload - Payload for reference-audio clone synthesis requests
#   SynthesisPayload - Union payload covering all normalized synthesis request variants
#   SynthesisRequest - Normalized synthesis request consumed by planner and family adapters
#   ExecutionPlan - Planner output describing family, model, backend, and selection rationale
#   capability_to_execution_mode - Forward lookup helper that delegates to the default capability registry.
#   execution_mode_to_capability - Reverse lookup helper that delegates to the default capability registry.
#   normalize_family_key - Normalize family labels from manifest metadata into stable family keys
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Phase 3.10: replaced the closed SynthesisCapability Literal with an open str alias backed by DEFAULT_CAPABILITY_REGISTRY; the module-level forward/reverse helpers now defer to the registry so new capabilities can be registered at runtime without editing this contract module]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.contracts.capabilities import (
    DEFAULT_CAPABILITY_REGISTRY,
    default_capability_for_mode,
    default_execution_mode_for,
)
from core.contracts.commands import (
    CustomVoiceCommand,
    GenerationCommand,
    VoiceCloneCommand,
    VoiceDesignCommand,
)
from core.models.manifest import ModelSpec

# Open string alias kept for backwards compatibility. The closed Literal that
# used to live here forced every new capability to be hardcoded in this module;
# concrete validation now lives in DEFAULT_CAPABILITY_REGISTRY.
SynthesisCapability = str


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


def capability_to_execution_mode(capability: SynthesisCapability) -> str:
    return default_execution_mode_for(capability)


def execution_mode_to_capability(mode: str) -> SynthesisCapability:
    return default_capability_for_mode(mode)


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
    def from_command(cls, command: GenerationCommand) -> SynthesisRequest:
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
    def execution_mode(self) -> str:
        return capability_to_execution_mode(self.capability)


@dataclass(frozen=True)
class ExecutionPlan:
    request: SynthesisRequest
    model_spec: ModelSpec
    backend_key: str
    backend_label: str
    family_key: str
    family_label: str
    selection_reason: str
    execution_mode: str


__all__ = [
    "DEFAULT_CAPABILITY_REGISTRY",
    "ExecutionPlan",
    "PresetSpeakerPayload",
    "SynthesisCapability",
    "SynthesisPayload",
    "SynthesisRequest",
    "VoiceClonePayload",
    "VoiceDesignPayload",
    "capability_to_execution_mode",
    "execution_mode_to_capability",
    "normalize_family_key",
]
