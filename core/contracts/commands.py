# FILE: core/contracts/commands.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define command DTOs for TTS synthesis requests.
#   SCOPE: CustomVoiceCommand, VoiceDesignCommand, VoiceCloneCommand dataclasses
#   DEPENDS: none
#   LINKS: M-CONTRACTS
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   GenerationCommand - Base command fields shared by synthesis requests
#   CustomVoiceCommand - Command for custom voice synthesis
#   VoiceDesignCommand - Command for voice design synthesis
#   VoiceCloneCommand - Command for voice clone synthesis
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.1 - Removed duplicate language field declaration while preserving command language normalization]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# START_CONTRACT: GenerationCommand
#   PURPOSE: Represent the shared base command fields for TTS generation requests.
#   INPUTS: { text: str - Source text to synthesize, model: Optional[str] - Optional requested model identifier, save_output: bool - Whether generated audio should be persisted, language: str - Requested language code or auto }
#   OUTPUTS: { instance - Immutable generation command base }
#   SIDE_EFFECTS: none
#   LINKS: M-CONTRACTS
# END_CONTRACT: GenerationCommand
@dataclass(frozen=True)
class GenerationCommand:
    text: str
    model: Optional[str] = None
    save_output: bool = False
    language: str = "auto"

    def __post_init__(self) -> None:
        normalized_language = self.language.strip().lower()
        if not normalized_language:
            raise ValueError("Language must not be empty")
        object.__setattr__(self, "language", normalized_language)


# START_CONTRACT: CustomVoiceCommand
#   PURPOSE: Represent a custom-voice synthesis request with speaker and instruction controls.
#   INPUTS: { text: str - Source text to synthesize, model: Optional[str] - Optional requested model identifier, save_output: bool - Whether generated audio should be persisted, language: str - Requested language code or auto, speaker: str - Speaker preset or identifier, instruct: str - Additional generation instruction, speed: float - Playback speed modifier }
#   OUTPUTS: { instance - Immutable custom voice generation command }
#   SIDE_EFFECTS: none
#   LINKS: M-CONTRACTS
# END_CONTRACT: CustomVoiceCommand
@dataclass(frozen=True)
class CustomVoiceCommand(GenerationCommand):
    speaker: str = "Vivian"
    instruct: str = "Normal tone"
    speed: float = 1.0


# START_CONTRACT: VoiceDesignCommand
#   PURPOSE: Represent a voice-design synthesis request using a natural language voice description.
#   INPUTS: { text: str - Source text to synthesize, model: Optional[str] - Optional requested model identifier, save_output: bool - Whether generated audio should be persisted, language: str - Requested language code or auto, voice_description: str - Natural language description of the target voice }
#   OUTPUTS: { instance - Immutable voice design generation command }
#   SIDE_EFFECTS: none
#   LINKS: M-CONTRACTS
# END_CONTRACT: VoiceDesignCommand
@dataclass(frozen=True)
class VoiceDesignCommand(GenerationCommand):
    voice_description: str = ""


# START_CONTRACT: VoiceCloneCommand
#   PURPOSE: Represent a voice-clone synthesis request using reference audio and optional reference text.
#   INPUTS: { text: str - Source text to synthesize, model: Optional[str] - Optional requested model identifier, save_output: bool - Whether generated audio should be persisted, language: str - Requested language code or auto, ref_audio_path: Optional[Path] - Reference audio path, ref_text: Optional[str] - Optional reference transcript }
#   OUTPUTS: { instance - Immutable voice clone generation command }
#   SIDE_EFFECTS: none
#   LINKS: M-CONTRACTS
# END_CONTRACT: VoiceCloneCommand
@dataclass(frozen=True)
class VoiceCloneCommand(GenerationCommand):
    ref_audio_path: Optional[Path] = None
    ref_text: Optional[str] = None


__all__ = [
    "GenerationCommand",
    "CustomVoiceCommand",
    "VoiceDesignCommand",
    "VoiceCloneCommand",
]
