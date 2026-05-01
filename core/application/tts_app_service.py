# FILE: core/application/tts_app_service.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide high-level synthesis facade for transport adapters.
#   SCOPE: TTSApplicationService class delegating to TTSService
#   DEPENDS: M-TTS-SERVICE, M-CONTRACTS
#   LINKS: M-APPLICATION
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   TTSApplicationService - High-level synthesis facade for transport adapters
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import dataclass

from core.contracts.commands import (
    CustomVoiceCommand,
    VoiceCloneCommand,
    VoiceDesignCommand,
)
from core.contracts.results import GenerationResult
from core.services.tts_service import TTSService


# START_CONTRACT: TTSApplicationService
#   PURPOSE: Provide the application-layer synthesis facade used by transport adapters.
#   INPUTS: { tts_service: TTSService - Core synthesis service handling model resolution and inference }
#   OUTPUTS: { instance - Immutable application service facade }
#   SIDE_EFFECTS: none
#   LINKS: M-APPLICATION
# END_CONTRACT: TTSApplicationService
@dataclass(frozen=True)
class TTSApplicationService:
    tts_service: TTSService

    def synthesize_custom(self, command: CustomVoiceCommand) -> GenerationResult:
        return self.tts_service.synthesize_custom(command)

    def synthesize_design(self, command: VoiceDesignCommand) -> GenerationResult:
        return self.tts_service.synthesize_design(command)

    def synthesize_clone(self, command: VoiceCloneCommand) -> GenerationResult:
        return self.tts_service.synthesize_clone(command)


__all__ = [
    "TTSApplicationService",
]
