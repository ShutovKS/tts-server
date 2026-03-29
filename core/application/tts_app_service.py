from __future__ import annotations

from dataclasses import dataclass

from core.contracts.commands import CustomVoiceCommand, VoiceCloneCommand, VoiceDesignCommand
from core.contracts.results import GenerationResult
from core.services.tts_service import TTSService


@dataclass(frozen=True)
class TTSApplicationService:
    tts_service: TTSService

    def synthesize_custom(self, command: CustomVoiceCommand) -> GenerationResult:
        return self.tts_service.synthesize_custom(command)

    def synthesize_design(self, command: VoiceDesignCommand) -> GenerationResult:
        return self.tts_service.synthesize_design(command)

    def synthesize_clone(self, command: VoiceCloneCommand) -> GenerationResult:
        return self.tts_service.synthesize_clone(command)
