# FILE: core/services/clone_preprocessing.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Stage and normalize reference audio for clone synthesis outside the synthesis coordinator.
#   SCOPE: CloneReferenceAudioPreprocessor context manager and prepared-reference DTO
#   DEPENDS: M-CONFIG, M-CONTRACTS, M-INFRASTRUCTURE, M-ERRORS
#   LINKS: M-TTS-COORDINATOR, M-INFRASTRUCTURE
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   PreparedCloneReference - Staged and normalized clone reference metadata
#   CloneReferenceAudioPreprocessor - Context-managed clone reference staging and WAV normalization service
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Extracted clone reference staging and WAV conversion from SynthesisCoordinator]
# END_CHANGE_SUMMARY

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from core.config import CoreSettings
from core.contracts.commands import VoiceCloneCommand
from core.errors import TTSGenerationError


# START_CONTRACT: PreparedCloneReference
#   PURPOSE: Describe a staged clone reference audio file after optional WAV normalization.
#   INPUTS: { source_audio: Path - Staged source copy, prepared_audio: Path - WAV-compatible path used for synthesis, converted: bool - Whether conversion created a new WAV file }
#   OUTPUTS: { instance - Prepared reference metadata consumed by SynthesisCoordinator }
#   SIDE_EFFECTS: none
#   LINKS: M-TTS-COORDINATOR
# END_CONTRACT: PreparedCloneReference
@dataclass(frozen=True)
class PreparedCloneReference:
    source_audio: Path
    prepared_audio: Path
    converted: bool


# START_CONTRACT: CloneReferenceAudioPreprocessor
#   PURPOSE: Own clone reference filesystem staging and format normalization while exposing a narrow prepared-reference context.
#   INPUTS: { settings: CoreSettings - Runtime audio conversion settings }
#   OUTPUTS: { instance - Clone reference preprocessing service }
#   SIDE_EFFECTS: none on construction
#   LINKS: M-TTS-COORDINATOR, M-INFRASTRUCTURE
# END_CONTRACT: CloneReferenceAudioPreprocessor
class CloneReferenceAudioPreprocessor:
    def __init__(self, settings: CoreSettings) -> None:
        self._settings = settings

    # START_CONTRACT: prepare
    #   PURPOSE: Stage a clone reference into an isolated temp directory and normalize it to the runtime WAV contract when needed.
    #   INPUTS: { command: VoiceCloneCommand - Clone request carrying ref_audio_path }
    #   OUTPUTS: { Iterator[PreparedCloneReference] - Context-managed prepared reference metadata }
    #   SIDE_EFFECTS: Copies the reference file, may run ffmpeg, and removes the temp directory on exit
    #   LINKS: M-TTS-COORDINATOR, M-INFRASTRUCTURE
    # END_CONTRACT: prepare
    @contextmanager
    def prepare(self, command: VoiceCloneCommand) -> Iterator[PreparedCloneReference]:
        ref_audio_path = command.ref_audio_path
        if ref_audio_path is None:
            raise TTSGenerationError(
                "Reference audio is required for clone synthesis",
                details={"mode": "clone", "reference_audio": None},
            )

        from core.infrastructure.audio_io import convert_audio_to_wav_if_needed, temporary_output_dir

        with temporary_output_dir(prefix="qwen3_tts_clone_input_") as temp_dir:
            source_audio = temp_dir / ref_audio_path.name
            source_audio.write_bytes(ref_audio_path.read_bytes())
            prepared_audio, converted = convert_audio_to_wav_if_needed(source_audio, self._settings)
            try:
                yield PreparedCloneReference(
                    source_audio=source_audio,
                    prepared_audio=prepared_audio,
                    converted=converted,
                )
            finally:
                if converted and prepared_audio.exists():
                    prepared_audio.unlink(missing_ok=True)


__all__ = ["CloneReferenceAudioPreprocessor", "PreparedCloneReference"]
