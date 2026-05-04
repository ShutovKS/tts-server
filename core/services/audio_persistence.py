# FILE: core/services/audio_persistence.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Persist generated synthesis audio artifacts for legacy backend and TTSEngine execution paths outside the synthesis coordinator.
#   SCOPE: AudioPersistenceService plus a small persisted-audio DTO for backend output directories and in-memory engine buffers
#   DEPENDS: M-CONFIG, M-CONTRACTS, M-INFRASTRUCTURE, M-MODELS
#   LINKS: M-TTS-COORDINATOR, M-INFRASTRUCTURE
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   PersistedAudio - Generated audio artifact plus optional saved-output path
#   AudioPersistenceService - Shared persistence helper for backend output dirs and engine audio buffers
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Extracted generated-audio persistence from SynthesisCoordinator]
# END_CHANGE_SUMMARY

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.config import CoreSettings
from core.contracts.results import AudioResult, GenerationResult
from core.models.catalog import ModelSpec


# START_CONTRACT: PersistedAudio
#   PURPOSE: Carry generated audio and the optional durable output path produced by save_output.
#   INPUTS: { audio: AudioResult - Generated audio artifact, saved_path: Path | None - Durable output path when persistence is enabled }
#   OUTPUTS: { instance - Persistence result consumed by SynthesisCoordinator }
#   SIDE_EFFECTS: none
#   LINKS: M-TTS-COORDINATOR
# END_CONTRACT: PersistedAudio
@dataclass(frozen=True)
class PersistedAudio:
    audio: AudioResult
    saved_path: Path | None


# START_CONTRACT: AudioPersistenceService
#   PURPOSE: Convert generated backend/engine artifacts into AudioResult plus optional saved output without mixing persistence into orchestration.
#   INPUTS: { settings: CoreSettings - Runtime output configuration }
#   OUTPUTS: { instance - Persistence helper }
#   SIDE_EFFECTS: none on construction
#   LINKS: M-TTS-COORDINATOR, M-INFRASTRUCTURE
# END_CONTRACT: AudioPersistenceService
class AudioPersistenceService:
    def __init__(self, settings: CoreSettings) -> None:
        self._settings = settings

    # START_CONTRACT: persist_backend_output
    #   PURPOSE: Read a backend-generated WAV artifact and optionally copy it to the configured outputs directory.
    #   INPUTS: { output_dir: Path - Backend output directory, spec: ModelSpec - Model metadata, text: str - Source text, save_output: bool - Whether durable persistence is requested }
    #   OUTPUTS: { PersistedAudio - Generated audio and optional saved path }
    #   SIDE_EFFECTS: Reads generated WAV and may copy it into settings.outputs_dir
    #   LINKS: M-TTS-COORDINATOR, M-INFRASTRUCTURE
    # END_CONTRACT: persist_backend_output
    def persist_backend_output(
        self,
        *,
        output_dir: Path,
        spec: ModelSpec,
        text: str,
        save_output: bool,
    ) -> PersistedAudio:
        from core.infrastructure.audio_io import read_generated_wav

        audio = read_generated_wav(output_dir)
        return PersistedAudio(
            audio=audio,
            saved_path=self._save_if_requested(audio=audio, spec=spec, text=text, save_output=save_output),
        )

    # START_CONTRACT: persist_engine_buffer
    #   PURPOSE: Materialize an in-memory engine WAV buffer into an output directory and optionally copy it to durable outputs.
    #   INPUTS: { output_dir: Path - Engine output directory, waveform: bytes - WAV bytes produced by TTSEngine, spec: ModelSpec - Model metadata, text: str - Source text, save_output: bool - Whether durable persistence is requested }
    #   OUTPUTS: { PersistedAudio - Generated audio and optional saved path }
    #   SIDE_EFFECTS: Writes audio_0001.wav and may copy it into settings.outputs_dir
    #   LINKS: M-TTS-COORDINATOR, M-INFRASTRUCTURE
    # END_CONTRACT: persist_engine_buffer
    def persist_engine_buffer(
        self,
        *,
        output_dir: Path,
        waveform: bytes,
        spec: ModelSpec,
        text: str,
        save_output: bool,
    ) -> PersistedAudio:
        output_path = output_dir / "audio_0001.wav"
        output_path.write_bytes(waveform)
        audio = AudioResult(path=output_path, bytes_data=waveform)
        return PersistedAudio(
            audio=audio,
            saved_path=self._save_if_requested(audio=audio, spec=spec, text=text, save_output=save_output),
        )

    # START_CONTRACT: materialize_engine_generation
    #   PURPOSE: Own the temporary output-directory lifecycle and convert a generated engine waveform into the final GenerationResult outside TTSService orchestration.
    #   INPUTS: { spec: ModelSpec - Model metadata, text: str - Source text, save_output: bool - Whether durable persistence is requested, backend_key: str - Backend key reported in the final result, generate_waveform: Callable[[Path], bytes] - Callback that produces the processed engine waveform for a temporary output directory }
    #   OUTPUTS: { GenerationResult - Fully materialized generated audio result }
    #   SIDE_EFFECTS: Creates a temporary output directory, writes audio_0001.wav through persist_engine_buffer, and may persist a durable saved output.
    #   LINKS: M-TTS-COORDINATOR, M-INFRASTRUCTURE
    # END_CONTRACT: materialize_engine_generation
    def materialize_engine_generation(
        self,
        *,
        spec: ModelSpec,
        text: str,
        save_output: bool,
        backend_key: str,
        generate_waveform: Callable[[Path], bytes],
    ) -> GenerationResult:
        from core.infrastructure.audio_io import temporary_output_dir

        with temporary_output_dir(prefix="tts_engine_output_") as output_dir:
            output_path = Path(output_dir)
            persisted = self.persist_engine_buffer(
                output_dir=output_path,
                waveform=generate_waveform(output_path),
                spec=spec,
                text=text,
                save_output=save_output,
            )
            return GenerationResult(
                audio=persisted.audio,
                saved_path=persisted.saved_path,
                model=spec.model_id,
                mode=spec.mode,
                backend=backend_key,
            )

    def _save_if_requested(
        self,
        *,
        audio: AudioResult,
        spec: ModelSpec,
        text: str,
        save_output: bool,
    ) -> Path | None:
        if not save_output:
            return None
        from core.infrastructure.audio_io import persist_output

        return persist_output(audio, spec.output_subfolder, text, self._settings)


__all__ = ["AudioPersistenceService", "PersistedAudio"]
