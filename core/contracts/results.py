# FILE: core/contracts/results.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define result types for completed TTS operations.
#   SCOPE: GenerationResult and AudioData types
#   DEPENDS: none
#   LINKS: M-CONTRACTS
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   AudioResult - Container for generated audio bytes and source path
#   GenerationResult - Result of a successful generation
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioResult:
    path: Path
    bytes_data: bytes
    media_type: str = "audio/wav"


# START_CONTRACT: GenerationResult
#   PURPOSE: Represent a successful synthesis result with generated audio and backend metadata.
#   INPUTS: { audio: AudioResult - Generated audio artifact, saved_path: Optional[Path] - Persisted output path when saved, model: str - Model identifier used for synthesis, mode: str - Synthesis mode, backend: str - Backend key that produced the result }
#   OUTPUTS: { instance - Immutable generation result }
#   SIDE_EFFECTS: none
#   LINKS: M-CONTRACTS
# END_CONTRACT: GenerationResult
@dataclass(frozen=True)
class GenerationResult:
    audio: AudioResult
    saved_path: Path | None
    model: str
    mode: str
    backend: str


__all__ = [
    "AudioResult",
    "GenerationResult",
]
