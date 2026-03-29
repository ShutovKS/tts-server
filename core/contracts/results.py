from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AudioResult:
    path: Path
    bytes_data: bytes
    media_type: str = "audio/wav"


@dataclass(frozen=True)
class GenerationResult:
    audio: AudioResult
    saved_path: Optional[Path]
    model: str
    mode: str
    backend: str
