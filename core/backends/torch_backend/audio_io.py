# FILE: core/backends/torch_backend/audio_io.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Persist Torch backend audio outputs to disk and enforce minimum-clone-duration safety.
#   SCOPE: WAV serialization helper, clone-duration guard, shared minimum-duration constant
#   DEPENDS: M-ERRORS
#   LINKS: M-BACKENDS
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   MIN_CLONE_AUDIO_SECONDS - Minimum allowed clone-output duration before the guard rejects.
#   persist_first_wav - Write the first generated waveform into the requested output directory.
#   assert_clone_audio_duration - Raise TTSGenerationError when clone audio is shorter than MIN_CLONE_AUDIO_SECONDS.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Extracted from monolithic torch_backend.py during Phase 1.4 strategy split]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.errors import TTSGenerationError

MIN_CLONE_AUDIO_SECONDS = 1.0


# START_CONTRACT: persist_first_wav
#   PURPOSE: Persist the first waveform from a generation result into the output directory as audio_0001.wav.
#   INPUTS: { backend_key: str - Backend identifier used in failure context, output_dir: Path - Directory to receive the audio file, wavs: list[Any] - Generated waveform sequence, sample_rate: int - Sample rate of the output WAV }
#   OUTPUTS: { None - Writes a single audio file to disk }
#   SIDE_EFFECTS: Creates output_dir if missing and writes audio_0001.wav into it
#   LINKS: M-BACKENDS
# END_CONTRACT: persist_first_wav
def persist_first_wav(
    *,
    backend_key: str,
    output_dir: Path,
    wavs: list[Any],
    sample_rate: int,
) -> None:
    if not wavs:
        raise TTSGenerationError(
            "Torch backend returned empty audio result",
            details={"backend": backend_key, "failure_kind": "empty_audio"},
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "audio_0001.wav"
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise TTSGenerationError(
            str(exc),
            details={
                "backend": backend_key,
                "failure_kind": "audio_write_dependency_missing",
                "output_path": str(target),
                "runtime_dependency": "soundfile",
            },
        ) from exc

    try:
        sf.write(target, wavs[0], sample_rate)
    except Exception as exc:  # pragma: no cover
        raise TTSGenerationError(
            str(exc),
            details={
                "backend": backend_key,
                "failure_kind": "audio_write_failed",
                "output_path": str(target),
            },
        ) from exc


# START_CONTRACT: assert_clone_audio_duration
#   PURPOSE: Reject clone outputs that fall under the minimum-duration threshold so misleading near-empty WAVs are not surfaced as success.
#   INPUTS: { backend_key: str - Backend identifier for failure context, wavs: list[Any] - Generated waveform sequence, sample_rate: int - Audio sample rate, text: str - Input text used for synthesis, family: str - Family key associated with the runtime model, ref_audio_path: Path - Reference audio used for the clone, ref_text: str | None - Optional reference transcript }
#   OUTPUTS: { None - Returns when the duration is acceptable }
#   SIDE_EFFECTS: Raises TTSGenerationError when the duration falls below MIN_CLONE_AUDIO_SECONDS
#   LINKS: M-BACKENDS
# END_CONTRACT: assert_clone_audio_duration
def assert_clone_audio_duration(
    *,
    backend_key: str,
    wavs: list[Any],
    sample_rate: int,
    text: str,
    family: str,
    ref_audio_path: Path,
    ref_text: str | None,
) -> None:
    if not wavs:
        return
    first_wav = np.asarray(wavs[0])
    frame_count = 0 if first_wav.ndim == 0 else int(first_wav.shape[0])
    duration_seconds = frame_count / float(sample_rate) if sample_rate > 0 else 0.0
    if duration_seconds >= MIN_CLONE_AUDIO_SECONDS:
        return
    raise TTSGenerationError(
        "Voice clone synthesis produced implausibly short audio",
        details={
            "backend": backend_key,
            "family": family,
            "failure_kind": "clone_audio_too_short",
            "duration_seconds": round(duration_seconds, 3),
            "minimum_expected_seconds": MIN_CLONE_AUDIO_SECONDS,
            "text_length": len(text),
            "ref_audio_path": str(ref_audio_path),
            "ref_text_provided": ref_text is not None,
            "hint": "Use a clean spoken reference clip and provide ref_text only when it exactly matches the reference audio transcript.",
        },
    )


__all__ = [
    "MIN_CLONE_AUDIO_SECONDS",
    "assert_clone_audio_duration",
    "persist_first_wav",
]
