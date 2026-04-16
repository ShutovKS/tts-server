# FILE: core/infrastructure/audio_io.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide audio file I/O utilities including format conversion and output persistence.
#   SCOPE: WAV conversion via ffmpeg, output persistence, temporary directory management
#   DEPENDS: M-CONFIG, M-ERRORS
#   LINKS: M-INFRASTRUCTURE
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   is_wav_reference_compatible - Check whether a WAV reference already matches the runtime clone format contract
#   convert_audio_to_wav_if_needed - Audio format normalization via ffmpeg
#   persist_output - Save generated audio to outputs directory
#   read_generated_wav - Read first WAV file from output directory
#   temporary_output_dir - Context manager for temporary output directories
#   check_ffmpeg_available - Report whether ffmpeg is available in PATH
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

import re
import shutil
import subprocess
import wave
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

from core.config import CoreSettings
from core.contracts.results import AudioResult
from core.errors import AudioArtifactNotFoundError, AudioConversionError


# START_CONTRACT: temporary_output_dir
#   PURPOSE: Create a temporary directory context for intermediate audio inputs or outputs.
#   INPUTS: { prefix: str - Prefix for the temporary directory name }
#   OUTPUTS: { Iterator[Path] - Context-managed temporary directory path }
#   SIDE_EFFECTS: Creates and removes a temporary directory on the local filesystem
#   LINKS: M-INFRASTRUCTURE
# END_CONTRACT: temporary_output_dir
@contextmanager
def temporary_output_dir(prefix: str = "qwen3_tts_") -> Iterator[Path]:
    with TemporaryDirectory(prefix=prefix) as temp_dir:
        yield Path(temp_dir)


# START_CONTRACT: is_wav_reference_compatible
#   PURPOSE: Check whether a WAV reference already matches the runtime clone-input contract.
#   INPUTS: { input_path: Path - Source WAV file to inspect, settings: CoreSettings - Runtime settings providing target sample rate }
#   OUTPUTS: { bool - True when the WAV already matches mono PCM16 at the configured sample rate }
#   SIDE_EFFECTS: none
#   LINKS: M-INFRASTRUCTURE
# END_CONTRACT: is_wav_reference_compatible
def is_wav_reference_compatible(input_path: Path, settings: CoreSettings) -> bool:
    try:
        with wave.open(str(input_path), "rb") as wav_file:
            return (
                wav_file.getnchannels() == 1
                and wav_file.getframerate() == settings.sample_rate
                and wav_file.getsampwidth() == 2
                and wav_file.getcomptype() == "NONE"
            )
    except wave.Error:
        return False


# START_CONTRACT: convert_audio_to_wav_if_needed
#   PURPOSE: Validate a reference audio file and convert it to mono WAV when required.
#   INPUTS: { input_path: Path - Source reference audio file, settings: CoreSettings - Runtime settings providing target sample rate }
#   OUTPUTS: { tuple[Path, bool] - Prepared WAV path and a flag indicating whether conversion occurred }
#   SIDE_EFFECTS: May invoke ffmpeg and create a converted WAV file on disk
#   LINKS: M-INFRASTRUCTURE
# END_CONTRACT: convert_audio_to_wav_if_needed
def convert_audio_to_wav_if_needed(
    input_path: Path, settings: CoreSettings
) -> tuple[Path, bool]:
    # START_BLOCK_CHECK_FORMAT
    if not input_path.exists():
        raise AudioConversionError(f"Reference audio file does not exist: {input_path}")

    if input_path.suffix.lower() == ".wav" and is_wav_reference_compatible(
        input_path, settings
    ):
        return input_path, False
    # END_BLOCK_CHECK_FORMAT

    temp_wav = input_path.parent / f"{input_path.stem}_converted.wav"
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-ar",
        str(settings.sample_rate),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(temp_wav),
    ]

    # START_BLOCK_RUN_FFMPEG
    try:
        subprocess.run(
            command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except FileNotFoundError as exc:
        raise AudioConversionError(
            "ffmpeg is not installed or not available in PATH"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise AudioConversionError(
            exc.stderr.decode("utf-8", errors="ignore") or "ffmpeg conversion failed"
        ) from exc
    # END_BLOCK_RUN_FFMPEG

    # START_BLOCK_VERIFY_OUTPUT
    # END_BLOCK_VERIFY_OUTPUT
    return temp_wav, True


# START_CONTRACT: read_generated_wav
#   PURPOSE: Read the first generated WAV artifact from a backend output directory.
#   INPUTS: { output_dir: Path - Directory containing generated audio artifacts }
#   OUTPUTS: { AudioResult - Generated audio bytes and source path }
#   SIDE_EFFECTS: Reads generated audio bytes from disk
#   LINKS: M-INFRASTRUCTURE
# END_CONTRACT: read_generated_wav
def read_generated_wav(output_dir: Path) -> AudioResult:
    wav_files = sorted(output_dir.glob("audio_*.wav"))
    if not wav_files:
        raise AudioArtifactNotFoundError(
            f"Generated audio file not found in {output_dir}"
        )

    path = wav_files[0]
    return AudioResult(path=path, bytes_data=path.read_bytes())


# START_CONTRACT: persist_output
#   PURPOSE: Persist a generated audio artifact into the configured outputs directory with a readable filename.
#   INPUTS: { audio_result: AudioResult - Generated audio artifact to persist, output_subfolder: str - Relative subdirectory for the persisted file, text_snippet: str - Source text used to derive the filename snippet, settings: CoreSettings - Runtime settings containing output paths and filename policy }
#   OUTPUTS: { Path - Final persisted output path }
#   SIDE_EFFECTS: Creates output directories and copies the generated audio artifact on disk
#   LINKS: M-INFRASTRUCTURE
# END_CONTRACT: persist_output
def persist_output(
    audio_result: AudioResult,
    output_subfolder: str,
    text_snippet: str,
    settings: CoreSettings,
) -> Path:
    save_path = settings.outputs_dir / output_subfolder
    save_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%H-%M-%S")
    clean_text = (
        re.sub(r"[^\w\s-]", "", text_snippet)[: settings.filename_max_len]
        .strip()
        .replace(" ", "_")
        or "audio"
    )
    final_path = save_path / f"{timestamp}_{clean_text}.wav"
    shutil.copy2(audio_result.path, final_path)
    return final_path


def check_ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

__all__ = [
    "temporary_output_dir",
    "convert_audio_to_wav_if_needed",
    "read_generated_wav",
    "persist_output",
    "check_ffmpeg_available",
]
