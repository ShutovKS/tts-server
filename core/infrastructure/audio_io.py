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


@contextmanager
def temporary_output_dir(prefix: str = "qwen3_tts_") -> Iterator[Path]:
    with TemporaryDirectory(prefix=prefix) as temp_dir:
        yield Path(temp_dir)


def convert_audio_to_wav_if_needed(input_path: Path, settings: CoreSettings) -> tuple[Path, bool]:
    if not input_path.exists():
        raise AudioConversionError(f"Reference audio file does not exist: {input_path}")

    if input_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(input_path), "rb") as wav_file:
                if wav_file.getnchannels() > 0:
                    return input_path, False
        except wave.Error:
            pass

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

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise AudioConversionError("ffmpeg is not installed or not available in PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise AudioConversionError(exc.stderr.decode("utf-8", errors="ignore") or "ffmpeg conversion failed") from exc

    return temp_wav, True


def read_generated_wav(output_dir: Path) -> AudioResult:
    wav_files = sorted(output_dir.glob("audio_*.wav"))
    if not wav_files:
        raise AudioArtifactNotFoundError(f"Generated audio file not found in {output_dir}")

    path = wav_files[0]
    return AudioResult(path=path, bytes_data=path.read_bytes())


def persist_output(audio_result: AudioResult, output_subfolder: str, text_snippet: str, settings: CoreSettings) -> Path:
    save_path = settings.outputs_dir / output_subfolder
    save_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%H-%M-%S")
    clean_text = re.sub(r"[^\w\s-]", "", text_snippet)[: settings.filename_max_len].strip().replace(" ", "_") or "audio"
    final_path = save_path / f"{timestamp}_{clean_text}.wav"
    shutil.copy2(audio_result.path, final_path)
    return final_path


def check_ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
