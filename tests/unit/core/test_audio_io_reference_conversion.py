# FILE: tests/unit/core/test_audio_io_reference_conversion.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Verify reference-audio normalization enforces the clone-runtime WAV contract.
#   SCOPE: compatibility detection for already-correct WAV inputs and ffmpeg normalization for incompatible WAV references
#   DEPENDS: M-INFRASTRUCTURE, M-CONFIG
#   LINKS: V-M-INFRASTRUCTURE
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _write_wav - Create a deterministic WAV fixture with the requested format.
#   test_convert_audio_to_wav_if_needed_keeps_matching_reference_wav - Verifies compatible mono PCM16 WAV references are reused as-is.
#   test_convert_audio_to_wav_if_needed_reencodes_incompatible_wav - Verifies incompatible WAV references are normalized through ffmpeg into mono PCM16 at the configured sample rate.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added coverage for strict WAV compatibility detection and normalization for clone reference audio]
# END_CHANGE_SUMMARY

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from core.config import CoreSettings
from core.infrastructure.audio_io import convert_audio_to_wav_if_needed


pytestmark = pytest.mark.unit


def _write_wav(path: Path, *, sample_rate: int, channels: int, sample_width: int) -> None:
    frames = sample_rate // 10
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        payload = bytearray()
        for index in range(frames):
            sample = int(12000 * math.sin(2 * math.pi * 440 * index / sample_rate))
            encoded = struct.pack("<h", sample)
            if sample_width == 1:
                encoded = bytes([(sample // 256) + 128])
            for _ in range(channels):
                payload.extend(encoded)
        wav_file.writeframes(bytes(payload))


def test_convert_audio_to_wav_if_needed_keeps_matching_reference_wav(tmp_path: Path):
    settings = CoreSettings(sample_rate=24000, models_dir=tmp_path / "models", outputs_dir=tmp_path / "outputs", voices_dir=tmp_path / "voices")
    source = tmp_path / "matching.wav"
    _write_wav(source, sample_rate=24000, channels=1, sample_width=2)

    converted_path, converted = convert_audio_to_wav_if_needed(source, settings)

    assert converted is False
    assert converted_path == source


def test_convert_audio_to_wav_if_needed_reencodes_incompatible_wav(tmp_path: Path):
    settings = CoreSettings(sample_rate=24000, models_dir=tmp_path / "models", outputs_dir=tmp_path / "outputs", voices_dir=tmp_path / "voices")
    source = tmp_path / "stereo_44100.wav"
    _write_wav(source, sample_rate=44100, channels=2, sample_width=2)

    converted_path, converted = convert_audio_to_wav_if_needed(source, settings)

    assert converted is True
    assert converted_path != source
    with wave.open(str(converted_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 24000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getcomptype() == "NONE"
