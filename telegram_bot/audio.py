"""
Audio conversion utilities for Telegram voice messages.

Telegram's sendVoice API requires:
- OGG format with OPUS codec
- Audio should be mono, typically 16kHz or 48kHz

This module provides conversion from WAV to Telegram-compatible OGG.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Tuple

from core.config import CoreSettings
from core.errors import AudioConversionError


def convert_wav_to_telegram_ogg(
    wav_bytes: bytes,
    settings: CoreSettings,
) -> Tuple[bytes, bool]:
    """
    Convert WAV audio to Telegram-compatible OGG format.
    
    Telegram sendVoice requires:
    - OGG container with OPUS codec
    - Typically mono audio
    - 16kHz or 48kHz sample rate (Telegram auto-converts if needed)
    
    Args:
        wav_bytes: Raw WAV audio bytes
        settings: Core settings with sample_rate
        
    Returns:
        Tuple of (ogg_bytes, was_converted). was_converted is False if input
        was already in valid OGG format (shouldn't happen normally).
        
    Raises:
        AudioConversionError: If conversion fails
    """
    # Check if ffmpeg is available
    if not _check_ffmpeg_available():
        raise AudioConversionError(
            "ffmpeg is not available. Cannot convert audio for Telegram.",
            details={"required_tool": "ffmpeg"},
        )
    
    try:
        # Use ffmpeg to convert WAV to OGG with OPUS codec
        # Telegram accepts OPUS-encoded audio in OGG container
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",  # Overwrite output
                "-v", "error",
                "-f", "wav",  # Input format
                "-i", "pipe:0",  # Read from stdin
                "-c:a", "libopus",  # OPUS codec
                "-b:a", "32k",  # Bitrate
                "-vbr", "on",  # Variable bitrate
                "-application", "voip",  # Optimized for voice
                "-f", "ogg",  # Output format
                "pipe:1",  # Write to stdout
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        ogg_bytes, stderr = process.communicate(input=wav_bytes, timeout=60)
        
        if process.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="ignore") if stderr else "Unknown error"
            raise AudioConversionError(
                f"ffmpeg conversion failed: {error_msg}",
                details={"ffmpeg_stderr": error_msg, "return_code": process.returncode},
            )
        
        if not ogg_bytes:
            raise AudioConversionError("ffmpeg produced empty output")
        
        return ogg_bytes, True
        
    except subprocess.TimeoutExpired:
        raise AudioConversionError("Audio conversion timed out after 60 seconds")
    except FileNotFoundError:
        raise AudioConversionError(
            "ffmpeg is not installed or not available in PATH",
            details={"required_tool": "ffmpeg"},
        )
    except Exception as exc:
        if isinstance(exc, AudioConversionError):
            raise
        raise AudioConversionError(
            f"Audio conversion failed: {exc}",
            details={"error_type": type(exc).__name__},
        ) from exc


def _check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available."""
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
