"""
Telegram media pipeline for Voice Cloning.

This module provides:
- Download of reference media from Telegram messages
- Media type/size validation
- Staging of downloaded files
- Automatic cleanup after processing

Features:
- Content-type validation aligned with HTTP clone API
- Size limits enforcement
- WAV conversion when needed
- Safe cleanup on success/failure
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from core.config import CoreSettings
from core.errors import AudioConversionError

if TYPE_CHECKING:
    from telegram_bot.client import TelegramClient


LOGGER = logging.getLogger(__name__)

# Content types aligned with HTTP clone API (routes_tts.py)
_ALLOWED_CLONE_UPLOAD_CONTENT_TYPES = frozenset(
    {
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/vnd.wave",
        "audio/mpeg",
        "audio/mp3",
        "audio/flac",
        "audio/x-flac",
        "audio/ogg",
        "audio/webm",
        "audio/mp4",
        "audio/x-m4a",
        "video/webm",
        "application/octet-stream",
    }
)

_ALLOWED_CLONE_UPLOAD_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a", ".mp4"})

# Telegram-specific media types
TELEGRAM_VOICE = "voice"
TELEGRAM_AUDIO = "audio"
TELEGRAM_DOCUMENT = "document"


class MediaType(Enum):
    """Telegram media types that can be used for voice cloning."""
    VOICE = "voice"
    AUDIO = "audio"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MediaValidationResult:
    """Result of media validation."""
    is_valid: bool
    media_type: MediaType
    error_message: Optional[str] = None
    content_type: Optional[str] = None
    file_size: int = 0


@dataclass(frozen=True)
class StagedMedia:
    """Staged media file with cleanup capability."""
    original_path: Path
    converted_path: Optional[Path]
    was_converted: bool
    is_wav: bool
    
    def get_audio_path(self) -> Path:
        """Get the audio path to use (converted if available)."""
        if self.converted_path and self.was_converted:
            return self.converted_path
        return self.original_path
    
    def cleanup(self) -> None:
        """Clean up staged files."""
        try:
            if self.original_path.exists():
                self.original_path.unlink()
        except OSError:
            pass
        
        if self.converted_path and self.converted_path.exists():
            try:
                self.converted_path.unlink()
            except OSError:
                pass


def get_telegram_media_type(message: dict) -> MediaType:
    """
    Determine the type of media in a Telegram message.
    
    Args:
        message: Telegram message dict
        
    Returns:
        MediaType enum value
    """
    if "voice" in message and message["voice"]:
        return MediaType.VOICE
    if "audio" in message and message["audio"]:
        return MediaType.AUDIO
    if "document" in message and message["document"]:
        return MediaType.DOCUMENT
    return MediaType.UNKNOWN


def get_telegram_content_type(message: dict, media_type: MediaType) -> Optional[str]:
    """
    Extract content type from Telegram message.
    
    Args:
        message: Telegram message dict
        media_type: Type of media
        
    Returns:
        Content type string or None
    """
    if media_type == MediaType.VOICE:
        voice = message.get("voice", {})
        return voice.get("mime_type")
    elif media_type == MediaType.AUDIO:
        audio = message.get("audio", {})
        return audio.get("mime_type")
    elif media_type == MediaType.DOCUMENT:
        doc = message.get("document", {})
        return doc.get("mime_type")
    return None


def get_telegram_file_id(message: dict, media_type: MediaType) -> Optional[str]:
    """
    Extract file_id from Telegram message.
    
    Args:
        message: Telegram message dict
        media_type: Type of media
        
    Returns:
        File ID string or None
    """
    if media_type == MediaType.VOICE:
        voice = message.get("voice", {})
        return voice.get("file_id")
    elif media_type == MediaType.AUDIO:
        audio = message.get("audio", {})
        return audio.get("file_id")
    elif media_type == MediaType.DOCUMENT:
        doc = message.get("document", {})
        return doc.get("file_id")
    return None


def get_telegram_file_size(message: dict, media_type: MediaType) -> int:
    """
    Extract file size from Telegram message.
    
    Args:
        message: Telegram message dict
        media_type: Type of media
        
    Returns:
        File size in bytes
    """
    if media_type == MediaType.VOICE:
        voice = message.get("voice", {})
        return voice.get("file_size", 0) or 0
    elif media_type == MediaType.AUDIO:
        audio = message.get("audio", {})
        return audio.get("file_size", 0) or 0
    elif media_type == MediaType.DOCUMENT:
        doc = message.get("document", {})
        return doc.get("file_size", 0) or 0
    return 0


def get_telegram_file_name(message: dict, media_type: MediaType) -> Optional[str]:
    """
    Extract file name from Telegram message.
    
    Args:
        message: Telegram message dict
        media_type: Type of media
        
    Returns:
        File name string or None
    """
    if media_type == MediaType.DOCUMENT:
        doc = message.get("document", {})
        return doc.get("file_name")
    return None


def validate_telegram_media(message: dict, max_size_bytes: int) -> MediaValidationResult:
    """
    Validate that a Telegram message contains valid clone media.
    
    Args:
        message: Telegram message dict
        max_size_bytes: Maximum allowed file size
        
    Returns:
        MediaValidationResult with validation status
    """
    media_type = get_telegram_media_type(message)
    
    if media_type == MediaType.UNKNOWN:
        return MediaValidationResult(
            is_valid=False,
            media_type=MediaType.UNKNOWN,
            error_message="No supported media found. Send a voice message, audio file, "
                         "or document with audio content.",
        )
    
    content_type = get_telegram_content_type(message, media_type)
    file_size = get_telegram_file_size(message, media_type)
    file_name = get_telegram_file_name(message, media_type)
    
    # Check file size
    if file_size > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        actual_mb = file_size / (1024 * 1024)
        return MediaValidationResult(
            is_valid=False,
            media_type=media_type,
            error_message=f"File too large: {actual_mb:.1f}MB. "
                         f"Maximum size: {max_mb:.1f}MB.",
            content_type=content_type,
            file_size=file_size,
        )
    
    # Validate content type or file extension
    is_valid_type = False
    
    if content_type:
        is_valid_type = content_type.lower() in _ALLOWED_CLONE_UPLOAD_CONTENT_TYPES
    
    # Also check by file extension
    if file_name:
        suffix = Path(file_name).suffix.lower()
        is_valid_type = is_valid_type or suffix in _ALLOWED_CLONE_UPLOAD_SUFFIXES
    
    # For voice messages without explicit mime type, assume they're valid ogg
    if media_type == MediaType.VOICE and content_type is None:
        is_valid_type = True
    
    if not is_valid_type:
        allowed = ", ".join(sorted(_ALLOWED_CLONE_UPLOAD_SUFFIXES))
        return MediaValidationResult(
            is_valid=False,
            media_type=media_type,
            error_message=f"Unsupported media format. "
                         f"Allowed formats: {allowed}",
            content_type=content_type,
            file_size=file_size,
        )
    
    return MediaValidationResult(
        is_valid=True,
        media_type=media_type,
        content_type=content_type,
        file_size=file_size,
    )


async def download_telegram_media(
    client: TelegramClient,
    message: dict,
    media_type: MediaType,
    staging_dir: Path,
) -> tuple[Path, Optional[str]]:
    """
    Download media from Telegram to staging directory.
    
    Args:
        client: Telegram client
        message: Telegram message dict
        media_type: Type of media to download
        staging_dir: Directory for staging files
        
    Returns:
        Tuple of (downloaded_path, content_type)
        
    Raises:
        DownloadError: If download fails
    """
    file_id = get_telegram_file_id(message, media_type)
    file_name = get_telegram_file_name(message, media_type)
    content_type = get_telegram_content_type(message, media_type)
    
    if not file_id:
        raise DownloadError("Could not extract file_id from message")
    
    # Generate unique filename
    suffix = ".audio"
    if file_name:
        suffix = Path(file_name).suffix.lower() or ".audio"
        if suffix not in _ALLOWED_CLONE_UPLOAD_SUFFIXES:
            suffix = ".audio"
    elif content_type:
        mime_to_ext = {
            "audio/wav": ".wav",
            "audio/mpeg": ".mp3",
            "audio/mp3": ".mp3",
            "audio/flac": ".flac",
            "audio/ogg": ".ogg",
            "audio/webm": ".webm",
            "audio/mp4": ".m4a",
        }
        suffix = mime_to_ext.get(content_type.lower(), ".audio")
    
    filename = f"clone_ref_{uuid.uuid4().hex[:8]}{suffix}"
    dest_path = staging_dir / filename
    
    try:
        await client.download_file(file_id, dest_path)
    except Exception as exc:
        raise DownloadError(f"Failed to download media: {exc}") from exc
    
    if not dest_path.exists() or dest_path.stat().st_size == 0:
        raise DownloadError("Downloaded file is empty or missing")
    
    return dest_path, content_type


async def stage_clone_media(
    client: TelegramClient,
    message: dict,
    settings: CoreSettings,
    staging_dir: Path | None = None,
) -> tuple[StagedMedia, MediaValidationResult]:
    """
    Download, validate and stage clone media from Telegram.
    
    This function:
    1. Validates the message contains valid clone media
    2. Downloads the media to a staging directory
    3. Converts to WAV if needed
    4. Returns StagedMedia with cleanup capability
    
    Args:
        client: Telegram client
        message: Telegram message dict (the replied message with media)
        settings: Core settings for conversion
        staging_dir: Optional staging directory (created if not provided)
        
    Returns:
        Tuple of (StagedMedia, MediaValidationResult)
        
    Raises:
        DownloadError: If download fails
        MediaValidationError: If media is invalid
    """
    # Validate media first
    max_size = settings.max_upload_size_bytes
    validation = validate_telegram_media(message, max_size)
    
    if not validation.is_valid:
        raise MediaValidationError(validation.error_message or "Invalid media")
    
    media_type = validation.media_type
    
    # Create staging directory if needed
    if staging_dir is None:
        staging_dir = Path(tempfile.gettempdir()) / f"qwen3_clone_{uuid.uuid4().hex[:8]}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download media
        downloaded_path, content_type = await download_telegram_media(
            client, message, media_type, staging_dir
        )
        
        # Check downloaded size
        actual_size = downloaded_path.stat().st_size
        if actual_size > max_size:
            downloaded_path.unlink(missing_ok=True)
            raise MediaValidationError(
                f"Downloaded file too large: {actual_size / (1024 * 1024):.1f}MB. "
                f"Maximum: {max_size / (1024 * 1024):.1f}MB."
            )
        
        # Try to convert to WAV if needed
        converted_path: Path | None = None
        was_converted = False
        
        try:
            converted_path, was_converted = convert_audio_to_wav_if_needed(
                downloaded_path, settings
            )
        except AudioConversionError as exc:
            # If conversion fails, use original if it's already WAV-compatible
            LOGGER.warning(f"Audio conversion failed, using original: {exc}")
            # Check if original is already WAV
            if downloaded_path.suffix.lower() != ".wav":
                downloaded_path.unlink(missing_ok=True)
                raise MediaValidationError(
                    f"Could not process audio file. "
                    f"Conversion failed: {exc}"
                )
        
        return StagedMedia(
            original_path=downloaded_path,
            converted_path=converted_path,
            was_converted=was_converted,
            is_wav=was_converted or downloaded_path.suffix.lower() == ".wav",
        ), validation
        
    except (DownloadError, MediaValidationError):
        raise
    except Exception as exc:
        raise DownloadError(f"Unexpected error during staging: {exc}") from exc


class DownloadError(Exception):
    """Error downloading media from Telegram."""
    pass


class MediaValidationError(Exception):
    """Error validating media."""
    pass


def convert_audio_to_wav_if_needed(input_path: Path, settings: CoreSettings) -> tuple[Path, bool]:
    """
    Convert audio to WAV format if needed.
    
    Args:
        input_path: Path to input audio file
        settings: Core settings
        
    Returns:
        Tuple of (path, was_converted) where path is the WAV file path
        and was_converted is True if conversion was performed
    """
    import subprocess
    import wave
    
    # Check if already WAV with valid channels
    if input_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(input_path), "rb") as wav_file:
                if wav_file.getnchannels() > 0:
                    return input_path, False
        except wave.Error:
            pass
    
    # Need to convert
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
        return temp_wav, True
    except FileNotFoundError as exc:
        raise AudioConversionError("ffmpeg is not installed or not available in PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise AudioConversionError(exc.stderr.decode("utf-8", errors="ignore") or "ffmpeg conversion failed") from exc
