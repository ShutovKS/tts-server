# FILE: telegram_bot/media.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Handle media extraction from Telegram messages for clone flows.
#   SCOPE: Reference audio download, format detection, temporary file management
#   DEPENDS: M-ERRORS
#   LINKS: M-TELEGRAM
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LOGGER - Module logger for Telegram media processing
#   TELEGRAM_VOICE - Telegram media key for voice attachments
#   TELEGRAM_AUDIO - Telegram media key for audio attachments
#   TELEGRAM_DOCUMENT - Telegram media key for document attachments
#   MediaType - Enum of supported Telegram media categories
#   MediaValidationResult - Validation outcome for Telegram clone media
#   StagedMedia - Downloaded and optionally converted clone media artifact
#   get_telegram_media_type - Detect supported Telegram media kind from a message
#   get_telegram_content_type - Extract MIME type from Telegram media metadata
#   get_telegram_file_id - Extract Telegram file identifier from media metadata
#   get_telegram_file_size - Extract Telegram media file size in bytes
#   get_telegram_file_name - Extract Telegram media file name when present
#   validate_telegram_media - Validate Telegram media for clone workflows
#   download_telegram_media - Download Telegram media into a staging directory
#   stage_clone_media - Download and normalize Telegram clone media
#   DownloadError - Raised when Telegram media download fails
#   MediaValidationError - Raised when Telegram media validation fails
#   convert_audio_to_wav_if_needed - Convert staged media to WAV when required
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, function contracts, semantic blocks, and migrated log events to block-reference format]
# END_CHANGE_SUMMARY

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

_ALLOWED_CLONE_UPLOAD_SUFFIXES = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a", ".mp4"}
)

# Telegram-specific media types
TELEGRAM_VOICE = "voice"
TELEGRAM_AUDIO = "audio"
TELEGRAM_DOCUMENT = "document"


# START_CONTRACT: MediaType
#   PURPOSE: Enumerate Telegram media categories accepted for clone reference audio.
#   INPUTS: {}
#   OUTPUTS: { MediaType - enum of supported Telegram media kinds }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: MediaType
class MediaType(Enum):
    """Telegram media types that can be used for voice cloning."""

    VOICE = "voice"
    AUDIO = "audio"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


# START_CONTRACT: MediaValidationResult
#   PURPOSE: Describe whether a Telegram media attachment is valid for clone workflows.
#   INPUTS: { is_valid: bool - validation result, media_type: MediaType - detected Telegram media kind, error_message: Optional[str] - validation failure detail, content_type: Optional[str] - MIME type, file_size: int - attachment size in bytes }
#   OUTPUTS: { MediaValidationResult - immutable validation result }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: MediaValidationResult
@dataclass(frozen=True)
class MediaValidationResult:
    """Result of media validation."""

    is_valid: bool
    media_type: MediaType
    error_message: Optional[str] = None
    content_type: Optional[str] = None
    file_size: int = 0


# START_CONTRACT: StagedMedia
#   PURPOSE: Represent downloaded clone reference media and its cleanup lifecycle.
#   INPUTS: { original_path: Path - downloaded file path, converted_path: Optional[Path] - normalized WAV path, was_converted: bool - conversion flag, is_wav: bool - readiness indicator }
#   OUTPUTS: { StagedMedia - immutable staged media descriptor }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: StagedMedia
@dataclass(frozen=True)
class StagedMedia:
    """Staged media file with cleanup capability."""

    original_path: Path
    converted_path: Optional[Path]
    was_converted: bool
    is_wav: bool

    # START_CONTRACT: get_audio_path
    #   PURPOSE: Return the best staged audio path for downstream clone synthesis.
    #   INPUTS: {}
    #   OUTPUTS: { Path - converted audio path when available, otherwise original path }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: get_audio_path
    def get_audio_path(self) -> Path:
        """Get the audio path to use (converted if available)."""
        if self.converted_path and self.was_converted:
            return self.converted_path
        return self.original_path

    # START_CONTRACT: cleanup
    #   PURPOSE: Remove temporary staged media files from disk after processing.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Deletes staged audio files from the filesystem when present.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: cleanup
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


# START_CONTRACT: get_telegram_media_type
#   PURPOSE: Detect which supported Telegram media kind is present in a message payload.
#   INPUTS: { message: dict - Telegram message payload }
#   OUTPUTS: { MediaType - detected media kind or UNKNOWN }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: get_telegram_media_type
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


# START_CONTRACT: get_telegram_content_type
#   PURPOSE: Extract the MIME type for supported Telegram media in a message payload.
#   INPUTS: { message: dict - Telegram message payload, media_type: MediaType - selected Telegram media kind }
#   OUTPUTS: { Optional[str] - MIME type when available }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: get_telegram_content_type
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


# START_CONTRACT: get_telegram_file_id
#   PURPOSE: Extract the Telegram file identifier for a supported media attachment.
#   INPUTS: { message: dict - Telegram message payload, media_type: MediaType - selected Telegram media kind }
#   OUTPUTS: { Optional[str] - Telegram file identifier when available }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: get_telegram_file_id
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


# START_CONTRACT: get_telegram_file_size
#   PURPOSE: Read the attachment size from a Telegram media payload.
#   INPUTS: { message: dict - Telegram message payload, media_type: MediaType - selected Telegram media kind }
#   OUTPUTS: { int - media size in bytes }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: get_telegram_file_size
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


# START_CONTRACT: get_telegram_file_name
#   PURPOSE: Extract the original file name from a Telegram document payload when present.
#   INPUTS: { message: dict - Telegram message payload, media_type: MediaType - selected Telegram media kind }
#   OUTPUTS: { Optional[str] - original attachment name when available }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: get_telegram_file_name
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


# START_CONTRACT: validate_telegram_media
#   PURPOSE: Validate Telegram reference media against clone workflow format and size rules.
#   INPUTS: { message: dict - Telegram message payload, max_size_bytes: int - maximum allowed attachment size }
#   OUTPUTS: { MediaValidationResult - structured validation outcome }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: validate_telegram_media
def validate_telegram_media(
    message: dict, max_size_bytes: int
) -> MediaValidationResult:
    """
    Validate that a Telegram message contains valid clone media.

    Args:
        message: Telegram message dict
        max_size_bytes: Maximum allowed file size

    Returns:
        MediaValidationResult with validation status
    """
    # START_BLOCK_DETECT_MEDIA_TYPE
    media_type = get_telegram_media_type(message)

    if media_type == MediaType.UNKNOWN:
        return MediaValidationResult(
            is_valid=False,
            media_type=MediaType.UNKNOWN,
            error_message="No supported media found. Send a voice message, audio file, "
            "or document with audio content.",
        )
    # END_BLOCK_DETECT_MEDIA_TYPE

    # START_BLOCK_VALIDATE_MEDIA_METADATA
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
            error_message=f"Unsupported media format. Allowed formats: {allowed}",
            content_type=content_type,
            file_size=file_size,
        )

    return MediaValidationResult(
        is_valid=True,
        media_type=media_type,
        content_type=content_type,
        file_size=file_size,
    )
    # END_BLOCK_VALIDATE_MEDIA_METADATA


# START_CONTRACT: download_telegram_media
#   PURPOSE: Download Telegram-hosted media to a local staging directory for clone processing.
#   INPUTS: { client: TelegramClient - Telegram API client, message: dict - Telegram message payload, media_type: MediaType - selected media kind, staging_dir: Path - local staging directory }
#   OUTPUTS: { tuple[Path, Optional[str]] - downloaded file path and detected content type }
#   SIDE_EFFECTS: Downloads media to the filesystem via Telegram API calls.
#   LINKS: M-TELEGRAM
# END_CONTRACT: download_telegram_media
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
    # START_BLOCK_RESOLVE_DOWNLOAD_TARGET
    file_id = get_telegram_file_id(message, media_type)
    file_name = get_telegram_file_name(message, media_type)
    content_type = get_telegram_content_type(message, media_type)

    if not file_id:
        raise DownloadError("Could not extract file_id from message")
    # END_BLOCK_RESOLVE_DOWNLOAD_TARGET

    # START_BLOCK_BUILD_DOWNLOAD_PATH
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
    # END_BLOCK_BUILD_DOWNLOAD_PATH

    # START_BLOCK_DOWNLOAD_MEDIA
    try:
        await client.download_file(file_id, dest_path)
    except Exception as exc:
        raise DownloadError(f"Failed to download media: {exc}") from exc

    if not dest_path.exists() or dest_path.stat().st_size == 0:
        raise DownloadError("Downloaded file is empty or missing")

    return dest_path, content_type
    # END_BLOCK_DOWNLOAD_MEDIA


# START_CONTRACT: stage_clone_media
#   PURPOSE: Validate, download, and normalize Telegram clone reference media for synthesis.
#   INPUTS: { client: TelegramClient - Telegram API client, message: dict - Telegram reply payload, settings: CoreSettings - runtime media settings, staging_dir: Path | None - optional staging directory }
#   OUTPUTS: { tuple[StagedMedia, MediaValidationResult] - staged media descriptor and validation outcome }
#   SIDE_EFFECTS: Creates staging directories, downloads files, and may invoke ffmpeg conversion.
#   LINKS: M-TELEGRAM
# END_CONTRACT: stage_clone_media
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
    # START_BLOCK_VALIDATE_STAGE_REQUEST
    # Validate media first
    max_size = settings.max_upload_size_bytes
    validation = validate_telegram_media(message, max_size)

    if not validation.is_valid:
        raise MediaValidationError(validation.error_message or "Invalid media")

    media_type = validation.media_type
    # END_BLOCK_VALIDATE_STAGE_REQUEST

    # START_BLOCK_PREPARE_STAGING_DIRECTORY
    # Create staging directory if needed
    if staging_dir is None:
        staging_dir = (
            Path(tempfile.gettempdir()) / f"qwen3_clone_{uuid.uuid4().hex[:8]}"
        )
    staging_dir.mkdir(parents=True, exist_ok=True)
    # END_BLOCK_PREPARE_STAGING_DIRECTORY

    try:
        # START_BLOCK_DOWNLOAD_MEDIA_FOR_STAGING
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
        # END_BLOCK_DOWNLOAD_MEDIA_FOR_STAGING

        # START_BLOCK_STAGE_CONVERT_TO_WAV
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
                    f"Could not process audio file. Conversion failed: {exc}"
                )

        return StagedMedia(
            original_path=downloaded_path,
            converted_path=converted_path,
            was_converted=was_converted,
            is_wav=was_converted or downloaded_path.suffix.lower() == ".wav",
        ), validation
        # END_BLOCK_STAGE_CONVERT_TO_WAV

    except (DownloadError, MediaValidationError):
        raise
    except Exception as exc:
        raise DownloadError(f"Unexpected error during staging: {exc}") from exc


# START_CONTRACT: DownloadError
#   PURPOSE: Represent failures while downloading Telegram reference media.
#   INPUTS: {}
#   OUTPUTS: { DownloadError - media download exception type }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: DownloadError
class DownloadError(Exception):
    """Error downloading media from Telegram."""

    pass


# START_CONTRACT: MediaValidationError
#   PURPOSE: Represent validation failures for Telegram reference media.
#   INPUTS: {}
#   OUTPUTS: { MediaValidationError - media validation exception type }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: MediaValidationError
class MediaValidationError(Exception):
    """Error validating media."""

    pass


# START_CONTRACT: convert_audio_to_wav_if_needed
#   PURPOSE: Normalize staged clone audio into WAV format when required by downstream synthesis.
#   INPUTS: { input_path: Path - source audio file, settings: CoreSettings - audio normalization settings }
#   OUTPUTS: { tuple[Path, bool] - WAV path and conversion flag }
#   SIDE_EFFECTS: May invoke ffmpeg and create a converted WAV file on disk.
#   LINKS: M-TELEGRAM
# END_CONTRACT: convert_audio_to_wav_if_needed
def convert_audio_to_wav_if_needed(
    input_path: Path, settings: CoreSettings
) -> tuple[Path, bool]:
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

    # START_BLOCK_DETECT_WAV_COMPATIBILITY
    # Check if already WAV with valid channels
    if input_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(input_path), "rb") as wav_file:
                if wav_file.getnchannels() > 0:
                    return input_path, False
        except wave.Error:
            pass
    # END_BLOCK_DETECT_WAV_COMPATIBILITY

    # START_BLOCK_FUNCTION_CONVERT_TO_WAV
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
        subprocess.run(
            command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        return temp_wav, True
    except FileNotFoundError as exc:
        raise AudioConversionError(
            "ffmpeg is not installed or not available in PATH"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise AudioConversionError(
            exc.stderr.decode("utf-8", errors="ignore") or "ffmpeg conversion failed"
        ) from exc
    # END_BLOCK_FUNCTION_CONVERT_TO_WAV

__all__ = [
    "LOGGER",
    "TELEGRAM_VOICE",
    "TELEGRAM_AUDIO",
    "TELEGRAM_DOCUMENT",
    "MediaType",
    "MediaValidationResult",
    "StagedMedia",
    "get_telegram_media_type",
    "get_telegram_content_type",
    "get_telegram_file_id",
    "get_telegram_file_size",
    "get_telegram_file_name",
    "validate_telegram_media",
    "download_telegram_media",
    "stage_clone_media",
    "DownloadError",
    "MediaValidationError",
    "convert_audio_to_wav_if_needed",
]
