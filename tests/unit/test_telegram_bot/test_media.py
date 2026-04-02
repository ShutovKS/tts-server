"""
Tests for telegram_bot.media module - Voice Cloning media pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from telegram_bot.media import (
    MediaType,
    MediaValidationResult,
    StagedMedia,
    validate_telegram_media,
    get_telegram_media_type,
    get_telegram_content_type,
    get_telegram_file_id,
    get_telegram_file_size,
    get_telegram_file_name,
    DownloadError,
    MediaValidationError,
    convert_audio_to_wav_if_needed,
    stage_clone_media,
    download_telegram_media,
    _ALLOWED_CLONE_UPLOAD_CONTENT_TYPES,
    _ALLOWED_CLONE_UPLOAD_SUFFIXES,
)


class TestMediaTypeEnum:
    """Tests for MediaType enum."""

    def test_media_type_values(self):
        """Verify MediaType enum has expected values."""
        assert MediaType.VOICE.value == "voice"
        assert MediaType.AUDIO.value == "audio"
        assert MediaType.DOCUMENT.value == "document"
        assert MediaType.UNKNOWN.value == "unknown"


class TestStagedMedia:
    """Tests for StagedMedia dataclass."""

    def test_staged_media_initialization(self):
        """Test StagedMedia can be created with required fields."""
        original_path = Path("/tmp/test.wav")
        staged = StagedMedia(
            original_path=original_path,
            converted_path=None,
            was_converted=False,
            is_wav=True,
        )
        assert staged.original_path == original_path
        assert staged.converted_path is None
        assert staged.was_converted is False
        assert staged.is_wav is True
        assert staged.get_audio_path() == original_path

    def test_staged_media_cleanup(self, tmp_path):
        """Test StagedMedia cleanup removes original file."""
        # Create a test file
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"test audio data")

        staged = StagedMedia(
            original_path=test_file,
            converted_path=None,
            was_converted=False,
            is_wav=True,
        )

        # Verify file exists before cleanup
        assert test_file.exists()

        # Cleanup
        staged.cleanup()

        # Verify file is removed
        assert not test_file.exists()

    def test_staged_media_with_converted_file(self, tmp_path):
        """Test StagedMedia returns converted path when was_converted=True."""
        original_path = tmp_path / "original.mp3"
        converted_path = tmp_path / "converted.wav"

        original_path.write_bytes(b"original")
        converted_path.write_bytes(b"converted")

        staged = StagedMedia(
            original_path=original_path,
            converted_path=converted_path,
            was_converted=True,
            is_wav=True,
        )

        # Should return converted path
        assert staged.get_audio_path() == converted_path
        assert staged.was_converted is True


class TestAllowedMediaConstants:
    """Tests for media type constants aligned with HTTP API."""

    def test_allowed_content_types_defined(self):
        """Verify allowed content types are defined and non-empty."""
        assert len(_ALLOWED_CLONE_UPLOAD_CONTENT_TYPES) > 0
        assert "audio/wav" in _ALLOWED_CLONE_UPLOAD_CONTENT_TYPES
        assert "audio/mpeg" in _ALLOWED_CLONE_UPLOAD_CONTENT_TYPES
        assert "audio/flac" in _ALLOWED_CLONE_UPLOAD_CONTENT_TYPES
        assert "audio/ogg" in _ALLOWED_CLONE_UPLOAD_CONTENT_TYPES

    def test_allowed_suffixes_defined(self):
        """Verify allowed file suffixes are defined and non-empty."""
        assert len(_ALLOWED_CLONE_UPLOAD_SUFFIXES) > 0
        assert ".wav" in _ALLOWED_CLONE_UPLOAD_SUFFIXES
        assert ".mp3" in _ALLOWED_CLONE_UPLOAD_SUFFIXES
        assert ".flac" in _ALLOWED_CLONE_UPLOAD_SUFFIXES
        assert ".ogg" in _ALLOWED_CLONE_UPLOAD_SUFFIXES


class TestGetTelegramMediaType:
    """Tests for get_telegram_media_type function."""

    def test_get_media_type_voice(self):
        """Test voice message detection."""
        message = {"voice": {"file_id": "abc123", "mime_type": "audio/ogg"}}
        assert get_telegram_media_type(message) == MediaType.VOICE

    def test_get_media_type_audio(self):
        """Test audio message detection."""
        message = {"audio": {"file_id": "def456", "mime_type": "audio/mpeg"}}
        assert get_telegram_media_type(message) == MediaType.AUDIO

    def test_get_media_type_document(self):
        """Test document message detection."""
        message = {"document": {"file_id": "ghi789", "mime_type": "audio/flac"}}
        assert get_telegram_media_type(message) == MediaType.DOCUMENT

    def test_get_media_type_no_audio(self):
        """Test message without audio returns UNKNOWN."""
        message = {"text": "Hello"}
        assert get_telegram_media_type(message) == MediaType.UNKNOWN

    def test_get_media_type_empty_message(self):
        """Test empty message returns UNKNOWN."""
        message = {}
        assert get_telegram_media_type(message) == MediaType.UNKNOWN


class TestGetTelegramContentType:
    """Tests for get_telegram_content_type function."""

    def test_content_type_from_voice(self):
        """Test extracting content type from voice message."""
        message = {"voice": {"mime_type": "audio/ogg"}}
        assert get_telegram_content_type(message, MediaType.VOICE) == "audio/ogg"

    def test_content_type_from_audio(self):
        """Test extracting content type from audio message."""
        message = {"audio": {"mime_type": "audio/mpeg"}}
        assert get_telegram_content_type(message, MediaType.AUDIO) == "audio/mpeg"

    def test_content_type_from_document(self):
        """Test extracting content type from document."""
        message = {"document": {"mime_type": "audio/flac"}}
        assert get_telegram_content_type(message, MediaType.DOCUMENT) == "audio/flac"

    def test_content_type_none_for_unknown(self):
        """Test unknown media type returns None."""
        message = {}
        assert get_telegram_content_type(message, MediaType.UNKNOWN) is None


class TestGetTelegramFileId:
    """Tests for get_telegram_file_id function."""

    def test_file_id_from_voice(self):
        """Test extracting file_id from voice message."""
        message = {"voice": {"file_id": "voice_file_id_123"}}
        assert get_telegram_file_id(message, MediaType.VOICE) == "voice_file_id_123"

    def test_file_id_from_audio(self):
        """Test extracting file_id from audio message."""
        message = {"audio": {"file_id": "audio_file_id_456"}}
        assert get_telegram_file_id(message, MediaType.AUDIO) == "audio_file_id_456"

    def test_file_id_from_document(self):
        """Test extracting file_id from document."""
        message = {"document": {"file_id": "doc_file_id_789"}}
        assert get_telegram_file_id(message, MediaType.DOCUMENT) == "doc_file_id_789"

    def test_file_id_none_for_unknown(self):
        """Test unknown media type returns None."""
        message = {}
        assert get_telegram_file_id(message, MediaType.UNKNOWN) is None


class TestGetTelegramFileSize:
    """Tests for get_telegram_file_size function."""

    def test_file_size_from_voice(self):
        """Test extracting file_size from voice message."""
        message = {"voice": {"file_size": 12345}}
        assert get_telegram_file_size(message, MediaType.VOICE) == 12345

    def test_file_size_from_audio(self):
        """Test extracting file_size from audio message."""
        message = {"audio": {"file_size": 67890}}
        assert get_telegram_file_size(message, MediaType.AUDIO) == 67890

    def test_file_size_from_document(self):
        """Test extracting file_size from document."""
        message = {"document": {"file_size": 111213}}
        assert get_telegram_file_size(message, MediaType.DOCUMENT) == 111213

    def test_file_size_zero_for_unknown(self):
        """Test unknown media type returns 0."""
        message = {}
        assert get_telegram_file_size(message, MediaType.UNKNOWN) == 0

    def test_file_size_handles_none(self):
        """Test file_size handles None values."""
        message = {"voice": {}}
        assert get_telegram_file_size(message, MediaType.VOICE) == 0


class TestGetTelegramFileName:
    """Tests for get_telegram_file_name function."""

    def test_file_name_from_document(self):
        """Test extracting file_name from document."""
        message = {"document": {"file_name": "my_voice.mp3"}}
        assert get_telegram_file_name(message, MediaType.DOCUMENT) == "my_voice.mp3"

    def test_file_name_none_for_voice(self):
        """Test voice messages don't have file_name."""
        message = {"voice": {}}
        assert get_telegram_file_name(message, MediaType.VOICE) is None

    def test_file_name_none_for_audio(self):
        """Test audio messages don't have file_name."""
        message = {"audio": {}}
        assert get_telegram_file_name(message, MediaType.AUDIO) is None


class TestValidateTelegramMedia:
    """Tests for validate_telegram_media function."""

    def test_validate_supported_content_type(self):
        """Test validation passes for supported audio/ogg."""
        message = {
            "voice": {
                "file_id": "abc123",
                "mime_type": "audio/ogg",
                "file_size": 1024,
            }
        }
        result = validate_telegram_media(message, max_size_bytes=10 * 1024 * 1024)

        assert result.is_valid is True
        assert result.media_type == MediaType.VOICE
        assert result.content_type == "audio/ogg"
        assert result.file_size == 1024
        assert result.error_message is None

    def test_validate_audio_mpeg(self):
        """Test validation passes for audio/mpeg."""
        message = {
            "audio": {
                "file_id": "def456",
                "mime_type": "audio/mpeg",
                "file_size": 1024,
            }
        }
        result = validate_telegram_media(message, max_size_bytes=10 * 1024 * 1024)

        assert result.is_valid is True
        assert result.media_type == MediaType.AUDIO
        assert result.content_type == "audio/mpeg"

    def test_validate_unsupported_content_type(self):
        """Test validation fails for unsupported content type."""
        message = {
            "document": {
                "file_id": "ghi789",
                "mime_type": "image/png",  # Not audio
                "file_size": 1024,
            }
        }
        result = validate_telegram_media(message, max_size_bytes=10 * 1024 * 1024)

        assert result.is_valid is False
        assert "Unsupported media format" in result.error_message

    def test_validate_file_too_large(self):
        """Test validation fails for oversized file."""
        message = {
            "voice": {
                "file_id": "abc123",
                "mime_type": "audio/ogg",
                "file_size": 20 * 1024 * 1024,  # 20 MB
            }
        }
        result = validate_telegram_media(message, max_size_bytes=10 * 1024 * 1024)

        assert result.is_valid is False
        assert "File too large" in result.error_message
        assert result.media_type == MediaType.VOICE

    def test_validate_exactly_at_limit(self):
        """Test validation passes when file size is exactly at limit."""
        message = {
            "voice": {
                "file_id": "abc123",
                "mime_type": "audio/ogg",
                "file_size": 10 * 1024 * 1024,  # Exactly 10 MB
            }
        }
        result = validate_telegram_media(message, max_size_bytes=10 * 1024 * 1024)

        assert result.is_valid is True

    def test_validate_no_media(self):
        """Test validation fails when no media in message."""
        message = {"text": "Hello"}
        result = validate_telegram_media(message, max_size_bytes=10 * 1024 * 1024)

        assert result.is_valid is False
        assert "No supported media found" in result.error_message
        assert result.media_type == MediaType.UNKNOWN

    def test_validate_voice_without_mime_type(self):
        """Test voice messages without mime type are valid (Telegram default is ogg)."""
        message = {
            "voice": {
                "file_id": "abc123",
                # No mime_type - common for voice messages
                "file_size": 1024,
            }
        }
        result = validate_telegram_media(message, max_size_bytes=10 * 1024 * 1024)

        # Voice messages without explicit mime type should be valid
        assert result.is_valid is True
        assert result.media_type == MediaType.VOICE


class TestMediaValidationEdgeCases:
    """Edge case tests for media validation."""

    def test_validate_none_content_type(self):
        """Test validation handles None content type with valid suffix."""
        message = {
            "document": {
                "file_id": "abc123",
                # No mime_type
                "file_name": "voice.wav",  # Valid suffix
                "file_size": 1024,
            }
        }
        result = validate_telegram_media(message, max_size_bytes=10 * 1024 * 1024)

        # Should be valid due to .wav suffix
        assert result.is_valid is True

    def test_validate_zero_file_size(self):
        """Test validation handles zero file size."""
        message = {
            "voice": {
                "file_id": "abc123",
                "mime_type": "audio/ogg",
                "file_size": 0,
            }
        }
        result = validate_telegram_media(message, max_size_bytes=10 * 1024 * 1024)

        # Zero size should pass validation (other checks would catch empty download)
        assert result.is_valid is True

    def test_validate_negative_file_size(self):
        """Test validation handles negative file size."""
        message = {
            "voice": {
                "file_id": "abc123",
                "mime_type": "audio/ogg",
                "file_size": -100,
            }
        }
        result = validate_telegram_media(message, max_size_bytes=10 * 1024 * 1024)

        # Should pass because -100 <= 10MB
        assert result.is_valid is True


class TestDownloadTelegramMedia:
    """Tests for download_telegram_media function."""

    @pytest.mark.asyncio
    async def test_download_success(self, tmp_path):
        """Test successful media download."""
        mock_client = AsyncMock()

        message = {
            "voice": {
                "file_id": "test_file_id",
                "mime_type": "audio/ogg",
            }
        }

        # Mock download_file to create the file
        async def mock_download(file_id, dest):
            dest.write_bytes(b"fake audio data")

        mock_client.download_file = mock_download

        result_path, content_type = await download_telegram_media(
            client=mock_client,
            message=message,
            media_type=MediaType.VOICE,
            staging_dir=tmp_path,
        )

        assert result_path.parent == tmp_path
        assert result_path.suffix == ".ogg"
        assert content_type == "audio/ogg"
        # Verify file was created
        assert result_path.exists()
        assert result_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_download_file_not_found(self, tmp_path):
        """Test download raises error when file_id missing."""
        mock_client = AsyncMock()

        # Message without file_id
        message = {"voice": {}}

        with pytest.raises(DownloadError, match="Could not extract file_id"):
            await download_telegram_media(
                client=mock_client,
                message=message,
                media_type=MediaType.VOICE,
                staging_dir=tmp_path,
            )

    @pytest.mark.asyncio
    async def test_download_no_audio_in_message(self, tmp_path):
        """Test download raises error when no audio in message."""
        mock_client = AsyncMock()
        message = {"text": "Hello"}

        with pytest.raises(DownloadError, match="Could not extract file_id"):
            await download_telegram_media(
                client=mock_client,
                message=message,
                media_type=MediaType.UNKNOWN,
                staging_dir=tmp_path,
            )

    @pytest.mark.asyncio
    async def test_download_staging_directory_structure(self, tmp_path):
        """Test download creates proper file naming."""
        mock_client = AsyncMock()

        message = {
            "document": {
                "file_id": "doc123",
                "mime_type": "audio/mpeg",
                "file_name": "my_voice_sample.mp3",
            }
        }

        # Mock download_file to create the file
        async def mock_download(file_id, dest):
            dest.write_bytes(b"fake audio data")

        mock_client.download_file = mock_download

        result_path, _ = await download_telegram_media(
            client=mock_client,
            message=message,
            media_type=MediaType.DOCUMENT,
            staging_dir=tmp_path,
        )

        assert result_path.name.startswith("clone_ref_")
        assert result_path.suffix == ".mp3"
        # Verify file was created
        assert result_path.exists()


class TestConversionToWav:
    """Tests for convert_audio_to_wav_if_needed function."""

    def _create_valid_wav(self, path: Path, duration_frames: int = 1000):
        """Helper to create a valid WAV file for testing."""
        import wave
        import struct
        
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            # Write silence - 16-bit samples
            wav.writeframes(b"\x00\x00" * duration_frames)

    def test_convert_wav_to_wav(self, tmp_path):
        """Test WAV files are not converted when output_format is wav."""
        # Create a valid WAV file
        wav_file = tmp_path / "test.wav"
        self._create_valid_wav(wav_file, duration_frames=1000)

        mock_settings = MagicMock()
        mock_settings.output_format = "wav"
        mock_settings.sample_rate = 24000

        result_path, was_converted = convert_audio_to_wav_if_needed(
            wav_file, mock_settings
        )

        # When output_format is wav and file is valid WAV, should not convert
        assert result_path == wav_file
        assert was_converted is False

    def test_convert_non_wav_to_wav(self, tmp_path):
        """Test non-WAV files trigger conversion attempt."""
        # Create a valid WAV file but with .mp3 extension to test conversion path
        mp3_file = tmp_path / "test.mp3"
        self._create_valid_wav(mp3_file, duration_frames=1000)

        mock_settings = MagicMock()
        mock_settings.output_format = "wav"
        mock_settings.sample_rate = 24000

        # The conversion should attempt to convert .mp3 to .wav
        # Result depends on ffmpeg availability
        try:
            result_path, was_converted = convert_audio_to_wav_if_needed(
                mp3_file, mock_settings
            )
            # If ffmpeg is available, converts to wav
            assert result_path.suffix == ".wav"
            assert was_converted is True
        except Exception as exc:
            # If ffmpeg is not available or fails, error is acceptable in test
            assert "ffmpeg" in str(exc).lower() or "conversion failed" in str(exc).lower()

    def test_convert_to_wav_nonexistent_file(self, tmp_path):
        """Test conversion handles nonexistent file gracefully."""
        nonexistent = tmp_path / "does_not_exist.mp3"

        mock_settings = MagicMock()
        mock_settings.output_format = "wav"
        mock_settings.sample_rate = 24000

        # Should handle gracefully - raise AudioConversionError
        with pytest.raises(Exception):  # Could be FileNotFoundError or AudioConversionError
            convert_audio_to_wav_if_needed(
                nonexistent, mock_settings
            )


class TestStageCloneMedia:
    """Tests for stage_clone_media function."""

    @pytest.mark.asyncio
    async def test_stage_valid_media(self, tmp_path):
        """Test staging valid WAV media."""
        mock_client = AsyncMock()

        # Create a fake downloaded file - use WAV format
        async def mock_download(file_id, dest):
            # Create a minimal valid WAV file
            import wave
            with wave.open(str(dest), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b"\x00" * 1000)

        mock_client.download_file = mock_download

        message = {
            "voice": {
                "file_id": "voice123",
                "mime_type": "audio/ogg",
                "file_size": 1024,
            }
        }

        mock_settings = MagicMock()
        mock_settings.max_upload_size_bytes = 10 * 1024 * 1024
        mock_settings.output_format = "wav"
        mock_settings.sample_rate = 24000

        staged_media, validation = await stage_clone_media(
            client=mock_client,
            message=message,
            settings=mock_settings,
            staging_dir=tmp_path,
        )

        assert validation.is_valid is True
        assert staged_media.original_path.exists()

    @pytest.mark.asyncio
    async def test_stage_invalid_media_raises(self, tmp_path):
        """Test staging invalid media raises MediaValidationError."""
        mock_client = AsyncMock()

        message = {"text": "Hello"}  # No audio

        mock_settings = MagicMock()
        mock_settings.max_upload_size_bytes = 10 * 1024 * 1024

        with pytest.raises(MediaValidationError, match="No supported media found"):
            await stage_clone_media(
                client=mock_client,
                message=message,
                settings=mock_settings,
                staging_dir=tmp_path,
            )

    @pytest.mark.asyncio
    async def test_stage_oversized_file_raises(self, tmp_path):
        """Test staging oversized file raises error."""
        mock_client = AsyncMock()

        message = {
            "voice": {
                "file_id": "voice123",
                "mime_type": "audio/ogg",
                "file_size": 20 * 1024 * 1024,  # 20 MB
            }
        }

        mock_settings = MagicMock()
        mock_settings.max_upload_size_bytes = 10 * 1024 * 1024  # 10 MB limit

        with pytest.raises(MediaValidationError, match="File too large"):
            await stage_clone_media(
                client=mock_client,
                message=message,
                settings=mock_settings,
                staging_dir=tmp_path,
            )


class TestMediaValidationResult:
    """Tests for MediaValidationResult dataclass."""

    def test_valid_result_creation(self):
        """Test creating valid validation result."""
        result = MediaValidationResult(
            is_valid=True,
            media_type=MediaType.VOICE,
            content_type="audio/ogg",
            file_size=1024,
        )

        assert result.is_valid is True
        assert result.media_type == MediaType.VOICE
        assert result.content_type == "audio/ogg"
        assert result.file_size == 1024
        assert result.error_message is None

    def test_invalid_result_creation(self):
        """Test creating invalid validation result."""
        result = MediaValidationResult(
            is_valid=False,
            media_type=MediaType.UNKNOWN,
            error_message="Unsupported format",
        )

        assert result.is_valid is False
        assert result.error_message == "Unsupported format"
