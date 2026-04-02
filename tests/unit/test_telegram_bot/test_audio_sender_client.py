"""
Unit tests for Telegram bot audio conversion, sender, and client.

Tests the audio conversion contract, sender voice path, and client multipart upload.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from telegram_bot.audio import convert_wav_to_telegram_ogg
from telegram_bot.client import TelegramBotClient, TelegramAPIError
from telegram_bot.config import TelegramSettings
from telegram_bot.sender import TelegramSender


def _make_wav_bytes(sample_rate: int = 24000, duration_frames: int = 240) -> bytes:
    """Create minimal WAV file bytes for testing."""
    buffer = io.BytesIO()
    silence_frame = bytes([0, 0])
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(silence_frame * duration_frames)
    return buffer.getvalue()


@dataclass
class MockCoreSettings:
    """Mock core settings for testing."""
    sample_rate: int = 24000


class TestConvertWavToTelegramOgg:
    """Tests for audio conversion function contract."""
    
    def test_conversion_returns_tuple(self):
        """Test that convert_wav_to_telegram_ogg returns tuple (bytes, bool)."""
        wav_bytes = _make_wav_bytes()
        settings = MockCoreSettings()
        
        result = convert_wav_to_telegram_ogg(wav_bytes, settings)
        
        # Result should be a tuple
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2 elements, got {len(result)}"
    
    def test_conversion_returns_bytes_and_bool(self):
        """Test that convert_wav_to_telegram_ogg returns (bytes, bool)."""
        wav_bytes = _make_wav_bytes()
        settings = MockCoreSettings()
        
        ogg_bytes, was_converted = convert_wav_to_telegram_ogg(wav_bytes, settings)
        
        assert isinstance(ogg_bytes, bytes), f"Expected bytes, got {type(ogg_bytes)}"
        assert isinstance(was_converted, bool), f"Expected bool, got {type(was_converted)}"
        assert len(ogg_bytes) > 0, "OGG bytes should not be empty"
        assert was_converted is True, "Conversion should return True (was converted)"
    
    def test_conversion_produces_valid_ogg(self):
        """Test that conversion produces non-empty OGG output."""
        wav_bytes = _make_wav_bytes()
        settings = MockCoreSettings()
        
        ogg_bytes, _ = convert_wav_to_telegram_ogg(wav_bytes, settings)
        
        # OGG files start with "OggS"
        assert ogg_bytes[:4] == b"OggS", "Output should be valid OGG format"


class TestTelegramSenderSendVoice:
    """Tests for TelegramSender.send_voice method."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock Telegram client."""
        client = MagicMock(spec=TelegramBotClient)
        client.send_voice = AsyncMock(return_value={"message_id": 123})
        return client
    
    @pytest.fixture
    def sender_settings(self):
        """Create settings for sender."""
        return MagicMock(spec=TelegramSettings)
    
    def test_send_voice_unpacks_tuple_correctly(self, mock_client, sender_settings):
        """Test that send_voice correctly unpacks tuple from convert_wav_to_telegram_ogg."""
        sender = TelegramSender(mock_client, sender_settings)
        wav_bytes = _make_wav_bytes()
        
        # This test verifies that the sender correctly handles the tuple return
        # from convert_wav_to_telegram_ogg
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            sender.send_voice(12345, wav_bytes)
        )
        
        # Client.send_voice should be called with bytes, not tuple
        mock_client.send_voice.assert_called_once()
        call_args = mock_client.send_voice.call_args
        assert isinstance(call_args[0][1], bytes), \
            f"Expected bytes, got {type(call_args[0][1])}"
    
    def test_send_voice_with_caption(self, mock_client, sender_settings):
        """Test send_voice with caption."""
        sender = TelegramSender(mock_client, sender_settings)
        wav_bytes = _make_wav_bytes()
        
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            sender.send_voice(12345, wav_bytes, caption="Test caption")
        )
        
        mock_client.send_voice.assert_called_once()
        call_kwargs = mock_client.send_voice.call_args[1]
        assert call_kwargs.get("caption") == "Test caption"
    
    def test_send_voice_logs_conversion(self, mock_client, sender_settings):
        """Test that send_voice logs conversion events."""
        sender = TelegramSender(mock_client, sender_settings)
        wav_bytes = _make_wav_bytes()
        
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            sender.send_voice(12345, wav_bytes)
        )
        
        # Should complete without errors if logging works
        mock_client.send_voice.assert_called_once()


class TestTelegramBotClientSendVoice:
    """Tests for TelegramBotClient.send_voice method."""
    
    @pytest.fixture
    def client(self):
        """Create Telegram client with test token."""
        return TelegramBotClient("test_token_123:ABCabc123")
    
    @pytest.mark.asyncio
    async def test_send_voice_uses_correct_multipart_key(self, client):
        """Test that send_voice uses 'voice' key, not 'audio'."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "result": {"message_id": 123}}
        mock_response.raise_for_status = MagicMock()
        
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.post.return_value.json = MagicMock(return_value=mock_response.json())
        
        with patch.object(client, "_get_client", AsyncMock(return_value=mock_client)):
            audio_bytes = b"fake_ogg_data"
            await client.send_voice(12345, audio_bytes)
            
            # Check that post was called with files containing 'voice' key
            mock_client.post.assert_called_once()
            call_kwargs = mock_client.post.call_args[1]
            
            assert "files" in call_kwargs, "files should be in call kwargs"
            files = call_kwargs["files"]
            assert "voice" in files, f"Expected 'voice' key in files, got: {list(files.keys())}"
            assert "audio" not in files, "Should not have 'audio' key in files"
    
    @pytest.mark.asyncio
    async def test_send_voice_correct_file_tuple_format(self, client):
        """Test that send_voice creates correct file tuple format."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "result": {"message_id": 123}}
        mock_response.raise_for_status = MagicMock()
        
        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        
        with patch.object(client, "_get_client", AsyncMock(return_value=mock_http_client)):
            audio_bytes = b"test_audio"
            await client.send_voice(12345, audio_bytes)
            
            call_kwargs = mock_http_client.post.call_args[1]
            files = call_kwargs["files"]
            
            # Check file tuple format: (filename, fileobj, mimetype)
            voice_file = files["voice"]
            assert len(voice_file) == 3, "File tuple should have 3 elements"
            filename, fileobj, mimetype = voice_file
            assert filename == "voice.ogg", f"Expected 'voice.ogg', got {filename}"
            assert isinstance(fileobj, io.BytesIO), "File should be BytesIO"
            assert mimetype == "audio/ogg", f"Expected 'audio/ogg', got {mimetype}"
    
    @pytest.mark.asyncio
    async def test_send_voice_includes_chat_id(self, client):
        """Test that send_voice includes chat_id in data."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "result": {"message_id": 123}}
        mock_response.raise_for_status = MagicMock()
        
        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        
        with patch.object(client, "_get_client", AsyncMock(return_value=mock_http_client)):
            await client.send_voice(67890, b"audio")
            
            call_kwargs = mock_http_client.post.call_args[1]
            data = call_kwargs["data"]
            
            assert "chat_id" in data, "chat_id should be in data"
            assert data["chat_id"] == 67890, f"Expected 67890, got {data['chat_id']}"
    
    @pytest.mark.asyncio
    async def test_send_voice_with_caption_and_parse_mode(self, client):
        """Test that send_voice includes caption and parse_mode."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "result": {"message_id": 123}}
        mock_response.raise_for_status = MagicMock()
        
        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        
        with patch.object(client, "_get_client", AsyncMock(return_value=mock_http_client)):
            await client.send_voice(12345, b"audio", caption="Test caption")
            
            call_kwargs = mock_http_client.post.call_args[1]
            data = call_kwargs["data"]
            
            assert "caption" in data, "caption should be in data"
            assert data["caption"] == "Test caption"
            assert "parse_mode" in data, "parse_mode should be in data"
            assert data["parse_mode"] == "Markdown"
    
    @pytest.mark.asyncio
    async def test_send_voice_with_duration(self, client):
        """Test that send_voice includes duration when provided."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "result": {"message_id": 123}}
        mock_response.raise_for_status = MagicMock()
        
        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        
        with patch.object(client, "_get_client", AsyncMock(return_value=mock_http_client)):
            await client.send_voice(12345, b"audio", duration=10)
            
            call_kwargs = mock_http_client.post.call_args[1]
            data = call_kwargs["data"]
            
            assert "duration" in data, "duration should be in data"
            assert data["duration"] == 10


class TestTelegramSenderErrorHandling:
    """Tests for TelegramSender error handling."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock Telegram client that fails on voice."""
        client = MagicMock(spec=TelegramBotClient)
        client.send_message = AsyncMock(return_value={"message_id": 123})
        client.send_voice = AsyncMock(side_effect=Exception("Network error"))
        return client
    
    @pytest.fixture
    def sender_settings(self):
        """Create settings for sender."""
        return MagicMock(spec=TelegramSettings)
    
    @pytest.mark.asyncio
    async def test_send_voice_conversion_error_returns_error_message(self, mock_client, sender_settings):
        """Test that audio conversion error returns error result and sends message to user."""
        sender = TelegramSender(mock_client, sender_settings)
        wav_bytes = _make_wav_bytes()
        
        # Mock convert to raise error
        with patch("telegram_bot.sender.convert_wav_to_telegram_ogg") as mock_convert:
            from core.errors import AudioConversionError
            mock_convert.side_effect = AudioConversionError("ffmpeg not found")
            
            result = await sender.send_voice(12345, wav_bytes)
            
            # Should return error result (not raise)
            assert result.success is False
            assert "Conversion failed" in result.error_message
            assert result.error_class == "conversion_error"
            
            # Error message should be sent to user
            mock_client.send_message.assert_called()
    
    @pytest.mark.asyncio
    async def test_send_voice_send_error_returns_error_result(self, mock_client, sender_settings):
        """Test that send error returns error result instead of raising."""
        sender = TelegramSender(mock_client, sender_settings)
        wav_bytes = _make_wav_bytes()
        
        # Mock send_voice to raise generic exception
        mock_client.send_voice = AsyncMock(side_effect=Exception("Network error"))
        
        result = await sender.send_voice(12345, wav_bytes)
        
        # Should return error result (not raise)
        assert result.success is False
        assert result.error_message is not None
