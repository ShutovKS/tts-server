"""
Integration tests for Telegram bot wiring.

Tests the full wiring from Telegram settings through to TTS service
without using the real Telegram API.
"""

# FILE: tests/integration/test_telegram_adapter/test_wiring.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Integration tests for Telegram adapter wiring and runtime assembly.
#   SCOPE: Settings bootstrap, dispatcher wiring, sender wiring, poller wiring
#   DEPENDS: M-TELEGRAM
#   LINKS: V-M-TELEGRAM
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _make_test_settings - Helper that builds minimal Telegram settings fixtures for integration wiring tests
#   _make_wav_bytes - Helper that creates deterministic WAV payloads for synthesizer wiring tests
#   MockTTSApplicationService - Fake application service used for runtime wiring verification
#   TestTelegramSettingsWiring - Verifies Telegram settings inherit and expose core/runtime fields
#   TestTelegramRuntimeWiring - Verifies Telegram runtime assembly wraps core runtime and settings
#   Dispatcher, synthesizer, polling, sender, and full-wiring tests - Verify adapter components wire together correctly
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT and MODULE_MAP]
# END_CHANGE_SUMMARY

from __future__ import annotations

import asyncio
import io
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from telegram_bot.bootstrap import build_telegram_runtime, TelegramRuntime
from telegram_bot.client import TelegramBotClient, TelegramAPIError
from telegram_bot.config import TelegramSettings
from telegram_bot.handlers.dispatcher import CommandDispatcher
from telegram_bot.handlers.tts_handler import TTSSynthesizer
from telegram_bot.polling import PollingAdapter
from telegram_bot.remote_client import RemoteServerClient
from telegram_bot.sender import TelegramSender


pytestmark = pytest.mark.integration


# Helper to create minimal settings for testing
def _make_test_settings(**overrides):
    """Create TelegramSettings with minimal required fields for testing."""
    defaults = {
        "telegram_bot_token": "test_token_123:ABCabc123",
        "models_dir": ".models",
        "outputs_dir": ".outputs",
        "voices_dir": ".voices",
        "telegram_dev_mode": True,  # Enable dev mode for tests with short tokens
    }
    defaults.update(overrides)
    return TelegramSettings(**defaults)


def _make_wav_bytes() -> bytes:
    """Create minimal WAV file bytes for testing."""
    buffer = io.BytesIO()
    silence_frame = bytes([0, 0])
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(silence_frame * 240)
    return buffer.getvalue()


@dataclass
class MockTTSApplicationService:
    """Mock TTS application service."""

    should_fail: bool = False
    fail_error: str = "Synthesis failed"
    last_command: Any = None

    def synthesize_custom(self, command):
        self.last_command = command
        if self.should_fail:
            from core.errors import TTSGenerationError

            raise TTSGenerationError(self.fail_error)

        from core.contracts.results import AudioResult, GenerationResult

        audio = AudioResult(
            path=Path("generated.wav"),
            bytes_data=_make_wav_bytes(),
        )
        return GenerationResult(
            audio=audio,
            saved_path=None,
            model="Qwen3-TTS",
            mode="custom",
            backend="mlx",
        )


class TestTelegramSettingsWiring:
    """Tests for Telegram settings wiring."""

    def test_settings_inherit_core_settings(self):
        """Test that Telegram settings inherit from CoreSettings."""
        settings = _make_test_settings(telegram_bot_token="test_token")

        # Should have core settings attributes
        assert hasattr(settings, "models_dir")
        assert hasattr(settings, "outputs_dir")
        assert hasattr(settings, "sample_rate")

    def test_settings_have_telegram_attributes(self):
        """Test that Telegram settings have Telegram-specific attributes."""
        settings = _make_test_settings(
            telegram_bot_token="test_token",
            telegram_allowed_user_ids=("123", "456"),
            telegram_default_speaker="Alex",
        )

        assert settings.telegram_bot_token == "test_token"
        assert settings.telegram_allowed_user_ids == ("123", "456")
        assert settings.telegram_default_speaker == "Alex"


class TestTelegramRuntimeWiring:
    """Tests for Telegram runtime wiring."""

    def test_telegram_runtime_has_remote_server_client(self):
        """Test that Telegram runtime contains the remote server client seam."""
        settings = _make_test_settings(
            telegram_bot_token="test_token",
            telegram_server_base_url="http://server.internal:8000",
        )

        runtime = build_telegram_runtime(settings)

        assert hasattr(runtime, "remote_server_client")
        assert runtime.remote_server_client is not None

    def test_telegram_runtime_has_settings(self):
        """Test that Telegram runtime contains settings."""
        settings = _make_test_settings(
            telegram_bot_token="test_token",
            telegram_server_base_url="http://server.internal:8000",
        )

        runtime = build_telegram_runtime(settings)

        assert hasattr(runtime, "settings")
        assert runtime.settings is settings

    def test_telegram_runtime_wires_remote_client_when_server_base_url_exists(self):
        """Test that runtime exposes the remote server client seam when configured."""
        settings = _make_test_settings(
            telegram_bot_token="test_token",
            telegram_server_base_url="http://server.internal:8000",
        )

        runtime = build_telegram_runtime(settings)

        assert isinstance(runtime.remote_server_client, RemoteServerClient)
        assert runtime.remote_server_client is not None
        assert runtime.remote_server_client.base_url == "http://server.internal:8000"


class TestDispatcherWiring:
    """Tests for dispatcher wiring with real components."""

    def test_dispatcher_requires_all_components(self):
        """Test that dispatcher initialization requires all components."""
        mock_synth = MagicMock(spec=TTSSynthesizer)
        mock_settings = MagicMock(spec=TelegramSettings)
        mock_sender = MagicMock()

        dispatcher = CommandDispatcher(mock_synth, mock_settings, mock_sender)

        assert dispatcher._synthesizer is mock_synth
        assert dispatcher._settings is mock_settings
        assert dispatcher._sender is mock_sender


class TestTTSSynthesizerWiring:
    """Tests for TTS synthesizer wiring with core."""

    def test_synthesizer_uses_settings_default_speaker(self):
        """Test that synthesizer uses settings for default speaker."""
        mock_app = MockTTSApplicationService()
        settings = _make_test_settings(
            telegram_bot_token="test",
            telegram_default_speaker="CustomVoice",
        )

        synthesizer = TTSSynthesizer(cast(Any, mock_app), settings)

        result = asyncio.run(synthesizer.synthesize("test"))

        assert result.success is True
        assert mock_app.last_command.speaker == "CustomVoice"


class TestPollingAdapterWiring:
    """Tests for polling adapter wiring."""

    @pytest.mark.anyio
    async def test_polling_adapter_initial_state(self):
        """Test polling adapter initial state."""
        mock_client = MagicMock(spec=TelegramBotClient)
        mock_dispatcher = MagicMock(spec=CommandDispatcher)
        mock_settings = _make_test_settings(telegram_bot_token="test")

        polling = PollingAdapter(mock_client, mock_dispatcher, mock_settings)

        assert polling._offset == 0
        assert polling._running is False


class TestClientAbstractInterface:
    """Tests for Telegram client abstract interface."""

    def test_client_interface_has_required_methods(self):
        """Test that client has all required methods."""
        methods = ["get_me", "get_updates", "send_message", "send_voice"]

        for method in methods:
            assert hasattr(TelegramBotClient, method)


class TestSenderWiring:
    """Tests for sender wiring."""

    def test_sender_uses_client(self):
        """Test that sender uses client for sending."""
        mock_client = MagicMock(spec=TelegramBotClient)
        mock_client.send_message = AsyncMock(return_value={"message_id": 123})
        settings = _make_test_settings(telegram_bot_token="test")

        sender = TelegramSender(mock_client, settings)

        asyncio.run(sender.send_text(12345, "Hello"))

        mock_client.send_message.assert_called_once()


class TestFullWiringFlow:
    """Integration tests for full wiring flow."""

    def test_settings_validation_before_build(self):
        """Test that invalid settings fail validation."""
        settings = _make_test_settings(telegram_bot_token="")

        errors = settings.validate()

        assert len(errors) > 0

    def test_allowlist_policy_in_settings(self):
        """Test that allowlist policy works in settings."""
        settings = _make_test_settings(
            telegram_bot_token="test",
            telegram_allowed_user_ids=("111", "222"),
        )

        assert settings.is_user_allowed(111) is True
        assert settings.is_user_allowed(222) is True
        assert settings.is_user_allowed(333) is False

    def test_max_text_length_in_settings(self):
        """Test that max text length is configurable."""
        settings = _make_test_settings(
            telegram_bot_token="test",
            telegram_max_text_length=500,
        )

        assert settings.telegram_max_text_length == 500


class TestErrorPropagation:
    """Tests for error propagation through layers."""

    @pytest.mark.anyio
    async def test_api_error_propagates(self):
        """Test that Telegram API errors propagate."""
        client = TelegramBotClient("test_token")

        with patch.object(client, "_request") as mock_req:
            mock_req.side_effect = TelegramAPIError("Test error", 400)

            with pytest.raises(TelegramAPIError) as exc_info:
                await client.get_me()

            assert "Test error" in str(exc_info.value)
