"""
Unit tests for Telegram bot operational readiness and polling.

Tests startup validation, lifecycle logging, polling metrics, and self-checks.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio


class TestPollingMetrics:
    """Tests for polling operational stats."""
    
    def test_operational_stats_has_expected_keys(self):
        """Test operational stats have expected keys."""
        from telegram_bot.polling import PollingAdapter
        
        mock_client = MagicMock()
        mock_dispatcher = MagicMock()
        mock_settings = MagicMock()
        
        adapter = PollingAdapter(
            client=mock_client,
            dispatcher=mock_dispatcher,
            settings=mock_settings,
        )
        
        stats = adapter.operational_stats
        
        # Check expected keys
        assert "running" in stats
        assert "offset" in stats
        assert "updates_processed" in stats
        assert "errors_count" in stats
        assert "last_update_time" in stats
    
    def test_operational_stats_defaults(self):
        """Test operational stats have correct defaults."""
        from telegram_bot.polling import PollingAdapter
        
        mock_client = MagicMock()
        mock_dispatcher = MagicMock()
        mock_settings = MagicMock()
        
        adapter = PollingAdapter(
            client=mock_client,
            dispatcher=mock_dispatcher,
            settings=mock_settings,
        )
        
        stats = adapter.operational_stats
        
        assert stats["running"] is False
        assert stats["offset"] == 0
        assert stats["updates_processed"] == 0
        assert stats["errors_count"] == 0
        assert stats["last_update_time"] is None
    
    def test_operational_stats_update_processing(self):
        """Test operational stats track updates processed."""
        from telegram_bot.polling import PollingAdapter
        
        mock_client = MagicMock()
        mock_dispatcher = MagicMock()
        mock_settings = MagicMock()
        
        adapter = PollingAdapter(
            client=mock_client,
            dispatcher=mock_dispatcher,
            settings=mock_settings,
        )
        
        # Simulate processing updates
        adapter._updates_processed = 10
        adapter._errors_count = 2
        adapter._last_update_time = 1234567890.0
        
        stats = adapter.operational_stats
        
        assert stats["updates_processed"] == 10
        assert stats["errors_count"] == 2
        assert stats["last_update_time"] == 1234567890.0


class TestStartupValidation:
    """Tests for startup validation and self-checks."""
    
    def test_main_function_exists(self):
        """Test main function exists."""
        from telegram_bot.__main__ import main
        
        assert main is not None
        assert callable(main)


class TestPollingLifecycle:
    """Tests for polling lifecycle management."""
    
    def test_polling_adapter_initialization(self):
        """Test PollingAdapter initialization."""
        from telegram_bot.polling import PollingAdapter
        
        mock_client = MagicMock()
        mock_dispatcher = MagicMock()
        mock_settings = MagicMock()
        
        adapter = PollingAdapter(
            client=mock_client,
            dispatcher=mock_dispatcher,
            settings=mock_settings,
        )
        
        assert adapter._client is mock_client
        assert adapter._dispatcher is mock_dispatcher
        assert adapter._settings is mock_settings
        assert adapter._running is False
    
    def test_polling_adapter_has_start_method(self):
        """Test PollingAdapter has start method."""
        from telegram_bot.polling import PollingAdapter
        
        assert hasattr(PollingAdapter, "start")
    
    def test_polling_adapter_has_stop_method(self):
        """Test PollingAdapter has stop method."""
        from telegram_bot.polling import PollingAdapter
        
        assert hasattr(PollingAdapter, "stop")


class TestPollingErrorTracking:
    """Tests for polling error tracking."""
    
    def test_errors_count_starts_at_zero(self):
        """Test errors count starts at zero."""
        from telegram_bot.polling import PollingAdapter
        
        mock_client = MagicMock()
        mock_dispatcher = MagicMock()
        mock_settings = MagicMock()
        
        adapter = PollingAdapter(
            client=mock_client,
            dispatcher=mock_dispatcher,
            settings=mock_settings,
        )
        
        assert adapter._errors_count == 0
    
    def test_updates_processed_starts_at_zero(self):
        """Test updates processed count starts at zero."""
        from telegram_bot.polling import PollingAdapter
        
        mock_client = MagicMock()
        mock_dispatcher = MagicMock()
        mock_settings = MagicMock()
        
        adapter = PollingAdapter(
            client=mock_client,
            dispatcher=mock_dispatcher,
            settings=mock_settings,
        )
        
        assert adapter._updates_processed == 0


class TestPollingConstants:
    """Tests for polling constants."""
    
    def test_poll_timeout_defined(self):
        """Test POLL_TIMEOUT_SECONDS is defined."""
        from telegram_bot.polling import PollingAdapter
        
        assert hasattr(PollingAdapter, "POLL_TIMEOUT_SECONDS")
        assert PollingAdapter.POLL_TIMEOUT_SECONDS > 0
    
    def test_poll_error_delay_defined(self):
        """Test POLL_ERROR_DELAY_SECONDS is defined."""
        from telegram_bot.polling import PollingAdapter
        
        assert hasattr(PollingAdapter, "POLL_ERROR_DELAY_SECONDS")
        assert PollingAdapter.POLL_ERROR_DELAY_SECONDS > 0
    
    def test_max_updates_per_batch_defined(self):
        """Test MAX_UPDATES_PER_BATCH is defined."""
        from telegram_bot.polling import PollingAdapter
        
        assert hasattr(PollingAdapter, "MAX_UPDATES_PER_BATCH")
        assert PollingAdapter.MAX_UPDATES_PER_BATCH > 0


class TestBotClassInterface:
    """Tests for TelegramBot class interface."""
    
    def test_bot_module_exists(self):
        """Test telegram_bot module exists."""
        import telegram_bot
        
        assert telegram_bot is not None
    
    def test_bot_has_main(self):
        """Test telegram_bot has main function."""
        from telegram_bot import __main__
        
        assert hasattr(__main__, "main")


class TestStructuredLoggingConstants:
    """Tests for structured logging constants in handlers."""
    
    def test_commands_module_exports_constants(self):
        """Test commands module has expected constants."""
        from telegram_bot.handlers.commands import (
            MIN_SPEED,
            MAX_SPEED,
            VALID_SPEAKERS,
        )
        
        # Constants should be defined
        assert MIN_SPEED > 0
        assert MAX_SPEED > MIN_SPEED
        assert isinstance(VALID_SPEAKERS, frozenset)
        assert len(VALID_SPEAKERS) > 0
    
    def test_log_constants_defined(self):
        """Test log constants are defined in dispatcher."""
        from telegram_bot.handlers import dispatcher
        
        # Check module has logger
        assert hasattr(dispatcher, "LOGGER")
    
    def test_polling_module_has_logger(self):
        """Test polling module has logger."""
        from telegram_bot.polling import LOGGER
        
        assert LOGGER is not None


class TestTelegramClientInterface:
    """Tests for TelegramClient interface."""
    
    def test_client_class_exists(self):
        """Test TelegramClient class exists."""
        from telegram_bot.client import TelegramClient
        
        assert TelegramClient is not None
    
    def test_client_has_get_me_method(self):
        """Test TelegramClient has get_me method."""
        from telegram_bot.client import TelegramClient
        
        assert hasattr(TelegramClient, "get_me")
    
    def test_client_has_send_message_method(self):
        """Test TelegramClient has send_message method."""
        from telegram_bot.client import TelegramClient
        
        assert hasattr(TelegramClient, "send_message")
    
    def test_client_has_send_voice_method(self):
        """Test TelegramClient has send_voice method."""
        from telegram_bot.client import TelegramClient
        
        assert hasattr(TelegramClient, "send_voice")
