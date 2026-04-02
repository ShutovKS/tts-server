"""
Tests for retry/backoff behavior in polling and sender.

These tests verify:
- Exponential backoff calculation
- Error classification for retry decisions
- Retry exhaustion handling
- Degraded state transitions
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from telegram_bot.polling import (
    BackoffConfig,
    ExponentialBackoff,
    PollingAdapter,
    PollingHealth,
    PollingState,
    classify_telegram_error,
)
from telegram_bot.client import TelegramAPIError


class TestExponentialBackoff:
    """Tests for ExponentialBackoff calculator."""
    
    def test_initial_delay(self):
        """First call returns initial delay."""
        backoff = ExponentialBackoff()
        delay = backoff.next_delay()
        assert delay == 1.0
    
    def test_exponential_increase(self):
        """Subsequent calls increase exponentially."""
        backoff = ExponentialBackoff(BackoffConfig(initial_delay=1.0, multiplier=2.0))
        
        backoff.next_delay()  # Reset to attempt 1
        d1 = backoff.next_delay()  # 1.0 * 2^1 = 2.0
        d2 = backoff.next_delay()  # 1.0 * 2^2 = 4.0
        d3 = backoff.next_delay()  # 1.0 * 2^3 = 8.0
        
        assert d1 == pytest.approx(2.0, abs=0.5)  # Allow for jitter
        assert d2 == pytest.approx(4.0, abs=0.5)
        assert d3 == pytest.approx(8.0, abs=1.0)
    
    def test_max_delay_respected(self):
        """Delay never exceeds max_delay (accounting for jitter)."""
        backoff = ExponentialBackoff(
            BackoffConfig(initial_delay=1.0, multiplier=10.0, max_delay=5.0, jitter=0.0)
        )
        
        for _ in range(10):
            delay = backoff.next_delay()
        
        # Without jitter, delay should not exceed max_delay
        assert delay <= 5.0
    
    def test_jitter_prevents_thundering_herd(self):
        """Jitter introduces randomness to delays."""
        backoff = ExponentialBackoff(BackoffConfig(jitter=0.2))
        backoff.next_delay()  # Reset
        
        delays = [backoff.next_delay() for _ in range(10)]
        
        # Not all delays should be identical
        assert len(set(int(d * 100) for d in delays)) > 1
    
    def test_reset_clears_attempt(self):
        """Reset returns to initial state."""
        backoff = ExponentialBackoff()
        backoff.next_delay()
        backoff.next_delay()
        backoff.next_delay()
        
        assert backoff.attempt > 1
        
        backoff.reset()
        
        assert backoff.attempt == 0
    
    def test_should_stop_after_max_retries(self):
        """should_stop returns True after max_retries."""
        backoff = ExponentialBackoff(
            BackoffConfig(max_retries=3, initial_delay=0.001)
        )
        
        for _ in range(3):
            backoff.next_delay()
        
        assert backoff.should_stop


class TestClassifyTelegramError:
    """Tests for error classification."""
    
    def test_rate_limit_is_retryable(self):
        """429 Rate limit is classified as retryable."""
        exc = TelegramAPIError("Too Many Requests", code=429)
        classified = classify_telegram_error(exc)
        
        assert classified.is_retryable
        assert classified.error_class.value == "retryable_rate_limit"
    
    def test_server_error_is_retryable(self):
        """5xx errors are retryable."""
        for code in [500, 502, 503, 504]:
            exc = TelegramAPIError("Server error", code=code)
            classified = classify_telegram_error(exc)
            
            assert classified.is_retryable
            assert classified.error_class.value == "retryable_network"
    
    def test_auth_error_is_fatal(self):
        """401 Unauthorized is fatal."""
        exc = TelegramAPIError("Unauthorized", code=401)
        classified = classify_telegram_error(exc)
        
        assert not classified.is_retryable
        assert classified.severity.value == "fatal"
        assert classified.error_class.value == "non_retryable_auth"
    
    def test_forbidden_is_critical(self):
        """403 Forbidden is critical but not fatal."""
        exc = TelegramAPIError("Forbidden", code=403)
        classified = classify_telegram_error(exc)
        
        assert not classified.is_retryable
        assert classified.severity.value == "critical"
    
    def test_bad_request_not_retryable(self):
        """4xx errors (except 429) are not retryable."""
        exc = TelegramAPIError("Bad Request", code=400)
        classified = classify_telegram_error(exc)
        
        assert not classified.is_retryable
    
    def test_timeout_is_retryable(self):
        """Timeout is classified as retryable network error."""
        exc = asyncio.TimeoutError("Request timeout")
        classified = classify_telegram_error(exc)
        
        assert classified.is_retryable
        assert classified.error_class.value == "retryable_network"
    
    def test_connection_error_is_retryable(self):
        """Connection errors are retryable."""
        exc = ConnectionError("Connection refused")
        classified = classify_telegram_error(exc)
        
        assert classified.is_retryable


class TestPollingHealth:
    """Tests for polling health tracking."""
    
    def test_initial_state(self):
        """Initial health state is STOPPED."""
        health = PollingHealth(state=PollingState.STOPPED, consecutive_errors=0)
        
        assert health.state == PollingState.STOPPED
        assert not health.is_healthy
        assert not health.is_degraded
    
    def test_healthy_state(self):
        """HEALTHY state is healthy and not degraded."""
        health = PollingHealth(state=PollingState.HEALTHY, consecutive_errors=0)
        
        assert health.is_healthy
        assert not health.is_degraded
    
    def test_degraded_state(self):
        """DEGRADED state is not healthy but is degraded."""
        health = PollingHealth(
            state=PollingState.DEGRADED,
            consecutive_errors=5,
            degradation_reason="connection_errors",
        )
        
        assert not health.is_healthy
        assert health.is_degraded
        assert health.consecutive_errors == 5
        assert health.degradation_reason == "connection_errors"
    
    def test_recovering_state(self):
        """RECOVERING state is degraded."""
        health = PollingHealth(state=PollingState.RECOVERING, consecutive_errors=0)
        
        assert health.is_degraded
    
    def test_to_dict(self):
        """to_dict returns complete health status."""
        health = PollingHealth(
            state=PollingState.HEALTHY,
            consecutive_errors=0,
            last_success_time=12345.0,
        )
        
        d = health.to_dict()
        
        assert d["state"] == "healthy"
        assert d["consecutive_errors"] == 0
        assert d["is_healthy"]
        assert not d["is_degraded"]


class TestPollingAdapterDegradedBehavior:
    """Tests for polling adapter degraded state behavior."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock Telegram client."""
        client = AsyncMock()
        client.get_me = AsyncMock(return_value={"id": 123, "username": "test_bot"})
        client.get_updates = AsyncMock(return_value=[])
        return client
    
    @pytest.fixture
    def mock_dispatcher(self):
        """Create mock dispatcher."""
        return AsyncMock()
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.telegram_bot_token = "test_token"
        settings.telegram_max_text_length = 1000
        return settings
    
    def test_adapter_uses_degradation_threshold(self, mock_client, mock_dispatcher, mock_settings):
        """Test that adapter uses degradation_threshold from config."""
        adapter = PollingAdapter(
            client=mock_client,
            dispatcher=mock_dispatcher,
            settings=mock_settings,
            backoff_config=BackoffConfig(initial_delay=0.05, degradation_threshold=5),
        )
        
        # Adapter should store the backoff config
        assert adapter._backoff_config.degradation_threshold == 5
    
    def test_on_error_increments_consecutive_errors(self, mock_client, mock_dispatcher, mock_settings):
        """Test that _on_error increments consecutive_errors counter."""
        adapter = PollingAdapter(
            client=mock_client,
            dispatcher=mock_dispatcher,
            settings=mock_settings,
            backoff_config=BackoffConfig(initial_delay=0.01, degradation_threshold=2),
        )
        
        # Set adapter to HEALTHY state (as if start() was called)
        adapter._set_state(PollingState.HEALTHY)
        
        # Manually trigger errors
        from telegram_bot.observability import ClassifiedError, ErrorClass, ErrorSeverity
        
        classified = ClassifiedError(
            message="Connection failed",
            error_class=ErrorClass.RETRYABLE_NETWORK,
            severity=ErrorSeverity.WARNING,
        )
        
        adapter._on_error(classified)
        assert adapter.health.consecutive_errors == 1
        
        adapter._on_error(classified)
        assert adapter.health.consecutive_errors == 2
        
        # After 2 errors with degradation_threshold=2, should be degraded
        assert adapter.health.state == PollingState.DEGRADED


class TestRetryBehavior:
    """Tests for retry behavior in API client."""
    
    @pytest.mark.asyncio
    async def test_client_retries_on_timeout(self):
        """Client retries on timeout with exponential backoff."""
        from telegram_bot.client import TelegramBotClient, RetryConfig
        
        call_count = 0
        
        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise asyncio.TimeoutError("Timeout")
            # Return mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"ok": True, "result": {}}
            return mock_response
        
        client = TelegramBotClient(
            bot_token="test_token",
            retry_config=RetryConfig(max_attempts=3, initial_delay=0.01),
        )
        
        with patch.object(client, '_get_client') as mock_get_client:
            mock_httpx_client = MagicMock()
            mock_httpx_client.post = mock_post
            mock_get_client.return_value = mock_httpx_client
            
            result = await client.get_me()
        
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_client_fails_after_max_retries(self):
        """Client fails after exhausting retries."""
        from telegram_bot.client import TelegramBotClient, RetryConfig
        
        async def mock_post(*args, **kwargs):
            raise asyncio.TimeoutError("Persistent timeout")
        
        client = TelegramBotClient(
            bot_token="test_token",
            retry_config=RetryConfig(max_attempts=2, initial_delay=0.01),
        )
        
        with patch.object(client, '_get_client') as mock_get_client:
            mock_httpx_client = MagicMock()
            mock_httpx_client.post = mock_post
            mock_get_client.return_value = mock_httpx_client
            
            with pytest.raises(TelegramAPIError):
                await client.get_me()


class TestSenderRetryBehavior:
    """Tests for sender retry behavior."""
    
    @pytest.mark.asyncio
    async def test_sender_retries_on_api_error(self):
        """Sender retries on transient API errors."""
        from telegram_bot.sender import TelegramSender, DeliveryRetryConfig
        
        call_count = 0
        
        async def mock_send_voice(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TelegramAPIError("Internal server error", code=500)
            return {"message_id": 123}
        
        mock_client = AsyncMock()
        mock_client.send_voice = mock_send_voice
        
        settings = MagicMock()
        settings.sample_rate = 24000
        
        sender = TelegramSender(
            client=mock_client,
            settings=settings,
            retry_config=DeliveryRetryConfig(max_attempts=3, initial_delay=0.01),
        )
        
        with patch('telegram_bot.sender.convert_wav_to_telegram_ogg') as mock_convert:
            mock_convert.return_value = (b"ogg_data", True)
            
            result = await sender.send_voice(
                chat_id=12345,
                audio_bytes=b"test_audio",
            )
        
        assert result.success
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_sender_fails_on_non_retryable_error(self):
        """Sender fails immediately on non-retryable errors."""
        from telegram_bot.sender import TelegramSender, DeliveryRetryConfig
        
        mock_client = AsyncMock()
        mock_client.send_voice = AsyncMock(
            side_effect=TelegramAPIError("Bad Request", code=400)
        )
        
        settings = MagicMock()
        settings.sample_rate = 24000
        
        sender = TelegramSender(
            client=mock_client,
            settings=settings,
            retry_config=DeliveryRetryConfig(max_attempts=3, initial_delay=0.01),
        )
        
        with patch('telegram_bot.sender.convert_wav_to_telegram_ogg') as mock_convert:
            mock_convert.return_value = (b"ogg_data", True)
            
            result = await sender.send_voice(
                chat_id=12345,
                audio_bytes=b"test_audio",
            )
        
        assert not result.success
        assert result.error_class == "non_retryable_input"
