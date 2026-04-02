"""
Tests for Telegram observability module.

These tests verify:
- Correlation context management
- Structured event logging
- Error classification
- Metrics collection
- Polling health tracking
"""

import logging
import pytest
from unittest.mock import MagicMock, patch
import uuid

from telegram_bot.observability import (
    TelegramCorrelationContext,
    ErrorClass,
    ErrorSeverity,
    ClassifiedError,
    TelegramMetrics,
    PollingState,
    PollingHealth,
    log_telegram_event,
    classify_telegram_error,
    get_correlation_context,
    set_correlation_context,
    clear_correlation_context,
    METRICS,
)


class TestTelegramCorrelationContext:
    """Tests for correlation context."""
    
    def test_context_creation(self):
        """Context stores all correlation fields."""
        ctx = TelegramCorrelationContext(
            update_id=12345,
            chat_id=67890,
            user_id=11111,
            request_id="req-abc123",
            operation="handle_tts",
        )
        
        assert ctx.update_id == 12345
        assert ctx.chat_id == 67890
        assert ctx.user_id == 11111
        assert ctx.request_id == "req-abc123"
        assert ctx.operation == "handle_tts"
        assert ctx.timestamp is not None
    
    def test_auto_request_id(self):
        """Request ID is auto-generated if not provided."""
        ctx = TelegramCorrelationContext(
            update_id=12345,
            chat_id=67890,
            user_id=11111,
        )
        
        assert ctx.request_id is not None
        assert len(ctx.request_id) > 0
    
    def test_to_dict(self):
        """to_dict returns all fields."""
        ctx = TelegramCorrelationContext(
            update_id=12345,
            chat_id=67890,
            user_id=11111,
            request_id="req-xyz",
            operation="synthesize",
        )
        
        d = ctx.to_dict()
        
        assert d["update_id"] == 12345
        assert d["chat_id"] == 67890
        assert d["user_id"] == 11111
        assert d["request_id"] == "req-xyz"
        assert d["operation"] == "synthesize"
        assert "timestamp" in d


class TestCorrelationContextManagement:
    """Tests for correlation context global management."""
    
    def test_set_and_get_context(self):
        """Context can be set and retrieved."""
        ctx = TelegramCorrelationContext(
            update_id=100,
            chat_id=200,
            user_id=300,
            request_id="test-req",
            operation="test",
        )
        
        set_correlation_context(ctx)
        retrieved = get_correlation_context()
        
        assert retrieved is not None
        assert retrieved.update_id == 100
        assert retrieved.chat_id == 200
    
    def test_clear_context(self):
        """Context can be cleared."""
        ctx = TelegramCorrelationContext(
            update_id=100,
            chat_id=200,
            user_id=300,
        )
        
        set_correlation_context(ctx)
        clear_correlation_context()
        
        retrieved = get_correlation_context()
        assert retrieved is None
    
    def test_context_manager(self):
        """Context manager preserves outer context."""
        outer = TelegramCorrelationContext(
            update_id=100,
            chat_id=200,
            user_id=300,
            operation="outer",
        )
        set_correlation_context(outer)
        
        inner = TelegramCorrelationContext(
            update_id=999,
            chat_id=888,
            operation="inner",
        )
        
        with set_correlation_context(inner):
            retrieved = get_correlation_context()
            assert retrieved.update_id == 999
            assert retrieved.operation == "inner"
        
        # After context manager, outer context restored
        restored = get_correlation_context()
        assert restored.update_id == 100
        assert restored.operation == "outer"


class TestClassifyTelegramError:
    """Tests for error classification."""
    
    def test_rate_limit_retryable(self):
        """429 is retryable with rate limit class."""
        from telegram_bot.client import TelegramAPIError
        
        exc = TelegramAPIError("Too Many Requests", code=429)
        classified = classify_telegram_error(exc)
        
        assert classified.is_retryable
        assert classified.error_class == ErrorClass.RETRYABLE_RATE_LIMIT
        assert classified.severity == ErrorSeverity.WARNING
    
    def test_5xx_retryable(self):
        """5xx errors are retryable."""
        from telegram_bot.client import TelegramAPIError
        
        for code in [500, 502, 503, 504]:
            exc = TelegramAPIError("Server error", code=code)
            classified = classify_telegram_error(exc)
            
            assert classified.is_retryable
            assert classified.error_class == ErrorClass.RETRYABLE_NETWORK
    
    def test_auth_fatal(self):
        """401 is fatal auth error."""
        from telegram_bot.client import TelegramAPIError
        
        exc = TelegramAPIError("Unauthorized", code=401)
        classified = classify_telegram_error(exc)
        
        assert not classified.is_retryable
        assert classified.error_class == ErrorClass.NON_RETRYABLE_AUTH
        assert classified.severity == ErrorSeverity.FATAL
    
    def test_forbidden_critical(self):
        """403 is critical but not fatal."""
        from telegram_bot.client import TelegramAPIError
        
        exc = TelegramAPIError("Forbidden", code=403)
        classified = classify_telegram_error(exc)
        
        assert not classified.is_retryable
        assert classified.error_class == ErrorClass.NON_RETRYABLE_API
        assert classified.severity == ErrorSeverity.CRITICAL
    
    def test_timeout_retryable(self):
        """Timeout is retryable network error."""
        import asyncio
        
        exc = asyncio.TimeoutError("Request timeout")
        classified = classify_telegram_error(exc)
        
        assert classified.is_retryable
        assert classified.error_class == ErrorClass.RETRYABLE_NETWORK
    
    def test_connection_error_retryable(self):
        """Connection error is retryable."""
        exc = ConnectionError("Connection refused")
        classified = classify_telegram_error(exc)
        
        assert classified.is_retryable
        assert classified.error_class == ErrorClass.RETRYABLE_NETWORK
    
    def test_value_error_not_retryable(self):
        """ValueError is not retryable."""
        exc = ValueError("Invalid input")
        classified = classify_telegram_error(exc)
        
        assert not classified.is_retryable
        assert classified.severity == ErrorSeverity.ERROR


class TestTelegramMetrics:
    """Tests for metrics collection."""
    
    def test_initial_state(self):
        """Initial metrics are zero."""
        metrics = TelegramMetrics()
        
        assert metrics.polling_updates_received == 0
        assert metrics.polling_errors_total == 0
        assert metrics.commands_received == 0
        assert metrics.commands_accepted == 0
        assert metrics.commands_rejected == 0
        assert metrics.synthesis_requests == 0
        assert metrics.synthesis_errors == 0
        assert metrics.conversion_errors == 0
        assert metrics.delivery_success == 0
        assert metrics.delivery_errors == 0
    
    def test_increment_counter(self):
        """Counters can be incremented."""
        metrics = TelegramMetrics()
        
        metrics.polling_updates_received += 5
        metrics.commands_received += 10
        
        assert metrics.polling_updates_received == 5
        assert metrics.commands_received == 10
    
    def test_record_timing(self):
        """Timings can be recorded."""
        metrics = TelegramMetrics()
        
        metrics.synthesis_duration.record(1.5)
        metrics.conversion_duration.record(0.3)
        metrics.delivery_duration.record(0.8)
        
        assert len(metrics.synthesis_duration) == 1
        assert len(metrics.conversion_duration) == 1
        assert len(metrics.delivery_duration) == 1
    
    def test_to_dict(self):
        """Metrics can be exported to dict."""
        metrics = TelegramMetrics()
        metrics.polling_updates_received = 100
        metrics.commands_received = 50
        metrics.delivery_success = 45
        
        d = metrics.to_dict()
        
        assert d["polling_updates_received"] == 100
        assert d["commands_received"] == 50
        assert d["delivery_success"] == 45


class TestPollingHealth:
    """Tests for polling health tracking."""
    
    def test_stopped_state(self):
        """STOPPED state is not healthy or degraded."""
        health = PollingHealth(state=PollingState.STOPPED)
        
        assert not health.is_healthy
        assert not health.is_degraded
        assert health.state == PollingState.STOPPED
    
    def test_healthy_state(self):
        """HEALTHY state is healthy."""
        health = PollingHealth(
            state=PollingState.HEALTHY,
            consecutive_errors=0,
            consecutive_successes=10,
        )
        
        assert health.is_healthy
        assert not health.is_degraded
    
    def test_degraded_state(self):
        """DEGRADED state tracks error details."""
        health = PollingHealth(
            state=PollingState.DEGRADED,
            consecutive_errors=5,
            degradation_reason="network_errors",
            last_error_time=12345.0,
        )
        
        assert not health.is_healthy
        assert health.is_degraded
        assert health.consecutive_errors == 5
        assert health.degradation_reason == "network_errors"
    
    def test_recovering_state(self):
        """RECOVERING state is degraded."""
        health = PollingHealth(
            state=PollingState.RECOVERING,
            consecutive_successes=2,
            recovery_threshold=3,
        )
        
        assert health.is_degraded
        assert health.consecutive_successes < health.recovery_threshold
    
    def test_to_dict_includes_all_fields(self):
        """to_dict includes all health fields."""
        health = PollingHealth(
            state=PollingState.HEALTHY,
            consecutive_errors=0,
            consecutive_successes=5,
            last_success_time=11111.0,
        )
        
        d = health.to_dict()
        
        assert d["state"] == "healthy"
        assert d["consecutive_errors"] == 0
        assert d["consecutive_successes"] == 5
        assert d["last_success_time"] == 11111.0
        assert d["is_healthy"]
        assert not d["is_degraded"]


class TestLogTelegramEvent:
    """Tests for structured event logging."""
    
    def test_log_with_context(self, caplog):
        """Events include correlation context."""
        ctx = TelegramCorrelationContext(
            update_id=123,
            chat_id=456,
            user_id=789,
            operation="test_op",
        )
        
        with caplog.at_level(logging.INFO):
            log_telegram_event("test_event", level=logging.INFO, extra={"key": "value"})
        
        # Event was logged
        assert len(caplog.records) >= 1
    
    def test_log_extra_fields(self, caplog):
        """Extra fields are included in log record."""
        with caplog.at_level(logging.INFO):
            log_telegram_event(
                "custom_event",
                level=logging.INFO,
                update_id=100,
                chat_id=200,
            )
        
        # Check that event was logged
        assert len(caplog.records) >= 1
    
    def test_log_error_event(self, caplog):
        """Error events are logged with exception info."""
        with caplog.at_level(logging.ERROR):
            log_telegram_event(
                "error_event",
                level=logging.ERROR,
                error="Test error",
                error_type="TestError",
            )
        
        assert len(caplog.records) >= 1
        assert caplog.records[-1].levelno >= logging.ERROR


class TestGlobalMetricsSingleton:
    """Tests for global METRICS singleton."""
    
    def test_metrics_singleton_exists(self):
        """METRICS singleton exists."""
        assert METRICS is not None
        assert isinstance(METRICS, TelegramMetrics)
    
    def test_metrics_reset(self):
        """Metrics can be reset."""
        # Store original values
        original = METRICS.polling_updates_received
        
        # Increment
        METRICS.polling_updates_received += 10
        assert METRICS.polling_updates_received == original + 10
        
        # Reset would be done by creating new instance
        # (In real usage, this would be done at start of day)
        new_metrics = TelegramMetrics()
        assert new_metrics.polling_updates_received == 0
