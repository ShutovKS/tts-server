"""
Telegram-specific observability and metrics.

This module provides structured logging events, correlation context,
and operational metrics tailored for the Telegram transport layer.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from core.metrics import (
    DEFAULT_METRICS_COLLECTOR,
    InMemoryMetricsCollector,
    MetricsCollector,
    OperationalMetricsRegistry,
)

if TYPE_CHECKING:
    pass


# ============================================================================
# Correlation Context
# ============================================================================

_UPDATE_ID: ContextVar[Optional[int]] = ContextVar("telegram_update_id", default=None)
_CHAT_ID: ContextVar[Optional[int]] = ContextVar("telegram_chat_id", default=None)
_USER_ID: ContextVar[Optional[int]] = ContextVar("telegram_user_id", default=None)
_REQUEST_ID: ContextVar[Optional[str]] = ContextVar("telegram_request_id", default=None)
_OPERATION: ContextVar[Optional[str]] = ContextVar("telegram_operation", default=None)


class TelegramCorrelationContext:
    """
    Manages correlation context for Telegram operations.
    
    Provides a stable set of correlation fields:
    - update_id: Telegram update identifier
    - chat_id: Target chat identifier
    - user_id: User who sent the message
    - request_id: Internal request tracking ID
    - operation: Current operation name
    - timestamp: Creation timestamp
    """
    
    def __init__(
        self,
        update_id: Optional[int] = None,
        chat_id: Optional[int] = None,
        user_id: Optional[int] = None,
        request_id: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        self.update_id = update_id
        self.chat_id = chat_id
        self.user_id = user_id
        self.request_id = request_id if request_id is not None else str(uuid.uuid4())[:12]
        self.operation = operation if operation is not None else "system"
        self.timestamp = time.time()
        self._tokens: dict[str, Any] = {}
    
    def bind(self) -> None:
        """Bind correlation context to context vars."""
        self._tokens["update_id"] = _UPDATE_ID.set(self.update_id)
        self._tokens["chat_id"] = _CHAT_ID.set(self.chat_id)
        self._tokens["user_id"] = _USER_ID.set(self.user_id)
        self._tokens["request_id"] = _REQUEST_ID.set(self.request_id)
        self._tokens["operation"] = _OPERATION.set(self.operation)
    
    def set_operation(self, operation: str) -> None:
        """Set current operation name."""
        self.operation = operation
        self._tokens["operation"] = _OPERATION.set(operation)
    
    def unbind(self) -> None:
        """Unbind correlation context from context vars."""
        for key, token in self._tokens.items():
            if key == "update_id":
                _UPDATE_ID.reset(token)
            elif key == "chat_id":
                _CHAT_ID.reset(token)
            elif key == "user_id":
                _USER_ID.reset(token)
            elif key == "request_id":
                _REQUEST_ID.reset(token)
            elif key == "operation":
                _OPERATION.reset(token)
        self._tokens.clear()
    
    def to_dict(self) -> dict[str, Any]:
        """Export correlation data as dict for logging."""
        return {
            "update_id": self.update_id,
            "chat_id": self.chat_id,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "operation": self.operation,
            "timestamp": self.timestamp,
        }


def get_correlation() -> dict[str, Any]:
    """Get current correlation context as dict."""
    return {
        "update_id": _UPDATE_ID.get(),
        "chat_id": _CHAT_ID.get(),
        "user_id": _USER_ID.get(),
        "request_id": _REQUEST_ID.get(),
        "operation": _OPERATION.get(),
    }


def get_correlation_context() -> Optional[TelegramCorrelationContext]:
    """Get current correlation context as TelegramCorrelationContext object."""
    # Check if context is set (has values)
    request_id = _REQUEST_ID.get()
    if request_id is None:
        return None
    
    ctx = TelegramCorrelationContext(
        update_id=_UPDATE_ID.get(),
        chat_id=_CHAT_ID.get(),
        user_id=_USER_ID.get(),
    )
    ctx.request_id = request_id
    ctx.operation = _OPERATION.get()
    return ctx


class _CorrelationContextManager:
    """Context manager for correlation context.
    
    Automatically enters context on creation (for simple usage without 'with').
    """
    
    def __init__(self, ctx: TelegramCorrelationContext):
        self.ctx = ctx
        self._previous: dict[str, Any] = {}
        self._entered: bool = False
        # Auto-enter on creation
        self.__enter__()
    
    def __enter__(self) -> TelegramCorrelationContext:
        # Save current values (if not already entered)
        if not self._entered:
            self._previous = {
                "update_id": _UPDATE_ID.get(),
                "chat_id": _CHAT_ID.get(),
                "user_id": _USER_ID.get(),
                "request_id": _REQUEST_ID.get(),
                "operation": _OPERATION.get(),
            }
            self._entered = True
        # Set new values
        _UPDATE_ID.set(self.ctx.update_id)
        _CHAT_ID.set(self.ctx.chat_id)
        _USER_ID.set(self.ctx.user_id)
        _REQUEST_ID.set(self.ctx.request_id)
        _OPERATION.set(self.ctx.operation)
        return self.ctx
    
    def __exit__(self, *args: Any) -> None:
        # Restore previous values explicitly
        _UPDATE_ID.set(self._previous.get("update_id"))
        _CHAT_ID.set(self._previous.get("chat_id"))
        _USER_ID.set(self._previous.get("user_id"))
        _REQUEST_ID.set(self._previous.get("request_id"))
        _OPERATION.set(self._previous.get("operation"))
        self._entered = False


def set_correlation_context(ctx: TelegramCorrelationContext) -> _CorrelationContextManager:
    """Set correlation context. Returns context manager for scoping.
    
    Usage:
        # Simple set
        set_correlation_context(ctx)
        
        # With context manager (restores previous on exit)
        with set_correlation_context(ctx):
            ...
    """
    return _CorrelationContextManager(ctx)


def clear_correlation_context() -> None:
    """Clear correlation context to defaults."""
    _UPDATE_ID.set(None)
    _CHAT_ID.set(None)
    _USER_ID.set(None)
    _REQUEST_ID.set(None)
    _OPERATION.set(None)


# ============================================================================
# Backoff Configuration
# ============================================================================

@dataclass
class BackoffConfig:
    """Configuration for exponential backoff."""
    initial_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: float = 0.1
    max_retries: int = 5
    degradation_threshold: int = 3


# ============================================================================
# Error Classification
# ============================================================================

class ErrorSeverity(Enum):
    """Severity levels for Telegram errors."""
    FATAL = "fatal"       # Must stop operation
    CRITICAL = "critical" # Requires immediate attention
    ERROR = "error"       # Operation failed
    WARNING = "warning"    # Degraded operation possible
    INFO = "info"         # Informational


class ErrorClass(Enum):
    """Classification of errors for retry decisions."""
    RETRYABLE_NETWORK = "retryable_network"      # Network timeout, 5xx
    RETRYABLE_RATE_LIMIT = "retryable_rate_limit"  # 429 Too Many Requests
    NON_RETRYABLE_API = "non_retryable_api"      # 4xx except 429
    NON_RETRYABLE_AUTH = "non_retryable_auth"    # Auth failures
    NON_RETRYABLE_INPUT = "non_retryable_input"  # Invalid input
    FATAL_CONFIG = "fatal_config"                # Config errors
    FATAL_RESOURCE = "fatal_resource"            # Resource exhaustion


@dataclass
class ClassifiedError:
    """Classified Telegram error with retry guidance."""
    error_class: ErrorClass
    severity: ErrorSeverity
    message: str
    code: Optional[int] = None
    retry_after: Optional[float] = None  # Seconds to wait for rate limits
    details: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_retryable(self) -> bool:
        """Whether this error should be retried."""
        return self.error_class in (
            ErrorClass.RETRYABLE_NETWORK,
            ErrorClass.RETRYABLE_RATE_LIMIT,
        )
    
    @property
    def should_stop(self) -> bool:
        """Whether this error should stop all operations."""
        return self.severity == ErrorSeverity.FATAL


# ============================================================================
# Telegram Metrics
# ============================================================================

class SimpleCounter:
    """Simple counter for metrics."""
    
    def __init__(self) -> None:
        self._value = 0
    
    def __iadd__(self, value: int) -> "SimpleCounter":
        self._value += value
        return self
    
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, int):
            return self._value == other
        if isinstance(other, SimpleCounter):
            return self._value == other._value
        return False
    
    def __int__(self) -> int:
        return self._value
    
    def __repr__(self) -> str:
        return f"SIMPLECOUNTER({self._value})"


class SimpleHistogram:
    """Simple histogram for metrics."""
    
    def __init__(self) -> None:
        self._values: list[float] = []
    
    def record(self, value: float) -> None:
        self._values.append(value)
    
    def __len__(self) -> int:
        return len(self._values)


class TelegramMetrics:
    """
    Telegram-specific operational metrics.
    
    Provides counters and gauges for:
    - Polling: active, degraded, errors
    - Delivery: success, failure (by type)
    - Conversion: success, failure
    - Commands: by type and result
    """
    
    def __init__(self, collector: Optional[MetricsCollector] = None):
        self._collector = collector or DEFAULT_METRICS_COLLECTOR
        self._registry = OperationalMetricsRegistry(self._collector)
        
        # Direct counter attributes for tests
        self.polling_updates_received = 0
        self.polling_errors_total = 0
        self.commands_received = 0
        self.commands_accepted = 0
        self.commands_rejected = 0
        self.synthesis_requests = 0
        self.synthesis_errors = 0
        self.conversion_errors = 0
        self.delivery_success = 0
        self.delivery_errors = 0
        
        # Timing histograms
        self.synthesis_duration = SimpleHistogram()
        self.conversion_duration = SimpleHistogram()
        self.delivery_duration = SimpleHistogram()
    
    # --- Polling Metrics ---
    
    def polling_started(self) -> None:
        """Record polling loop started."""
        self._collector.increment(
            "telegram.polling.started",
            tags={"instance": "telegram_bot"},
        )
    
    def polling_stopped(self) -> None:
        """Record polling loop stopped."""
        self._collector.increment(
            "telegram.polling.stopped",
            tags={"instance": "telegram_bot"},
        )
    
    def polling_degraded(self, reason: str) -> None:
        """Record polling entering degraded mode."""
        self._collector.increment(
            "telegram.polling.degraded",
            tags={"reason": reason},
        )
    
    def polling_recovered(self) -> None:
        """Record polling recovered from degraded mode."""
        self._collector.increment(
            "telegram.polling.recovered",
            tags={"instance": "telegram_bot"},
        )
    
    def polling_error(self, error_class: str, fatal: bool = False) -> None:
        """Record polling error."""
        self._collector.increment(
            "telegram.polling.errors",
            tags={"error_class": error_class, "fatal": str(fatal).lower()},
        )
    
    def updates_received(self, count: int) -> None:
        """Record updates received in batch."""
        self.polling_updates_received += count
        self._collector.increment(
            "telegram.updates.received",
            count,
            tags={"instance": "telegram_bot"},
        )
    
    def updates_processed(self, count: int) -> None:
        """Record updates successfully processed."""
        self._collector.increment(
            "telegram.updates.processed",
            count,
            tags={"instance": "telegram_bot"},
        )
    
    # --- Command Metrics ---
    
    def command_received(self, command: str) -> None:
        """Record command received."""
        self.commands_received += 1
        self._collector.increment(
            "telegram.commands.received",
            tags={"command": command},
        )
    
    def command_accepted(self, command: str) -> None:
        """Record command accepted for processing."""
        self.commands_accepted += 1
        self._collector.increment(
            "telegram.commands.accepted",
            tags={"command": command},
        )
    
    def command_rejected(self, command: str, reason: str) -> None:
        """Record command rejected."""
        self.commands_rejected += 1
        self._collector.increment(
            "telegram.commands.rejected",
            tags={"command": command, "reason": reason},
        )
    
    # --- Synthesis Metrics ---
    
    def synthesis_started(self, speaker: str) -> None:
        """Record TTS synthesis started."""
        self.synthesis_requests += 1
        self._collector.increment(
            "telegram.synthesis.started",
            tags={"speaker": speaker},
        )
    
    def synthesis_completed(self, speaker: str, duration_ms: float) -> None:
        """Record TTS synthesis completed."""
        self._collector.increment(
            "telegram.synthesis.completed",
            tags={"speaker": speaker},
        )
        self._collector.observe_timing(
            "telegram.synthesis.duration_ms",
            duration_ms,
            tags={"speaker": speaker},
        )
    
    def synthesis_failed(self, speaker: str, error_type: str) -> None:
        """Record TTS synthesis failed."""
        self.synthesis_errors += 1
        self._collector.increment(
            "telegram.synthesis.failed",
            tags={"speaker": speaker, "error_type": error_type},
        )
    
    # --- Conversion Metrics ---
    
    def conversion_started(self) -> None:
        """Record audio conversion started."""
        self._collector.increment("telegram.conversion.started")
    
    def conversion_completed(self, duration_ms: float) -> None:
        """Record audio conversion completed."""
        self._collector.increment("telegram.conversion.completed")
        self._collector.observe_timing(
            "telegram.conversion.duration_ms",
            duration_ms,
        )
    
    def conversion_failed(self, error_type: str) -> None:
        """Record audio conversion failed."""
        self.conversion_errors += 1
        self._collector.increment(
            "telegram.conversion.failed",
            tags={"error_type": error_type},
        )
    
    # --- Delivery Metrics ---
    
    def delivery_started(self) -> None:
        """Record voice delivery started."""
        self._collector.increment("telegram.delivery.started")
    
    def delivery_completed(self, duration_ms: float) -> None:
        """Record voice delivery completed."""
        self.delivery_success += 1
        self._collector.increment("telegram.delivery.completed")
        self._collector.observe_timing(
            "telegram.delivery.duration_ms",
            duration_ms,
        )
    
    def delivery_failed(self, error_class: str, retryable: bool) -> None:
        """Record voice delivery failed."""
        self.delivery_errors += 1
        self._collector.increment(
            "telegram.delivery.failed",
            tags={
                "error_class": error_class,
                "retryable": str(retryable).lower(),
            },
        )
    
    def delivery_retried(self, attempt: int) -> None:
        """Record delivery retry attempt."""
        self._collector.increment(
            "telegram.delivery.retried",
            tags={"attempt": str(attempt)},
        )
    
    def delivery_exhausted(self) -> None:
        """Record delivery retry exhaustion."""
        self._collector.increment("telegram.delivery.exhausted")
    
    # --- Job Integration Metrics (Stage 2) ---
    
    def jobs_submitted(self) -> None:
        """Record job submitted through job model."""
        self._collector.increment("telegram.jobs.submitted")
    
    def jobs_submission_failed(self) -> None:
        """Record job submission failure."""
        self._collector.increment("telegram.jobs.submission_failed")
    
    def jobs_duplicate(self) -> None:
        """Record duplicate job detection (idempotency hit)."""
        self._collector.increment("telegram.jobs.duplicate")
    
    def jobs_completed(self) -> None:
        """Record job completion detected."""
        self._collector.increment("telegram.jobs.completed")
    
    def jobs_failed(self) -> None:
        """Record job failure detected."""
        self._collector.increment("telegram.jobs.failed")
    
    def job_delivery_completed(self) -> None:
        """Record job result delivery to user."""
        self._collector.increment("telegram.jobs.delivery_completed")
    
    def job_delivery_recovered(self) -> None:
        """Record job delivery recovered from restart."""
        self._collector.increment("telegram.jobs.delivery_recovered")
    
    def voice_sent(self) -> None:
        """Record voice message sent."""
        self._collector.increment("telegram.voice.sent")
    
    def voice_send_failed(self) -> None:
        """Record voice message send failure."""
        self._collector.increment("telegram.voice.send_failed")
    
    # --- Summary ---
    
    def summary(self) -> dict[str, Any]:
        """Get metrics summary."""
        return {
            "polling_updates_received": self.polling_updates_received,
            "polling_errors_total": self.polling_errors_total,
            "commands_received": self.commands_received,
            "commands_accepted": self.commands_accepted,
            "commands_rejected": self.commands_rejected,
            "synthesis_requests": self.synthesis_requests,
            "synthesis_errors": self.synthesis_errors,
            "conversion_errors": self.conversion_errors,
            "delivery_success": self.delivery_success,
            "delivery_errors": self.delivery_errors,
            # Job integration metrics (Stage 2)
            "jobs_submitted": getattr(self, '_jobs_submitted_count', 0),
            "jobs_submission_failed": getattr(self, '_jobs_submission_failed_count', 0),
            "jobs_duplicate": getattr(self, '_jobs_duplicate_count', 0),
            "jobs_completed": getattr(self, '_jobs_completed_count', 0),
            "jobs_failed": getattr(self, '_jobs_failed_count', 0),
            "jobs_delivery_completed": getattr(self, '_jobs_delivery_completed_count', 0),
            "jobs_delivery_recovered": getattr(self, '_jobs_delivery_recovered_count', 0),
            "voice_sent": getattr(self, '_voice_sent_count', 0),
            "voice_send_failed": getattr(self, '_voice_send_failed_count', 0),
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Export metrics as dict."""
        return self.summary()


# Default metrics instance
METRICS = TelegramMetrics()


# ============================================================================
# Structured Events
# ============================================================================

def log_telegram_event(
    event_or_logger: Any,
    level: Optional[int] = None,
    message: str = "",
    _logger: Optional[logging.Logger] = None,
    **fields: Any,
) -> None:
    """
    Log structured Telegram event with correlation context.
    
    Supports two calling conventions:
    - log_telegram_event(event, level, message, **fields)  # For tests
    - log_telegram_event(logger, level=level, event=event, message=message, **fields)  # For production
    
    Args:
        event_or_logger: Event name or logger instance
        level: Log level (e.g., logging.INFO, logging.ERROR)
        message: Human-readable message
        _logger: Optional logger instance (uses default if not provided)
        **fields: Additional fields to include in the log
    """
    import json
    
    # Detect calling convention based on first argument type
    if isinstance(event_or_logger, logging.Logger):
        # Called as: log_telegram_event(logger, level=..., event=..., message=..., **fields)
        logger = event_or_logger
        if level is not None:
            event = fields.pop("event", "")
        else:
            event = ""
        message = fields.pop("message", message)
    else:
        # Called as: log_telegram_event(event, level, message, **fields)
        event = event_or_logger
        logger = _logger or logging.getLogger("telegram_bot")
    
    # Get correlation context
    correlation = get_correlation()
    
    payload = {
        "event": event,
        "message": message,
        **fields,
    }
    
    # Add correlation fields if available and not already set
    if correlation.get("request_id") is not None and "request_id" not in fields:
        payload["request_id"] = correlation["request_id"]
    if correlation.get("operation") is not None and "operation" not in fields:
        payload["operation"] = correlation["operation"]
    if correlation.get("update_id") is not None and "update_id" not in fields:
        payload["update_id"] = correlation["update_id"]
    if correlation.get("chat_id") is not None and "chat_id" not in fields:
        payload["chat_id"] = correlation["chat_id"]
    if correlation.get("user_id") is not None and "user_id" not in fields:
        payload["user_id"] = correlation["user_id"]
    
    logger.log(level, json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


# ============================================================================
# Operational States
# ============================================================================

class PollingState(Enum):
    """Polling operational states."""
    STOPPED = "stopped"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    STOPPING = "stopping"


@dataclass
class PollingHealth:
    """Health status of polling loop."""
    state: PollingState
    consecutive_errors: int = 0
    consecutive_successes: int = 0
    recovery_threshold: int = 3
    last_success_time: Optional[float] = None
    last_error_time: Optional[float] = None
    degradation_reason: Optional[str] = None
    error_samples: list[str] = field(default_factory=list)
    
    @property
    def is_healthy(self) -> bool:
        return self.state == PollingState.HEALTHY
    
    @property
    def is_degraded(self) -> bool:
        return self.state in (PollingState.DEGRADED, PollingState.RECOVERING)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "consecutive_errors": self.consecutive_errors,
            "consecutive_successes": self.consecutive_successes,
            "last_success_time": self.last_success_time,
            "last_error_time": self.last_error_time,
            "degradation_reason": self.degradation_reason,
            "recovery_threshold": self.recovery_threshold,
            "is_healthy": self.is_healthy,
            "is_degraded": self.is_degraded,
        }


# ============================================================================
# Error Classification
# ============================================================================

def classify_telegram_error(exc: Exception) -> ClassifiedError:
    """
    Classify a Telegram-related error.
    
    Determines retry strategy and severity based on error type and code.
    
    Args:
        exc: The exception to classify
        
    Returns:
        ClassifiedError with retry guidance
    """
    error_msg = str(exc).lower()
    error_type = type(exc).__name__
    
    # Import here to avoid circular reference
    from telegram_bot.client import TelegramAPIError
    
    # Check if it's a Telegram API error
    if isinstance(exc, TelegramAPIError):
        code = exc.code
        
        # Rate limiting - retry after wait
        if code == 429:
            return ClassifiedError(
                error_class=ErrorClass.RETRYABLE_RATE_LIMIT,
                severity=ErrorSeverity.WARNING,
                message=f"Rate limited by Telegram API: {exc}",
                code=code,
                retry_after=5.0,  # Default, actual may come from response
            )
        
        # Server errors - retryable
        if code and 500 <= code < 600:
            return ClassifiedError(
                error_class=ErrorClass.RETRYABLE_NETWORK,
                severity=ErrorSeverity.WARNING,
                message=f"Telegram server error: {exc}",
                code=code,
            )
        
        # Auth failures - non-retryable, potentially fatal
        if code == 401 or "unauthorized" in error_msg:
            return ClassifiedError(
                error_class=ErrorClass.NON_RETRYABLE_AUTH,
                severity=ErrorSeverity.FATAL,
                message=f"Authentication failed: {exc}",
                code=code,
            )
        
        # Forbidden - may be config issue
        if code == 403:
            return ClassifiedError(
                error_class=ErrorClass.NON_RETRYABLE_API,
                severity=ErrorSeverity.CRITICAL,
                message=f"Access forbidden: {exc}",
                code=code,
            )
        
        # Bad request - non-retryable
        if code and 400 <= code < 500:
            return ClassifiedError(
                error_class=ErrorClass.NON_RETRYABLE_INPUT,
                severity=ErrorSeverity.WARNING,
                message=f"Invalid request: {exc}",
                code=code,
            )
    
    # Network errors - retryable
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError)):
        return ClassifiedError(
            error_class=ErrorClass.RETRYABLE_NETWORK,
            severity=ErrorSeverity.WARNING,
            message=f"Network error: {exc}",
        )
    
    # HTTP library errors
    if "timeout" in error_msg.lower():
        return ClassifiedError(
            error_class=ErrorClass.RETRYABLE_NETWORK,
            severity=ErrorSeverity.WARNING,
            message=f"Request timeout: {exc}",
        )
    
    if "connection" in error_msg.lower():
        return ClassifiedError(
            error_class=ErrorClass.RETRYABLE_NETWORK,
            severity=ErrorSeverity.WARNING,
            message=f"Connection error: {exc}",
        )
    
    # Import httpx errors
    try:
        import httpx
        if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
            return ClassifiedError(
                error_class=ErrorClass.RETRYABLE_NETWORK,
                severity=ErrorSeverity.WARNING,
                message=f"HTTP connection error: {exc}",
            )
    except ImportError:
        pass
    
    # ValueError and other input errors - non-retryable
    if isinstance(exc, ValueError):
        return ClassifiedError(
            error_class=ErrorClass.NON_RETRYABLE_INPUT,
            severity=ErrorSeverity.ERROR,
            message=f"Invalid input: {exc}",
            details={"error_type": error_type},
        )
    
    # Unknown errors - be conservative, mark as non-retryable
    return ClassifiedError(
        error_class=ErrorClass.NON_RETRYABLE_API,
        severity=ErrorSeverity.WARNING,
        message=f"Unknown error: {exc}",
        details={"error_type": error_type},
    )
