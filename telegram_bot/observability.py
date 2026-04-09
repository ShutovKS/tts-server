# FILE: telegram_bot/observability.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide Telegram-specific observability helpers and degraded-state tracking.
#   SCOPE: Bot-specific logging helpers, health state tracking
#   DEPENDS: M-OBSERVABILITY
#   LINKS: M-TELEGRAM
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   TelegramCorrelationContext - Correlation context for Telegram operations
#   get_correlation - Read current Telegram correlation fields
#   get_correlation_context - Rehydrate the current Telegram correlation context
#   set_correlation_context - Bind a Telegram correlation context for a scope
#   clear_correlation_context - Clear Telegram correlation context variables
#   BackoffConfig - Retry and degradation policy for Telegram observability
#   ErrorSeverity - Severity levels for Telegram error reporting
#   ErrorClass - Classification categories for Telegram retry decisions
#   ClassifiedError - Structured Telegram error classification payload
#   SimpleCounter - Lightweight counter helper for Telegram metrics
#   SimpleHistogram - Lightweight histogram helper for Telegram metrics
#   TelegramMetrics - Telegram operational metrics facade
#   METRICS - Default Telegram metrics collector instance
#   log_telegram_event - Emit structured Telegram transport log events
#   PollingState - Enum of Telegram polling loop states
#   PollingHealth - Health snapshot for Telegram polling
#   classify_telegram_error - Classify Telegram exceptions for retry handling
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, function contracts, semantic blocks, and migrated log events to block-reference format]
# END_CHANGE_SUMMARY

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


# START_CONTRACT: TelegramCorrelationContext
#   PURPOSE: Carry request correlation metadata across Telegram bot operations.
#   INPUTS: { update_id: Optional[int] - Telegram update identifier, chat_id: Optional[int] - Telegram chat identifier, user_id: Optional[int] - Telegram user identifier, request_id: Optional[str] - internal request identifier, operation: Optional[str] - operation name }
#   OUTPUTS: { TelegramCorrelationContext - mutable correlation context object }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: TelegramCorrelationContext
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
        self.request_id = (
            request_id if request_id is not None else str(uuid.uuid4())[:12]
        )
        self.operation = operation if operation is not None else "system"
        self.timestamp = time.time()
        self._tokens: dict[str, Any] = {}

    # START_CONTRACT: bind
    #   PURPOSE: Bind Telegram correlation fields into active context variables.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Updates context-local correlation state.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: bind
    def bind(self) -> None:
        """Bind correlation context to context vars."""
        self._tokens["update_id"] = _UPDATE_ID.set(self.update_id)
        self._tokens["chat_id"] = _CHAT_ID.set(self.chat_id)
        self._tokens["user_id"] = _USER_ID.set(self.user_id)
        self._tokens["request_id"] = _REQUEST_ID.set(self.request_id)
        self._tokens["operation"] = _OPERATION.set(self.operation)

    # START_CONTRACT: set_operation
    #   PURPOSE: Update the current Telegram operation name in the correlation context.
    #   INPUTS: { operation: str - operation name }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Updates context-local operation state.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: set_operation
    def set_operation(self, operation: str) -> None:
        """Set current operation name."""
        self.operation = operation
        self._tokens["operation"] = _OPERATION.set(operation)

    # START_CONTRACT: unbind
    #   PURPOSE: Restore previous context state after a Telegram operation finishes.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Clears context-local correlation bindings set by this instance.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: unbind
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

    # START_CONTRACT: to_dict
    #   PURPOSE: Export Telegram correlation metadata as a dictionary for logging.
    #   INPUTS: {}
    #   OUTPUTS: { dict[str, Any] - correlation metadata mapping }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: to_dict
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


# START_CONTRACT: get_correlation
#   PURPOSE: Read the currently bound Telegram correlation fields from context variables.
#   INPUTS: {}
#   OUTPUTS: { dict[str, Any] - active correlation metadata mapping }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: get_correlation
def get_correlation() -> dict[str, Any]:
    """Get current correlation context as dict."""
    return {
        "update_id": _UPDATE_ID.get(),
        "chat_id": _CHAT_ID.get(),
        "user_id": _USER_ID.get(),
        "request_id": _REQUEST_ID.get(),
        "operation": _OPERATION.get(),
    }


# START_CONTRACT: get_correlation_context
#   PURPOSE: Rehydrate the currently bound Telegram correlation context when present.
#   INPUTS: {}
#   OUTPUTS: { Optional[TelegramCorrelationContext] - reconstructed correlation context or None }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: get_correlation_context
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


# START_CONTRACT: set_correlation_context
#   PURPOSE: Bind a Telegram correlation context and return a scope manager for restoration.
#   INPUTS: { ctx: TelegramCorrelationContext - correlation context to bind }
#   OUTPUTS: { _CorrelationContextManager - context manager for scoped binding }
#   SIDE_EFFECTS: Updates context-local correlation state.
#   LINKS: M-TELEGRAM
# END_CONTRACT: set_correlation_context
def set_correlation_context(
    ctx: TelegramCorrelationContext,
) -> _CorrelationContextManager:
    """Set correlation context. Returns context manager for scoping.

    Usage:
        # Simple set
        set_correlation_context(ctx)

        # With context manager (restores previous on exit)
        with set_correlation_context(ctx):
            ...
    """
    return _CorrelationContextManager(ctx)


# START_CONTRACT: clear_correlation_context
#   PURPOSE: Reset all Telegram correlation context variables to empty values.
#   INPUTS: {}
#   OUTPUTS: { None - no return value }
#   SIDE_EFFECTS: Clears context-local correlation state.
#   LINKS: M-TELEGRAM
# END_CONTRACT: clear_correlation_context
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


# START_CONTRACT: BackoffConfig
#   PURPOSE: Configure Telegram observability backoff and degradation thresholds.
#   INPUTS: { initial_delay: float - first retry delay, max_delay: float - maximum retry delay, multiplier: float - exponential factor, jitter: float - randomization factor, max_retries: int - retry cap, degradation_threshold: int - degraded-state threshold }
#   OUTPUTS: { BackoffConfig - immutable observability retry policy }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: BackoffConfig
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


# START_CONTRACT: ErrorSeverity
#   PURPOSE: Enumerate severity levels used for Telegram error reporting.
#   INPUTS: {}
#   OUTPUTS: { ErrorSeverity - error severity enum }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: ErrorSeverity
class ErrorSeverity(Enum):
    """Severity levels for Telegram errors."""

    FATAL = "fatal"  # Must stop operation
    CRITICAL = "critical"  # Requires immediate attention
    ERROR = "error"  # Operation failed
    WARNING = "warning"  # Degraded operation possible
    INFO = "info"  # Informational


# START_CONTRACT: ErrorClass
#   PURPOSE: Enumerate Telegram error classes used for retry decisions.
#   INPUTS: {}
#   OUTPUTS: { ErrorClass - error classification enum }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: ErrorClass
class ErrorClass(Enum):
    """Classification of errors for retry decisions."""

    RETRYABLE_NETWORK = "retryable_network"  # Network timeout, 5xx
    RETRYABLE_RATE_LIMIT = "retryable_rate_limit"  # 429 Too Many Requests
    NON_RETRYABLE_API = "non_retryable_api"  # 4xx except 429
    NON_RETRYABLE_AUTH = "non_retryable_auth"  # Auth failures
    NON_RETRYABLE_INPUT = "non_retryable_input"  # Invalid input
    FATAL_CONFIG = "fatal_config"  # Config errors
    FATAL_RESOURCE = "fatal_resource"  # Resource exhaustion


# START_CONTRACT: ClassifiedError
#   PURPOSE: Store Telegram error classification details and retry guidance.
#   INPUTS: { error_class: ErrorClass - error category, severity: ErrorSeverity - severity level, message: str - human-readable message, code: Optional[int] - optional API code, retry_after: Optional[float] - optional retry delay, details: dict[str, Any] - extra metadata }
#   OUTPUTS: { ClassifiedError - immutable classified error payload }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: ClassifiedError
@dataclass
class ClassifiedError:
    """Classified Telegram error with retry guidance."""

    error_class: ErrorClass
    severity: ErrorSeverity
    message: str
    code: Optional[int] = None
    retry_after: Optional[float] = None  # Seconds to wait for rate limits
    details: dict[str, Any] = field(default_factory=dict)

    # START_CONTRACT: is_retryable
    #   PURPOSE: Report whether the classified Telegram error should be retried.
    #   INPUTS: {}
    #   OUTPUTS: { bool - True when the error class is retryable }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: is_retryable
    @property
    def is_retryable(self) -> bool:
        """Whether this error should be retried."""
        return self.error_class in (
            ErrorClass.RETRYABLE_NETWORK,
            ErrorClass.RETRYABLE_RATE_LIMIT,
        )

    # START_CONTRACT: should_stop
    #   PURPOSE: Report whether the classified Telegram error should stop processing entirely.
    #   INPUTS: {}
    #   OUTPUTS: { bool - True when the severity is fatal }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: should_stop
    @property
    def should_stop(self) -> bool:
        """Whether this error should stop all operations."""
        return self.severity == ErrorSeverity.FATAL


# ============================================================================
# Telegram Metrics
# ============================================================================


# START_CONTRACT: SimpleCounter
#   PURPOSE: Provide a lightweight integer-like counter for Telegram metrics tests and summaries.
#   INPUTS: {}
#   OUTPUTS: { SimpleCounter - mutable counter helper }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: SimpleCounter
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


# START_CONTRACT: SimpleHistogram
#   PURPOSE: Collect timing samples for lightweight Telegram metric summaries.
#   INPUTS: {}
#   OUTPUTS: { SimpleHistogram - mutable histogram helper }
#   SIDE_EFFECTS: Stores recorded timing values in memory.
#   LINKS: M-TELEGRAM
# END_CONTRACT: SimpleHistogram
class SimpleHistogram:
    """Simple histogram for metrics."""

    def __init__(self) -> None:
        self._values: list[float] = []

    # START_CONTRACT: record
    #   PURPOSE: Add a numeric timing sample to the histogram.
    #   INPUTS: { value: float - measurement value to store }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Appends the sample to the in-memory histogram.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: record
    def record(self, value: float) -> None:
        self._values.append(value)

    def __len__(self) -> int:
        return len(self._values)


# START_CONTRACT: TelegramMetrics
#   PURPOSE: Track Telegram adapter operational counters, timings, and delivery metrics.
#   INPUTS: { collector: Optional[MetricsCollector] - optional metrics collector backend }
#   OUTPUTS: { TelegramMetrics - Telegram metrics facade }
#   SIDE_EFFECTS: Increments counters and records timings in the backing metrics collector.
#   LINKS: M-TELEGRAM
# END_CONTRACT: TelegramMetrics
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

    # START_CONTRACT: polling_started
    #   PURPOSE: Record that the Telegram polling loop has started.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: polling_started
    def polling_started(self) -> None:
        """Record polling loop started."""
        self._collector.increment(
            "telegram.polling.started",
            tags={"instance": "telegram_bot"},
        )

    # START_CONTRACT: polling_stopped
    #   PURPOSE: Record that the Telegram polling loop has stopped.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: polling_stopped
    def polling_stopped(self) -> None:
        """Record polling loop stopped."""
        self._collector.increment(
            "telegram.polling.stopped",
            tags={"instance": "telegram_bot"},
        )

    # START_CONTRACT: polling_degraded
    #   PURPOSE: Record that Telegram polling entered a degraded state.
    #   INPUTS: { reason: str - degradation reason label }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: polling_degraded
    def polling_degraded(self, reason: str) -> None:
        """Record polling entering degraded mode."""
        self._collector.increment(
            "telegram.polling.degraded",
            tags={"reason": reason},
        )

    # START_CONTRACT: polling_recovered
    #   PURPOSE: Record that Telegram polling recovered from a degraded state.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: polling_recovered
    def polling_recovered(self) -> None:
        """Record polling recovered from degraded mode."""
        self._collector.increment(
            "telegram.polling.recovered",
            tags={"instance": "telegram_bot"},
        )

    # START_CONTRACT: polling_error
    #   PURPOSE: Record a Telegram polling error with classification metadata.
    #   INPUTS: { error_class: str - classified polling error name, fatal: bool - fatal error flag }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: polling_error
    def polling_error(self, error_class: str, fatal: bool = False) -> None:
        """Record polling error."""
        self._collector.increment(
            "telegram.polling.errors",
            tags={"error_class": error_class, "fatal": str(fatal).lower()},
        )

    # START_CONTRACT: updates_received
    #   PURPOSE: Record how many Telegram updates were received in a polling batch.
    #   INPUTS: { count: int - batch size }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments counters in memory and the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: updates_received
    def updates_received(self, count: int) -> None:
        """Record updates received in batch."""
        self.polling_updates_received += count
        self._collector.increment(
            "telegram.updates.received",
            count,
            tags={"instance": "telegram_bot"},
        )

    # START_CONTRACT: updates_processed
    #   PURPOSE: Record how many Telegram updates were successfully processed.
    #   INPUTS: { count: int - processed update count }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: updates_processed
    def updates_processed(self, count: int) -> None:
        """Record updates successfully processed."""
        self._collector.increment(
            "telegram.updates.processed",
            count,
            tags={"instance": "telegram_bot"},
        )

    # --- Command Metrics ---

    # START_CONTRACT: command_received
    #   PURPOSE: Record receipt of a Telegram command.
    #   INPUTS: { command: str - command name }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments counters in memory and the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: command_received
    def command_received(self, command: str) -> None:
        """Record command received."""
        self.commands_received += 1
        self._collector.increment(
            "telegram.commands.received",
            tags={"command": command},
        )

    # START_CONTRACT: command_accepted
    #   PURPOSE: Record that a Telegram command was accepted for processing.
    #   INPUTS: { command: str - command name }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments counters in memory and the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: command_accepted
    def command_accepted(self, command: str) -> None:
        """Record command accepted for processing."""
        self.commands_accepted += 1
        self._collector.increment(
            "telegram.commands.accepted",
            tags={"command": command},
        )

    # START_CONTRACT: command_rejected
    #   PURPOSE: Record that a Telegram command was rejected and why.
    #   INPUTS: { command: str - command name, reason: str - rejection reason label }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments counters in memory and the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: command_rejected
    def command_rejected(self, command: str, reason: str) -> None:
        """Record command rejected."""
        self.commands_rejected += 1
        self._collector.increment(
            "telegram.commands.rejected",
            tags={"command": command, "reason": reason},
        )

    # --- Synthesis Metrics ---

    # START_CONTRACT: synthesis_started
    #   PURPOSE: Record that a Telegram synthesis request has started.
    #   INPUTS: { speaker: str - speaker or workflow label }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments counters in memory and the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: synthesis_started
    def synthesis_started(self, speaker: str) -> None:
        """Record TTS synthesis started."""
        self.synthesis_requests += 1
        self._collector.increment(
            "telegram.synthesis.started",
            tags={"speaker": speaker},
        )

    # START_CONTRACT: synthesis_completed
    #   PURPOSE: Record successful completion of a Telegram synthesis request.
    #   INPUTS: { speaker: str - speaker or workflow label, duration_ms: float - synthesis duration }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments and times metrics in the backing collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: synthesis_completed
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

    # START_CONTRACT: synthesis_failed
    #   PURPOSE: Record failure of a Telegram synthesis request.
    #   INPUTS: { speaker: str - speaker or workflow label, error_type: str - failure type name }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments counters in memory and the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: synthesis_failed
    def synthesis_failed(self, speaker: str, error_type: str) -> None:
        """Record TTS synthesis failed."""
        self.synthesis_errors += 1
        self._collector.increment(
            "telegram.synthesis.failed",
            tags={"speaker": speaker, "error_type": error_type},
        )

    # --- Conversion Metrics ---

    # START_CONTRACT: conversion_started
    #   PURPOSE: Record that Telegram audio conversion has started.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: conversion_started
    def conversion_started(self) -> None:
        """Record audio conversion started."""
        self._collector.increment("telegram.conversion.started")

    # START_CONTRACT: conversion_completed
    #   PURPOSE: Record successful completion of Telegram audio conversion.
    #   INPUTS: { duration_ms: float - conversion duration }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments and times metrics in the backing collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: conversion_completed
    def conversion_completed(self, duration_ms: float) -> None:
        """Record audio conversion completed."""
        self._collector.increment("telegram.conversion.completed")
        self._collector.observe_timing(
            "telegram.conversion.duration_ms",
            duration_ms,
        )

    # START_CONTRACT: conversion_failed
    #   PURPOSE: Record failure of Telegram audio conversion.
    #   INPUTS: { error_type: str - conversion failure type }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments counters in memory and the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: conversion_failed
    def conversion_failed(self, error_type: str) -> None:
        """Record audio conversion failed."""
        self.conversion_errors += 1
        self._collector.increment(
            "telegram.conversion.failed",
            tags={"error_type": error_type},
        )

    # --- Delivery Metrics ---

    # START_CONTRACT: delivery_started
    #   PURPOSE: Record that Telegram voice delivery has started.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: delivery_started
    def delivery_started(self) -> None:
        """Record voice delivery started."""
        self._collector.increment("telegram.delivery.started")

    # START_CONTRACT: delivery_completed
    #   PURPOSE: Record successful completion of Telegram voice delivery.
    #   INPUTS: { duration_ms: float - delivery duration }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments counters in memory and records timing metrics.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: delivery_completed
    def delivery_completed(self, duration_ms: float) -> None:
        """Record voice delivery completed."""
        self.delivery_success += 1
        self._collector.increment("telegram.delivery.completed")
        self._collector.observe_timing(
            "telegram.delivery.duration_ms",
            duration_ms,
        )

    # START_CONTRACT: delivery_failed
    #   PURPOSE: Record failure of Telegram voice delivery.
    #   INPUTS: { error_class: str - classified delivery error, retryable: bool - retry guidance flag }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments counters in memory and the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: delivery_failed
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

    # START_CONTRACT: delivery_retried
    #   PURPOSE: Record that Telegram voice delivery is being retried.
    #   INPUTS: { attempt: int - retry attempt number }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: delivery_retried
    def delivery_retried(self, attempt: int) -> None:
        """Record delivery retry attempt."""
        self._collector.increment(
            "telegram.delivery.retried",
            tags={"attempt": str(attempt)},
        )

    # START_CONTRACT: delivery_exhausted
    #   PURPOSE: Record that Telegram voice delivery retries were exhausted.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: delivery_exhausted
    def delivery_exhausted(self) -> None:
        """Record delivery retry exhaustion."""
        self._collector.increment("telegram.delivery.exhausted")

    # --- Job Integration Metrics (Stage 2) ---

    # START_CONTRACT: jobs_submitted
    #   PURPOSE: Record submission of a Telegram job through the job model.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: jobs_submitted
    def jobs_submitted(self) -> None:
        """Record job submitted through job model."""
        self._collector.increment("telegram.jobs.submitted")

    # START_CONTRACT: jobs_submission_failed
    #   PURPOSE: Record failure while submitting a Telegram job.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: jobs_submission_failed
    def jobs_submission_failed(self) -> None:
        """Record job submission failure."""
        self._collector.increment("telegram.jobs.submission_failed")

    # START_CONTRACT: jobs_duplicate
    #   PURPOSE: Record reuse of an existing Telegram job via idempotency.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: jobs_duplicate
    def jobs_duplicate(self) -> None:
        """Record duplicate job detection (idempotency hit)."""
        self._collector.increment("telegram.jobs.duplicate")

    # START_CONTRACT: jobs_completed
    #   PURPOSE: Record completion of a Telegram job.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: jobs_completed
    def jobs_completed(self) -> None:
        """Record job completion detected."""
        self._collector.increment("telegram.jobs.completed")

    # START_CONTRACT: jobs_failed
    #   PURPOSE: Record failure of a Telegram job.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: jobs_failed
    def jobs_failed(self) -> None:
        """Record job failure detected."""
        self._collector.increment("telegram.jobs.failed")

    # START_CONTRACT: job_delivery_completed
    #   PURPOSE: Record successful delivery of a completed Telegram job result.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: job_delivery_completed
    def job_delivery_completed(self) -> None:
        """Record job result delivery to user."""
        self._collector.increment("telegram.jobs.delivery_completed")

    # START_CONTRACT: job_delivery_recovered
    #   PURPOSE: Record recovered delivery of a Telegram job after restart.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: job_delivery_recovered
    def job_delivery_recovered(self) -> None:
        """Record job delivery recovered from restart."""
        self._collector.increment("telegram.jobs.delivery_recovered")

    # START_CONTRACT: voice_sent
    #   PURPOSE: Record successful sending of a Telegram voice message.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: voice_sent
    def voice_sent(self) -> None:
        """Record voice message sent."""
        self._collector.increment("telegram.voice.sent")

    # START_CONTRACT: voice_send_failed
    #   PURPOSE: Record failure to send a Telegram voice message.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Increments the backing metrics collector.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: voice_send_failed
    def voice_send_failed(self) -> None:
        """Record voice message send failure."""
        self._collector.increment("telegram.voice.send_failed")

    # --- Summary ---

    # START_CONTRACT: summary
    #   PURPOSE: Return a structured snapshot of accumulated Telegram metrics.
    #   INPUTS: {}
    #   OUTPUTS: { dict[str, Any] - summary of Telegram operational metrics }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: summary
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
            "jobs_submitted": getattr(self, "_jobs_submitted_count", 0),
            "jobs_submission_failed": getattr(self, "_jobs_submission_failed_count", 0),
            "jobs_duplicate": getattr(self, "_jobs_duplicate_count", 0),
            "jobs_completed": getattr(self, "_jobs_completed_count", 0),
            "jobs_failed": getattr(self, "_jobs_failed_count", 0),
            "jobs_delivery_completed": getattr(
                self, "_jobs_delivery_completed_count", 0
            ),
            "jobs_delivery_recovered": getattr(
                self, "_jobs_delivery_recovered_count", 0
            ),
            "voice_sent": getattr(self, "_voice_sent_count", 0),
            "voice_send_failed": getattr(self, "_voice_send_failed_count", 0),
        }

    # START_CONTRACT: to_dict
    #   PURPOSE: Export Telegram metrics as a dictionary.
    #   INPUTS: {}
    #   OUTPUTS: { dict[str, Any] - serialized metrics snapshot }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: to_dict
    def to_dict(self) -> dict[str, Any]:
        """Export metrics as dict."""
        return self.summary()


# Default metrics instance
METRICS = TelegramMetrics()


# ============================================================================
# Structured Events
# ============================================================================


# START_CONTRACT: log_telegram_event
#   PURPOSE: Emit a structured Telegram log event augmented with correlation metadata.
#   INPUTS: { event_or_logger: Any - event name or logger instance, level: Optional[int] - log level, message: str - human-readable message, _logger: Optional[logging.Logger] - fallback logger, fields: Any - structured event fields }
#   OUTPUTS: { None - no return value }
#   SIDE_EFFECTS: Writes a structured log record through the configured logger.
#   LINKS: M-TELEGRAM
# END_CONTRACT: log_telegram_event
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

    logger.log(
        level, json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    )


# ============================================================================
# Operational States
# ============================================================================


# START_CONTRACT: PollingState
#   PURPOSE: Enumerate lifecycle states for Telegram polling health reporting.
#   INPUTS: {}
#   OUTPUTS: { PollingState - polling state enum }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: PollingState
class PollingState(Enum):
    """Polling operational states."""

    STOPPED = "stopped"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    STOPPING = "stopping"


# START_CONTRACT: PollingHealth
#   PURPOSE: Store health information and recent failure context for the Telegram polling loop.
#   INPUTS: { state: PollingState - current polling state, consecutive_errors: int - recent error streak, consecutive_successes: int - recent success streak, recovery_threshold: int - recovery threshold, last_success_time: Optional[float] - last success timestamp, last_error_time: Optional[float] - last error timestamp, degradation_reason: Optional[str] - degraded-state reason, error_samples: list[str] - recent error samples }
#   OUTPUTS: { PollingHealth - mutable polling health snapshot }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: PollingHealth
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

    # START_CONTRACT: is_healthy
    #   PURPOSE: Report whether Telegram polling is currently healthy.
    #   INPUTS: {}
    #   OUTPUTS: { bool - True when polling state is healthy }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: is_healthy
    @property
    def is_healthy(self) -> bool:
        return self.state == PollingState.HEALTHY

    # START_CONTRACT: is_degraded
    #   PURPOSE: Report whether Telegram polling is currently degraded or recovering.
    #   INPUTS: {}
    #   OUTPUTS: { bool - True when polling is degraded }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: is_degraded
    @property
    def is_degraded(self) -> bool:
        return self.state in (PollingState.DEGRADED, PollingState.RECOVERING)

    # START_CONTRACT: to_dict
    #   PURPOSE: Export polling health fields as a dictionary.
    #   INPUTS: {}
    #   OUTPUTS: { dict[str, Any] - serialized polling health snapshot }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: to_dict
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


# START_CONTRACT: classify_telegram_error
#   PURPOSE: Classify Telegram-related exceptions into retry and severity categories for observability.
#   INPUTS: { exc: Exception - Telegram-related exception }
#   OUTPUTS: { ClassifiedError - structured retry guidance and severity classification }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: classify_telegram_error
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

__all__ = [
    "TelegramCorrelationContext",
    "get_correlation",
    "get_correlation_context",
    "set_correlation_context",
    "clear_correlation_context",
    "BackoffConfig",
    "ErrorSeverity",
    "ErrorClass",
    "ClassifiedError",
    "SimpleCounter",
    "SimpleHistogram",
    "TelegramMetrics",
    "METRICS",
    "log_telegram_event",
    "PollingState",
    "PollingHealth",
    "classify_telegram_error",
]
