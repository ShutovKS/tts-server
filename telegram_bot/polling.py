"""
Telegram polling adapter with retry, backoff, and degraded state handling.

This module provides long polling implementation for receiving updates
from Telegram Bot API with:
- Predictable backoff policy for retryable errors
- Error classification (retryable vs non-retryable)
- Degraded state tracking and recovery
- Structured observability events
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from telegram_bot.observability import (
    METRICS,
    ClassifiedError,
    ErrorClass,
    ErrorSeverity,
    PollingHealth,
    PollingState,
    TelegramCorrelationContext,
    TelegramMetrics,
    classify_telegram_error,
    get_correlation,
    log_telegram_event,
)


LOGGER = logging.getLogger("telegram_bot.polling")


# ============================================================================
# Backoff Configuration
# ============================================================================

@dataclass(frozen=True)
class BackoffConfig:
    """Backoff policy configuration."""
    initial_delay: float = 1.0       # Initial retry delay (seconds)
    max_delay: float = 60.0          # Maximum retry delay (seconds)
    multiplier: float = 2.0          # Exponential multiplier
    jitter: float = 0.1              # Random jitter factor (0-1)
    max_retries: int = 10            # Maximum consecutive retries
    degradation_threshold: int = 3   # Errors before degraded state


class ExponentialBackoff:
    """
    Exponential backoff calculator with jitter.
    
    Provides predictable backoff delays for retry scenarios.
    """
    
    def __init__(self, config: Optional[BackoffConfig] = None):
        self._config = config or BackoffConfig()
        self._attempt = 0
    
    def reset(self) -> None:
        """Reset backoff state."""
        self._attempt = 0
    
    def next_delay(self) -> float:
        """
        Calculate next delay with exponential backoff and jitter.
        
        Returns:
            Delay in seconds until next retry
        """
        if self._attempt == 0:
            self._attempt = 1
            return self._config.initial_delay
        
        # Exponential calculation
        delay = min(
            self._config.initial_delay * (self._config.multiplier ** self._attempt),
            self._config.max_delay,
        )
        
        # Add jitter to prevent thundering herd
        import random
        jitter_range = delay * self._config.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        
        self._attempt += 1
        return max(0.1, delay)
    
    @property
    def attempt(self) -> int:
        """Current retry attempt number."""
        return self._attempt
    
    @property
    def should_stop(self) -> bool:
        """Check if max retries exceeded."""
        return self._attempt >= self._config.max_retries


# ============================================================================
# Telegram Client Interface
# ============================================================================

class TelegramClient(ABC):
    """
    Abstract Telegram client interface.
    
    This abstraction allows for easy testing and swapping between
    real Telegram API and mock implementations.
    """
    
    @abstractmethod
    async def get_me(self) -> dict[str, Any]:
        """Get bot information."""
        ...
    
    @abstractmethod
    async def get_updates(
        self,
        offset: int = 0,
        timeout: int = 0,
        allowed_updates: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get updates from Telegram."""
        ...
    
    @abstractmethod
    async def send_message(self, chat_id: int, text: str, parse_mode: str = "Markdown") -> dict[str, Any]:
        """Send text message."""
        ...
    
    @abstractmethod
    async def send_voice(
        self,
        chat_id: int,
        audio: bytes,
        caption: str | None = None,
        duration: int | None = None,
    ) -> dict[str, Any]:
        """Send voice message."""
        ...


# ============================================================================
# Polling Adapter
# ============================================================================

@dataclass
class Update:
    """Represents a Telegram update."""
    update_id: int
    message: dict[str, Any] | None
    edited_message: dict[str, Any] | None
    callback_query: dict[str, Any] | None
    
    @property
    def text(self) -> str | None:
        """Get text from message."""
        if self.message:
            return self.message.get("text")
        return None
    
    @property
    def user_id(self) -> int | None:
        """Get user ID from message."""
        if self.message:
            return self.message.get("from", {}).get("id")
        return None
    
    @property
    def chat_id(self) -> int | None:
        """Get chat ID from message."""
        if self.message:
            return self.message.get("chat", {}).get("id")
        return None
    
    @property
    def chat_type(self) -> str | None:
        """Get chat type from message."""
        if self.message:
            return self.message.get("chat", {}).get("type")
        return None
    
    @property
    def message_id(self) -> int | None:
        """Get message ID from message."""
        if self.message:
            return self.message.get("message_id")
        return None


class PollingAdapter:
    """
    Long polling adapter for Telegram updates.
    
    Features:
    - Exponential backoff with jitter for retryable errors
    - Error classification (retryable vs non-retryable)
    - Degraded state tracking with automatic recovery
    - Structured logging with correlation context
    - Operational metrics collection
    
    Error handling strategy:
    - Retryable errors (network timeout, 5xx, rate limit): exponential backoff
    - Non-retryable errors (auth, invalid input): fail fast, log error
    - Fatal errors (config, resource exhaustion): stop polling
    """
    
    # Polling configuration
    POLL_TIMEOUT_SECONDS = 30
    MAX_UPDATES_PER_BATCH = 100
    POLL_ERROR_DELAY_SECONDS = 5  # Delay between polling attempts on error
    
    def __init__(
        self,
        client: TelegramClient,
        dispatcher: telegram_bot.handlers.dispatcher.CommandDispatcher,
        settings: telegram_bot.config.TelegramSettings,
        logger: logging.Logger | None = None,
        metrics: Optional[TelegramMetrics] = None,
        backoff_config: Optional[BackoffConfig] = None,
    ):
        """
        Initialize polling adapter.
        
        Args:
            client: Telegram API client
            dispatcher: Command dispatcher
            settings: Telegram settings
            logger: Optional logger instance
            metrics: Optional metrics collector (uses global if None)
            backoff_config: Optional backoff configuration
        """
        self._client = client
        self._dispatcher = dispatcher
        self._settings = settings
        self._logger = logger or LOGGER
        self._metrics = metrics or METRICS
        self._backoff_config = backoff_config or BackoffConfig()
        self._backoff = ExponentialBackoff(self._backoff_config)
        self._offset = 0
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Operational state
        self._health = PollingHealth(
            state=PollingState.STOPPED,
            consecutive_errors=0,
        )
        
        # Metrics
        self._updates_processed = 0
        self._errors_count = 0
        self._last_update_time: float | None = None
    
    @property
    def health(self) -> PollingHealth:
        """Get current polling health status."""
        return self._health
    
    @property
    def operational_stats(self) -> dict[str, Any]:
        """Get current operational statistics."""
        return {
            "running": self._running,
            "state": self._health.state.value,
            "offset": self._offset,
            "updates_processed": self._updates_processed,
            "errors_count": self._errors_count,
            "consecutive_errors": self._health.consecutive_errors,
            "last_update_time": self._last_update_time,
            "health": self._health.to_dict(),
        }
    
    # ------------------------------------------------------------------------
    # Lifecycle Methods
    # ------------------------------------------------------------------------
    
    async def start(self) -> None:
        """
        Start the polling loop.
        
        Performs initial self-check, then enters the polling loop.
        Handles degraded states and recovery automatically.
        """
        self._running = True
        self._shutdown_event.clear()
        self._set_state(PollingState.STARTING)
        
        self._metrics.polling_started()
        
        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="telegram.polling.starting",
            message="Telegram polling loop starting",
            poll_timeout=self.POLL_TIMEOUT_SECONDS,
            max_updates_per_batch=self.MAX_UPDATES_PER_BATCH,
        )
        
        # Initial connectivity check
        try:
            bot_info = await self._client.get_me()
            self._on_success()
            
            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="telegram.polling.connected",
                message="Telegram API connection verified",
                bot_username=bot_info.get("username"),
                bot_name=bot_info.get("first_name"),
                bot_id=bot_info.get("id"),
            )
        except Exception as exc:
            classified = classify_telegram_error(exc)
            
            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.polling.connection_failed",
                message="Failed to connect to Telegram API",
                error=str(exc),
                error_type=type(exc).__name__,
                error_class=classified.error_class.value,
                severity=classified.severity.value,
            )
            
            self._metrics.polling_error(classified.error_class.value, classified.should_stop)
            self._on_error(classified)
            
            if classified.should_stop:
                self._set_state(PollingState.STOPPED)
                raise RuntimeError(f"Telegram connection failed (fatal): {exc}") from exc
            
            # Non-fatal: try to continue with degraded state
            self._set_degraded(f"connection_failed: {exc}")
        
        self._set_state(PollingState.HEALTHY)
        
        try:
            while self._running:
                try:
                    await self._poll_once()
                except asyncio.CancelledError:
                    log_telegram_event(
                        self._logger,
                        level=logging.INFO,
                        event="telegram.polling.cancelled",
                        message="Polling loop cancelled",
                    )
                    break
                    
        finally:
            self._set_state(PollingState.STOPPING)
            self._metrics.polling_stopped()
            
            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="telegram.polling.stopped",
                message="Telegram polling loop stopped",
                total_updates_processed=self._updates_processed,
                total_errors=self._errors_count,
            )
    
    async def stop(self) -> None:
        """Stop the polling loop gracefully."""
        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="telegram.polling.stop_requested",
            message="Stopping Telegram polling loop",
            operational_stats=self.operational_stats,
        )
        
        self._running = False
        self._shutdown_event.set()
    
    # ------------------------------------------------------------------------
    # Polling Logic
    # ------------------------------------------------------------------------
    
    async def _poll_once(self) -> None:
        """Execute one polling iteration with retry logic."""
        try:
            updates = await self._get_updates_batch()
            
            if updates:
                batch_size = len(updates)
                self._metrics.updates_received(batch_size)
                
                log_telegram_event(
                    self._logger,
                    level=logging.DEBUG,
                    event="telegram.polling.batch_received",
                    message=f"Received batch of {batch_size} updates",
                    batch_size=batch_size,
                )
                
                for update_dict in updates:
                    await self._process_update(update_dict)
                
                # Check if recovering
                if self._health.state == PollingState.RECOVERING:
                    self._on_recovery()
            
            # Reset backoff on successful poll
            self._backoff.reset()
            self._on_success()
            
        except Exception as exc:
            classified = classify_telegram_error(exc)
            
            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.polling.error",
                message=f"Error in polling loop: {classified.message}",
                error=str(exc),
                error_type=type(exc).__name__,
                error_class=classified.error_class.value,
                severity=classified.severity.value,
                consecutive_errors=self._health.consecutive_errors,
            )
            
            self._metrics.polling_error(classified.error_class.value, classified.should_stop)
            self._on_error(classified)
            
            if classified.should_stop:
                self._logger.error("Fatal polling error, stopping")
                self._running = False
                raise
            
            # Calculate backoff delay
            delay = self._backoff.next_delay()
            
            if self._backoff.should_stop:
                self._logger.warning(
                    f"Polling retry exhausted after {self._backoff.attempt} attempts, "
                    f"will continue with degraded state"
                )
                self._metrics.polling_degraded("retry_exhausted")
                self._set_degraded("retry_exhausted")
            else:
                log_telegram_event(
                    self._logger,
                    level=logging.WARNING,
                    event="telegram.polling.retrying",
                    message=f"Retrying after {delay:.2f}s",
                    delay_seconds=delay,
                    retry_attempt=self._backoff.attempt,
                )
                await asyncio.sleep(delay)
    
    async def _get_updates_batch(self) -> list[dict[str, Any]]:
        """Get a batch of updates with polling timeout."""
        try:
            updates = await self._client.get_updates(
                offset=self._offset,
                timeout=self.POLL_TIMEOUT_SECONDS,
                allowed_updates=["message"],
            )
            return updates
            
        except Exception as exc:
            classified = classify_telegram_error(exc)
            
            log_telegram_event(
                self._logger,
                level=logging.WARNING,
                event="telegram.polling.fetch_error",
                message=f"Failed to fetch updates: {classified.message}",
                error=str(exc),
                error_type=type(exc).__name__,
                error_class=classified.error_class.value,
                current_offset=self._offset,
            )
            
            raise  # Let caller handle retry
    
    async def _process_update(self, update_dict: dict[str, Any]) -> None:
        """Process a single update with correlation context."""
        update_id = update_dict.get("update_id", 0)
        message = update_dict.get("message")
        
        # Update offset to acknowledge this update
        self._offset = update_id + 1
        
        if message is None:
            return
        
        # Create correlation context for this update
        user_id = message.get("from", {}).get("id")
        chat_id = message.get("chat", {}).get("id")
        
        ctx = TelegramCorrelationContext(
            update_id=update_id,
            chat_id=chat_id,
            user_id=user_id,
        )
        ctx.bind()
        
        try:
            # Track update timing
            self._last_update_time = time.time()
            
            if user_id is None or chat_id is None:
                log_telegram_event(
                    self._logger,
                    level=logging.WARNING,
                    event="telegram.update.invalid",
                    message="Update missing user_id or chat_id, skipping",
                    update_id=update_id,
                )
                return
            
            # Track processed updates
            self._updates_processed += 1
            self._metrics.updates_processed(1)
            
            text = message.get("text", "")
            chat_type = message.get("chat", {}).get("type", "")
            message_id = message.get("message_id")
            
            log_telegram_event(
                self._logger,
                level=logging.DEBUG,
                event="telegram.update.processing",
                message="Processing update",
                update_id=update_id,
                user_id=user_id,
                chat_id=chat_id,
                chat_type=chat_type,
                message_length=len(text) if text else 0,
            )
            
            # Delegate to dispatcher
            await self._dispatcher.handle_update(
                text=text or "",
                user_id=user_id,
                chat_id=chat_id,
                message_id=message_id or 0,
                chat_type=chat_type,
                reply_to_message=message.get("reply_to_message"),
            )
            
        except Exception as exc:
            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.update.processing_failed",
                message="Failed to process update in dispatcher",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            
        finally:
            ctx.unbind()
    
    # ------------------------------------------------------------------------
    # State Management
    # ------------------------------------------------------------------------
    
    def _set_state(self, state: PollingState) -> None:
        """Set polling state."""
        old_state = self._health.state
        self._health.state = state
        
        if old_state != state:
            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="telegram.polling.state_changed",
                message=f"Polling state changed: {old_state.value} -> {state.value}",
                old_state=old_state.value,
                new_state=state.value,
            )
    
    def _set_degraded(self, reason: str) -> None:
        """Enter degraded state."""
        self._health.state = PollingState.DEGRADED
        self._health.degradation_reason = reason
        self._metrics.polling_degraded(reason)
        
        log_telegram_event(
            self._logger,
            level=logging.WARNING,
            event="telegram.polling.degraded",
            message="Polling entered degraded state",
            reason=reason,
            consecutive_errors=self._health.consecutive_errors,
        )
    
    def _on_error(self, classified: ClassifiedError) -> None:
        """Handle error, update state and counters."""
        self._errors_count += 1
        self._health.consecutive_errors += 1
        self._health.last_error_time = time.time()
        self._health.error_samples.append(classified.message)
        
        # Keep only last 5 error samples
        if len(self._health.error_samples) > 5:
            self._health.error_samples = self._health.error_samples[-5:]
        
        # Check for degradation threshold
        if (self._health.consecutive_errors >= self._backoff_config.degradation_threshold
                and self._health.state == PollingState.HEALTHY):
            self._set_degraded(f"consecutive_errors:{self._health.consecutive_errors}")
    
    def _on_success(self) -> None:
        """Handle successful operation."""
        self._health.consecutive_errors = 0
        self._health.last_success_time = time.time()
        
        if self._health.state == PollingState.DEGRADED:
            self._health.state = PollingState.RECOVERING
            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="telegram.polling.recovering",
                message="Polling entering recovery mode after success",
            )
        elif self._health.state == PollingState.RECOVERING:
            self._set_state(PollingState.HEALTHY)
    
    def _on_recovery(self) -> None:
        """Handle recovery completion."""
        self._metrics.polling_recovered()
        self._set_state(PollingState.HEALTHY)
        
        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="telegram.polling.recovered",
            message="Polling fully recovered",
        )


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
    
    # Unknown errors - be conservative, mark as non-retryable
    return ClassifiedError(
        error_class=ErrorClass.NON_RETRYABLE_API,
        severity=ErrorSeverity.WARNING,
        message=f"Unknown error: {exc}",
        details={"error_type": error_type},
    )


# Import here to avoid circular reference
import telegram_bot.handlers.dispatcher
import telegram_bot.config
