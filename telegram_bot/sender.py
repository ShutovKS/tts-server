"""
Telegram message sender with audio conversion and retry logic.

This module provides the MessageSender implementation that integrates
with the Telegram client and audio conversion utilities, featuring:
- Retry logic for transient delivery failures
- Error classification for voice send operations
- Structured logging with timing metrics
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol

from core.errors import AudioConversionError
from core.observability import get_logger, log_event, Timer
from telegram_bot.audio import convert_wav_to_telegram_ogg
from telegram_bot.client import TelegramBotClient, TelegramAPIError
from telegram_bot.observability import (
    METRICS,
    TelegramMetrics,
    classify_telegram_error,
    log_telegram_event,
)

if TYPE_CHECKING:
    from telegram_bot.config import TelegramSettings


LOGGER = get_logger(__name__)


# ============================================================================
# Retry Configuration
# ============================================================================

@dataclass(frozen=True)
class DeliveryRetryConfig:
    """Configuration for voice delivery retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 15.0
    multiplier: float = 2.0


# ============================================================================
# Delivery Result
# ============================================================================

@dataclass
class DeliveryResult:
    """Result of voice delivery attempt."""
    success: bool
    error_message: Optional[str] = None
    attempts: int = 1
    duration_ms: float = 0.0
    error_class: Optional[str] = None
    is_retryable: bool = False


# ============================================================================
# Message Sender Protocol
# ============================================================================

class MessageSender(Protocol):
    """Protocol for sending messages to Telegram."""
    
    async def send_text(self, chat_id: int, text: str) -> None:
        """Send text message to chat."""
        ...
    
    async def send_voice(
        self,
        chat_id: int,
        audio_bytes: bytes,
        caption: str | None = None,
    ) -> DeliveryResult:
        """Send voice message to chat."""
        ...


# ============================================================================
# Telegram Sender
# ============================================================================

class TelegramSender:
    """
    Telegram message sender with audio conversion and retry logic.
    
    Features:
    - Automatic retry for transient voice delivery failures
    - Audio conversion from WAV to Telegram-compatible OGG format
    - Structured logging with timing and error classification
    - Error user notification on failures
    
    Retry strategy:
    - Network timeouts and 5xx errors: exponential backoff
    - Rate limits (429): wait and retry
    - Client errors (4xx): fail immediately
    """
    
    def __init__(
        self,
        client: TelegramBotClient,
        settings: TelegramSettings,
        logger: logging.Logger | None = None,
        metrics: Optional[TelegramMetrics] = None,
        retry_config: Optional[DeliveryRetryConfig] = None,
    ):
        """
        Initialize Telegram sender.
        
        Args:
            client: Telegram bot client
            settings: Telegram settings including sample_rate
            logger: Optional logger instance
            metrics: Optional metrics collector
            retry_config: Optional retry configuration
        """
        self._client = client
        self._settings = settings
        self._logger = logger or LOGGER
        self._metrics = metrics or METRICS
        self._retry_config = retry_config or DeliveryRetryConfig()
    
    async def send_text(self, chat_id: int, text: str) -> None:
        """
        Send text message to chat.
        
        Args:
            chat_id: Target chat ID
            text: Message text (supports Markdown)
        """
        try:
            await self._client.send_message(chat_id, text, parse_mode="Markdown")
            
            log_telegram_event(
                self._logger,
                level=logging.DEBUG,
                event="telegram.message.sent",
                message="Telegram text message sent",
                chat_id=chat_id,
                text_length=len(text),
            )
        except Exception as exc:
            classified = classify_telegram_error(exc)
            
            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.message.failed",
                message=f"Failed to send text message: {classified.message}",
                chat_id=chat_id,
                error=str(exc),
                error_class=classified.error_class.value,
            )
            raise
    
    async def send_voice(
        self,
        chat_id: int,
        audio_bytes: bytes,
        caption: str | None = None,
    ) -> DeliveryResult:
        """
        Send voice message to chat with retry logic.
        
        This method:
        1. Converts WAV audio to Telegram-compatible OGG format
        2. Attempts to send with automatic retry on transient failures
        3. Returns detailed result with timing and error info
        
        Args:
            chat_id: Target chat ID
            audio_bytes: WAV audio bytes
            caption: Optional caption for the voice message
            
        Returns:
            DeliveryResult with success status and details
        """
        timer = Timer()
        attempt = 0
        last_error: Exception | None = None
        last_classified = None
        
        # Step 1: Audio conversion (no retry)
        try:
            self._metrics.conversion_started()
            log_telegram_event(
                self._logger,
                level=logging.DEBUG,
                event="telegram.voice.converting",
                message="Converting audio for Telegram voice",
                chat_id=chat_id,
                input_size=len(audio_bytes),
            )
            
            ogg_bytes, _ = convert_wav_to_telegram_ogg(audio_bytes, self._settings)
            conversion_duration = timer.elapsed_ms
            
            self._metrics.conversion_completed(conversion_duration)
            
            log_telegram_event(
                self._logger,
                level=logging.DEBUG,
                event="telegram.voice.conversion_completed",
                message="Audio conversion completed",
                chat_id=chat_id,
                input_size=len(audio_bytes),
                ogg_size=len(ogg_bytes),
                conversion_duration_ms=conversion_duration,
            )
            
        except AudioConversionError as exc:
            self._metrics.conversion_failed(type(exc).__name__)
            
            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.voice.conversion_failed",
                message=f"Audio conversion failed: {exc}",
                chat_id=chat_id,
                error=str(exc),
            )
            
            # Send error message to user
            await self._notify_error(
                chat_id,
                "❌ *Audio Conversion Error*\n\n"
                "Failed to convert audio for Telegram. Please try again later.",
            )
            
            return DeliveryResult(
                success=False,
                error_message=f"Conversion failed: {exc}",
                error_class="conversion_error",
                is_retryable=False,
            )
        
        # Step 2: Voice delivery with retry
        while attempt < self._retry_config.max_attempts:
            attempt += 1
            self._metrics.delivery_started()
            
            log_telegram_event(
                self._logger,
                level=logging.DEBUG,
                event="telegram.voice.delivery_attempt",
                message=f"Voice delivery attempt {attempt}/{self._retry_config.max_attempts}",
                chat_id=chat_id,
                attempt=attempt,
                ogg_size=len(ogg_bytes),
            )
            
            try:
                await self._client.send_voice(
                    chat_id,
                    ogg_bytes,
                    caption=caption,
                )
                
                delivery_duration = timer.elapsed_ms
                
                self._metrics.delivery_completed(delivery_duration)
                
                log_telegram_event(
                    self._logger,
                    level=logging.INFO,
                    event="telegram.voice.sent",
                    message="Telegram voice message sent",
                    chat_id=chat_id,
                    input_size=len(audio_bytes),
                    ogg_size=len(ogg_bytes),
                    duration_ms=delivery_duration,
                    attempts=attempt,
                )
                
                return DeliveryResult(
                    success=True,
                    attempts=attempt,
                    duration_ms=delivery_duration,
                )
                
            except TelegramAPIError as exc:
                last_error = exc
                last_classified = classify_telegram_error(exc)
                
                log_telegram_event(
                    self._logger,
                    level=logging.WARNING,
                    event="telegram.voice.delivery_error",
                    message=f"Voice delivery error: {last_classified.message}",
                    chat_id=chat_id,
                    attempt=attempt,
                    error=str(exc),
                    error_code=exc.code,
                    error_class=last_classified.error_class.value,
                    severity=last_classified.severity.value,
                    retryable=last_classified.is_retryable,
                )
                
                self._metrics.delivery_failed(
                    last_classified.error_class.value,
                    last_classified.is_retryable,
                )
                
                # Check if we should retry
                if not last_classified.is_retryable:
                    # Non-retryable error - fail immediately
                    self._metrics.delivery_exhausted()
                    
                    await self._notify_error(
                        chat_id,
                        "❌ *Send Error*\n\n"
                        "Failed to send voice message. Please try again later.",
                    )
                    
                    return DeliveryResult(
                        success=False,
                        error_message=last_classified.message,
                        attempts=attempt,
                        duration_ms=timer.elapsed_ms,
                        error_class=last_classified.error_class.value,
                        is_retryable=False,
                    )
                
                # Retryable error - wait and retry
                if attempt < self._retry_config.max_attempts:
                    delay = self._calculate_delay(attempt, last_classified)
                    self._metrics.delivery_retried(attempt)
                    
                    log_telegram_event(
                        self._logger,
                        level=logging.INFO,
                        event="telegram.voice.retrying",
                        message=f"Retrying voice delivery after {delay:.2f}s",
                        chat_id=chat_id,
                        delay_seconds=delay,
                        attempt=attempt,
                    )
                    
                    await asyncio.sleep(delay)
                    continue
                    
            except Exception as exc:
                last_error = exc
                last_classified = classify_telegram_error(exc)
                
                log_telegram_event(
                    self._logger,
                    level=logging.ERROR,
                    event="telegram.voice.delivery_exception",
                    message=f"Voice delivery exception: {exc}",
                    chat_id=chat_id,
                    attempt=attempt,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                
                self._metrics.delivery_failed(
                    last_classified.error_class.value,
                    last_classified.is_retryable,
                )
                
                # For unknown errors, be conservative - don't retry indefinitely
                if attempt >= self._retry_config.max_attempts:
                    self._metrics.delivery_exhausted()
                    
                    await self._notify_error(
                        chat_id,
                        "❌ *Send Error*\n\n"
                        "Failed to send voice message. Please try again later.",
                    )
                    
                    return DeliveryResult(
                        success=False,
                        error_message=str(exc),
                        attempts=attempt,
                        duration_ms=timer.elapsed_ms,
                        error_class=last_classified.error_class.value,
                        is_retryable=False,
                    )
                
                delay = self._calculate_delay(attempt, last_classified)
                await asyncio.sleep(delay)
        
        # All retries exhausted
        self._metrics.delivery_exhausted()
        
        log_telegram_event(
            self._logger,
            level=logging.ERROR,
            event="telegram.voice.delivery_exhausted",
            message="Voice delivery retry attempts exhausted",
            chat_id=chat_id,
            total_attempts=attempt,
        )
        
        await self._notify_error(
            chat_id,
            "❌ *Send Error*\n\n"
            "Failed to send voice message after multiple attempts. Please try again later.",
        )
        
        return DeliveryResult(
            success=False,
            error_message=last_classified.message if last_classified else "Max retries exceeded",
            attempts=attempt,
            duration_ms=timer.elapsed_ms,
            error_class=last_classified.error_class.value if last_classified else "unknown",
            is_retryable=False,
        )
    
    def _calculate_delay(
        self,
        attempt: int,
        classified: telegram_bot.observability.ClassifiedError,
    ) -> float:
        """Calculate delay before next retry attempt."""
        # Use retry_after from rate limit if available
        if classified.retry_after:
            return min(classified.retry_after, self._retry_config.max_delay)
        
        # Exponential backoff
        delay = self._retry_config.initial_delay * (
            self._retry_config.multiplier ** (attempt - 1)
        )
        return min(delay, self._retry_config.max_delay)
    
    async def _notify_error(self, chat_id: int, message: str) -> None:
        """Send error notification to user (best effort)."""
        try:
            await self.send_text(chat_id, message)
        except Exception:
            # Ignore nested errors - don't fail if we can't notify
            log_telegram_event(
                self._logger,
                level=logging.WARNING,
                event="telegram.notification.failed",
                message="Failed to send error notification to user",
                chat_id=chat_id,
            )


# Import Protocol for type hints
from typing import Protocol
import telegram_bot.observability
