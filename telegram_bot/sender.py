# FILE: telegram_bot/sender.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Send audio results back to Telegram users as voice messages.
#   SCOPE: Audio delivery, voice message formatting, delivery metadata persistence
#   DEPENDS: M-TELEGRAM
#   LINKS: M-TELEGRAM
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LOGGER - Module logger for Telegram sender events
#   DeliveryRetryConfig - Retry policy for Telegram voice delivery attempts
#   DeliveryResult - Outcome payload for Telegram voice delivery
#   MessageSender - Protocol for Telegram text and voice delivery
#   TelegramSender - Telegram delivery service with conversion and retry logic
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, function contracts, semantic blocks, and migrated log events to block-reference format]
# END_CHANGE_SUMMARY

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


# START_CONTRACT: DeliveryRetryConfig
#   PURPOSE: Configure retry behavior for Telegram voice delivery attempts.
#   INPUTS: { max_attempts: int - maximum delivery attempts, initial_delay: float - initial retry delay, max_delay: float - delay ceiling, multiplier: float - exponential factor }
#   OUTPUTS: { DeliveryRetryConfig - immutable delivery retry policy }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: DeliveryRetryConfig
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


# START_CONTRACT: DeliveryResult
#   PURPOSE: Describe the outcome of sending a Telegram voice message.
#   INPUTS: { success: bool - delivery result, error_message: Optional[str] - failure detail, attempts: int - attempt count, duration_ms: float - total elapsed time, error_class: Optional[str] - classified failure type, is_retryable: bool - retry guidance }
#   OUTPUTS: { DeliveryResult - delivery outcome payload }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: DeliveryResult
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


# START_CONTRACT: MessageSender
#   PURPOSE: Define the public Telegram message delivery interface used by handlers and pollers.
#   INPUTS: {}
#   OUTPUTS: { MessageSender - protocol for text and voice delivery }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: MessageSender
class MessageSender(Protocol):
    """Protocol for sending messages to Telegram."""

    # START_CONTRACT: send_text
    #   PURPOSE: Send a text message to a Telegram chat through the delivery interface.
    #   INPUTS: { chat_id: int - target Telegram chat identifier, text: str - message body }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Sends a Telegram API message.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: send_text
    async def send_text(self, chat_id: int, text: str) -> None:
        """Send text message to chat."""
        ...

    # START_CONTRACT: send_voice
    #   PURPOSE: Send a voice message to a Telegram chat through the delivery interface.
    #   INPUTS: { chat_id: int - target Telegram chat identifier, audio_bytes: bytes - voice payload bytes, caption: str | None - optional caption }
    #   OUTPUTS: { DeliveryResult - delivery outcome payload }
    #   SIDE_EFFECTS: Sends a Telegram API voice message.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: send_voice
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


# START_CONTRACT: TelegramSender
#   PURPOSE: Convert generated audio and deliver Telegram text or voice messages with retry handling.
#   INPUTS: { client: TelegramBotClient - Telegram API client, settings: TelegramSettings - Telegram runtime settings, logger: logging.Logger | None - optional logger, metrics: Optional[TelegramMetrics] - optional metrics collector, retry_config: Optional[DeliveryRetryConfig] - delivery retry policy }
#   OUTPUTS: { TelegramSender - configured Telegram delivery service }
#   SIDE_EFFECTS: Performs audio conversion, sends Telegram API requests, and may notify users of failures.
#   LINKS: M-TELEGRAM
# END_CONTRACT: TelegramSender
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

    # START_CONTRACT: send_text
    #   PURPOSE: Send a formatted text message to a Telegram chat.
    #   INPUTS: { chat_id: int - target Telegram chat identifier, text: str - Markdown message body }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Performs a Telegram API request and logs delivery results.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: send_text
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
                event="[Sender][send_text][send_text]",
                message="Telegram text message sent",
                chat_id=chat_id,
                text_length=len(text),
            )
        except Exception as exc:
            classified = classify_telegram_error(exc)

            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="[Sender][send_text][send_text]",
                message=f"Failed to send text message: {classified.message}",
                chat_id=chat_id,
                error=str(exc),
                error_class=classified.error_class.value,
            )
            raise

    # START_CONTRACT: send_voice
    #   PURPOSE: Convert WAV audio and deliver it as a Telegram voice message with retries.
    #   INPUTS: { chat_id: int - target Telegram chat identifier, audio_bytes: bytes - WAV audio payload, caption: str | None - optional caption }
    #   OUTPUTS: { DeliveryResult - delivery outcome payload }
    #   SIDE_EFFECTS: Invokes ffmpeg conversion, sends Telegram API requests, and may send fallback error text.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: send_voice
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
        # START_BLOCK_PREPARE_DELIVERY
        timer = Timer()
        attempt = 0
        last_error: Exception | None = None
        last_classified = None
        # END_BLOCK_PREPARE_DELIVERY

        # START_BLOCK_FORMAT_AUDIO
        # Step 1: Audio conversion (no retry)
        try:
            self._metrics.conversion_started()
            log_telegram_event(
                self._logger,
                level=logging.DEBUG,
                event="[Sender][send_voice][BLOCK_FORMAT_AUDIO]",
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
                event="[Sender][send_voice][BLOCK_FORMAT_AUDIO]",
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
                event="[Sender][send_voice][BLOCK_FORMAT_AUDIO]",
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
        # END_BLOCK_FORMAT_AUDIO

        # START_BLOCK_SEND_VOICE_MESSAGE
        # Step 2: Voice delivery with retry
        while attempt < self._retry_config.max_attempts:
            attempt += 1
            self._metrics.delivery_started()

            log_telegram_event(
                self._logger,
                level=logging.DEBUG,
                event="[Sender][send_voice][BLOCK_SEND_VOICE_MESSAGE]",
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
                    event="[Sender][send_voice][BLOCK_SEND_VOICE_MESSAGE]",
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
                    event="[Sender][send_voice][BLOCK_SEND_VOICE_MESSAGE]",
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
                        event="[Sender][send_voice][BLOCK_SEND_VOICE_MESSAGE]",
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
                    event="[Sender][send_voice][BLOCK_SEND_VOICE_MESSAGE]",
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
            event="[Sender][send_voice][BLOCK_SEND_VOICE_MESSAGE]",
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
            error_message=last_classified.message
            if last_classified
            else "Max retries exceeded",
            attempts=attempt,
            duration_ms=timer.elapsed_ms,
            error_class=last_classified.error_class.value
            if last_classified
            else "unknown",
            is_retryable=False,
        )
        # END_BLOCK_SEND_VOICE_MESSAGE

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
                event="[Sender][_notify_error][_notify_error]",
                message="Failed to send error notification to user",
                chat_id=chat_id,
            )


# Import Protocol for type hints
from typing import Protocol
import telegram_bot.observability

__all__ = [
    "LOGGER",
    "DeliveryRetryConfig",
    "DeliveryResult",
    "MessageSender",
    "TelegramSender",
]
