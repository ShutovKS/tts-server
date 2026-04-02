"""
Telegram bot bootstrap and runtime assembly.

This module wires together the Telegram transport layer with the existing
core TTS infrastructure, providing a clean separation of concerns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

from core.bootstrap import CoreRuntime, build_runtime
from core.observability import get_logger, log_event
from telegram_bot.config import TelegramSettings
from telegram_bot.rate_limiter import TelegramRateLimiter, create_telegram_rate_limiter

if TYPE_CHECKING:
    from telegram_bot.rate_limiter import RateLimitDecision


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class TelegramRuntime:
    """Runtime container for Telegram bot transport layer."""
    
    settings: TelegramSettings
    core: CoreRuntime
    rate_limiter: TelegramRateLimiter = field(default=None)
    
    def check_rate_limit(self, user_id: int | str) -> "RateLimitDecision":
        """Check rate limit for user."""
        if self.rate_limiter is None:
            # Return allowed if no rate limiter configured
            from telegram_bot.rate_limiter import RateLimitDecision
            return RateLimitDecision(
                allowed=True,
                limit=0,
                window_seconds=0,
                current_count=0,
            )
        return self.rate_limiter.check_and_consume(user_id)


@lru_cache(maxsize=1)
def get_telegram_settings() -> TelegramSettings:
    """Get Telegram settings from environment with caching."""
    settings = TelegramSettings.from_env()
    settings.ensure_directories()
    return settings


def _validate_telegram_settings(settings: TelegramSettings) -> list[str]:
    """
    Validate Telegram-specific settings beyond basic validation.
    
    Returns:
        List of warning messages (non-fatal issues)
    """
    warnings = []
    
    # Check for potential configuration issues
    if not settings.telegram_allowed_user_ids:
        warnings.append(
            "ALLOWLIST_WARNING: telegram_allowed_user_ids is empty. "
            "Consider restricting access in production."
        )
    
    # Check default speaker
    from telegram_bot.handlers.commands import get_valid_speakers, VALID_SPEAKERS
    if settings.telegram_default_speaker not in VALID_SPEAKERS:
        warnings.append(
            f"DEFAULT_SPEAKER_WARNING: '{settings.telegram_default_speaker}' is not in "
            f"the list of available speakers. Available: {', '.join(sorted(VALID_SPEAKERS))}"
        )
    
    # Check text length limits
    if settings.telegram_max_text_length < 10:
        warnings.append(
            f"TEXT_LENGTH_WARNING: telegram_max_text_length is very small ({settings.telegram_max_text_length}). "
            "Users may not be able to send meaningful text."
        )
    
    return warnings


def build_telegram_runtime(settings: Optional[TelegramSettings] = None) -> TelegramRuntime:
    """
    Build Telegram runtime by composing core runtime with Telegram settings.
    
    Args:
        settings: Optional Telegram settings. If None, loads from environment.
        
    Returns:
        TelegramRuntime containing core runtime and Telegram-specific settings.
        
    Raises:
        ValueError: If required settings are missing or invalid
    """
    log_event(
        LOGGER,
        level=logging.INFO,
        event="telegram.bootstrap.starting",
        message="Building Telegram runtime",
    )
    
    resolved_settings = settings or get_telegram_settings()
    
    # Validate required settings
    errors = resolved_settings.validate()
    if errors:
        log_event(
            LOGGER,
            level=logging.ERROR,
            event="telegram.bootstrap.validation_failed",
            message="Telegram settings validation failed",
            errors=errors,
        )
        raise ValueError(f"Invalid Telegram settings: {', '.join(errors)}")
    
    log_event(
        LOGGER,
        level=logging.INFO,
        event="telegram.bootstrap.settings_loaded",
        message="Telegram settings loaded",
        default_speaker=resolved_settings.telegram_default_speaker,
        max_text_length=resolved_settings.telegram_max_text_length,
        allowed_users_count=len(resolved_settings.telegram_allowed_user_ids),
    )
    
    # Perform additional validation with warnings
    validation_warnings = _validate_telegram_settings(resolved_settings)
    for warning in validation_warnings:
        log_event(
            LOGGER,
            level=logging.WARNING,
            event="telegram.bootstrap.setting_warning",
            message=warning,
        )
    
    # Build core runtime using inherited settings
    log_event(
        LOGGER,
        level=logging.INFO,
        event="telegram.bootstrap.core_building",
        message="Building core runtime",
        backend=resolved_settings.backend,
    )
    
    core_runtime = build_runtime(resolved_settings)
    
    log_event(
        LOGGER,
        level=logging.INFO,
        event="telegram.bootstrap.complete",
        message="Telegram runtime built successfully",
    )
    
    # Create rate limiter
    rate_limiter = create_telegram_rate_limiter(resolved_settings)
    
    if rate_limiter.is_enabled:
        log_event(
            LOGGER,
            level=logging.INFO,
            event="telegram.bootstrap.rate_limiter_enabled",
            message="Rate limiter enabled",
            limit_per_minute=rate_limiter.limit_per_minute,
        )
    
    return TelegramRuntime(
        settings=resolved_settings,
        core=core_runtime,
        rate_limiter=rate_limiter,
    )
