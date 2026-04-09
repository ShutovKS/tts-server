# FILE: telegram_bot/bootstrap.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Build Telegram bot runtime by wiring core runtime with Telegram-specific components.
#   SCOPE: Bot runtime assembly, self-checks, startup validation
#   DEPENDS: M-BOOTSTRAP, M-CONFIG
#   LINKS: M-TELEGRAM
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LOGGER - Module logger for Telegram bootstrap events
#   TelegramRuntime - Runtime container for Telegram settings and core services
#   get_telegram_settings - Load and cache Telegram settings from environment
#   build_telegram_runtime - Factory for Telegram bot runtime assembly
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, function contracts, semantic blocks, and migrated log events to block-reference format]
# END_CHANGE_SUMMARY

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


# START_CONTRACT: TelegramRuntime
#   PURPOSE: Hold Telegram adapter settings, core runtime, and rate limiter dependencies.
#   INPUTS: { settings: TelegramSettings - resolved Telegram configuration, core: CoreRuntime - shared application runtime, rate_limiter: TelegramRateLimiter | None - optional throttling component }
#   OUTPUTS: { TelegramRuntime - immutable Telegram runtime container }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: TelegramRuntime
@dataclass(frozen=True)
class TelegramRuntime:
    """Runtime container for Telegram bot transport layer."""

    settings: TelegramSettings
    core: CoreRuntime
    rate_limiter: TelegramRateLimiter = field(default=None)

    # START_CONTRACT: check_rate_limit
    #   PURPOSE: Evaluate whether a Telegram user can issue another command right now.
    #   INPUTS: { user_id: int | str - Telegram user identifier }
    #   OUTPUTS: { RateLimitDecision - allow or deny decision with quota metadata }
    #   SIDE_EFFECTS: Consumes a rate-limit slot when a limiter is configured.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: check_rate_limit
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


# START_CONTRACT: get_telegram_settings
#   PURPOSE: Load and cache Telegram settings from the environment.
#   INPUTS: {}
#   OUTPUTS: { TelegramSettings - validated settings with ensured directories }
#   SIDE_EFFECTS: Ensures configured runtime directories exist on disk.
#   LINKS: M-TELEGRAM
# END_CONTRACT: get_telegram_settings
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


# START_CONTRACT: build_telegram_runtime
#   PURPOSE: Assemble the Telegram adapter runtime from settings and shared core services.
#   INPUTS: { settings: Optional[TelegramSettings] - optional prebuilt Telegram settings }
#   OUTPUTS: { TelegramRuntime - runtime container for Telegram execution }
#   SIDE_EFFECTS: Emits startup logs and initializes the Telegram rate limiter.
#   LINKS: M-TELEGRAM
# END_CONTRACT: build_telegram_runtime
def build_telegram_runtime(
    settings: Optional[TelegramSettings] = None,
) -> TelegramRuntime:
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
        event="[TelegramBootstrap][build_telegram_runtime][BUILD_TELEGRAM_RUNTIME]",
        message="Building Telegram runtime",
    )

    # START_BLOCK_INIT_SETTINGS
    resolved_settings = settings or get_telegram_settings()

    # Validate required settings
    errors = resolved_settings.validate()
    if errors:
        log_event(
            LOGGER,
            level=logging.ERROR,
            event="[TelegramBootstrap][build_telegram_runtime][BLOCK_INIT_SETTINGS]",
            message="Telegram settings validation failed",
            errors=errors,
        )
        raise ValueError(f"Invalid Telegram settings: {', '.join(errors)}")

    log_event(
        LOGGER,
        level=logging.INFO,
        event="[TelegramBootstrap][build_telegram_runtime][BLOCK_INIT_SETTINGS]",
        message="Telegram settings loaded",
        default_speaker=resolved_settings.telegram_default_speaker,
        max_text_length=resolved_settings.telegram_max_text_length,
        allowed_users_count=len(resolved_settings.telegram_allowed_user_ids),
    )
    # END_BLOCK_INIT_SETTINGS

    # START_BLOCK_RUN_SELF_CHECKS
    # Perform additional validation with warnings
    validation_warnings = _validate_telegram_settings(resolved_settings)
    for warning in validation_warnings:
        log_event(
            LOGGER,
            level=logging.WARNING,
            event="[TelegramBootstrap][build_telegram_runtime][BLOCK_RUN_SELF_CHECKS]",
            message=warning,
        )
    # END_BLOCK_RUN_SELF_CHECKS

    # START_BLOCK_BUILD_RUNTIME
    # Build core runtime using inherited settings
    log_event(
        LOGGER,
        level=logging.INFO,
        event="[TelegramBootstrap][build_telegram_runtime][BLOCK_BUILD_RUNTIME]",
        message="Building core runtime",
        backend=resolved_settings.backend,
    )

    core_runtime = build_runtime(resolved_settings)

    log_event(
        LOGGER,
        level=logging.INFO,
        event="[TelegramBootstrap][build_telegram_runtime][BLOCK_BUILD_RUNTIME]",
        message="Telegram runtime built successfully",
    )

    # Create rate limiter
    rate_limiter = create_telegram_rate_limiter(resolved_settings)

    if rate_limiter.is_enabled:
        log_event(
            LOGGER,
            level=logging.INFO,
            event="[TelegramBootstrap][build_telegram_runtime][BLOCK_BUILD_RUNTIME]",
            message="Rate limiter enabled",
            limit_per_minute=rate_limiter.limit_per_minute,
        )

    return TelegramRuntime(
        settings=resolved_settings,
        core=core_runtime,
        rate_limiter=rate_limiter,
    )
    # END_BLOCK_BUILD_RUNTIME

__all__ = [
    "LOGGER",
    "TelegramRuntime",
    "get_telegram_settings",
    "build_telegram_runtime",
]
