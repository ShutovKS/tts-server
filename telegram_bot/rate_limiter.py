# FILE: telegram_bot/rate_limiter.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Implement per-user rate limiting for Telegram bot commands.
#   SCOPE: Token bucket rate limiter with per-user tracking
#   DEPENDS: M-CONFIG
#   LINKS: M-TELEGRAM
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LOGGER - Module logger for Telegram rate limiting events
#   RateLimitDecision - Outcome payload for Telegram rate-limit checks
#   UserRateLimitState - Per-user sliding window request state
#   TelegramRateLimiter - Per-user rate limiter for bot commands
#   create_telegram_rate_limiter - Build a Telegram rate limiter from settings
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, function contracts, semantic blocks, and migrated log events to block-reference format]
# END_CHANGE_SUMMARY

"""
Per-user rate limiting for Telegram bot.

This module provides rate limiting aligned with admission control policies
from core/application/admission_control.py.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from telegram_bot.config import TelegramSettings


LOGGER = logging.getLogger(__name__)


# START_CONTRACT: RateLimitDecision
#   PURPOSE: Describe the outcome of a Telegram rate-limit check.
#   INPUTS: { allowed: bool - request admission result, limit: int - configured request cap, window_seconds: int - sliding window size, current_count: int - active request count, retry_after_seconds: float | None - optional retry delay }
#   OUTPUTS: { RateLimitDecision - rate-limit decision payload }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: RateLimitDecision
@dataclass
class RateLimitDecision:
    """Decision from rate limit check."""

    allowed: bool
    limit: int
    window_seconds: int
    current_count: int
    retry_after_seconds: float | None = None


# START_CONTRACT: UserRateLimitState
#   PURPOSE: Track recent Telegram request timestamps for a single user.
#   INPUTS: { requests: deque[float] - recorded request times in monotonic seconds }
#   OUTPUTS: { UserRateLimitState - mutable per-user throttling state }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: UserRateLimitState
@dataclass
class UserRateLimitState:
    """Rate limit state for a single user."""

    requests: deque[float] = field(default_factory=deque)

    # START_CONTRACT: is_allowed
    #   PURPOSE: Decide whether another request fits within the sliding rate-limit window.
    #   INPUTS: { limit: int - max requests allowed in the window, window_seconds: int - sliding window duration }
    #   OUTPUTS: { tuple[bool, int, float | None] - allow flag, current count, and optional retry delay }
    #   SIDE_EFFECTS: Mutates stored request timestamps when the request is accepted.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: is_allowed
    def is_allowed(
        self, limit: int, window_seconds: int
    ) -> tuple[bool, int, float | None]:
        """Check if request is allowed.

        Returns:
            Tuple of (allowed, current_count, retry_after_seconds)
        """
        now = time.monotonic()
        cutoff = now - window_seconds

        # Remove expired entries
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

        # Check limit
        if len(self.requests) >= limit:
            # Calculate retry after
            oldest = self.requests[0]
            retry_after = oldest + window_seconds - now
            return False, len(self.requests), max(0.0, retry_after)

        # Allow and record
        self.requests.append(now)
        return True, len(self.requests), None


# START_CONTRACT: TelegramRateLimiter
#   PURPOSE: Enforce in-memory per-user Telegram command throttling.
#   INPUTS: { settings: TelegramSettings - Telegram configuration with rate-limit policy }
#   OUTPUTS: { TelegramRateLimiter - configured limiter instance }
#   SIDE_EFFECTS: Maintains mutable in-memory request history for Telegram users.
#   LINKS: M-TELEGRAM
# END_CONTRACT: TelegramRateLimiter
class TelegramRateLimiter:
    """
    Per-user rate limiter for Telegram bot.

    This implementation is self-contained and doesn't require external
    rate limit backends. It uses in-memory sliding window algorithm.
    """

    def __init__(self, settings: TelegramSettings):
        """Initialize rate limiter with settings."""
        self._settings = settings
        self._user_states: dict[int, UserRateLimitState] = defaultdict(
            UserRateLimitState
        )
        self._lock = asyncio.Lock()
        self._window_seconds = 60  # 1 minute window
        self._limit = settings.telegram_rate_limit_per_user_per_minute
        self._enabled = settings.telegram_rate_limit_enabled
        self._dev_mode = settings.telegram_dev_mode

    # START_CONTRACT: check_and_consume
    #   PURPOSE: Evaluate and consume a Telegram user's rate-limit quota for a command.
    #   INPUTS: { user_id: int | str - Telegram user identifier }
    #   OUTPUTS: { RateLimitDecision - allow or deny decision with quota details }
    #   SIDE_EFFECTS: Updates in-memory request history for non-admin users when enabled.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: check_and_consume
    def check_and_consume(self, user_id: int | str) -> RateLimitDecision:
        """Check and consume rate limit for user.

        Args:
            user_id: Telegram user ID

        Returns:
            RateLimitDecision with allowed status and metadata
        """
        if not self._enabled:
            return RateLimitDecision(
                allowed=True,
                limit=self._limit,
                window_seconds=self._window_seconds,
                current_count=0,
            )

        # Admins bypass rate limiting always (both dev and production)
        if self._settings.is_admin_user(user_id):
            return RateLimitDecision(
                allowed=True,
                limit=self._limit,
                window_seconds=self._window_seconds,
                current_count=0,
            )

        uid = int(user_id)
        state = self._user_states[uid]

        allowed, count, retry_after = state.is_allowed(
            self._limit,
            self._window_seconds,
        )

        if not allowed:
            LOGGER.warning(
                f"Rate limit exceeded for user {user_id}",
                extra={"limit": self._limit, "retry_after": retry_after},
            )

        return RateLimitDecision(
            allowed=allowed,
            limit=self._limit,
            window_seconds=self._window_seconds,
            current_count=count,
            retry_after_seconds=retry_after,
        )

    # START_CONTRACT: reset_user
    #   PURPOSE: Clear stored rate-limit history for a Telegram user.
    #   INPUTS: { user_id: int | str - Telegram user identifier }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Removes the user's in-memory throttle state.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: reset_user
    def reset_user(self, user_id: int | str) -> None:
        """Reset rate limit state for user."""
        uid = int(user_id)
        if uid in self._user_states:
            del self._user_states[uid]

    # START_CONTRACT: get_stats
    #   PURPOSE: Report current rate-limit statistics for a Telegram user.
    #   INPUTS: { user_id: int | str - Telegram user identifier }
    #   OUTPUTS: { dict - current throttle metadata for the user }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: get_stats
    def get_stats(self, user_id: int | str) -> dict:
        """Get rate limit stats for user."""
        uid = int(user_id)
        state = self._user_states.get(uid)

        if state is None:
            return {"enabled": self._enabled, "user_known": False}

        now = time.monotonic()
        cutoff = now - self._window_seconds

        # Count active requests
        active = sum(1 for ts in state.requests if ts >= cutoff)

        return {
            "enabled": self._enabled,
            "user_known": True,
            "active_requests": active,
            "limit": self._limit,
            "window_seconds": self._window_seconds,
        }

    # START_CONTRACT: check_and_consume_async
    #   PURPOSE: Perform a concurrency-safe asynchronous Telegram rate-limit check.
    #   INPUTS: { user_id: int | str - Telegram user identifier }
    #   OUTPUTS: { RateLimitDecision - allow or deny decision with quota details }
    #   SIDE_EFFECTS: Serializes access to and may update in-memory request history.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: check_and_consume_async
    async def check_and_consume_async(self, user_id: int | str) -> RateLimitDecision:
        """Async version of check_and_consume."""
        async with self._lock:
            return self.check_and_consume(user_id)

    # START_CONTRACT: is_enabled
    #   PURPOSE: Expose whether Telegram rate limiting is currently enabled.
    #   INPUTS: {}
    #   OUTPUTS: { bool - True when throttling is active }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: is_enabled
    @property
    def is_enabled(self) -> bool:
        """Check if rate limiting is enabled."""
        return self._enabled

    # START_CONTRACT: limit_per_minute
    #   PURPOSE: Expose the configured per-minute Telegram request quota.
    #   INPUTS: {}
    #   OUTPUTS: { int - allowed requests per minute }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: limit_per_minute
    @property
    def limit_per_minute(self) -> int:
        """Get configured limit per minute."""
        return self._limit


# START_CONTRACT: create_telegram_rate_limiter
#   PURPOSE: Build a Telegram rate limiter from resolved settings.
#   INPUTS: { settings: TelegramSettings - Telegram runtime settings }
#   OUTPUTS: { TelegramRateLimiter - configured rate limiter instance }
#   SIDE_EFFECTS: Initializes in-memory limiter state.
#   LINKS: M-TELEGRAM
# END_CONTRACT: create_telegram_rate_limiter
def create_telegram_rate_limiter(settings: TelegramSettings) -> TelegramRateLimiter:
    """Factory function to create rate limiter from settings."""
    return TelegramRateLimiter(settings)

__all__ = [
    "LOGGER",
    "RateLimitDecision",
    "UserRateLimitState",
    "TelegramRateLimiter",
    "create_telegram_rate_limiter",
]
