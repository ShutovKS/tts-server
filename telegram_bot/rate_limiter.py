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


@dataclass
class RateLimitDecision:
    """Decision from rate limit check."""
    
    allowed: bool
    limit: int
    window_seconds: int
    current_count: int
    retry_after_seconds: float | None = None


@dataclass
class UserRateLimitState:
    """Rate limit state for a single user."""
    
    requests: deque[float] = field(default_factory=deque)
    
    def is_allowed(self, limit: int, window_seconds: int) -> tuple[bool, int, float | None]:
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


class TelegramRateLimiter:
    """
    Per-user rate limiter for Telegram bot.
    
    This implementation is self-contained and doesn't require external
    rate limit backends. It uses in-memory sliding window algorithm.
    """
    
    def __init__(self, settings: TelegramSettings):
        """Initialize rate limiter with settings."""
        self._settings = settings
        self._user_states: dict[int, UserRateLimitState] = defaultdict(UserRateLimitState)
        self._lock = asyncio.Lock()
        self._window_seconds = 60  # 1 minute window
        self._limit = settings.telegram_rate_limit_per_user_per_minute
        self._enabled = settings.telegram_rate_limit_enabled
        self._dev_mode = settings.telegram_dev_mode
    
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
    
    def reset_user(self, user_id: int | str) -> None:
        """Reset rate limit state for user."""
        uid = int(user_id)
        if uid in self._user_states:
            del self._user_states[uid]
    
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
    
    async def check_and_consume_async(self, user_id: int | str) -> RateLimitDecision:
        """Async version of check_and_consume."""
        async with self._lock:
            return self.check_and_consume(user_id)
    
    @property
    def is_enabled(self) -> bool:
        """Check if rate limiting is enabled."""
        return self._enabled
    
    @property
    def limit_per_minute(self) -> int:
        """Get configured limit per minute."""
        return self._limit


def create_telegram_rate_limiter(settings: TelegramSettings) -> TelegramRateLimiter:
    """Factory function to create rate limiter from settings."""
    return TelegramRateLimiter(settings)
