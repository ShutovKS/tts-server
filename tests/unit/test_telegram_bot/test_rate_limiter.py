"""
Tests for Telegram rate limiter.
"""

import time
from unittest.mock import MagicMock

import pytest

from telegram_bot.config import TelegramSettings
from telegram_bot.rate_limiter import (
    TelegramRateLimiter,
    UserRateLimitState,
    RateLimitDecision,
    create_telegram_rate_limiter,
)


class TestUserRateLimitState:
    """Tests for UserRateLimitState."""
    
    def test_first_request_allowed(self):
        """First request should always be allowed."""
        state = UserRateLimitState()
        allowed, count, retry_after = state.is_allowed(limit=5, window_seconds=60)
        
        assert allowed is True
        assert count == 1
        assert retry_after is None
    
    def test_under_limit_allowed(self):
        """Requests under limit should be allowed."""
        state = UserRateLimitState()
        
        # Make 4 requests
        for _ in range(4):
            state.is_allowed(limit=5, window_seconds=60)
        
        allowed, count, retry_after = state.is_allowed(limit=5, window_seconds=60)
        
        assert allowed is True
        assert count == 5
    
    def test_at_limit_rejected(self):
        """Requests at limit should be rejected."""
        state = UserRateLimitState()
        
        # Make 5 requests (at limit)
        for _ in range(5):
            state.is_allowed(limit=5, window_seconds=60)
        
        allowed, count, retry_after = state.is_allowed(limit=5, window_seconds=60)
        
        assert allowed is False
        assert count == 5
        assert retry_after is not None
        assert retry_after > 0
    
    def test_expired_entries_removed(self):
        """Expired entries should be removed from window."""
        state = UserRateLimitState()
        
        # Add request with old timestamp
        old_timestamp = time.monotonic() - 120  # 2 minutes ago
        state.requests.append(old_timestamp)
        
        # Add 4 recent requests
        for _ in range(4):
            state.is_allowed(limit=5, window_seconds=60)
        
        # Old entry should be expired, so we should be allowed
        allowed, count, retry_after = state.is_allowed(limit=5, window_seconds=60)
        
        assert allowed is True
        assert count == 5  # Only 4 recent + 1 new


class TestRateLimitDecision:
    """Tests for RateLimitDecision dataclass."""
    
    def test_allowed_decision(self):
        """Test allowed rate limit decision."""
        decision = RateLimitDecision(
            allowed=True,
            limit=20,
            window_seconds=60,
            current_count=5,
        )
        
        assert decision.allowed is True
        assert decision.retry_after_seconds is None
    
    def test_rejected_decision(self):
        """Test rejected rate limit decision."""
        decision = RateLimitDecision(
            allowed=False,
            limit=20,
            window_seconds=60,
            current_count=20,
            retry_after_seconds=30.0,
        )
        
        assert decision.allowed is False
        assert decision.retry_after_seconds == 30.0


class TestTelegramRateLimiter:
    """Tests for TelegramRateLimiter."""
    
    def _create_settings(
        self,
        rate_limit_enabled: bool = True,
        rate_limit_per_user_per_minute: int = 20,
        dev_mode: bool = False,
        admin_user_ids: tuple = (),
    ) -> TelegramSettings:
        """Create test settings."""
        settings = MagicMock(spec=TelegramSettings)
        settings.telegram_rate_limit_enabled = rate_limit_enabled
        settings.telegram_rate_limit_per_user_per_minute = rate_limit_per_user_per_minute
        settings.telegram_dev_mode = dev_mode
        settings.telegram_admin_user_ids = admin_user_ids
        settings.is_admin_user = lambda uid: str(uid) in admin_user_ids
        return settings
    
    def test_rate_limiter_disabled(self):
        """When rate limiting is disabled, all requests are allowed."""
        settings = self._create_settings(rate_limit_enabled=False)
        limiter = TelegramRateLimiter(settings)
        
        decision = limiter.check_and_consume(123)
        
        assert decision.allowed is True
        assert decision.current_count == 0
    
    def test_first_request_allowed(self):
        """First request should be allowed."""
        settings = self._create_settings(rate_limit_enabled=True, rate_limit_per_user_per_minute=20)
        limiter = TelegramRateLimiter(settings)
        
        decision = limiter.check_and_consume(123)
        
        assert decision.allowed is True
        assert decision.current_count == 1
    
    def test_under_limit_allowed(self):
        """Requests under limit should be allowed."""
        settings = self._create_settings(rate_limit_enabled=True, rate_limit_per_user_per_minute=5)
        limiter = TelegramRateLimiter(settings)
        
        # Make 4 requests
        for _ in range(4):
            limiter.check_and_consume(123)
        
        # 5th request should still be allowed
        decision = limiter.check_and_consume(123)
        
        assert decision.allowed is True
    
    def test_at_limit_rejected(self):
        """Requests at limit should be rejected."""
        settings = self._create_settings(rate_limit_enabled=True, rate_limit_per_user_per_minute=5)
        limiter = TelegramRateLimiter(settings)
        
        # Make 5 requests (at limit)
        for _ in range(5):
            limiter.check_and_consume(123)
        
        # 6th request should be rejected
        decision = limiter.check_and_consume(123)
        
        assert decision.allowed is False
        assert decision.retry_after_seconds is not None
    
    def test_different_users_separate_limits(self):
        """Different users should have separate rate limits."""
        settings = self._create_settings(rate_limit_enabled=True, rate_limit_per_user_per_minute=2)
        limiter = TelegramRateLimiter(settings)
        
        # User 1 makes 2 requests
        for _ in range(2):
            limiter.check_and_consume(1)
        
        # User 1 at limit, should be rejected
        decision1 = limiter.check_and_consume(1)
        assert decision1.allowed is False
        
        # User 2 should still be allowed
        decision2 = limiter.check_and_consume(2)
        assert decision2.allowed is True
    
    def test_admin_bypasses_in_production(self):
        """Admins should bypass rate limiting in production (non-dev mode)."""
        settings = self._create_settings(
            rate_limit_enabled=True,
            rate_limit_per_user_per_minute=1,
            dev_mode=False,
            admin_user_ids=("999",),
        )
        limiter = TelegramRateLimiter(settings)
        
        # Admin makes many requests
        for _ in range(10):
            limiter.check_and_consume(999)
        
        # Admin should still be allowed
        decision = limiter.check_and_consume(999)
        assert decision.allowed is True
    
    def test_admin_bypasses_in_dev_mode(self):
        """Admins should bypass rate limiting in dev mode too."""
        settings = self._create_settings(
            rate_limit_enabled=True,
            rate_limit_per_user_per_minute=1,
            dev_mode=True,
            admin_user_ids=("999",),
        )
        limiter = TelegramRateLimiter(settings)
        
        # Admin makes many requests
        for _ in range(10):
            limiter.check_and_consume(999)
        
        decision = limiter.check_and_consume(999)
        assert decision.allowed is True
    
    def test_non_admin_respects_limit_in_dev_mode(self):
        """Non-admins should still respect rate limits in dev mode."""
        settings = self._create_settings(
            rate_limit_enabled=True,
            rate_limit_per_user_per_minute=2,
            dev_mode=True,  # Dev mode on
            admin_user_ids=("999",),  # Different admin
        )
        limiter = TelegramRateLimiter(settings)
        
        # Regular user makes 2 requests
        for _ in range(2):
            limiter.check_and_consume(123)
        
        # At limit, should be rejected
        decision = limiter.check_and_consume(123)
        assert decision.allowed is False
    
    def test_reset_user(self):
        """Reset should clear user's rate limit state."""
        settings = self._create_settings(rate_limit_enabled=True, rate_limit_per_user_per_minute=2)
        limiter = TelegramRateLimiter(settings)
        
        # Use up the limit
        for _ in range(2):
            limiter.check_and_consume(123)
        
        # Should be rejected
        assert limiter.check_and_consume(123).allowed is False
        
        # Reset
        limiter.reset_user(123)
        
        # Should be allowed again
        decision = limiter.check_and_consume(123)
        assert decision.allowed is True
    
    def test_get_stats_unknown_user(self):
        """Stats for unknown user should indicate user not known."""
        settings = self._create_settings(rate_limit_enabled=True)
        limiter = TelegramRateLimiter(settings)
        
        stats = limiter.get_stats(999)
        
        assert stats["enabled"] is True
        assert stats["user_known"] is False
    
    def test_get_stats_known_user(self):
        """Stats for known user should show request count."""
        settings = self._create_settings(rate_limit_enabled=True, rate_limit_per_user_per_minute=10)
        limiter = TelegramRateLimiter(settings)
        
        # Make some requests
        limiter.check_and_consume(123)
        limiter.check_and_consume(123)
        
        stats = limiter.get_stats(123)
        
        assert stats["enabled"] is True
        assert stats["user_known"] is True
        assert stats["active_requests"] == 2
        assert stats["limit"] == 10
    
    def test_is_enabled_property(self):
        """Test is_enabled property."""
        settings = self._create_settings(rate_limit_enabled=True)
        limiter = TelegramRateLimiter(settings)
        assert limiter.is_enabled is True
        
        settings = self._create_settings(rate_limit_enabled=False)
        limiter = TelegramRateLimiter(settings)
        assert limiter.is_enabled is False
    
    def test_limit_per_minute_property(self):
        """Test limit_per_minute property."""
        settings = self._create_settings(rate_limit_enabled=True, rate_limit_per_user_per_minute=30)
        limiter = TelegramRateLimiter(settings)
        assert limiter.limit_per_minute == 30


class TestCreateRateLimiter:
    """Tests for create_telegram_rate_limiter factory function."""
    
    def test_create_rate_limiter(self):
        """Test factory function creates rate limiter."""
        settings = MagicMock(spec=TelegramSettings)
        settings.telegram_rate_limit_enabled = True
        settings.telegram_rate_limit_per_user_per_minute = 20
        settings.telegram_dev_mode = False
        settings.telegram_admin_user_ids = ()
        
        limiter = create_telegram_rate_limiter(settings)
        
        assert isinstance(limiter, TelegramRateLimiter)
        assert limiter.is_enabled is True
        assert limiter.limit_per_minute == 20
