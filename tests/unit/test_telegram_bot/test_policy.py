"""
Unit tests for policy enforcement (private-only, allowlist).

Tests Telegram settings policy including user allowlist and chat type filtering.
"""

# FILE: tests/unit/test_telegram_bot/test_policy.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for Telegram settings policy enforcement and allowlists.
#   SCOPE: Allowlist checks, settings validation, env parsing, determinism
#   DEPENDS: M-TELEGRAM
#   LINKS: V-M-TELEGRAM
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _make_test_settings - Helper that builds minimal TelegramSettings fixtures for policy tests
#   TestAllowlistPolicy - Verifies allowlist matching semantics for ids and empty policies
#   TestSettingsValidation - Verifies Telegram settings validation failures and success cases
#   TestSettingsFromEnv - Verifies Telegram policy settings parsing from environment variables
#   TestPolicyDeterminism - Verifies allowlist results stay deterministic across repeated checks and ordering changes
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT and MODULE_MAP]
# END_CHANGE_SUMMARY

from __future__ import annotations

import pytest

from telegram_bot.config import TelegramSettings


# Helper to create minimal settings for testing policy logic
def _make_test_settings(**overrides):
    """Create TelegramSettings with minimal required fields for policy testing."""
    defaults = {
        "telegram_bot_token": "test_token_123:ABCabc123",
        "telegram_allowed_user_ids": (),
        "models_dir": ".models",
        "outputs_dir": ".outputs",
        "voices_dir": ".voices",
    }
    defaults.update(overrides)
    return TelegramSettings(**defaults)


class TestAllowlistPolicy:
    """Tests for user allowlist policy."""

    def test_empty_allowlist_allows_all_users(self):
        """Test that empty allowlist allows all users."""
        settings = _make_test_settings(
            telegram_allowed_user_ids=(),
        )

        # Any user should be allowed
        assert settings.is_user_allowed(12345) is True
        assert settings.is_user_allowed(67890) is True
        assert settings.is_user_allowed("12345") is True

    def test_allowlist_with_single_user(self):
        """Test allowlist with single user."""
        settings = _make_test_settings(
            telegram_allowed_user_ids=("12345",),
        )

        assert settings.is_user_allowed(12345) is True
        assert settings.is_user_allowed("12345") is True
        assert settings.is_user_allowed(67890) is False

    def test_allowlist_with_multiple_users(self):
        """Test allowlist with multiple users."""
        settings = _make_test_settings(
            telegram_allowed_user_ids=("12345", "67890", "11111"),
        )

        assert settings.is_user_allowed(12345) is True
        assert settings.is_user_allowed(67890) is True
        assert settings.is_user_allowed(11111) is True
        assert settings.is_user_allowed(99999) is False

    def test_allowlist_with_string_ids(self):
        """Test allowlist accepts string user IDs."""
        settings = _make_test_settings(
            telegram_allowed_user_ids=("111", "222", "333"),
        )

        # Both int and string lookups should work
        assert settings.is_user_allowed(111) is True
        assert settings.is_user_allowed("111") is True
        assert settings.is_user_allowed(444) is False

    def test_allowlist_checks_are_case_insensitive(self):
        """Test allowlist checks normalize to strings."""
        settings = _make_test_settings(
            telegram_allowed_user_ids=("12345",),
        )

        # Both int and string of same value should match
        assert settings.is_user_allowed(12345) is True
        assert settings.is_user_allowed("12345") is True


class TestSettingsValidation:
    """Tests for settings validation."""

    def test_valid_settings_no_errors(self):
        """Test that valid settings have no errors."""
        settings = _make_test_settings(
            telegram_bot_token="valid_token_123:ABCdefGHI",
        )

        errors = settings.validate()

        assert errors == []

    def test_missing_token_has_error(self):
        """Test that missing bot token has error."""
        settings = _make_test_settings(
            telegram_bot_token="",
        )

        errors = settings.validate()

        assert len(errors) == 1
        assert "TOKEN" in errors[0].upper()

    def test_whitespace_only_token_has_error(self):
        """Test that whitespace-only token has error."""
        settings = _make_test_settings(
            telegram_bot_token="   ",
        )

        errors = settings.validate()

        assert len(errors) == 1
        assert "TOKEN" in errors[0].upper()

    def test_invalid_max_text_length(self):
        """Test that non-positive max text length has error."""
        settings = _make_test_settings(
            telegram_max_text_length=0,
        )

        errors = settings.validate()

        assert any("MAX_TEXT_LENGTH" in e.upper() for e in errors)

    def test_negative_max_text_length(self):
        """Test that negative max text length has error."""
        settings = _make_test_settings(
            telegram_max_text_length=-100,
        )

        errors = settings.validate()

        assert any("MAX_TEXT_LENGTH" in e.upper() for e in errors)


class TestSettingsFromEnv:
    """Tests for settings parsing from environment."""

    def test_parse_telegram_settings_from_env(self):
        """Test parsing Telegram settings from environment."""
        environ = {
            # CoreSettings required fields
            "TTS_MODELS_DIR": ".models",
            "TTS_OUTPUTS_DIR": ".outputs",
            "TTS_VOICES_DIR": ".voices",
            # Telegram settings
            "TTS_TELEGRAM_BOT_TOKEN": "test_token",
            "TTS_TELEGRAM_ALLOWED_USER_IDS": "123,456,789",
            "TTS_TELEGRAM_LOG_LEVEL": "DEBUG",
            "TTS_TELEGRAM_DEFAULT_SPEAKER": "Alex",
            "TTS_TELEGRAM_MAX_TEXT_LENGTH": "500",
        }

        settings = TelegramSettings.from_env(environ)

        assert settings.telegram_bot_token == "test_token"
        assert settings.telegram_allowed_user_ids == ("123", "456", "789")
        assert settings.telegram_log_level == "DEBUG"
        assert settings.telegram_default_speaker == "Alex"
        assert settings.telegram_max_text_length == 500

    def test_parse_empty_allowlist(self):
        """Test parsing empty allowlist from environment."""
        environ = {
            "TTS_MODELS_DIR": ".models",
            "TTS_OUTPUTS_DIR": ".outputs",
            "TTS_VOICES_DIR": ".voices",
            "TTS_TELEGRAM_BOT_TOKEN": "test_token",
            "TTS_TELEGRAM_ALLOWED_USER_IDS": "",
        }

        settings = TelegramSettings.from_env(environ)

        assert settings.telegram_allowed_user_ids == ()

    def test_parse_default_speaker(self):
        """Test parsing default speaker from environment."""
        environ = {
            "TTS_MODELS_DIR": ".models",
            "TTS_OUTPUTS_DIR": ".outputs",
            "TTS_VOICES_DIR": ".voices",
            "TTS_TELEGRAM_BOT_TOKEN": "test_token",
        }

        settings = TelegramSettings.from_env(environ)

        assert settings.telegram_default_speaker == "Vivian"

    def test_parse_default_max_text_length(self):
        """Test parsing default max text length from environment."""
        environ = {
            "TTS_MODELS_DIR": ".models",
            "TTS_OUTPUTS_DIR": ".outputs",
            "TTS_VOICES_DIR": ".voices",
            "TTS_TELEGRAM_BOT_TOKEN": "test_token",
        }

        settings = TelegramSettings.from_env(environ)

        assert settings.telegram_max_text_length == 1000


class TestPolicyDeterminism:
    """Tests for policy determinism - same input always produces same result."""

    def test_same_user_always_allowed(self):
        """Test that same user ID always has same result."""
        settings = _make_test_settings(
            telegram_allowed_user_ids=("12345",),
        )

        # Multiple checks should always return same result
        for _ in range(10):
            assert settings.is_user_allowed(12345) is True
            assert settings.is_user_allowed(67890) is False

    def test_policy_not_affected_by_order(self):
        """Test that allowlist order doesn't affect results."""
        settings1 = _make_test_settings(
            telegram_allowed_user_ids=("1", "2", "3"),
        )
        settings2 = _make_test_settings(
            telegram_allowed_user_ids=("3", "1", "2"),
        )

        for user_id in ["1", "2", "3", "4", "5"]:
            assert settings1.is_user_allowed(user_id) == settings2.is_user_allowed(
                user_id
            )
