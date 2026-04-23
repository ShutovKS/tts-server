"""
Unit tests for Telegram bot configuration and bootstrap.

Tests TelegramSettings.from_env, parse_core_settings_from_env integration,
and bootstrap/runtime assembly.
"""

# FILE: tests/unit/test_telegram_bot/test_config_bootstrap.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for Telegram settings parsing and runtime bootstrap.
#   SCOPE: Env parsing, validation, runtime construction, operational settings
#   DEPENDS: M-TELEGRAM
#   LINKS: V-M-TELEGRAM
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _make_env - Helper that builds environment dictionaries for Telegram settings tests
#   TestTelegramSettingsFromEnv - Verifies Telegram and core settings parsing from environment variables
#   TestTelegramSettingsValidation - Verifies Telegram settings validation behavior
#   TestAllowlistPolicy - Verifies allowlist behavior through settings helpers
#   TestTelegramRuntimeBootstrap - Verifies Telegram runtime bootstrap and validation before build
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT and MODULE_MAP]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from telegram_bot.bootstrap import build_telegram_runtime, TelegramRuntime
from telegram_bot.config import TelegramSettings
from telegram_bot.remote_client import RemoteServerClient


def _make_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Create test environment with required variables."""
    env = {
        "TTS_TELEGRAM_BOT_TOKEN": "test_token_123:ABCabc123",
        "TTS_TELEGRAM_SERVER_BASE_URL": "http://server.internal:8000",
        "TTS_MODELS_DIR": ".models",
        "TTS_OUTPUTS_DIR": ".outputs",
        "TTS_VOICES_DIR": ".voices",
    }
    if overrides:
        env.update(overrides)
    return env


class TestTelegramSettingsFromEnv:
    """Tests for TelegramSettings.from_env method."""

    def test_from_env_returns_telegram_settings(self):
        """Test that from_env returns TelegramSettings instance."""
        env = _make_env()
        settings = TelegramSettings.from_env(env)

        assert isinstance(settings, TelegramSettings)

    def test_from_env_parses_telegram_specific_vars(self):
        """Test that from_env parses Telegram-specific environment variables."""
        env = _make_env(
            {
                "TTS_TELEGRAM_BOT_TOKEN": "my_token",
                "TTS_TELEGRAM_ALLOWED_USER_IDS": "111,222,333",
                "TTS_TELEGRAM_LOG_LEVEL": "debug",
                "TTS_TELEGRAM_DEFAULT_SPEAKER": "Alex",
                "TTS_TELEGRAM_MAX_TEXT_LENGTH": "500",
            }
        )

        settings = TelegramSettings.from_env(env)

        assert settings.telegram_bot_token == "my_token"
        assert settings.telegram_allowed_user_ids == ("111", "222", "333")
        assert settings.telegram_log_level == "debug"
        assert settings.telegram_default_speaker == "Alex"
        assert settings.telegram_max_text_length == 500

    def test_from_env_uses_parse_core_settings_from_env(self):
        """Test that from_env uses parse_core_settings_from_env for core settings."""
        env = _make_env(
            {
                "TTS_SAMPLE_RATE": "48000",
                "TTS_BACKEND": "mlx",
                "TTS_AUTH_MODE": "static_bearer",
                "TTS_AUTH_STATIC_BEARER_TOKEN": "secret_token",
                "TTS_RATE_LIMIT_ENABLED": "true",
            }
        )

        settings = TelegramSettings.from_env(env)

        # These should be set via parse_core_settings_from_env
        assert settings.sample_rate == 48000
        assert settings.backend == "mlx"
        assert settings.auth_mode == "static_bearer"
        assert settings.auth_static_bearer_token == "secret_token"
        assert settings.rate_limit_enabled is True

    def test_from_env_has_all_core_settings(self):
        """Test that from_env sets all core settings fields."""
        env = _make_env()

        settings = TelegramSettings.from_env(env)

        # Core settings that should be available
        assert hasattr(settings, "models_dir")
        assert hasattr(settings, "mlx_models_dir")
        assert hasattr(settings, "outputs_dir")
        assert hasattr(settings, "voices_dir")
        assert hasattr(settings, "upload_staging_dir")
        assert hasattr(settings, "model_manifest_path")
        assert hasattr(settings, "backend")
        assert hasattr(settings, "backend_autoselect")
        assert hasattr(settings, "model_preload_policy")
        assert hasattr(settings, "model_preload_ids")
        assert hasattr(settings, "job_execution_backend")
        assert hasattr(settings, "job_metadata_backend")
        assert hasattr(settings, "job_artifact_backend")
        assert hasattr(settings, "auth_mode")
        assert hasattr(settings, "auth_static_bearer_token")
        assert hasattr(settings, "auth_static_bearer_principal_id")
        assert hasattr(settings, "auth_static_bearer_credential_id")
        assert hasattr(settings, "rate_limit_enabled")
        assert hasattr(settings, "rate_limit_backend")
        assert hasattr(settings, "rate_limit_sync_tts_per_minute")
        assert hasattr(settings, "rate_limit_async_submit_per_minute")
        assert hasattr(settings, "rate_limit_job_read_per_minute")
        assert hasattr(settings, "rate_limit_job_cancel_per_minute")
        assert hasattr(settings, "rate_limit_control_plane_per_minute")
        assert hasattr(settings, "quota_enabled")
        assert hasattr(settings, "quota_backend")
        assert hasattr(settings, "quota_compute_requests_per_window")
        assert hasattr(settings, "quota_compute_window_seconds")
        assert hasattr(settings, "quota_max_active_jobs_per_principal")
        assert hasattr(settings, "default_save_output")
        assert hasattr(settings, "max_upload_size_bytes")
        assert hasattr(settings, "max_input_text_chars")
        assert hasattr(settings, "request_timeout_seconds")
        assert hasattr(settings, "inference_busy_status_code")
        assert hasattr(settings, "sample_rate")
        assert hasattr(settings, "filename_max_len")
        assert hasattr(settings, "auto_play_cli")

    def test_from_env_defaults(self):
        """Test that from_env applies correct defaults."""
        # Create env without Telegram-specific vars to test defaults
        env = {
            "TTS_TELEGRAM_BOT_TOKEN": "test_token_123:ABCabc123",
            "TTS_MODELS_DIR": ".models",
            "TTS_OUTPUTS_DIR": ".outputs",
            "TTS_VOICES_DIR": ".voices",
        }

        settings = TelegramSettings.from_env(env)

        assert settings.telegram_allowed_user_ids == ()
        assert settings.telegram_log_level == "info"
        assert settings.telegram_default_speaker == "Vivian"
        assert settings.telegram_max_text_length == 1000


class TestTelegramSettingsValidation:
    """Tests for TelegramSettings validation."""

    def test_validate_requires_bot_token(self):
        """Test that validation fails without bot token."""
        env = _make_env({"TTS_TELEGRAM_BOT_TOKEN": ""})
        settings = TelegramSettings.from_env(env)

        errors = settings.validate()

        assert len(errors) > 0
        assert any("token" in e.lower() for e in errors)

    def test_validate_rejects_negative_max_text_length(self):
        """Test that validation rejects negative max text length."""
        env = _make_env({"TTS_TELEGRAM_MAX_TEXT_LENGTH": "-1"})
        settings = TelegramSettings.from_env(env)

        errors = settings.validate()

        assert len(errors) > 0
        assert any("length" in e.lower() for e in errors)

    def test_validate_passes_with_valid_settings(self):
        """Test that validation passes with valid settings."""
        env = _make_env()
        settings = TelegramSettings.from_env(env)

        errors = settings.validate()

        assert len(errors) == 0


class TestAllowlistPolicy:
    """Tests for allowlist policy in settings."""

    def test_is_user_allowed_empty_allowlist(self):
        """Test that empty allowlist allows all users."""
        env = _make_env()
        settings = TelegramSettings.from_env(env)

        assert settings.is_user_allowed(123) is True
        assert settings.is_user_allowed(456) is True
        assert settings.is_user_allowed("789") is True

    def test_is_user_allowed_with_allowlist(self):
        """Test that allowlist restricts access."""
        env = _make_env({"TTS_TELEGRAM_ALLOWED_USER_IDS": "111,222"})
        settings = TelegramSettings.from_env(env)

        assert settings.is_user_allowed(111) is True
        assert settings.is_user_allowed(222) is True
        assert settings.is_user_allowed("111") is True  # String comparison
        assert settings.is_user_allowed(333) is False
        assert settings.is_user_allowed("333") is False


class TestTelegramRuntimeBootstrap:
    """Tests for Telegram runtime bootstrap."""

    def test_build_telegram_runtime_returns_telegram_runtime(self):
        """Test that build_telegram_runtime returns TelegramRuntime."""
        env = _make_env({"TTS_TELEGRAM_SERVER_BASE_URL": "http://server.internal:8000"})
        settings = TelegramSettings.from_env(env)

        runtime = build_telegram_runtime(settings)

        assert isinstance(runtime, TelegramRuntime)
        assert runtime.settings is settings
        assert runtime.remote_server_client is not None

    def test_build_telegram_runtime_validates_settings(self):
        """Test that build_telegram_runtime validates settings before building."""
        env = _make_env({"TTS_TELEGRAM_BOT_TOKEN": ""})
        settings = TelegramSettings.from_env(env)

        with pytest.raises(ValueError) as exc_info:
            build_telegram_runtime(settings)

        assert "Invalid Telegram settings" in str(exc_info.value)

    def test_telegram_runtime_has_required_attributes(self):
        """Test that TelegramRuntime has required attributes."""
        env = _make_env({"TTS_TELEGRAM_SERVER_BASE_URL": "http://server.internal:8000"})
        settings = TelegramSettings.from_env(env)

        runtime = build_telegram_runtime(settings)

        assert hasattr(runtime, "settings")
        assert hasattr(runtime, "remote_server_client")
        assert runtime.settings is settings
        assert runtime.remote_server_client is not None

    def test_from_env_parses_delivery_store_and_poll_interval(self):
        """Telegram operational settings should be parsed from env."""
        env = _make_env(
            {
                "TTS_TELEGRAM_DELIVERY_STORE_PATH": "/tmp/telegram-delivery.json",
                "TTS_TELEGRAM_POLL_INTERVAL_SECONDS": "2.5",
                "TTS_TELEGRAM_MAX_RETRIES": "7",
                "TTS_TELEGRAM_SERVER_BASE_URL": "http://server.internal:8000/",
            }
        )
        settings = TelegramSettings.from_env(env)

        assert settings.telegram_delivery_store_path == "/tmp/telegram-delivery.json"
        assert settings.telegram_poll_interval_seconds == 2.5
        assert settings.telegram_max_retries == 7
        assert settings.telegram_server_base_url == "http://server.internal:8000"

    def test_from_env_requires_remote_server_base_url(self):
        """Telegram remote server base URL should be required."""
        env = _make_env({"TTS_TELEGRAM_SERVER_BASE_URL": ""})

        settings = TelegramSettings.from_env(env)

        assert settings.telegram_server_base_url == ""
        assert "TTS_TELEGRAM_SERVER_BASE_URL is required" in settings.validate()

    def test_build_telegram_runtime_wires_remote_server_client_when_configured(self):
        """Configured remote server URL should build a reusable runtime client."""
        env = _make_env(
            {"TTS_TELEGRAM_SERVER_BASE_URL": "http://server.internal:8000"}
        )
        settings = TelegramSettings.from_env(env)

        runtime = build_telegram_runtime(settings)

        assert isinstance(runtime.remote_server_client, RemoteServerClient)
        assert runtime.remote_server_client is not None
        assert runtime.remote_server_client.base_url == "http://server.internal:8000"

    def test_build_telegram_runtime_rejects_missing_remote_server_base_url(self):
        """Without a configured server URL, runtime bootstrap fails clearly."""
        env = _make_env({"TTS_TELEGRAM_SERVER_BASE_URL": ""})
        settings = TelegramSettings.from_env(env)

        with pytest.raises(ValueError) as exc_info:
            build_telegram_runtime(settings)

        assert "TTS_TELEGRAM_SERVER_BASE_URL is required" in str(exc_info.value)


class TestConfigPathResolution:
    """Tests for path resolution in configuration."""

    def test_models_dir_resolved_from_env(self):
        """Test that models_dir is resolved from environment variable."""
        env = _make_env({"TTS_MODELS_DIR": "/custom/models"})
        settings = TelegramSettings.from_env(env)

        assert settings.models_dir == Path("/custom/models").resolve()

    def test_mlx_models_dir_resolved_from_env(self):
        """Test that mlx_models_dir is resolved from environment variable."""
        env = _make_env({"TTS_MLX_MODELS_DIR": "/custom/mlx-models"})
        settings = TelegramSettings.from_env(env)

        assert settings.mlx_models_dir == Path("/custom/mlx-models").resolve()

    def test_outputs_dir_resolved_from_env(self):
        """Test that outputs_dir is resolved from environment variable."""
        env = _make_env({"TTS_OUTPUTS_DIR": "/custom/outputs"})
        settings = TelegramSettings.from_env(env)

        assert settings.outputs_dir == Path("/custom/outputs").resolve()

    def test_voices_dir_resolved_from_env(self):
        """Test that voices_dir is resolved from environment variable."""
        env = _make_env({"TTS_VOICES_DIR": "/custom/voices"})
        settings = TelegramSettings.from_env(env)

        assert settings.voices_dir == Path("/custom/voices").resolve()
