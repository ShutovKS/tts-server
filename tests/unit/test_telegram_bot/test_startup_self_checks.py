"""
Tests for startup self-check behavior.

These tests verify:
- Startup check phases and validation
- Fatal vs warning errors
- Bot token validation
- Backend availability check
- FFmpeg availability check
- Allowlist validation
"""

# FILE: tests/unit/test_telegram_bot/test_startup_self_checks.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for Telegram startup self-checks and validation gates.
#   SCOPE: Startup phases, token validation, backend checks, ffmpeg checks
#   DEPENDS: M-TELEGRAM
#   LINKS: V-M-TELEGRAM
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   TestStartupCheckPhase - Verifies startup self-check phase ordering and constants
#   TestStartupCheckResult - Verifies startup self-check result state, warnings, and failures
#   TestRunStartupSelfChecks - Verifies settings validation, token, ffmpeg, allowlist, and runtime startup validation flows
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Expanded startup self-check coverage for invalid Telegram settings and runtime validation failures]
# END_CHANGE_SUMMARY

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from telegram_bot.__main__ import (
    StartupCheckResult,
    StartupCheckPhase,
    run_startup_self_checks,
)


class TestStartupCheckPhase:
    """Tests for StartupCheckPhase enum."""

    def test_phase_order(self):
        """Phases are properly ordered by their int values."""
        assert StartupCheckPhase.CONFIG.value < StartupCheckPhase.TELEGRAM_API.value
        assert StartupCheckPhase.TELEGRAM_API.value < StartupCheckPhase.REMOTE_SERVER.value
        assert StartupCheckPhase.REMOTE_SERVER.value < StartupCheckPhase.ALLOWLIST.value
        assert StartupCheckPhase.ALLOWLIST.value < StartupCheckPhase.COMPLETE.value

    def test_phase_values(self):
        """Phase values are correct."""
        assert StartupCheckPhase.CONFIG.value == 1
        assert StartupCheckPhase.TELEGRAM_API.value == 2
        assert StartupCheckPhase.REMOTE_SERVER.value == 3
        assert StartupCheckPhase.ALLOWLIST.value == 4
        assert StartupCheckPhase.COMPLETE.value == 5


class TestStartupCheckResult:
    """Tests for StartupCheckResult class."""

    def test_success_result(self):
        """Successful check has all success flags."""
        result = StartupCheckResult(
            success=True,
            phase=StartupCheckPhase.COMPLETE,
        )
        result.checks_passed = ["bot_token_configured", "backend_available:mlx"]

        assert result.success
        assert result.is_success
        assert result.phase == StartupCheckPhase.COMPLETE

    def test_failed_result(self):
        """Failed check has failure details."""
        result = StartupCheckResult(
            success=False,
            phase=StartupCheckPhase.CONFIG,
        )
        result.errors.append("FATAL: TTS_TELEGRAM_BOT_TOKEN not set")

        assert not result.success
        assert not result.is_success
        assert "TOKEN" in result.errors[0]

    def test_warning_result(self):
        """Warning doesn't make check fatal."""
        result = StartupCheckResult(
            success=True,
            phase=StartupCheckPhase.COMPLETE,
        )
        result.warnings.append(
            "WARNING: ALLOWLIST_EMPTY - No user restrictions configured"
        )

        assert result.success
        assert len(result.warnings) == 1

    def test_mixed_result(self):
        """Mixed result with warnings and errors."""
        result = StartupCheckResult(
            success=False,
            phase=StartupCheckPhase.REMOTE_SERVER,
        )
        result.checks_passed = ["bot_token_configured"]
        result.warnings.append("WARNING: ALLOWLIST_EMPTY")
        result.errors.append("FATAL: ffmpeg is not available")

        assert not result.success
        assert len(result.checks_passed) == 1
        assert len(result.warnings) == 1
        assert len(result.errors) == 1

    def test_summary_handles_string_checks(self):
        """Summary maps stored string check names to passed entries."""
        result = StartupCheckResult(
            success=True,
            phase=StartupCheckPhase.COMPLETE,
        )
        result.checks_passed = ["bot_token_configured", "remote_server_ready"]

        summary = result.summary()

        assert summary["success"] is True
        assert summary["phase"] == StartupCheckPhase.COMPLETE.value
        assert summary["bot_token_configured"] == {"passed": True, "message": None}
        assert summary["remote_server_ready"] == {"passed": True, "message": None}

    def test_is_success_property(self):
        """is_success is an alias for success."""
        result = StartupCheckResult(success=True)
        assert result.is_success == result.success


class TestRunStartupSelfChecks:
    """Tests for run_startup_self_checks function."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings matching TelegramSettings interface."""
        settings = MagicMock(
            spec=[
                "telegram_bot_token",
                "telegram_allowed_user_ids",
                "telegram_default_speaker",
                "telegram_max_text_length",
                "telegram_server_base_url",
                "validate",
            ]
        )
        settings.telegram_bot_token = "valid_test_token"
        settings.telegram_allowed_user_ids = [12345, 67890]
        settings.telegram_default_speaker = "af_bella"
        settings.telegram_max_text_length = 1000
        settings.telegram_server_base_url = "http://server.internal:8000"
        settings.validate.return_value = []
        return settings

    @pytest.fixture
    def mock_settings_no_allowlist(self):
        """Create mock settings with empty allowlist."""
        settings = MagicMock(
            spec=[
                "telegram_bot_token",
                "telegram_allowed_user_ids",
                "telegram_default_speaker",
                "telegram_max_text_length",
                "telegram_server_base_url",
                "validate",
            ]
        )
        settings.telegram_bot_token = "valid_test_token"
        settings.telegram_allowed_user_ids = []
        settings.telegram_default_speaker = "af_bella"
        settings.telegram_max_text_length = 1000
        settings.telegram_server_base_url = "http://server.internal:8000"
        settings.validate.return_value = []
        return settings

    @pytest.fixture
    def mock_settings_no_token(self):
        """Create mock settings without bot token."""
        settings = MagicMock(
            spec=[
                "telegram_bot_token",
                "telegram_allowed_user_ids",
                "telegram_default_speaker",
                "telegram_max_text_length",
                "telegram_server_base_url",
                "validate",
            ]
        )
        settings.telegram_bot_token = ""
        settings.telegram_allowed_user_ids = [12345]
        settings.telegram_default_speaker = "af_bella"
        settings.telegram_max_text_length = 1000
        settings.telegram_server_base_url = "http://server.internal:8000"
        settings.validate.return_value = ["TTS_TELEGRAM_BOT_TOKEN is required"]
        return settings

    @pytest.mark.anyio
    async def test_all_checks_pass(self, mock_settings):
        """All startup checks pass."""
        mock_runtime = MagicMock()
        mock_runtime.settings = mock_settings
        mock_runtime.remote_server_client = MagicMock()
        mock_runtime.remote_server_client.get_readiness = AsyncMock(return_value=MagicMock(status="ok"))

        with patch("telegram_bot.__main__.verify_telegram_connectivity", AsyncMock(return_value=True)):
            result = await run_startup_self_checks(mock_runtime, client=MagicMock())

        assert result.success
        assert result.is_success
        assert result.phase == StartupCheckPhase.COMPLETE
        assert "bot_token_configured" in result.checks_passed
        assert "remote_server_ready" in result.checks_passed
        assert "telegram_api_reachable" in result.checks_passed
        assert len(result.errors) == 0

    @pytest.mark.anyio
    async def test_missing_bot_token(self, mock_settings_no_token):
        """Missing bot token is fatal."""
        mock_runtime = MagicMock()
        mock_runtime.settings = mock_settings_no_token
        mock_runtime.remote_server_client = MagicMock()
        mock_runtime.remote_server_client.get_readiness = AsyncMock(return_value=MagicMock(status="ok"))
        client = MagicMock()
        client.get_me = AsyncMock(return_value={"id": 1, "username": "bot", "first_name": "bot"})

        result = await run_startup_self_checks(mock_runtime, client=client)

        assert not result.success
        assert not result.is_success
        assert len(result.errors) > 0
        assert any("TOKEN" in e.upper() for e in result.errors)

    @pytest.mark.anyio
    async def test_short_bot_token_validation_is_fatal(self, mock_settings):
        """Invalid short Telegram token from settings validation is fatal."""
        mock_settings.validate.return_value = [
            "TTS_TELEGRAM_BOT_TOKEN appears to be invalid (too short)"
        ]
        mock_runtime = MagicMock()
        mock_runtime.settings = mock_settings
        mock_runtime.remote_server_client = MagicMock()
        mock_runtime.remote_server_client.get_readiness = AsyncMock(return_value=MagicMock(status="ok"))
        client = MagicMock()
        client.get_me = AsyncMock(return_value={"id": 1, "username": "bot", "first_name": "bot"})

        result = await run_startup_self_checks(mock_runtime, client=client)

        assert not result.success
        assert any("too short" in error.lower() for error in result.errors)

    @pytest.mark.anyio
    async def test_invalid_text_length_validation_is_fatal(self, mock_settings):
        """Settings validation errors for impossible text length abort startup."""
        mock_settings.validate.return_value = [
            "TTS_TELEGRAM_MAX_TEXT_LENGTH must be positive"
        ]
        mock_settings.telegram_max_text_length = -1
        mock_runtime = MagicMock()
        mock_runtime.settings = mock_settings
        mock_runtime.remote_server_client = MagicMock()
        mock_runtime.remote_server_client.get_readiness = AsyncMock(return_value=MagicMock(status="ok"))
        client = MagicMock()
        client.get_me = AsyncMock(return_value={"id": 1, "username": "bot", "first_name": "bot"})

        result = await run_startup_self_checks(mock_runtime, client=client)

        assert not result.success
        assert any("MAX_TEXT_LENGTH" in error for error in result.errors)

    @pytest.mark.anyio
    async def test_settings_validation_exception_is_fatal(self, mock_settings):
        """Unexpected settings validation failure aborts startup."""
        mock_settings.validate.side_effect = RuntimeError("validation exploded")
        mock_runtime = MagicMock()
        mock_runtime.settings = mock_settings
        mock_runtime.remote_server_client = MagicMock()
        mock_runtime.remote_server_client.get_readiness = AsyncMock(return_value=MagicMock(status="ok"))

        client = MagicMock()
        client.get_me = AsyncMock(return_value={"id": 1, "username": "bot", "first_name": "bot"})

        result = await run_startup_self_checks(mock_runtime, client=client)

        assert not result.success
        assert any("validation failed" in error.lower() for error in result.errors)

    @pytest.mark.anyio
    async def test_remote_server_readiness_failure_is_fatal(self, mock_settings):
        """Remote readiness failures abort startup."""
        mock_runtime = MagicMock()
        mock_runtime.settings = mock_settings
        mock_runtime.remote_server_client = MagicMock()
        mock_runtime.remote_server_client.get_readiness = AsyncMock(side_effect=RuntimeError("readiness probe failed"))

        client = MagicMock()
        client.get_me = AsyncMock(return_value={"id": 1, "username": "bot", "first_name": "bot"})

        result = await run_startup_self_checks(mock_runtime, client=client)

        assert not result.success
        assert any("readiness" in error.lower() for error in result.errors)

    @pytest.mark.anyio
    async def test_missing_remote_server_base_url_is_fatal(self, mock_settings):
        """Missing remote server configuration aborts startup."""
        mock_settings.telegram_server_base_url = ""
        mock_runtime = MagicMock()
        mock_runtime.settings = mock_settings
        mock_runtime.remote_server_client = None

        client = MagicMock()
        client.get_me = AsyncMock(return_value={"id": 1, "username": "bot", "first_name": "bot"})

        result = await run_startup_self_checks(mock_runtime, client=client)

        assert not result.success
        assert any("server_base_url" in error.lower() for error in result.errors)

    @pytest.mark.anyio
    async def test_remote_server_readiness_success(self, mock_settings):
        """Remote readiness success is recorded."""
        mock_runtime = MagicMock()
        mock_runtime.settings = mock_settings
        mock_runtime.remote_server_client = MagicMock()
        mock_runtime.remote_server_client.get_readiness = AsyncMock(return_value=MagicMock(status="ok"))
        client = MagicMock()
        client.get_me = AsyncMock(return_value={"id": 1, "username": "bot", "first_name": "bot"})

        result = await run_startup_self_checks(mock_runtime, client=client)

        assert result.success
        assert "remote_server_ready" in result.checks_passed

    @pytest.mark.anyio
    async def test_empty_allowlist_warning(self, mock_settings_no_allowlist):
        """Empty allowlist generates warning but continues."""
        mock_runtime = MagicMock()
        mock_runtime.settings = mock_settings_no_allowlist
        mock_runtime.remote_server_client = MagicMock()
        mock_runtime.remote_server_client.get_readiness = AsyncMock(return_value=MagicMock(status="ok"))
        client = MagicMock()
        client.get_me = AsyncMock(return_value={"id": 1, "username": "bot", "first_name": "bot"})

        result = await run_startup_self_checks(mock_runtime, client=client)

        # Empty allowlist is a warning, not fatal
        assert result.success
        assert len(result.warnings) > 0
        assert any("allowlist" in w.lower() for w in result.warnings)

    @pytest.mark.anyio
    async def test_telegram_runtime_object(self, mock_settings):
        """Handles TelegramRuntime object correctly."""
        mock_runtime = MagicMock()
        mock_runtime.settings = mock_settings
        mock_runtime.remote_server_client = MagicMock()
        mock_runtime.remote_server_client.get_readiness = AsyncMock(return_value=MagicMock(status="ok"))
        client = MagicMock()
        client.get_me = AsyncMock(return_value={"id": 1, "username": "bot", "first_name": "bot"})

        result = await run_startup_self_checks(mock_runtime, client=client)

        # Result should be successful (empty allowlist is just a warning)
        assert result.success
        assert result.is_success
        assert result.phase == StartupCheckPhase.COMPLETE
        assert "remote_server_ready" in result.checks_passed

    @pytest.mark.anyio
    async def test_default_speaker_warning(self, mock_settings):
        """Empty default speaker generates warning."""
        mock_settings.telegram_default_speaker = ""
        mock_runtime = MagicMock()
        mock_runtime.settings = mock_settings
        mock_runtime.remote_server_client = MagicMock()
        mock_runtime.remote_server_client.get_readiness = AsyncMock(return_value=MagicMock(status="ok"))

        result = await run_startup_self_checks(mock_runtime, client=MagicMock())

        # Empty default speaker should add warning
        assert any("speaker" in w.lower() for w in result.warnings)

    @pytest.mark.anyio
    async def test_text_length_warning_small(self, mock_settings):
        """Very small text length generates warning."""
        mock_settings.telegram_max_text_length = 5
        mock_runtime = MagicMock()
        mock_runtime.settings = mock_settings
        mock_runtime.remote_server_client = MagicMock()
        mock_runtime.remote_server_client.get_readiness = AsyncMock(return_value=MagicMock(status="ok"))

        result = await run_startup_self_checks(mock_runtime, client=MagicMock())

        # Very small text length should add warning
        assert any(
            "text" in w.lower() and "small" in w.lower() for w in result.warnings
        )

    @pytest.mark.anyio
    async def test_text_length_warning_large(self, mock_settings):
        """Very large text length generates warning."""
        mock_settings.telegram_max_text_length = 10000
        mock_runtime = MagicMock()
        mock_runtime.settings = mock_settings
        mock_runtime.remote_server_client = MagicMock()
        mock_runtime.remote_server_client.get_readiness = AsyncMock(return_value=MagicMock(status="ok"))

        result = await run_startup_self_checks(mock_runtime, client=MagicMock())

        # Very large text length should add warning
        assert any(
            "text" in w.lower() and "large" in w.lower() for w in result.warnings
        )
