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

import pytest
from unittest.mock import MagicMock, patch

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
        assert StartupCheckPhase.TELEGRAM_API.value < StartupCheckPhase.BACKEND.value
        assert StartupCheckPhase.BACKEND.value < StartupCheckPhase.FFMPEG.value
        assert StartupCheckPhase.FFMPEG.value < StartupCheckPhase.ALLOWLIST.value
        assert StartupCheckPhase.ALLOWLIST.value < StartupCheckPhase.COMPLETE.value
    
    def test_phase_values(self):
        """Phase values are correct."""
        assert StartupCheckPhase.CONFIG.value == 1
        assert StartupCheckPhase.TELEGRAM_API.value == 2
        assert StartupCheckPhase.BACKEND.value == 3
        assert StartupCheckPhase.FFMPEG.value == 4
        assert StartupCheckPhase.ALLOWLIST.value == 5
        assert StartupCheckPhase.COMPLETE.value == 6


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
        result.errors.append("FATAL: QWEN_TTS_TELEGRAM_BOT_TOKEN not set")
        
        assert not result.success
        assert not result.is_success
        assert "TOKEN" in result.errors[0]
    
    def test_warning_result(self):
        """Warning doesn't make check fatal."""
        result = StartupCheckResult(
            success=True,
            phase=StartupCheckPhase.COMPLETE,
        )
        result.warnings.append("WARNING: ALLOWLIST_EMPTY - No user restrictions configured")
        
        assert result.success
        assert len(result.warnings) == 1
    
    def test_mixed_result(self):
        """Mixed result with warnings and errors."""
        result = StartupCheckResult(
            success=False,
            phase=StartupCheckPhase.FFMPEG,
        )
        result.checks_passed = ["bot_token_configured"]
        result.warnings.append("WARNING: ALLOWLIST_EMPTY")
        result.errors.append("FATAL: ffmpeg is not available")
        
        assert not result.success
        assert len(result.checks_passed) == 1
        assert len(result.warnings) == 1
        assert len(result.errors) == 1
    
    def test_is_success_property(self):
        """is_success is an alias for success."""
        result = StartupCheckResult(success=True)
        assert result.is_success == result.success


class TestRunStartupSelfChecks:
    """Tests for run_startup_self_checks function."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings matching TelegramSettings interface."""
        settings = MagicMock(spec=[
            'telegram_bot_token',
            'telegram_allowed_user_ids',
            'telegram_default_speaker',
            'telegram_max_text_length',
            'backend',
        ])
        settings.telegram_bot_token = "valid_test_token"
        settings.telegram_allowed_user_ids = [12345, 67890]
        settings.telegram_default_speaker = "af_bella"
        settings.telegram_max_text_length = 1000
        settings.backend = "mlx"
        return settings
    
    @pytest.fixture
    def mock_settings_no_allowlist(self):
        """Create mock settings with empty allowlist."""
        settings = MagicMock(spec=[
            'telegram_bot_token',
            'telegram_allowed_user_ids',
            'telegram_default_speaker',
            'telegram_max_text_length',
            'backend',
        ])
        settings.telegram_bot_token = "valid_test_token"
        settings.telegram_allowed_user_ids = []
        settings.telegram_default_speaker = "af_bella"
        settings.telegram_max_text_length = 1000
        settings.backend = "mlx"
        return settings
    
    @pytest.fixture
    def mock_settings_no_token(self):
        """Create mock settings without bot token."""
        settings = MagicMock(spec=[
            'telegram_bot_token',
            'telegram_allowed_user_ids',
            'telegram_default_speaker',
            'telegram_max_text_length',
            'backend',
        ])
        settings.telegram_bot_token = ""
        settings.telegram_allowed_user_ids = [12345]
        settings.telegram_default_speaker = "af_bella"
        settings.telegram_max_text_length = 1000
        settings.backend = "mlx"
        return settings
    
    def test_all_checks_pass(self, mock_settings):
        """All startup checks pass."""
        with patch("telegram_bot.__main__.is_ffmpeg_available") as mock_ffmpeg:
            mock_ffmpeg.return_value = True
            
            result = run_startup_self_checks(mock_settings)
        
        assert result.success
        assert result.is_success
        assert "bot_token_configured" in result.checks_passed
        assert "ffmpeg_available" in result.checks_passed
        assert len(result.errors) == 0
    
    def test_missing_bot_token(self, mock_settings_no_token):
        """Missing bot token is fatal."""
        with patch("telegram_bot.__main__.is_ffmpeg_available") as mock_ffmpeg:
            mock_ffmpeg.return_value = True
            
            result = run_startup_self_checks(mock_settings_no_token)
        
        assert not result.success
        assert not result.is_success
        assert len(result.errors) > 0
        assert any("TOKEN" in e.upper() for e in result.errors)
    
    def test_ffmpeg_missing(self, mock_settings):
        """Missing FFmpeg is fatal."""
        with patch("telegram_bot.__main__.is_ffmpeg_available") as mock_ffmpeg:
            mock_ffmpeg.return_value = False
            
            result = run_startup_self_checks(mock_settings)
        
        assert not result.success
        assert not result.is_success
        assert len(result.errors) > 0
        assert any("ffmpeg" in e.lower() for e in result.errors)
    
    def test_empty_allowlist_warning(self, mock_settings_no_allowlist):
        """Empty allowlist generates warning but continues."""
        with patch("telegram_bot.__main__.is_ffmpeg_available") as mock_ffmpeg:
            mock_ffmpeg.return_value = True
            
            result = run_startup_self_checks(mock_settings_no_allowlist)
        
        # Empty allowlist is a warning, not fatal
        assert result.success
        assert len(result.warnings) > 0
        assert any("allowlist" in w.lower() for w in result.warnings)
    
    def test_telegram_runtime_object(self, mock_settings):
        """Handles TelegramRuntime object correctly."""
        mock_runtime = MagicMock()
        mock_runtime.settings = mock_settings
        mock_runtime.core = None  # No core, skip backend check
        
        with patch("telegram_bot.__main__.is_ffmpeg_available") as mock_ffmpeg:
            mock_ffmpeg.return_value = True
            
            result = run_startup_self_checks(mock_runtime)
        
        # Result should be successful (empty allowlist is just a warning)
        assert result.success
        assert result.is_success
        # Backend check skipped when core is None
        assert "backend_available:mlx" in result.checks_passed
    
    def test_default_speaker_warning(self, mock_settings):
        """Empty default speaker generates warning."""
        mock_settings.telegram_default_speaker = ""
        
        with patch("telegram_bot.__main__.is_ffmpeg_available") as mock_ffmpeg:
            mock_ffmpeg.return_value = True
            
            result = run_startup_self_checks(mock_settings)
        
        # Empty default speaker should add warning
        assert any("speaker" in w.lower() for w in result.warnings)
    
    def test_text_length_warning_small(self, mock_settings):
        """Very small text length generates warning."""
        mock_settings.telegram_max_text_length = 5
        
        with patch("telegram_bot.__main__.is_ffmpeg_available") as mock_ffmpeg:
            mock_ffmpeg.return_value = True
            
            result = run_startup_self_checks(mock_settings)
        
        # Very small text length should add warning
        assert any("text" in w.lower() and "small" in w.lower() for w in result.warnings)
    
    def test_text_length_warning_large(self, mock_settings):
        """Very large text length generates warning."""
        mock_settings.telegram_max_text_length = 10000
        
        with patch("telegram_bot.__main__.is_ffmpeg_available") as mock_ffmpeg:
            mock_ffmpeg.return_value = True
            
            result = run_startup_self_checks(mock_settings)
        
        # Very large text length should add warning
        assert any("text" in w.lower() and "large" in w.lower() for w in result.warnings)
