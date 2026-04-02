"""
Telegram bot configuration settings.

Environment variables:
- QWEN_TTS_TELEGRAM_BOT_TOKEN: Bot token from @BotFather (required)
- QWEN_TTS_TELEGRAM_ALLOWED_USER_IDS: Comma-separated list of Telegram user IDs (optional, empty = all users allowed)
- QWEN_TTS_TELEGRAM_LOG_LEVEL: Log level (default: info)
- QWEN_TTS_TELEGRAM_DEFAULT_SPEAKER: Default speaker name (default: Vivian)
- QWEN_TTS_TELEGRAM_MAX_TEXT_LENGTH: Maximum text length for /tts command (default: 1000)
- QWEN_TTS_TELEGRAM_DEV_MODE: Enable dev mode with relaxed security (default: false)
- QWEN_TTS_TELEGRAM_RATE_LIMIT_ENABLED: Enable per-user rate limiting (default: true)
- QWEN_TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE: Max requests per user per minute (default: 20)
- QWEN_TTS_TELEGRAM_JOB_TIMEOUT_SECONDS: Job execution timeout (default: from core settings)
- QWEN_TTS_TELEGRAM_DELIVERY_STORE_PATH: Path for delivery metadata store (optional)
- QWEN_TTS_TELEGRAM_POLL_INTERVAL_SECONDS: Job poller interval (default: 1.0)
- QWEN_TTS_TELEGRAM_MAX_RETRIES: Max retry attempts for API calls (default: 3)
- QWEN_TTS_TELEGRAM_ADMIN_USER_IDS: Comma-separated admin user IDs with elevated access (optional)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from core.config import (
    CoreSettings,
    DEFAULT_MODELS_DIR,
    DEFAULT_OUTPUTS_DIR,
    DEFAULT_VOICES_DIR,
    parse_core_settings_from_env,
    env_text,
    env_int,
    env_bool,
    env_path,
    _parse_csv_env,
)


# Default values for Telegram-specific settings
DEFAULT_TELEGRAM_LOG_LEVEL = "info"
DEFAULT_TELEGRAM_DEFAULT_SPEAKER = "Vivian"
DEFAULT_TELEGRAM_MAX_TEXT_LENGTH = 1000
DEFAULT_TELEGRAM_DEV_MODE = False
DEFAULT_TELEGRAM_RATE_LIMIT_ENABLED = True
DEFAULT_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE = 20
DEFAULT_TELEGRAM_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_TELEGRAM_MAX_RETRIES = 3


@dataclass(frozen=True)
class TelegramSecurityPolicy:
    """Security policy configuration for Telegram bot.
    
    This aligns with admission control policies from core/application/admission_control.py
    while providing Telegram-specific enforcement.
    """
    
    # Rate limiting
    rate_limit_enabled: bool = DEFAULT_TELEGRAM_RATE_LIMIT_ENABLED
    rate_limit_per_user_per_minute: int = DEFAULT_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE
    
    # Dev mode - relaxes certain security checks for development
    dev_mode: bool = DEFAULT_TELEGRAM_DEV_MODE
    
    # Admin users with elevated access
    admin_user_ids: tuple[str, ...] = field(default_factory=tuple)
    
    # Token security
    require_token_validation: bool = True
    token_min_length: int = 20  # Bot tokens are typically 40+ chars
    
    # Allowlist enforcement
    allowlist_strict_mode: bool = True  # If true, empty allowlist = only admins allowed
    
    def is_admin(self, user_id: int | str) -> bool:
        """Check if user is an admin."""
        return str(user_id) in self.admin_user_ids
    
    def should_enforce_rate_limit(self, user_id: int | str) -> bool:
        """Check if rate limiting should be enforced for user."""
        if not self.rate_limit_enabled:
            return False
        # Admins bypass rate limiting in production
        if not self.dev_mode and self.is_admin(user_id):
            return False
        return True
    
    def allow_empty_allowlist(self) -> bool:
        """Check if empty allowlist should allow all users."""
        return not self.allowlist_strict_mode or self.dev_mode


@dataclass(frozen=True)
class TelegramSettings(CoreSettings):
    """Telegram bot settings extending core settings."""
    
    # Telegram-specific settings
    telegram_bot_token: str = ""
    telegram_allowed_user_ids: tuple[str, ...] = field(default_factory=tuple)
    telegram_log_level: str = DEFAULT_TELEGRAM_LOG_LEVEL
    telegram_default_speaker: str = DEFAULT_TELEGRAM_DEFAULT_SPEAKER
    telegram_max_text_length: int = DEFAULT_TELEGRAM_MAX_TEXT_LENGTH
    
    # Security policy settings
    telegram_dev_mode: bool = DEFAULT_TELEGRAM_DEV_MODE
    telegram_rate_limit_enabled: bool = DEFAULT_TELEGRAM_RATE_LIMIT_ENABLED
    telegram_rate_limit_per_user_per_minute: int = DEFAULT_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE
    telegram_admin_user_ids: tuple[str, ...] = field(default_factory=tuple)
    
    # Operational settings
    telegram_delivery_store_path: str = ""
    telegram_poll_interval_seconds: float = DEFAULT_TELEGRAM_POLL_INTERVAL_SECONDS
    telegram_max_retries: int = DEFAULT_TELEGRAM_MAX_RETRIES
    
    @property
    def security_policy(self) -> TelegramSecurityPolicy:
        """Get security policy from current settings."""
        return TelegramSecurityPolicy(
            rate_limit_enabled=self.telegram_rate_limit_enabled,
            rate_limit_per_user_per_minute=self.telegram_rate_limit_per_user_per_minute,
            dev_mode=self.telegram_dev_mode,
            admin_user_ids=self.telegram_admin_user_ids,
            allowlist_strict_mode=True,
        )
    
    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "TelegramSettings":
        """Parse Telegram settings from environment variables.
        
        Uses parse_core_settings_from_env() to ensure consistency with core settings
        parsing, then adds Telegram-specific settings.
        """
        bot_token = env_text("QWEN_TTS_TELEGRAM_BOT_TOKEN", "", environ).strip()
        
        # Parse core settings using the shared parser for consistency
        core_settings = parse_core_settings_from_env(environ)
        
        return cls(
            # Core settings from shared parser
            models_dir=core_settings["models_dir"],
            outputs_dir=core_settings["outputs_dir"],
            voices_dir=core_settings["voices_dir"],
            upload_staging_dir=core_settings["upload_staging_dir"],
            model_manifest_path=core_settings["model_manifest_path"],
            backend=core_settings["backend"],
            backend_autoselect=core_settings["backend_autoselect"],
            model_preload_policy=core_settings["model_preload_policy"],
            model_preload_ids=core_settings["model_preload_ids"],
            job_execution_backend=core_settings["job_execution_backend"],
            job_metadata_backend=core_settings["job_metadata_backend"],
            job_artifact_backend=core_settings["job_artifact_backend"],
            auth_mode=core_settings["auth_mode"],
            auth_static_bearer_token=core_settings["auth_static_bearer_token"],
            auth_static_bearer_principal_id=core_settings["auth_static_bearer_principal_id"],
            auth_static_bearer_credential_id=core_settings["auth_static_bearer_credential_id"],
            rate_limit_enabled=core_settings["rate_limit_enabled"],
            rate_limit_backend=core_settings["rate_limit_backend"],
            rate_limit_sync_tts_per_minute=core_settings["rate_limit_sync_tts_per_minute"],
            rate_limit_async_submit_per_minute=core_settings["rate_limit_async_submit_per_minute"],
            rate_limit_job_read_per_minute=core_settings["rate_limit_job_read_per_minute"],
            rate_limit_job_cancel_per_minute=core_settings["rate_limit_job_cancel_per_minute"],
            rate_limit_control_plane_per_minute=core_settings["rate_limit_control_plane_per_minute"],
            quota_enabled=core_settings["quota_enabled"],
            quota_backend=core_settings["quota_backend"],
            quota_compute_requests_per_window=core_settings["quota_compute_requests_per_window"],
            quota_compute_window_seconds=core_settings["quota_compute_window_seconds"],
            quota_max_active_jobs_per_principal=core_settings["quota_max_active_jobs_per_principal"],
            default_save_output=core_settings["default_save_output"],
            enable_streaming=core_settings["enable_streaming"],
            max_upload_size_bytes=core_settings["max_upload_size_bytes"],
            max_input_text_chars=core_settings["max_input_text_chars"],
            request_timeout_seconds=core_settings["request_timeout_seconds"],
            inference_busy_status_code=core_settings["inference_busy_status_code"],
            sample_rate=core_settings["sample_rate"],
            filename_max_len=core_settings["filename_max_len"],
            auto_play_cli=core_settings["auto_play_cli"],
            # Telegram-specific settings
            telegram_bot_token=bot_token,
            telegram_allowed_user_ids=_parse_csv_env("QWEN_TTS_TELEGRAM_ALLOWED_USER_IDS", environ),
            telegram_log_level=env_text("QWEN_TTS_TELEGRAM_LOG_LEVEL", DEFAULT_TELEGRAM_LOG_LEVEL, environ),
            telegram_default_speaker=env_text("QWEN_TTS_TELEGRAM_DEFAULT_SPEAKER", DEFAULT_TELEGRAM_DEFAULT_SPEAKER, environ),
            telegram_max_text_length=env_int("QWEN_TTS_TELEGRAM_MAX_TEXT_LENGTH", DEFAULT_TELEGRAM_MAX_TEXT_LENGTH, environ),
            # Security policy settings
            telegram_dev_mode=env_bool("QWEN_TTS_TELEGRAM_DEV_MODE", DEFAULT_TELEGRAM_DEV_MODE, environ),
            telegram_rate_limit_enabled=env_bool("QWEN_TTS_TELEGRAM_RATE_LIMIT_ENABLED", DEFAULT_TELEGRAM_RATE_LIMIT_ENABLED, environ),
            telegram_rate_limit_per_user_per_minute=env_int("QWEN_TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE", DEFAULT_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE, environ),
            telegram_admin_user_ids=_parse_csv_env("QWEN_TTS_TELEGRAM_ADMIN_USER_IDS", environ),
            # Operational settings
            telegram_delivery_store_path=env_text("QWEN_TTS_TELEGRAM_DELIVERY_STORE_PATH", "", environ),
            telegram_poll_interval_seconds=float(env_text("QWEN_TTS_TELEGRAM_POLL_INTERVAL_SECONDS", str(DEFAULT_TELEGRAM_POLL_INTERVAL_SECONDS), environ)),
            telegram_max_retries=env_int("QWEN_TTS_TELEGRAM_MAX_RETRIES", DEFAULT_TELEGRAM_MAX_RETRIES, environ),
        )
    
    def is_user_allowed(self, user_id: int | str) -> bool:
        """Check if user is in the allowlist. Empty allowlist means all users allowed."""
        # Admins are always allowed
        if self.security_policy.is_admin(user_id):
            return True
        # Empty allowlist allows all users (with warning in strict mode)
        if not self.telegram_allowed_user_ids:
            return True
        return str(user_id) in self.telegram_allowed_user_ids
    
    def is_admin_user(self, user_id: int | str) -> bool:
        """Check if user is an admin."""
        return self.security_policy.is_admin(user_id)
    
    def should_enforce_rate_limit(self, user_id: int | str) -> bool:
        """Check if rate limiting should be enforced for user."""
        return self.security_policy.should_enforce_rate_limit(user_id)
    
    def validate(self) -> list[str]:
        """Validate settings and return list of errors. Empty list means valid."""
        errors = []
        
        # Token validation
        if not self.telegram_bot_token or not self.telegram_bot_token.strip():
            errors.append("QWEN_TTS_TELEGRAM_BOT_TOKEN is required")
        elif len(self.telegram_bot_token) < 20 and not self.telegram_dev_mode:
            errors.append("QWEN_TTS_TELEGRAM_BOT_TOKEN appears to be invalid (too short)")
        
        # Text length validation
        if self.telegram_max_text_length <= 0:
            errors.append("QWEN_TTS_TELEGRAM_MAX_TEXT_LENGTH must be positive")
        elif self.telegram_max_text_length > 5000:
            errors.append("QWEN_TTS_TELEGRAM_MAX_TEXT_LENGTH exceeds maximum (5000)")
        
        # Rate limit validation
        if self.telegram_rate_limit_per_user_per_minute <= 0:
            errors.append("QWEN_TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE must be positive")
        
        # Poll interval validation
        if self.telegram_poll_interval_seconds <= 0:
            errors.append("QWEN_TTS_TELEGRAM_POLL_INTERVAL_SECONDS must be positive")
        
        # Max retries validation
        if self.telegram_max_retries < 0:
            errors.append("QWEN_TTS_TELEGRAM_MAX_RETRIES must be non-negative")
        
        # Dev mode with empty allowlist is acceptable for testing/development
        # In production, users should set proper allowlist or admin users
        
        return errors
