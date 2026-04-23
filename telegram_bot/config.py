# FILE: telegram_bot/config.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Define Telegram-specific configuration parsed from environment variables.
#   SCOPE: TelegramSettings dataclass, environment parsing
#   DEPENDS: M-CONFIG
#   LINKS: M-TELEGRAM
#   ROLE: CONFIG
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   DEFAULT_TELEGRAM_LOG_LEVEL - Default Telegram log verbosity
#   DEFAULT_TELEGRAM_DEFAULT_SPEAKER - Default Telegram speaker name
#   DEFAULT_TELEGRAM_MAX_TEXT_LENGTH - Default Telegram command text limit
#   DEFAULT_TELEGRAM_DEV_MODE - Default Telegram development mode flag
#   DEFAULT_TELEGRAM_RATE_LIMIT_ENABLED - Default Telegram rate limiting toggle
#   DEFAULT_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE - Default Telegram per-user rate limit
#   DEFAULT_TELEGRAM_POLL_INTERVAL_SECONDS - Default Telegram job polling interval
#   DEFAULT_TELEGRAM_MAX_RETRIES - Default Telegram API retry count
#   DEFAULT_TELEGRAM_SERVER_BASE_URL - Default remote server base URL for canonical HTTP client wiring
#   TelegramSecurityPolicy - Telegram-specific security and admission policy
#   TelegramSettings - Telegram-specific configuration
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Added injectable remote server base URL configuration for the Telegram-side canonical HTTP client layer]
# END_CHANGE_SUMMARY

"""
Telegram bot configuration settings.

Environment variables:
- TTS_TELEGRAM_BOT_TOKEN: Bot token from @BotFather (required)
- TTS_TELEGRAM_ALLOWED_USER_IDS: Comma-separated list of Telegram user IDs (optional, empty = all users allowed)
- TTS_TELEGRAM_LOG_LEVEL: Log level (default: info)
- TTS_TELEGRAM_DEFAULT_SPEAKER: Default speaker name (default: Vivian)
- TTS_TELEGRAM_MAX_TEXT_LENGTH: Maximum text length for /tts command (default: 1000)
- TTS_TELEGRAM_DEV_MODE: Enable dev mode with relaxed security (default: false)
- TTS_TELEGRAM_RATE_LIMIT_ENABLED: Enable per-user rate limiting (default: true)
- TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE: Max requests per user per minute (default: 20)
- TTS_TELEGRAM_JOB_TIMEOUT_SECONDS: Job execution timeout (default: from core settings)
- TTS_TELEGRAM_DELIVERY_STORE_PATH: Path for delivery metadata store (optional)
- TTS_TELEGRAM_POLL_INTERVAL_SECONDS: Job poller interval (default: 1.0)
- TTS_TELEGRAM_MAX_RETRIES: Max retry attempts for API calls (default: 3)
- TTS_TELEGRAM_SERVER_BASE_URL: Optional canonical remote server base URL for Telegram HTTP client flows
- TTS_TELEGRAM_ADMIN_USER_IDS: Comma-separated admin user IDs with elevated access (optional)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from core.config import (
    CoreSettings,
    parse_core_settings_from_env,
    env_text,
    env_int,
    env_bool,
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
DEFAULT_TELEGRAM_SERVER_BASE_URL = ""


# START_CONTRACT: TelegramSecurityPolicy
#   PURPOSE: Represent Telegram-specific admission and security policy decisions.
#   INPUTS: { rate_limit_enabled: bool - enables per-user throttling, rate_limit_per_user_per_minute: int - request limit, dev_mode: bool - relaxed development mode flag, admin_user_ids: tuple[str, ...] - privileged Telegram users, require_token_validation: bool - token validation toggle, token_min_length: int - minimum token size, allowlist_strict_mode: bool - empty allowlist policy }
#   OUTPUTS: { TelegramSecurityPolicy - immutable security policy configuration }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: TelegramSecurityPolicy
@dataclass(frozen=True)
class TelegramSecurityPolicy:
    """Security policy configuration for Telegram bot.

    This aligns with admission control policies from core/application/admission_control.py
    while providing Telegram-specific enforcement.
    """

    # Rate limiting
    rate_limit_enabled: bool = DEFAULT_TELEGRAM_RATE_LIMIT_ENABLED
    rate_limit_per_user_per_minute: int = (
        DEFAULT_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE
    )

    # Dev mode - relaxes certain security checks for development
    dev_mode: bool = DEFAULT_TELEGRAM_DEV_MODE

    # Admin users with elevated access
    admin_user_ids: tuple[str, ...] = field(default_factory=tuple)

    # Token security
    require_token_validation: bool = True
    token_min_length: int = 20  # Bot tokens are typically 40+ chars

    # Allowlist enforcement
    allowlist_strict_mode: bool = True  # If true, empty allowlist = only admins allowed

    # START_CONTRACT: is_admin
    #   PURPOSE: Check whether a Telegram user is configured as an administrator.
    #   INPUTS: { user_id: int | str - Telegram user identifier }
    #   OUTPUTS: { bool - True when the user has admin access }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: is_admin
    def is_admin(self, user_id: int | str) -> bool:
        """Check if user is an admin."""
        return str(user_id) in self.admin_user_ids

    # START_CONTRACT: should_enforce_rate_limit
    #   PURPOSE: Decide whether rate limiting applies to a specific Telegram user.
    #   INPUTS: { user_id: int | str - Telegram user identifier }
    #   OUTPUTS: { bool - True when the user should be throttled }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: should_enforce_rate_limit
    def should_enforce_rate_limit(self, user_id: int | str) -> bool:
        """Check if rate limiting should be enforced for user."""
        if not self.rate_limit_enabled:
            return False
        # Admins bypass rate limiting in production
        if not self.dev_mode and self.is_admin(user_id):
            return False
        return True

    # START_CONTRACT: allow_empty_allowlist
    #   PURPOSE: Report whether an empty Telegram allowlist is permitted under current policy.
    #   INPUTS: {}
    #   OUTPUTS: { bool - True when empty allowlists are accepted }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: allow_empty_allowlist
    def allow_empty_allowlist(self) -> bool:
        """Check if empty allowlist should allow all users."""
        return not self.allowlist_strict_mode or self.dev_mode


# START_CONTRACT: TelegramSettings
#   PURPOSE: Extend shared runtime settings with Telegram adapter-local and remote client configuration values.
#   INPUTS: { telegram_bot_token: str - bot API token, telegram_allowed_user_ids: tuple[str, ...] - user allowlist, telegram_log_level: str - logging level, telegram_default_speaker: str - fallback voice, telegram_max_text_length: int - message length limit, telegram_dev_mode: bool - development mode flag, telegram_rate_limit_enabled: bool - throttling toggle, telegram_rate_limit_per_user_per_minute: int - user quota, telegram_admin_user_ids: tuple[str, ...] - admin allowlist, telegram_delivery_store_path: str - delivery metadata path, telegram_poll_interval_seconds: float - job polling cadence, telegram_max_retries: int - retry cap, telegram_server_base_url: str - canonical remote server base URL }
#   OUTPUTS: { TelegramSettings - immutable Telegram runtime settings }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: TelegramSettings
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
    telegram_rate_limit_per_user_per_minute: int = (
        DEFAULT_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE
    )
    telegram_admin_user_ids: tuple[str, ...] = field(default_factory=tuple)

    # Operational settings
    telegram_delivery_store_path: str = ""
    telegram_poll_interval_seconds: float = DEFAULT_TELEGRAM_POLL_INTERVAL_SECONDS
    telegram_max_retries: int = DEFAULT_TELEGRAM_MAX_RETRIES
    telegram_server_base_url: str = DEFAULT_TELEGRAM_SERVER_BASE_URL

    # START_CONTRACT: security_policy
    #   PURPOSE: Build a Telegram security policy view from the current settings.
    #   INPUTS: {}
    #   OUTPUTS: { TelegramSecurityPolicy - derived policy object for admission checks }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: security_policy
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

    # START_CONTRACT: from_env
    #   PURPOSE: Parse Telegram settings from environment variables and shared core defaults.
    #   INPUTS: { environ: Mapping[str, str] | None - optional environment mapping override }
    #   OUTPUTS: { TelegramSettings - parsed Telegram configuration }
    #   SIDE_EFFECTS: Reads process environment variables when no mapping is provided.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: from_env
    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "TelegramSettings":
        """Parse Telegram settings from environment variables.

        Uses parse_core_settings_from_env() to ensure consistency with core settings
        parsing, then adds Telegram-specific settings.
        """
        bot_token = env_text("TTS_TELEGRAM_BOT_TOKEN", "", environ).strip()

        # Parse core settings using the shared parser for consistency
        core_settings = parse_core_settings_from_env(environ)

        return cls(
            **core_settings,  # type: ignore[arg-type]
            # Telegram-specific settings
            telegram_bot_token=bot_token,
            telegram_allowed_user_ids=_parse_csv_env(
                "TTS_TELEGRAM_ALLOWED_USER_IDS", environ
            ),
            telegram_log_level=env_text(
                "TTS_TELEGRAM_LOG_LEVEL", DEFAULT_TELEGRAM_LOG_LEVEL, environ
            ),
            telegram_default_speaker=env_text(
                "TTS_TELEGRAM_DEFAULT_SPEAKER",
                DEFAULT_TELEGRAM_DEFAULT_SPEAKER,
                environ,
            ),
            telegram_max_text_length=env_int(
                "TTS_TELEGRAM_MAX_TEXT_LENGTH",
                DEFAULT_TELEGRAM_MAX_TEXT_LENGTH,
                environ,
            ),
            # Security policy settings
            telegram_dev_mode=env_bool(
                "TTS_TELEGRAM_DEV_MODE", DEFAULT_TELEGRAM_DEV_MODE, environ
            ),
            telegram_rate_limit_enabled=env_bool(
                "TTS_TELEGRAM_RATE_LIMIT_ENABLED",
                DEFAULT_TELEGRAM_RATE_LIMIT_ENABLED,
                environ,
            ),
            telegram_rate_limit_per_user_per_minute=env_int(
                "TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE",
                DEFAULT_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE,
                environ,
            ),
            telegram_admin_user_ids=_parse_csv_env(
                "TTS_TELEGRAM_ADMIN_USER_IDS", environ
            ),
            # Operational settings
            telegram_delivery_store_path=env_text(
                "TTS_TELEGRAM_DELIVERY_STORE_PATH", "", environ
            ),
            telegram_poll_interval_seconds=float(
                env_text(
                    "TTS_TELEGRAM_POLL_INTERVAL_SECONDS",
                    str(DEFAULT_TELEGRAM_POLL_INTERVAL_SECONDS),
                    environ,
                )
            ),
            telegram_max_retries=env_int(
                "TTS_TELEGRAM_MAX_RETRIES", DEFAULT_TELEGRAM_MAX_RETRIES, environ
            ),
            telegram_server_base_url=env_text(
                "TTS_TELEGRAM_SERVER_BASE_URL",
                DEFAULT_TELEGRAM_SERVER_BASE_URL,
                environ,
            ).rstrip("/"),
        )

    # START_CONTRACT: is_user_allowed
    #   PURPOSE: Determine whether a Telegram user may access the bot.
    #   INPUTS: { user_id: int | str - Telegram user identifier }
    #   OUTPUTS: { bool - True when the user passes allowlist checks }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: is_user_allowed
    def is_user_allowed(self, user_id: int | str) -> bool:
        """Check if user is in the allowlist. Empty allowlist means all users allowed."""
        # Admins are always allowed
        if self.security_policy.is_admin(user_id):
            return True
        # Empty allowlist allows all users (with warning in strict mode)
        if not self.telegram_allowed_user_ids:
            return True
        return str(user_id) in self.telegram_allowed_user_ids

    # START_CONTRACT: is_admin_user
    #   PURPOSE: Check whether a Telegram user has administrator privileges.
    #   INPUTS: { user_id: int | str - Telegram user identifier }
    #   OUTPUTS: { bool - True when the user is an admin }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: is_admin_user
    def is_admin_user(self, user_id: int | str) -> bool:
        """Check if user is an admin."""
        return self.security_policy.is_admin(user_id)

    # START_CONTRACT: should_enforce_rate_limit
    #   PURPOSE: Delegate Telegram user throttling policy to the derived security policy.
    #   INPUTS: { user_id: int | str - Telegram user identifier }
    #   OUTPUTS: { bool - True when rate limiting should be applied }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: should_enforce_rate_limit
    def should_enforce_rate_limit(self, user_id: int | str) -> bool:
        """Check if rate limiting should be enforced for user."""
        return self.security_policy.should_enforce_rate_limit(user_id)

    # START_CONTRACT: validate
    #   PURPOSE: Validate Telegram configuration values and report any blocking issues.
    #   INPUTS: {}
    #   OUTPUTS: { list[str] - validation error messages, empty when valid }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: validate
    def validate(self) -> list[str]:
        """Validate settings and return list of errors. Empty list means valid."""
        errors = []

        # Token validation
        if not self.telegram_bot_token or not self.telegram_bot_token.strip():
            errors.append("TTS_TELEGRAM_BOT_TOKEN is required")
        elif len(self.telegram_bot_token) < 20 and not self.telegram_dev_mode:
            errors.append(
                "TTS_TELEGRAM_BOT_TOKEN appears to be invalid (too short)"
            )

        # Text length validation
        if self.telegram_max_text_length <= 0:
            errors.append("TTS_TELEGRAM_MAX_TEXT_LENGTH must be positive")
        elif self.telegram_max_text_length > 5000:
            errors.append("TTS_TELEGRAM_MAX_TEXT_LENGTH exceeds maximum (5000)")

        # Rate limit validation
        if self.telegram_rate_limit_per_user_per_minute <= 0:
            errors.append(
                "TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE must be positive"
            )

        # Poll interval validation
        if self.telegram_poll_interval_seconds <= 0:
            errors.append("TTS_TELEGRAM_POLL_INTERVAL_SECONDS must be positive")

        # Max retries validation
        if self.telegram_max_retries < 0:
            errors.append("TTS_TELEGRAM_MAX_RETRIES must be non-negative")

        # Remote server validation
        if not self.telegram_server_base_url or not self.telegram_server_base_url.strip():
            errors.append("TTS_TELEGRAM_SERVER_BASE_URL is required")

        # Dev mode with empty allowlist is acceptable for testing/development
        # In production, users should set proper allowlist or admin users

        return errors

__all__ = [
    "DEFAULT_TELEGRAM_LOG_LEVEL",
    "DEFAULT_TELEGRAM_DEFAULT_SPEAKER",
    "DEFAULT_TELEGRAM_MAX_TEXT_LENGTH",
    "DEFAULT_TELEGRAM_DEV_MODE",
    "DEFAULT_TELEGRAM_RATE_LIMIT_ENABLED",
    "DEFAULT_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE",
    "DEFAULT_TELEGRAM_POLL_INTERVAL_SECONDS",
    "DEFAULT_TELEGRAM_MAX_RETRIES",
    "DEFAULT_TELEGRAM_SERVER_BASE_URL",
    "TelegramSecurityPolicy",
    "TelegramSettings",
]
