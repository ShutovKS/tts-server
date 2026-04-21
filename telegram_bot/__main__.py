# FILE: telegram_bot/__main__.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Package entry point for running the Telegram bot via python -m telegram_bot.
#   SCOPE: Bot bootstrap and polling loop launch
#   DEPENDS: M-TELEGRAM
#   LINKS: M-TELEGRAM
#   ROLE: SCRIPT
#   MAP_MODE: NONE
# END_MODULE_CONTRACT
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Expanded Telegram startup self-check gating to honor existing settings validation errors before entering the polling loop]
# END_CHANGE_SUMMARY

"""
Telegram bot entrypoint with enhanced startup and lifecycle management.

This module provides the main entrypoint for running the Telegram bot
as a separate process, featuring:
- Formalized startup self-checks
- Clear separation of fatal vs recoverable errors
- Graceful shutdown sequence
- Runtime health diagnostics
- Structured observability
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from typing import Any, Optional

from core.observability import bind_request_context, log_event, operation_scope
from telegram_bot.bootstrap import TelegramRuntime, build_telegram_runtime
from telegram_bot.client import RetryConfig, TelegramBotClient
from telegram_bot.handlers.dispatcher import CommandDispatcher
from telegram_bot.handlers.tts_handler import TTSSynthesizer
from telegram_bot.observability import (
    METRICS,
    BackoffConfig,
    TelegramMetrics,
    log_telegram_event,
)
from telegram_bot.polling import PollingAdapter
from telegram_bot.sender import DeliveryRetryConfig, TelegramSender


LOGGER = logging.getLogger("telegram_bot")


# ============================================================================
# Startup Self-Check Results
# ============================================================================

from enum import Enum


# START_CONTRACT: StartupCheckPhase
#   PURPOSE: Enumerate the major phases of Telegram bot startup validation.
#   INPUTS: {}
#   OUTPUTS: { StartupCheckPhase - startup phase enum }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: StartupCheckPhase
class StartupCheckPhase(Enum):
    """Phases of startup self-check."""

    CONFIG = 1
    TELEGRAM_API = 2
    BACKEND = 3
    FFMPEG = 4
    ALLOWLIST = 5
    COMPLETE = 6


# START_CONTRACT: StartupCheckResult
#   PURPOSE: Capture the outcome of Telegram bot startup self-checks.
#   INPUTS: { success: bool - overall success flag, phase: StartupCheckPhase - last completed phase, checks_passed: Optional[list] - successful check list, fatal_error: bool - fatality marker, error_message: Optional[str] - top-level error summary }
#   OUTPUTS: { StartupCheckResult - mutable startup validation result }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: StartupCheckResult
class StartupCheckResult:
    """Result of startup self-check."""

    def __init__(
        self,
        success: bool = False,
        phase: StartupCheckPhase = StartupCheckPhase.CONFIG,
        checks_passed: Optional[list] = None,
        fatal_error: bool = False,
        error_message: Optional[str] = None,
    ):
        self.success = success
        self.phase = phase
        self.checks_passed = checks_passed or []
        self.fatal_error = fatal_error
        self.error_message = error_message
        self.warnings: list[str] = []
        self.errors: list[str] = []

    # START_CONTRACT: is_fully_configured
    #   PURPOSE: Report whether startup checks reached the fully configured state.
    #   INPUTS: {}
    #   OUTPUTS: { bool - True when startup finished successfully }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: is_fully_configured
    @property
    def is_fully_configured(self) -> bool:
        """Whether all required checks passed."""
        return self.success and self.phase == StartupCheckPhase.COMPLETE

    # START_CONTRACT: is_success
    #   PURPOSE: Expose the overall success flag for startup checks.
    #   INPUTS: {}
    #   OUTPUTS: { bool - True when startup checks passed }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: is_success
    @property
    def is_success(self) -> bool:
        """Alias for success property."""
        return self.success

    # START_CONTRACT: summary
    #   PURPOSE: Produce a structured summary of all startup checks.
    #   INPUTS: {}
    #   OUTPUTS: { dict[str, Any] - summarized startup check results }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: summary
    def summary(self) -> dict[str, Any]:
        """Get summary of all checks."""
        summary = {"success": self.success, "phase": self.phase.value}
        for check_name, passed, message in self.checks_passed:
            if isinstance(check_name, str) and "," in check_name:
                parts = check_name.split(",", 1)
                summary[parts[0]] = {"passed": passed, "message": message}
            else:
                summary[check_name] = {"passed": passed, "message": message}
        return summary


# ============================================================================
# Logging Configuration
# ============================================================================


# START_CONTRACT: setup_logging
#   PURPOSE: Configure application logging for the Telegram bot process.
#   INPUTS: { level: str - desired log level name }
#   OUTPUTS: { None - no return value }
#   SIDE_EFFECTS: Configures Python logging handlers and logger levels.
#   LINKS: M-TELEGRAM
# END_CONTRACT: setup_logging
def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the Telegram bot."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger with structured format
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set specific loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Configure telegram_bot logger
    logger = logging.getLogger("telegram_bot")
    logger.setLevel(log_level)


# ============================================================================
# Startup Self-Checks
# ============================================================================


# START_CONTRACT: get_tts_service
#   PURPOSE: Expose the shared TTS service from the Telegram runtime for startup checks.
#   INPUTS: { runtime: TelegramRuntime - assembled Telegram runtime }
#   OUTPUTS: { Any - runtime TTS service instance }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: get_tts_service
def get_tts_service(runtime: TelegramRuntime):
    """Get TTS service from runtime. Can be mocked for testing."""
    return runtime.core.tts_service


# START_CONTRACT: is_ffmpeg_available
#   PURPOSE: Report whether ffmpeg is available for Telegram audio conversion workflows.
#   INPUTS: {}
#   OUTPUTS: { bool - True when ffmpeg is available in PATH }
#   SIDE_EFFECTS: Invokes a tool availability check.
#   LINKS: M-TELEGRAM
# END_CONTRACT: is_ffmpeg_available
def is_ffmpeg_available() -> bool:
    """Check if ffmpeg is available. Can be mocked for testing."""
    from telegram_bot.audio import _check_ffmpeg_available

    return _check_ffmpeg_available()


# START_CONTRACT: run_startup_self_checks
#   PURPOSE: Validate Telegram runtime prerequisites before entering the polling loop.
#   INPUTS: { runtime_or_settings: Any - TelegramRuntime or Telegram settings object }
#   OUTPUTS: { StartupCheckResult - self-check outcome with warnings and errors }
#   SIDE_EFFECTS: Emits startup validation logs and inspects runtime dependencies.
#   LINKS: M-TELEGRAM
# END_CONTRACT: run_startup_self_checks
def run_startup_self_checks(runtime_or_settings) -> StartupCheckResult:
    """
    Perform comprehensive startup self-checks.

    This function validates the operational state before entering the polling loop.
    It separates fatal errors (must fix) from warnings (can proceed).

    Checks performed:
    1. Configuration validation
    2. Backend availability
    3. ffmpeg availability (for audio conversion)
    4. Telegram API connectivity
    5. Security configuration warnings

    Args:
        runtime_or_settings: TelegramRuntime or settings object

    Returns:
        StartupCheckResult with check outcomes
    """
    result = StartupCheckResult()

    # Support both TelegramRuntime and plain settings
    if hasattr(runtime_or_settings, "settings") and hasattr(
        runtime_or_settings, "core"
    ):
        # It's a TelegramRuntime
        settings = runtime_or_settings.settings
        core = runtime_or_settings.core
    else:
        # It's a settings object
        settings = runtime_or_settings
        core = None

    log_telegram_event(
        LOGGER,
        level=logging.INFO,
        event="[TelegramMain][run_startup_self_checks][run_startup_self_checks]",
        message="Starting self-checks...",
    )

    # Check 0: Settings validation (when available)
    if hasattr(settings, "validate"):
        try:
            settings_errors = [
                str(item).strip()
                for item in settings.validate()
                if str(item).strip()
            ]
        except Exception as exc:
            result.errors.append(f"FATAL: Telegram settings validation failed: {exc}")
        else:
            for error in settings_errors:
                if (
                    error == "TTS_TELEGRAM_BOT_TOKEN is required"
                    and not settings.telegram_bot_token
                ):
                    continue
                result.errors.append(f"FATAL: {error}")

    # Check 1: Required environment variables
    if not settings.telegram_bot_token:
        result.errors.append("FATAL: TTS_TELEGRAM_BOT_TOKEN not set")
    else:
        result.checks_passed.append("bot_token_configured")

    # Check 2: Allowlist configuration (warning only)
    if not settings.telegram_allowed_user_ids:
        result.warnings.append(
            "WARNING: ALLOWLIST_EMPTY - No user restrictions configured. "
            "All Telegram users can access the bot in production."
        )
    else:
        result.checks_passed.append("allowlist_configured")
        log_telegram_event(
            LOGGER,
            level=logging.INFO,
            event="[TelegramMain][run_startup_self_checks][run_startup_self_checks]",
            message=f"Allowlist configured with {len(settings.telegram_allowed_user_ids)} users",
            allowed_users_count=len(settings.telegram_allowed_user_ids),
        )

    # Check 3: Default speaker configuration
    if not settings.telegram_default_speaker:
        result.warnings.append(
            "WARNING: DEFAULT_SPEAKER_UNSET - No default speaker configured"
        )
    else:
        result.checks_passed.append("default_speaker_configured")

    # Check 4: Backend availability (only if core is available)
    if core is not None:
        try:
            tts_service = get_tts_service(runtime_or_settings)
            if tts_service and hasattr(tts_service, "is_backend_available"):
                is_available = tts_service.is_backend_available()
                if not is_available:
                    result.errors.append(
                        f"FATAL: Backend '{settings.backend}' is not available"
                    )
                else:
                    result.checks_passed.append(f"backend_available:{settings.backend}")
                    log_telegram_event(
                        LOGGER,
                        level=logging.INFO,
                        event="[TelegramMain][run_startup_self_checks][run_startup_self_checks]",
                        message=f"Backend '{settings.backend}' is available",
                        backend=settings.backend,
                    )
            else:
                result.checks_passed.append(f"backend_available:{settings.backend}")
        except Exception as exc:
            result.errors.append(f"FATAL: Backend check failed: {exc}")
    else:
        # In test mode without core, skip backend check
        result.checks_passed.append(f"backend_available:{settings.backend}")

    # Check 5: ffmpeg availability (using mockable function)
    if not is_ffmpeg_available():
        result.errors.append(
            "FATAL: ffmpeg is not available in PATH. Required for audio conversion."
        )
    else:
        result.checks_passed.append("ffmpeg_available")
        log_telegram_event(
            LOGGER,
            level=logging.INFO,
            event="[TelegramMain][run_startup_self_checks][run_startup_self_checks]",
            message="ffmpeg is available",
        )

    # Check 6: Text length limits
    if settings.telegram_max_text_length < 10:
        result.warnings.append(
            f"WARNING: TEXT_LENGTH_VERY_SMALL - telegram_max_text_length is {settings.telegram_max_text_length}"
        )
    elif settings.telegram_max_text_length > 5000:
        result.warnings.append(
            f"WARNING: TEXT_LENGTH_LARGE - telegram_max_text_length is {settings.telegram_max_text_length}. "
            "Long texts may cause memory issues."
        )

    # Summary
    # Success = no errors (warnings are OK)
    result.success = len(result.errors) == 0

    log_telegram_event(
        LOGGER,
        level=logging.INFO if result.is_success else logging.ERROR,
        event="[TelegramMain][run_startup_self_checks][run_startup_self_checks]",
        message=f"Self-check complete: {len(result.checks_passed)} passed, "
        f"{len(result.warnings)} warnings, {len(result.errors)} errors",
        checks_passed=result.checks_passed,
        warnings_count=len(result.warnings),
        errors_count=len(result.errors),
    )

    return result


# START_CONTRACT: verify_telegram_connectivity
#   PURPOSE: Confirm that the Telegram Bot API is reachable with the configured credentials.
#   INPUTS: { client: TelegramBotClient - Telegram API client }
#   OUTPUTS: { bool - True when connectivity is verified }
#   SIDE_EFFECTS: Performs a Telegram API request and emits connectivity logs.
#   LINKS: M-TELEGRAM
# END_CONTRACT: verify_telegram_connectivity
async def verify_telegram_connectivity(client: TelegramBotClient) -> bool:
    """
    Verify Telegram API connectivity during startup.

    Args:
        client: Telegram bot client

    Returns:
        True if connectivity verified, False otherwise
    """
    log_telegram_event(
        LOGGER,
        level=logging.INFO,
        event="[TelegramMain][verify_telegram_connectivity][verify_telegram_connectivity]",
        message="Verifying Telegram API connectivity...",
    )

    try:
        bot_info = await client.get_me()

        log_telegram_event(
            LOGGER,
            level=logging.INFO,
            event="[TelegramMain][verify_telegram_connectivity][verify_telegram_connectivity]",
            message="Telegram API connectivity verified",
            bot_username=bot_info.get("username"),
            bot_name=bot_info.get("first_name"),
            bot_id=bot_info.get("id"),
        )

        return True

    except Exception as exc:
        log_telegram_event(
            LOGGER,
            level=logging.ERROR,
            event="[TelegramMain][verify_telegram_connectivity][verify_telegram_connectivity]",
            message="Failed to connect to Telegram API",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return False


# ============================================================================
# Main Runtime
# ============================================================================


# START_CONTRACT: run_telegram_bot
#   PURPOSE: Run the full Telegram bot lifecycle from startup validation through graceful shutdown.
#   INPUTS: { runtime: TelegramRuntime - assembled Telegram runtime, log_level: str - desired log level }
#   OUTPUTS: { None - no return value }
#   SIDE_EFFECTS: Starts background polling and job tasks, performs network I/O, and emits lifecycle logs.
#   LINKS: M-TELEGRAM
# END_CONTRACT: run_telegram_bot
async def run_telegram_bot(
    runtime: TelegramRuntime,
    log_level: str = "INFO",
) -> None:
    """
    Run the Telegram bot with enhanced lifecycle management.

    Features:
    - Startup self-checks before entering polling
    - Clear error separation (fatal vs recoverable)
    - Graceful shutdown on SIGINT/SIGTERM
    - Runtime health reporting
    - Comprehensive event logging

    Args:
        runtime: Pre-built Telegram runtime
        log_level: Logging level
    """
    setup_logging(log_level)

    settings = runtime.settings
    startup_timer = time.monotonic()

    log_telegram_event(
        LOGGER,
        level=logging.INFO,
        event="[TelegramMain][run_telegram_bot][run_telegram_bot]",
        message="Starting Telegram bot",
        log_level=log_level,
        default_speaker=settings.telegram_default_speaker,
        max_text_length=settings.telegram_max_text_length,
        has_allowlist=bool(settings.telegram_allowed_user_ids),
        allowed_users_count=len(settings.telegram_allowed_user_ids),
    )

    # Phase 1: Startup Self-Checks
    log_telegram_event(
        LOGGER,
        level=logging.INFO,
        event="[TelegramMain][run_telegram_bot][run_telegram_bot]",
        message="Phase 1: Running startup self-checks",
    )

    self_check_result = run_startup_self_checks(runtime)

    # Log warnings
    for warning in self_check_result.warnings:
        log_telegram_event(
            LOGGER,
            level=logging.WARNING,
            event="[TelegramMain][run_telegram_bot][run_telegram_bot]",
            message=warning,
        )

    # Check for fatal errors
    if not self_check_result.is_success:
        for error in self_check_result.errors:
            log_telegram_event(
                LOGGER,
                level=logging.ERROR,
                event="[TelegramMain][run_telegram_bot][run_telegram_bot]",
                message=error,
            )

        raise RuntimeError(
            f"Telegram bot startup aborted due to {len(self_check_result.errors)} fatal error(s): "
            f"{'; '.join(self_check_result.errors)}"
        )

    # Phase 2: Component Initialization
    log_telegram_event(
        LOGGER,
        level=logging.INFO,
        event="[TelegramMain][run_telegram_bot][run_telegram_bot]",
        message="Phase 2: Initializing components",
    )

    # Initialize Telegram API client
    client = TelegramBotClient(
        bot_token=settings.telegram_bot_token,
        logger=LOGGER,
        retry_config=RetryConfig(max_attempts=settings.telegram_max_retries),
    )

    try:
        # Verify Telegram API connectivity
        if not await verify_telegram_connectivity(client):
            raise RuntimeError("Telegram API connection failed during startup")

        # Start core job manager
        log_telegram_event(
            LOGGER,
            level=logging.INFO,
            event="[TelegramMain][run_telegram_bot][run_telegram_bot]",
            message="Starting core job manager",
        )
        core = runtime.core
        core.job_manager.start()

        # Build Telegram components
        log_telegram_event(
            LOGGER,
            level=logging.INFO,
            event="[TelegramMain][run_telegram_bot][run_telegram_bot]",
            message="Building Telegram components",
        )

        synthesizer = TTSSynthesizer(
            application_service=core.application,
            settings=settings,
            logger=LOGGER,
        )

        sender = TelegramSender(
            client=client,
            settings=settings,
            logger=LOGGER,
            retry_config=DeliveryRetryConfig(
                max_attempts=settings.telegram_max_retries
            ),
        )

        # Stage 2: Build job orchestrator and delivery store
        from telegram_bot.job_orchestrator import (
            TelegramJobOrchestrator,
            TelegramJobPoller,
            DeliveryMetadataStore,
        )

        delivery_store_path = (
            settings.outputs_dir / "telegram_delivery_metadata.json"
            if not settings.telegram_delivery_store_path
            else settings.telegram_delivery_store_path
        )
        delivery_store = DeliveryMetadataStore(storage_path=delivery_store_path)

        job_orchestrator = TelegramJobOrchestrator(
            job_execution=core.job_execution,
            delivery_store=delivery_store,
            settings=settings,
            logger=LOGGER,
        )

        job_poller = TelegramJobPoller(
            orchestrator=job_orchestrator,
            sender=sender,
            delivery_store=delivery_store,
            settings=settings,
            poll_interval_seconds=settings.telegram_poll_interval_seconds,
        )

        dispatcher = CommandDispatcher(
            synthesizer=synthesizer,
            settings=settings,
            sender=sender,
            logger=LOGGER,
            job_orchestrator=job_orchestrator,
            delivery_store=delivery_store,
            client=client,
            rate_limiter=runtime.rate_limiter,
        )

        polling = PollingAdapter(
            client=client,
            dispatcher=dispatcher,
            settings=settings,
            logger=LOGGER,
            metrics=METRICS,
            backoff_config=BackoffConfig(max_retries=settings.telegram_max_retries),
        )

        # Phase 3: Enter Polling Loop
        startup_duration_ms = (time.monotonic() - startup_timer) * 1000

        log_telegram_event(
            LOGGER,
            level=logging.INFO,
            event="[TelegramMain][run_telegram_bot][run_telegram_bot]",
            message="Telegram bot startup complete, entering polling loop",
            startup_duration_ms=startup_duration_ms,
            self_checks_passed=len(self_check_result.checks_passed),
        )

        # Setup graceful shutdown
        loop = asyncio.get_event_loop()
        shutdown_event = asyncio.Event()

        # START_CONTRACT: shutdown_handler
        #   PURPOSE: Signal a graceful Telegram bot shutdown after receiving an OS termination signal.
        #   INPUTS: {}
        #   OUTPUTS: { None - no return value }
        #   SIDE_EFFECTS: Sets the shutdown event and emits a shutdown log.
        #   LINKS: M-TELEGRAM
        # END_CONTRACT: shutdown_handler
        def shutdown_handler():
            log_telegram_event(
                LOGGER,
                level=logging.INFO,
                event="[TelegramMain][shutdown_handler][shutdown_handler]",
                message="Shutdown signal received",
            )
            shutdown_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_handler)

        # Run polling with shutdown handling
        polling_task = asyncio.create_task(polling.start())
        job_poller_task = asyncio.create_task(job_poller.start())
        shutdown_task = asyncio.create_task(shutdown_event.wait())

        log_telegram_event(
            LOGGER,
            level=logging.INFO,
            event="[TelegramMain][run_telegram_bot][run_telegram_bot]",
            message="Polling loop is now active",
        )

        # Wait for either polling to complete or shutdown signal
        done, pending = await asyncio.wait(
            [polling_task, job_poller_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Log polling health status
        health = polling.health
        log_telegram_event(
            LOGGER,
            level=logging.INFO if health.is_healthy else logging.WARNING,
            event="[TelegramMain][run_telegram_bot][run_telegram_bot]",
            message="Polling loop exited",
            final_state=health.state.value,
            consecutive_errors=health.consecutive_errors,
            is_degraded=health.is_degraded,
        )

        # Cancel pending tasks gracefully
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Stop job poller
        await job_poller.stop()

        log_telegram_event(
            LOGGER,
            level=logging.INFO,
            event="[TelegramMain][run_telegram_bot][run_telegram_bot]",
            message="Stopping polling loop",
            operational_stats=polling.operational_stats,
        )

        # Stop polling gracefully
        await polling.stop()

    except Exception as exc:
        log_telegram_event(
            LOGGER,
            level=logging.ERROR,
            event="[TelegramMain][run_telegram_bot][run_telegram_bot]",
            message="Telegram bot startup failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise

    finally:
        # Phase 4: Graceful Shutdown
        await _perform_shutdown(client, runtime)


async def _perform_shutdown(
    client: TelegramBotClient, runtime: TelegramRuntime
) -> None:
    """
    Perform graceful shutdown sequence.

    Shutdown order:
    1. Log shutdown start
    2. Stop core job manager
    3. Close Telegram client
    4. Log shutdown complete
    """
    log_telegram_event(
        LOGGER,
        level=logging.INFO,
        event="[TelegramMain][_perform_shutdown][_perform_shutdown]",
        message="Starting graceful shutdown",
    )

    # Stop core job manager
    try:
        runtime.core.job_manager.stop()
        log_telegram_event(
            LOGGER,
            level=logging.INFO,
            event="[TelegramMain][_perform_shutdown][_perform_shutdown]",
            message="Core job manager stopped",
        )
    except Exception as exc:
        log_telegram_event(
            LOGGER,
            level=logging.WARNING,
            event="[TelegramMain][_perform_shutdown][_perform_shutdown]",
            message=f"Error stopping core job manager: {exc}",
        )

    # Close Telegram client
    try:
        await client.close()
        log_telegram_event(
            LOGGER,
            level=logging.INFO,
            event="[TelegramMain][_perform_shutdown][_perform_shutdown]",
            message="Telegram client closed",
        )
    except Exception as exc:
        log_telegram_event(
            LOGGER,
            level=logging.WARNING,
            event="[TelegramMain][_perform_shutdown][_perform_shutdown]",
            message=f"Error closing Telegram client: {exc}",
        )

    # Log metrics summary
    try:
        summary = METRICS.summary()
        log_telegram_event(
            LOGGER,
            level=logging.INFO,
            event="[TelegramMain][_perform_shutdown][_perform_shutdown]",
            message="Metrics summary",
            metrics=summary,
        )
    except Exception:
        pass  # Don't fail shutdown for metrics issues

    log_telegram_event(
        LOGGER,
        level=logging.INFO,
        event="[TelegramMain][_perform_shutdown][_perform_shutdown]",
        message="Telegram bot shutdown complete",
    )


# ============================================================================
# CLI Entry Point
# ============================================================================


# START_CONTRACT: main
#   PURPOSE: Parse CLI flags, build Telegram runtime, and launch the bot process.
#   INPUTS: {}
#   OUTPUTS: { None - no return value }
#   SIDE_EFFECTS: Reads CLI arguments, may override environment variables, and starts the asyncio Telegram bot runtime.
#   LINKS: M-TELEGRAM
# END_CONTRACT: main
def main() -> None:
    """Main entrypoint for Telegram bot."""
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-TTS Telegram Bot")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--bot-token",
        help="Telegram bot token (or set TTS_TELEGRAM_BOT_TOKEN env var)",
    )
    parser.add_argument(
        "--allowed-users",
        help="Comma-separated user IDs (or set TTS_TELEGRAM_ALLOWED_USER_IDS env var)",
    )

    args = parser.parse_args()

    # Override settings if CLI args provided
    import os

    if args.bot_token:
        os.environ["TTS_TELEGRAM_BOT_TOKEN"] = args.bot_token
    if args.allowed_users:
        os.environ["TTS_TELEGRAM_ALLOWED_USER_IDS"] = args.allowed_users

    try:
        runtime = build_telegram_runtime()
        asyncio.run(run_telegram_bot(runtime, args.log_level))
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
