# FILE: telegram_bot/handlers/dispatcher.py
# VERSION: 1.1.2
# START_MODULE_CONTRACT
#   PURPOSE: Route incoming Telegram updates to appropriate command handlers.
#   SCOPE: Update dispatcher with command routing and user allowlist
#   DEPENDS: M-TELEGRAM
#   LINKS: M-TELEGRAM
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LOGGER - Module logger for Telegram dispatcher events
#   MessageSender - Protocol for Telegram text and voice delivery
#   CommandDispatcher - Route Telegram updates to command handlers
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.2 - Prevented duplicate Telegram submits from resetting already persisted delivery state]
# END_CHANGE_SUMMARY

"""
Command dispatcher for Telegram bot with observability.

This module routes parsed commands to appropriate handlers and provides
the async UX with acknowledgment and result delivery, featuring:
- Command metrics (received, accepted, rejected)
- Correlation context propagation
- Structured logging with operation tracking
- Job integration with canonical remote async HTTP jobs
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol

from core.observability import get_logger
from telegram_bot.handlers.commands import (
    CommandType,
    ParsedCommand,
    get_valid_speakers,
    is_private_chat,
    parse_command,
    parse_design_args,
    parse_tts_args,
    validate_design_command,
    validate_tts_command,
    parse_clone_args,
    validate_clone_command,
    MAX_TEXT_LENGTH as MAX_CLONE_TEXT_LENGTH,
)
from telegram_bot.handlers.tts_handler import TTSSynthesizer
from telegram_bot.media import DownloadError, MediaValidationError, stage_clone_media
from telegram_bot.observability import (
    METRICS,
    log_telegram_event,
)


if TYPE_CHECKING:
    from telegram_bot.config import TelegramSettings
    from telegram_bot.polling import TelegramClient
    from telegram_bot.rate_limiter import TelegramRateLimiter


LOGGER = get_logger(__name__)


# START_CONTRACT: MessageSender
#   PURPOSE: Define the message delivery interface required by the Telegram command dispatcher.
#   INPUTS: {}
#   OUTPUTS: { MessageSender - protocol for Telegram text and voice delivery }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: MessageSender
class MessageSender(Protocol):
    """Protocol for sending messages to Telegram."""

    # START_CONTRACT: send_text
    #   PURPOSE: Send a Telegram text message through the dispatcher's delivery interface.
    #   INPUTS: { chat_id: int - target Telegram chat identifier, text: str - message body }
    #   OUTPUTS: { Any - transport-specific send result }
    #   SIDE_EFFECTS: Sends a Telegram API message in concrete implementations.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: send_text
    async def send_text(self, chat_id: int, text: str) -> Any:
        """Send text message to chat."""
        ...

    # START_CONTRACT: send_voice
    #   PURPOSE: Send a Telegram voice message through the dispatcher's delivery interface.
    #   INPUTS: { chat_id: int - target Telegram chat identifier, audio_bytes: bytes - voice payload bytes, caption: str | None - optional caption }
    #   OUTPUTS: { Any - transport-specific send result }
    #   SIDE_EFFECTS: Sends a Telegram voice message in concrete implementations.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: send_voice
    async def send_voice(
        self, chat_id: int, audio_bytes: bytes, caption: str | None = None
    ) -> Any:
        """Send voice message to chat."""
        ...


# START_CONTRACT: CommandDispatcher
#   PURPOSE: Route Telegram updates to command handlers and coordinate bot responses.
#   INPUTS: { synthesizer: TTSSynthesizer - synthesis service, settings: TelegramSettings - Telegram configuration, sender: MessageSender - delivery service, logger: logging.Logger | None - optional logger, job_orchestrator: Any - optional job orchestration service, delivery_store: Any - optional delivery metadata store, client: TelegramClient | None - optional Telegram media client, rate_limiter: TelegramRateLimiter | None - optional per-user throttler }
#   OUTPUTS: { CommandDispatcher - configured update dispatcher }
#   SIDE_EFFECTS: Sends Telegram replies, records metrics, and may submit background jobs.
#   LINKS: M-TELEGRAM
# END_CONTRACT: CommandDispatcher
class CommandDispatcher:
    """
    Command dispatcher that routes Telegram commands to handlers.

    This class implements the async UX pattern with full observability:
    1. Acknowledge command immediately
    2. Process in background
    3. Send result when ready

    Features:
    - Command metrics (received, accepted, rejected)
    - Correlation context for tracing
    - Structured logging
    """

    # Response templates
    START_MESSAGE = (
        "👋 *Добро пожаловать в Qwen3 TTS Bot*\n\n"
        "Бот работает только в *личном чате* и умеет:\n"
        "• озвучивать текст командой `/tts`\n"
        "• создавать новый голос командой `/design`\n"
        "• клонировать голос из вашего аудио командой `/clone`\n\n"
        "Результат всегда приходит отдельным *voice* сообщением после завершения генерации.\n\n"
        "*Команды:*\n"
        "• `/start` — краткое описание\n"
        "• `/help` — подробная справка и примеры\n"
        "• `/tts` — обычный синтез речи\n"
        "• `/design` — создание нового голоса по описанию\n"
        "• `/clone` — клонирование голоса из reply на audio\n\n"
        "*Быстрый старт:*\n"
        "`/tts -- Привет, это тест`\n"
        "`/tts speaker=Ryan speed=0.9 lang=ru -- Привет, это тест`\n"
        "`/design lang=ru calm narrator -- Привет, мир`\n\n"
        "Для `/clone` сначала отправьте `voice`, `audio` или `document`, затем ответьте на это сообщение командой `/clone ...`."
    )

    HELP_MESSAGE = (
        "📖 *Справка по командам*\n\n"
        "Бот работает только в *личном чате*. Все результаты приходят отдельным *voice* сообщением после завершения генерации.\n\n"
        "*1. Обычный TTS*\n"
        "`/tts -- <текст>`\n"
        "`/tts speaker=<speaker> -- <текст>`\n"
        "`/tts speed=<speed> -- <текст>`\n"
        "`/tts speaker=<speaker> speed=<speed> lang=<language> -- <текст>`\n\n"
        "*Параметры /tts:*\n"
        "• `speaker` — имя голоса\n"
        "• `speed` — скорость от 0.5 до 2.0\n"
        "• `lang` — язык, по умолчанию `auto`\n"
        "• если параметры не нужны, используйте `/tts -- текст`\n\n"
        "*Доступные голоса:*\n"
        "{speakers}\n\n"
        "*2. Voice Design*\n"
        "`/design [lang=<language>] <описание_голоса> -- <текст>`\n"
        "Пример: `/design lang=ru calm documentary narrator -- Расскажи о космосе`\n\n"
        "*3. Voice Cloning*\n"
        "Сначала отправьте `voice`, `audio` или `document` с поддерживаемым аудио, затем *ответьте* на это сообщение одной из команд:\n"
        "`/clone [lang=<language>] -- <текст>`\n"
        "`/clone [lang=<language>] ref=<транскрипт> -- <текст>`\n"
        "Пример: reply на audio → `/clone lang=ru ref=This is my sample -- Скажи это моим голосом`\n\n"
        "*Важно:*\n"
        "• разделитель `--` обязателен для `/tts`, `/design` и `/clone`\n"
        "• `/clone` работает только как reply на аудио\n"
        "• в `/clone` не поддерживаются `speaker`, `speed` и `model`\n"
        "• если команда отклонена, бот пришлёт понятную ошибку с подсказкой\n"
        "• если запрос принят, результат придёт отдельным сообщением"
    )

    PROCESSING_MESSAGE = "🎙️ *Обработка запущена*\nСейчас подготовлю результат и пришлю его отдельным voice сообщением."

    ACCEPTED_MESSAGE = (
        "✅ *Запрос принят*\n\n"
        "Озвучиваю текст голосом *{speaker}* со скоростью *{speed}x* и языком *{language}*.\n"
        "Когда генерация завершится, я пришлю отдельное voice сообщение."
    )

    DESIGN_ACCEPTED_MESSAGE = (
        "✅ *Запрос принят*\n\n"
        "Создаю голос по описанию: *{voice_description}*. Язык: *{language}*.\n"
        "Когда генерация завершится, я пришлю отдельное voice сообщение."
    )

    CLONE_ACCEPTED_MESSAGE = (
        "✅ *Запрос принят*\n\n"
        "Использую аудио из reply-сообщения как reference, язык *{language}*, и запускаю клонирование.\n"
        "Когда генерация завершится, я пришлю отдельное voice сообщение или понятную ошибку, если reference audio не подошёл."
    )

    SUCCESS_TEMPLATE = (
        "✅ *Готово*\n\n"
        "Сгенерировано voice сообщение длительностью около *{duration:.1f} с* голосом *{speaker}*.\n\n"
        'Текст: "{text}"'
    )

    ERROR_TEMPLATE = "❌ *Ошибка*\n{error}\n\nОткройте `/help`, чтобы посмотреть корректный синтаксис команд и примеры."

    def __init__(
        self,
        synthesizer: Any | None,
        settings: TelegramSettings,
        sender: MessageSender,
        logger: logging.Logger | None = None,
        job_orchestrator=None,
        delivery_store=None,
        client: "TelegramClient | None" = None,
        rate_limiter: "TelegramRateLimiter | None" = None,
    ):
        """
        Initialize command dispatcher.

        Args:
            synthesizer: TTS synthesizer instance or None in remote-only command flow
            settings: Telegram settings
            sender: Message sender for Telegram API
            logger: Optional logger instance
            job_orchestrator: Optional job orchestrator for Stage 2 job integration
            delivery_store: Optional delivery metadata store for Stage 2
            client: Optional Telegram client for clone media download
            rate_limiter: Optional Telegram rate limiter for per-user throttling
        """
        self._synthesizer = synthesizer
        self._settings = settings
        self._sender = sender
        self._logger = logger or LOGGER
        # Stage 2: Job integration
        self._job_orchestrator = job_orchestrator
        self._delivery_store = delivery_store
        self._client = client
        self._rate_limiter = rate_limiter

    @property
    def _use_job_model(self) -> bool:
        """Whether to use job model for TTS (Stage 2)."""
        return self._job_orchestrator is not None and self._delivery_store is not None

    def _get_help_message(self) -> str:
        """Get formatted help message with available speakers."""
        speakers = ", ".join(get_valid_speakers())
        return self.HELP_MESSAGE.format(speakers=speakers) + self._runtime_capability_summary()

    def _runtime_capability_available(self, mode: str) -> bool:
        return self._settings.resolve_runtime_model_binding(mode) is not None

    def _runtime_capability_summary(self) -> str:
        family = self._settings.active_family or "unbound"
        return (
            "\n\n*Runtime bindings:*\n"
            f"• family: `{family}`\n"
            f"• custom: `{'on' if self._runtime_capability_available('custom') else 'off'}`\n"
            f"• design: `{'on' if self._runtime_capability_available('design') else 'off'}`\n"
            f"• clone: `{'on' if self._runtime_capability_available('clone') else 'off'}`"
        )

    # START_CONTRACT: handle_update
    #   PURPOSE: Validate and route a Telegram update to the correct command handler.
    #   INPUTS: { text: str - incoming message text, user_id: int - Telegram user identifier, chat_id: int - Telegram chat identifier, message_id: int - Telegram message identifier, chat_type: str - Telegram chat type, reply_to_message: dict[str, Any] | None - optional replied message payload }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Emits logs and metrics, sends Telegram responses, and may start synthesis or job workflows.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: handle_update
    async def handle_update(
        self,
        text: str,
        user_id: int,
        chat_id: int,
        message_id: int,
        chat_type: str,
        reply_to_message: dict[str, Any] | None = None,
    ) -> None:
        """
        Handle incoming Telegram update.

        Args:
            text: Message text
            user_id: Telegram user ID
            chat_id: Telegram chat ID
            message_id: Telegram message ID
            chat_type: Telegram chat type
            reply_to_message: Optional replied Telegram message payload
        """
        # START_BLOCK_LOG_INCOMING_UPDATE
        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="[Dispatcher][handle_update][BLOCK_LOG_INCOMING_UPDATE]",
            message="Received Telegram update",
            user_id=user_id,
            chat_id=chat_id,
            chat_type=chat_type,
            message_length=len(text) if text else 0,
        )
        # END_BLOCK_LOG_INCOMING_UPDATE

        # START_BLOCK_VALIDATE_USER
        # Check if private chat
        if not is_private_chat(chat_type):
            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="[Dispatcher][handle_update][BLOCK_VALIDATE_USER]",
                message="Ignoring non-private chat message",
                user_id=user_id,
                chat_id=chat_id,
                chat_type=chat_type,
            )
            # Silently ignore non-private chats for security
            return

        # Check user allowlist
        if not self._settings.is_user_allowed(user_id):
            log_telegram_event(
                self._logger,
                level=logging.WARNING,
                event="[Dispatcher][handle_update][BLOCK_VALIDATE_USER]",
                message="User not in allowlist",
                user_id=user_id,
                chat_id=chat_id,
            )
            self._metrics.command_rejected("any", "not_in_allowlist")
            await self._sender.send_text(
                chat_id,
                "⛔ *Access Denied*\n\nYou are not authorized to use this bot.",
            )
            return
        # END_BLOCK_VALIDATE_USER

        # START_BLOCK_PARSE_COMMAND
        parsed = parse_command(
            text,
            user_id,
            chat_id,
            message_id,
            reply_to_message=reply_to_message,
        )
        if parsed is None:
            # Not a command, ignore
            return
        # END_BLOCK_PARSE_COMMAND

        # START_BLOCK_APPLY_RATE_LIMIT
        # Track command received
        command_name = parsed.command.value
        self._metrics.command_received(command_name)

        if self._rate_limiter is not None and self._rate_limiter.is_enabled:
            decision = self._rate_limiter.check_and_consume(user_id)
            if not decision.allowed:
                retry_after_seconds = max(
                    1, math.ceil(decision.retry_after_seconds or 0.0)
                )
                log_telegram_event(
                    self._logger,
                    level=logging.WARNING,
                    event="[Dispatcher][handle_update][BLOCK_APPLY_RATE_LIMIT]",
                    message="User exceeded Telegram command rate limit",
                    user_id=user_id,
                    chat_id=chat_id,
                    command=command_name,
                    retry_after_seconds=retry_after_seconds,
                    limit=decision.limit,
                )
                self._metrics.command_rejected(command_name, "rate_limited")
                await self._sender.send_text(
                    chat_id,
                    (
                        "⏳ *Слишком много запросов*\n\n"
                        f"Подождите *{retry_after_seconds}* сек. перед следующей командой."
                    ),
                )
                return
        # END_BLOCK_APPLY_RATE_LIMIT

        # START_BLOCK_ROUTE_COMMAND
        # Route to appropriate handler
        await self._route_command(parsed)
        # END_BLOCK_ROUTE_COMMAND

    async def _route_command(self, parsed: ParsedCommand) -> None:
        """Route parsed command to handler."""
        # START_BLOCK_BUILD_HANDLER_MAP
        handlers: dict[CommandType, Callable[[ParsedCommand], Awaitable[None]]] = {
            CommandType.START: self._handle_start,
            CommandType.HELP: self._handle_help,
            CommandType.TTS: self._handle_tts,
            CommandType.DESIGN: self._handle_design,
            CommandType.CLONE: self._handle_clone,
            CommandType.UNKNOWN: self._handle_unknown,
        }
        # END_BLOCK_BUILD_HANDLER_MAP

        # START_BLOCK_ROUTE_COMMAND_DISPATCH
        handler = handlers.get(parsed.command, self._handle_unknown)

        # Track command accepted
        self._metrics.command_accepted(parsed.command.value)

        await handler(parsed)
        # END_BLOCK_ROUTE_COMMAND_DISPATCH

    async def _handle_start(self, parsed: ParsedCommand) -> None:
        """Handle /start command."""
        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="[Dispatcher][_handle_start][_handle_start]",
            message="Handling /start command",
            user_id=parsed.user_id,
            chat_id=parsed.chat_id,
        )

        await self._sender.send_text(
            parsed.chat_id, self.START_MESSAGE + self._runtime_capability_summary()
        )

    async def _handle_help(self, parsed: ParsedCommand) -> None:
        """Handle /help command."""
        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="[Dispatcher][_handle_help][_handle_help]",
            message="Handling /help command",
            user_id=parsed.user_id,
            chat_id=parsed.chat_id,
        )

        await self._sender.send_text(parsed.chat_id, self._get_help_message())

    async def _handle_tts(self, parsed: ParsedCommand) -> None:
        """Handle /tts command with async UX."""
        if not self._runtime_capability_available("custom"):
            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(
                    error="Обычный синтез недоступен: custom capability не привязана к текущему runtime."
                ),
            )
            return
        # START_BLOCK_LOG_TTS_COMMAND
        command_name = parsed.command.value

        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="[Dispatcher][_handle_tts][BLOCK_LOG_TTS_COMMAND]",
            message="TTS command received",
            user_id=parsed.user_id,
            chat_id=parsed.chat_id,
            text_length=len(parsed.args),
        )
        # END_BLOCK_LOG_TTS_COMMAND

        # START_BLOCK_VALIDATE_TTS_COMMAND
        # Validate command syntax first
        validation = validate_tts_command(
            parsed, self._settings.telegram_max_text_length
        )
        if not validation.is_valid:
            log_telegram_event(
                self._logger,
                level=logging.WARNING,
                event="[Dispatcher][_handle_tts][BLOCK_VALIDATE_TTS_COMMAND]",
                message="TTS command rejected due to validation failure",
                user_id=parsed.user_id,
                chat_id=parsed.chat_id,
                reason="validation_failed",
                error=validation.error_message,
            )

            self._metrics.command_rejected(command_name, "validation_failed")

            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(error=validation.error_message),
            )
            return
        # END_BLOCK_VALIDATE_TTS_COMMAND

        # START_BLOCK_PARSE_TTS_COMMAND
        # Parse TTS arguments for speaker and speed
        tts_args = parse_tts_args(parsed.args)
        if tts_args is None:
            # This shouldn't happen if validation passed, but handle it
            log_telegram_event(
                self._logger,
                level=logging.WARNING,
                event="[Dispatcher][_handle_tts][BLOCK_PARSE_TTS_COMMAND]",
                message="TTS command rejected due to parsing failure",
                user_id=parsed.user_id,
                chat_id=parsed.chat_id,
                reason="parse_failed",
            )

            self._metrics.command_rejected(command_name, "parse_failed")

            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(
                    error="Failed to parse /tts arguments. Use `/help` for usage."
                ),
            )
            return
        # END_BLOCK_PARSE_TTS_COMMAND

        # START_BLOCK_ACKNOWLEDGE_TTS_COMMAND
        # Track command accepted
        self._metrics.command_accepted(command_name)

        # Determine effective speaker and speed
        effective_speaker = tts_args.speaker or self._settings.telegram_default_speaker
        effective_speed = tts_args.speed or 1.0

        # Acknowledge with details
        await self._sender.send_text(
            parsed.chat_id,
            self.ACCEPTED_MESSAGE.format(
                speaker=effective_speaker,
                speed=effective_speed,
                language=tts_args.language,
            ),
        )
        # END_BLOCK_ACKNOWLEDGE_TTS_COMMAND

        # START_BLOCK_ROUTE_TTS_EXECUTION
        if not self._use_job_model:
            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="[Dispatcher][_handle_tts][BLOCK_ROUTE_TTS_EXECUTION]",
                message="TTS command requires remote async job orchestration but none is configured",
                user_id=parsed.user_id,
                chat_id=parsed.chat_id,
                message_id=parsed.message_id,
            )
            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(
                    error="Обычный синтез временно недоступен: не настроен удалённый async server."
                ),
            )
            return

        await self._handle_tts_via_job(
            parsed.chat_id,
            parsed.message_id,
            tts_args.text,
            effective_speaker,
            effective_speed,
            tts_args.language,
        )
        # END_BLOCK_ROUTE_TTS_EXECUTION

    async def _handle_tts_via_job(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        speaker: str,
        speed: float,
        language: str,
    ) -> None:
        """
        Handle TTS via job model (Stage 2).

        Creates a job for TTS synthesis and delivery metadata.
        Job completion and delivery is handled by TelegramJobPoller.
        """
        from telegram_bot.observability import METRICS

        assert self._job_orchestrator is not None
        assert self._delivery_store is not None
        job_orchestrator = self._job_orchestrator
        delivery_store = self._delivery_store

        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="[Dispatcher][_handle_tts_via_job][_handle_tts_via_job]",
            message="TTS request queued for processing via job model",
            chat_id=chat_id,
            message_id=message_id,
            text_length=len(text),
            speaker=speaker,
            speed=speed,
            language=language,
        )

        # Submit job through orchestrator
        result = await job_orchestrator.submit_tts_job(
            text=text,
            speaker=speaker,
            speed=speed,
            language=language,
            chat_id=chat_id,
            message_id=message_id,
        )

        if result.success:
            if result.is_duplicate:
                await self._ensure_duplicate_delivery_metadata(
                    chat_id=chat_id,
                    message_id=message_id,
                    job_id=result.job_id,
                    submit_request_id=result.submit_request_id,
                )
                METRICS.jobs_duplicate()
                log_telegram_event(
                    self._logger,
                    level=logging.INFO,
                    event="[Dispatcher][_handle_tts_via_job][_handle_tts_via_job]",
                    message="Duplicate job submission detected, re-queuing delivery for existing job",
                    chat_id=chat_id,
                    message_id=message_id,
                    job_id=result.job_id,
                    delivery_requeued=True,
                )
            else:
                # Create delivery metadata
                await delivery_store.create(
                    chat_id=chat_id,
                    message_id=message_id,
                    job_id=result.job_id,
                    submit_request_id=result.submit_request_id,
                )
                METRICS.jobs_submitted()
        else:
            # Submission failed
            METRICS.jobs_submission_failed()
            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="[Dispatcher][_handle_tts_via_job][_handle_tts_via_job]",
                message="Job submission failed",
                chat_id=chat_id,
                message_id=message_id,
                error=result.error_message,
            )
            # Send error message to user
            await self._sender.send_text(
                chat_id,
                self.ERROR_TEMPLATE.format(
                    error=f"Failed to submit job: {result.error_message}"
                ),
            )

    async def _handle_design(self, parsed: ParsedCommand) -> None:
        """
        Handle /design command with async UX (Stage 3 Voice Design).

        Creates a job for voice design synthesis and delivery metadata.
        Job completion and delivery is handled by TelegramJobPoller.
        """
        if not self._runtime_capability_available("design"):
            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(
                    error="Voice Design недоступен: design capability не привязана к текущему runtime."
                ),
            )
            return
        from telegram_bot.handlers.commands import (
            MAX_TEXT_LENGTH,
            MAX_VOICE_DESCRIPTION_LENGTH,
        )

        # START_BLOCK_LOG_DESIGN_COMMAND
        command_name = parsed.command.value

        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="[Dispatcher][_handle_design][BLOCK_LOG_DESIGN_COMMAND]",
            message="Voice Design command received",
            user_id=parsed.user_id,
            chat_id=parsed.chat_id,
            text_length=len(parsed.args),
        )
        # END_BLOCK_LOG_DESIGN_COMMAND

        # START_BLOCK_VALIDATE_DESIGN_COMMAND
        # Validate command syntax
        validation = validate_design_command(
            parsed,
            max_voice_description_length=MAX_VOICE_DESCRIPTION_LENGTH,
            max_text_length=MAX_TEXT_LENGTH,
        )
        if not validation.is_valid:
            log_telegram_event(
                self._logger,
                level=logging.WARNING,
                event="[Dispatcher][_handle_design][BLOCK_VALIDATE_DESIGN_COMMAND]",
                message="Design command rejected due to validation failure",
                user_id=parsed.user_id,
                chat_id=parsed.chat_id,
                reason="validation_failed",
                error=validation.error_message,
            )

            self._metrics.command_rejected(command_name, "validation_failed")

            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(error=validation.error_message),
            )
            return
        # END_BLOCK_VALIDATE_DESIGN_COMMAND

        # START_BLOCK_PARSE_DESIGN_COMMAND
        # Parse design arguments
        design_args = parse_design_args(parsed.args)
        if design_args is None:
            log_telegram_event(
                self._logger,
                level=logging.WARNING,
                event="[Dispatcher][_handle_design][BLOCK_PARSE_DESIGN_COMMAND]",
                message="Design command rejected due to parsing failure",
                user_id=parsed.user_id,
                chat_id=parsed.chat_id,
                reason="parse_failed",
            )

            self._metrics.command_rejected(command_name, "parse_failed")

            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(
                    error="Failed to parse /design arguments. Use `/help` for usage."
                ),
            )
            return
        # END_BLOCK_PARSE_DESIGN_COMMAND

        # START_BLOCK_ACKNOWLEDGE_DESIGN_COMMAND
        # Track command accepted
        self._metrics.command_accepted(command_name)

        # Acknowledge with voice description preview
        voice_preview = (
            design_args.voice_description[:50] + "..."
            if len(design_args.voice_description) > 50
            else design_args.voice_description
        )
        await self._sender.send_text(
            parsed.chat_id,
            self.DESIGN_ACCEPTED_MESSAGE.format(
                voice_description=voice_preview,
                language=design_args.language,
            ),
        )
        # END_BLOCK_ACKNOWLEDGE_DESIGN_COMMAND

        # START_BLOCK_ROUTE_DESIGN_EXECUTION
        if not self._use_job_model:
            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="[Dispatcher][_handle_design][BLOCK_ROUTE_DESIGN_EXECUTION]",
                message="Design command requires remote async job orchestration but none is configured",
                user_id=parsed.user_id,
                chat_id=parsed.chat_id,
                message_id=parsed.message_id,
            )
            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(
                    error="Voice Design временно недоступен: не настроен удалённый async server."
                ),
            )
            return

        await self._handle_design_via_job(
            parsed.chat_id,
            parsed.message_id,
            design_args.voice_description,
            design_args.text,
            design_args.language,
        )
        # END_BLOCK_ROUTE_DESIGN_EXECUTION

    async def _handle_design_via_job(
        self,
        chat_id: int,
        message_id: int,
        voice_description: str,
        text: str,
        language: str,
    ) -> None:
        """
        Handle Voice Design via job model (Stage 3).

        Creates a job for voice design synthesis and delivery metadata.
        Job completion and delivery is handled by TelegramJobPoller.
        """
        from telegram_bot.observability import METRICS

        assert self._job_orchestrator is not None
        assert self._delivery_store is not None
        job_orchestrator = self._job_orchestrator
        delivery_store = self._delivery_store

        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="[Dispatcher][_handle_design_via_job][_handle_design_via_job]",
            message="Voice Design request queued for processing via job model",
            chat_id=chat_id,
            message_id=message_id,
            voice_description_length=len(voice_description),
            text_length=len(text),
            language=language,
        )

        # Submit job through orchestrator
        result = await job_orchestrator.submit_design_job(
            voice_description=voice_description,
            text=text,
            language=language,
            chat_id=chat_id,
            message_id=message_id,
        )

        if result.success:
            if result.is_duplicate:
                await self._ensure_duplicate_delivery_metadata(
                    chat_id=chat_id,
                    message_id=message_id,
                    job_id=result.job_id,
                    submit_request_id=result.submit_request_id,
                )
                METRICS.jobs_duplicate()
                log_telegram_event(
                    self._logger,
                    level=logging.INFO,
                    event="[Dispatcher][_handle_design_via_job][_handle_design_via_job]",
                    message="Duplicate design job submission detected, re-queuing delivery for existing job",
                    chat_id=chat_id,
                    message_id=message_id,
                    job_id=result.job_id,
                    delivery_requeued=True,
                )
            else:
                await delivery_store.create(
                    chat_id=chat_id,
                    message_id=message_id,
                    job_id=result.job_id,
                    submit_request_id=result.submit_request_id,
                )
                METRICS.jobs_submitted()
        else:
            METRICS.jobs_submission_failed()
            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="[Dispatcher][_handle_design_via_job][_handle_design_via_job]",
                message="Design job submission failed",
                chat_id=chat_id,
                message_id=message_id,
                error=result.error_message,
            )
            await self._sender.send_text(
                chat_id,
                self.ERROR_TEMPLATE.format(
                    error=f"Failed to submit design job: {result.error_message}"
                ),
            )

    async def _handle_clone(self, parsed: ParsedCommand) -> None:
        """
        Handle /clone command with async UX (Stage 4 Voice Cloning).

        This command requires a reply to a message with media (voice, audio, or document).
        The reply should contain: /clone [-- ref=<transcript>] -- <text>

        Creates a job for voice cloning synthesis and delivery metadata.
        Job completion and delivery is handled by TelegramJobPoller.
        """
        if not self._runtime_capability_available("clone"):
            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(
                    error="Voice cloning недоступен: clone capability не привязана к текущему runtime."
                ),
            )
            return
        # START_BLOCK_LOG_CLONE_COMMAND
        command_name = parsed.command.value

        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="[Dispatcher][_handle_clone][BLOCK_LOG_CLONE_COMMAND]",
            message="Voice Clone command received",
            user_id=parsed.user_id,
            chat_id=parsed.chat_id,
            text_length=len(parsed.args),
        )
        # END_BLOCK_LOG_CLONE_COMMAND

        # START_BLOCK_VALIDATE_CLONE_COMMAND
        # Validate command syntax
        validation = validate_clone_command(
            parsed,
            max_text_length=MAX_CLONE_TEXT_LENGTH,
        )
        if not validation.is_valid:
            log_telegram_event(
                self._logger,
                level=logging.WARNING,
                event="[Dispatcher][_handle_clone][BLOCK_VALIDATE_CLONE_COMMAND]",
                message="Clone command rejected due to validation failure",
                user_id=parsed.user_id,
                chat_id=parsed.chat_id,
                reason="validation_failed",
                error=validation.error_message,
            )

            self._metrics.command_rejected(command_name, "validation_failed")

            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(error=validation.error_message),
            )
            return
        # END_BLOCK_VALIDATE_CLONE_COMMAND

        # START_BLOCK_PARSE_CLONE_COMMAND
        # Parse clone arguments
        clone_args = parse_clone_args(parsed.args)
        if clone_args is None:
            log_telegram_event(
                self._logger,
                level=logging.WARNING,
                event="[Dispatcher][_handle_clone][BLOCK_PARSE_CLONE_COMMAND]",
                message="Clone command rejected due to parsing failure",
                user_id=parsed.user_id,
                chat_id=parsed.chat_id,
                reason="parse_failed",
            )

            self._metrics.command_rejected(command_name, "parse_failed")

            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(
                    error="Failed to parse /clone arguments. Use `/help` for usage."
                ),
            )
            return
        # END_BLOCK_PARSE_CLONE_COMMAND

        # START_BLOCK_VALIDATE_CLONE_RUNTIME
        if not self._use_job_model:
            # Clone is only supported via job model
            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="[Dispatcher][_handle_clone][BLOCK_VALIDATE_CLONE_RUNTIME]",
                message="Clone command requires job model but none is configured",
                user_id=parsed.user_id,
                chat_id=parsed.chat_id,
            )

            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(
                    error="Voice cloning is not available. Please try again later."
                ),
            )
            return

        reply_message = parsed.reply_to_message
        if reply_message is None:
            self._metrics.command_rejected(command_name, "missing_reply_media")
            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(
                    error="Для `/clone` нужно ответить на сообщение с `voice`, `audio` или `document` с аудио."
                ),
            )
            return

        telegram_client = self._client or getattr(self._sender, "_client", None)
        if telegram_client is None:
            self._metrics.command_rejected(
                command_name, "clone_media_client_unavailable"
            )
            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(
                    error="Клонирование временно недоступно: не настроен доступ к Telegram media API."
                ),
            )
            return
        # END_BLOCK_VALIDATE_CLONE_RUNTIME

        # START_BLOCK_SUBMIT_TTS_JOB
        command_message_id = parsed.message_id
        reply_message_id = reply_message.get("message_id") or parsed.message_id
        media_kinds = [
            kind for kind in ("voice", "audio", "document") if reply_message.get(kind)
        ]
        media_kind = media_kinds[0] if media_kinds else "unknown"

        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="[Dispatcher][_handle_clone][BLOCK_SUBMIT_TTS_JOB]",
            message="Starting clone reference media staging",
            user_id=parsed.user_id,
            chat_id=parsed.chat_id,
            message_id=command_message_id,
            reply_message_id=reply_message_id,
            media_kind=media_kind,
        )

        try:
            staged_media, validation = await stage_clone_media(
                client=telegram_client,
                message=reply_message,
                settings=self._settings,
            )
            ref_audio_path = str(staged_media.get_audio_path())

            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="[Dispatcher][_handle_clone][BLOCK_SUBMIT_TTS_JOB]",
                message="Clone reference media staged successfully",
                user_id=parsed.user_id,
                chat_id=parsed.chat_id,
                message_id=command_message_id,
                reply_message_id=reply_message_id,
                media_kind=media_kind,
                staged_audio_path=ref_audio_path,
                was_converted=staged_media.was_converted,
                content_type=validation.content_type,
                file_size=validation.file_size,
            )
        except MediaValidationError as exc:
            self._metrics.command_rejected(command_name, "invalid_reference_media")
            log_telegram_event(
                self._logger,
                level=logging.WARNING,
                event="[Dispatcher][_handle_clone][BLOCK_SUBMIT_TTS_JOB]",
                message="Clone reference media validation failed",
                user_id=parsed.user_id,
                chat_id=parsed.chat_id,
                message_id=command_message_id,
                reply_message_id=reply_message_id,
                media_kind=media_kind,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(error=str(exc)),
            )
            return
        except DownloadError as exc:
            self._metrics.command_rejected(
                command_name, "reference_media_download_failed"
            )
            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="[Dispatcher][_handle_clone][BLOCK_SUBMIT_TTS_JOB]",
                message="Clone reference media download failed",
                user_id=parsed.user_id,
                chat_id=parsed.chat_id,
                message_id=command_message_id,
                reply_message_id=reply_message_id,
                media_kind=media_kind,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(
                    error=f"Не удалось скачать reference audio из reply-сообщения: {exc}"
                ),
            )
            return
        except Exception as exc:
            self._metrics.command_rejected(command_name, "reference_media_stage_failed")
            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="[Dispatcher][_handle_clone][BLOCK_SUBMIT_TTS_JOB]",
                message="Clone reference media staging failed unexpectedly",
                user_id=parsed.user_id,
                chat_id=parsed.chat_id,
                message_id=command_message_id,
                reply_message_id=reply_message_id,
                media_kind=media_kind,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(
                    error="Не удалось подготовить reference audio для клонирования. Попробуйте отправить аудио ещё раз."
                ),
            )
            return
        # END_BLOCK_SUBMIT_TTS_JOB

        # START_BLOCK_DELIVER_RESULT
        # Track command accepted
        self._metrics.command_accepted(command_name)

        # Acknowledge
        await self._sender.send_text(
            parsed.chat_id,
            self.CLONE_ACCEPTED_MESSAGE.format(language=clone_args.language),
        )

        await self._handle_clone_via_job(
            parsed.chat_id,
            command_message_id,
            clone_args.text,
            clone_args.ref_text,
            clone_args.language,
            ref_audio_path,
            validation.content_type or "application/octet-stream",
        )
        # END_BLOCK_DELIVER_RESULT

    async def _handle_clone_via_job(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        ref_text: str | None,
        language: str,
        ref_audio_path: str,
        ref_audio_content_type: str,
    ) -> None:
        """
        Handle Voice Clone via job model (Stage 4).

        Creates a job for voice cloning synthesis and delivery metadata.
        Job completion and delivery is handled by TelegramJobPoller.
        """
        from telegram_bot.observability import METRICS

        assert self._job_orchestrator is not None
        assert self._delivery_store is not None
        job_orchestrator = self._job_orchestrator
        delivery_store = self._delivery_store

        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="[Dispatcher][_handle_clone_via_job][_handle_clone_via_job]",
            message="Voice Clone request queued for processing via job model",
            chat_id=chat_id,
            message_id=message_id,
            ref_text_provided=ref_text is not None,
            text_length=len(text),
            language=language,
            ref_audio_path=ref_audio_path,
        )

        # Submit job through orchestrator
        result = await job_orchestrator.submit_clone_job(
            text=text,
            ref_text=ref_text,
            language=language,
            chat_id=chat_id,
            message_id=message_id,
            ref_audio_path=ref_audio_path,
            ref_audio_content_type=ref_audio_content_type,
        )

        if result.success:
            if result.is_duplicate:
                METRICS.jobs_duplicate()
                await self._ensure_duplicate_delivery_metadata(
                    chat_id=chat_id,
                    message_id=message_id,
                    job_id=result.job_id,
                    submit_request_id=result.submit_request_id,
                )
                log_telegram_event(
                    self._logger,
                    level=logging.INFO,
                    event="[Dispatcher][_handle_clone_via_job][_handle_clone_via_job]",
                    message="Duplicate clone job submission detected, re-queuing delivery for existing job",
                    chat_id=chat_id,
                    message_id=message_id,
                    job_id=result.job_id,
                    delivery_requeued=True,
                )
            else:
                await delivery_store.create(
                    chat_id=chat_id,
                    message_id=message_id,
                    job_id=result.job_id,
                    submit_request_id=result.submit_request_id,
                )
                METRICS.jobs_submitted()
        else:
            METRICS.jobs_submission_failed()
            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="[Dispatcher][_handle_clone_via_job][_handle_clone_via_job]",
                message="Clone job submission failed",
                chat_id=chat_id,
                message_id=message_id,
                error=result.error_message,
            )
            await self._sender.send_text(
                chat_id,
                self.ERROR_TEMPLATE.format(
                    error=f"Failed to submit clone job: {result.error_message}"
                ),
            )

    # START_CONTRACT: _ensure_duplicate_delivery_metadata
    #   PURPOSE: Preserve existing Telegram delivery state when a duplicate remote submit is observed.
    #   INPUTS: { chat_id: int - Telegram chat identifier, message_id: int - Telegram message identifier, job_id: str | None - remote job identifier, submit_request_id: str | None - original submit correlation id }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: May create pending delivery metadata when no prior record exists.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: _ensure_duplicate_delivery_metadata
    async def _ensure_duplicate_delivery_metadata(
        self,
        *,
        chat_id: int,
        message_id: int,
        job_id: str | None,
        submit_request_id: str | None,
    ) -> None:
        assert self._delivery_store is not None
        if job_id is None:
            return

        existing_metadata = await self._delivery_store.get(chat_id, message_id)
        if existing_metadata is not None:
            return

        await self._delivery_store.create(
            chat_id=chat_id,
            message_id=message_id,
            job_id=job_id,
            submit_request_id=submit_request_id,
        )

    async def _handle_unknown(self, parsed: ParsedCommand) -> None:
        """Handle unknown commands."""
        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="[Dispatcher][_handle_unknown][_handle_unknown]",
            message="Handling unknown command",
            user_id=parsed.user_id,
            chat_id=parsed.chat_id,
            raw_command=parsed.raw_text,
        )

        self._metrics.command_rejected("unknown", "unknown_command")

        await self._sender.send_text(
            parsed.chat_id,
            "🤔 *Unknown command*\n\nUse /help to see available commands.",
        )

    # START_CONTRACT: _metrics
    #   PURPOSE: Expose the shared Telegram metrics collector used by the dispatcher.
    #   INPUTS: {}
    #   OUTPUTS: { TelegramMetrics - shared Telegram metrics singleton }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: _metrics
    @property
    def _metrics(self):
        """Access metrics singleton."""
        return METRICS

__all__ = [
    "LOGGER",
    "MessageSender",
    "CommandDispatcher",
]
