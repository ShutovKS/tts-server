"""
Command dispatcher for Telegram bot with observability.

This module routes parsed commands to appropriate handlers and provides
the async UX with acknowledgment and result delivery, featuring:
- Command metrics (received, accepted, rejected)
- Correlation context propagation
- Structured logging with operation tracking
- Job integration with core job model (Stage 2)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol

from core.observability import get_logger, log_event
from telegram_bot.handlers.commands import (
    CommandType,
    ParsedCommand,
    ParsedTTSArgs,
    get_valid_speakers,
    is_private_chat,
    parse_command,
    parse_design_args,
    parse_tts_args,
    validate_design_args,
    validate_design_command,
    validate_tts_args,
    validate_tts_command,
    parse_clone_args,
    validate_clone_command,
    MAX_TEXT_LENGTH as MAX_CLONE_TEXT_LENGTH,
)
from telegram_bot.handlers.tts_handler import TTSSynthesizer
from telegram_bot.media import DownloadError, MediaValidationError, stage_clone_media
from telegram_bot.observability import (
    METRICS,
    TelegramCorrelationContext,
    log_telegram_event,
)


if TYPE_CHECKING:
    from telegram_bot.config import TelegramSettings
    from telegram_bot.polling import TelegramClient
    from telegram_bot.rate_limiter import TelegramRateLimiter
    from telegram_bot.sender import TelegramSender


LOGGER = get_logger(__name__)


class MessageSender(Protocol):
    """Protocol for sending messages to Telegram."""
    
    async def send_text(self, chat_id: int, text: str) -> None:
        """Send text message to chat."""
        ...
    
    async def send_voice(self, chat_id: int, audio_bytes: bytes, caption: str | None = None) -> None:
        """Send voice message to chat."""
        ...


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
        "`/tts speaker=Ryan speed=0.9 -- Привет, это тест`\n"
        "`/design calm narrator -- Привет, мир`\n\n"
        "Для `/clone` сначала отправьте `voice`, `audio` или `document`, затем ответьте на это сообщение командой `/clone ...`."
    )
    
    HELP_MESSAGE = (
        "📖 *Справка по командам*\n\n"
        "Бот работает только в *личном чате*. Все результаты приходят отдельным *voice* сообщением после завершения генерации.\n\n"
        "*1. Обычный TTS*\n"
        "`/tts -- <текст>`\n"
        "`/tts speaker=<speaker> -- <текст>`\n"
        "`/tts speed=<speed> -- <текст>`\n"
        "`/tts speaker=<speaker> speed=<speed> -- <текст>`\n\n"
        "*Параметры /tts:*\n"
        "• `speaker` — имя голоса\n"
        "• `speed` — скорость от 0.5 до 2.0\n"
        "• если параметры не нужны, используйте `/tts -- текст`\n\n"
        "*Доступные голоса:*\n"
        "{speakers}\n\n"
        "*2. Voice Design*\n"
        "`/design <описание_голоса> -- <текст>`\n"
        "Пример: `/design calm documentary narrator -- Расскажи о космосе`\n\n"
        "*3. Voice Cloning*\n"
        "Сначала отправьте `voice`, `audio` или `document` с поддерживаемым аудио, затем *ответьте* на это сообщение одной из команд:\n"
        "`/clone -- <текст>`\n"
        "`/clone ref=<транскрипт> -- <текст>`\n"
        "Пример: reply на audio → `/clone ref=This is my sample -- Скажи это моим голосом`\n\n"
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
        "Озвучиваю текст голосом *{speaker}* со скоростью *{speed}x*.\n"
        "Когда генерация завершится, я пришлю отдельное voice сообщение."
    )
    
    DESIGN_ACCEPTED_MESSAGE = (
        "✅ *Запрос принят*\n\n"
        "Создаю голос по описанию: *{voice_description}*.\n"
        "Когда генерация завершится, я пришлю отдельное voice сообщение."
    )
    
    CLONE_ACCEPTED_MESSAGE = (
        "✅ *Запрос принят*\n\n"
        "Использую аудио из reply-сообщения как reference и запускаю клонирование.\n"
        "Когда генерация завершится, я пришлю отдельное voice сообщение или понятную ошибку, если reference audio не подошёл."
    )
    
    SUCCESS_TEMPLATE = (
        "✅ *Готово*\n\n"
        "Сгенерировано voice сообщение длительностью около *{duration:.1f} с* голосом *{speaker}*.\n\n"
        "Текст: \"{text}\""
    )
    
    ERROR_TEMPLATE = "❌ *Ошибка*\n{error}\n\nОткройте `/help`, чтобы посмотреть корректный синтаксис команд и примеры."
    
    def __init__(
        self,
        synthesizer: TTSSynthesizer,
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
            synthesizer: TTS synthesizer instance (used for direct synthesis fallback)
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
        return self.HELP_MESSAGE.format(speakers=speakers)
    
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
        # Log incoming update
        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="telegram.update.received",
            message="Received Telegram update",
            user_id=user_id,
            chat_id=chat_id,
            chat_type=chat_type,
            message_length=len(text) if text else 0,
        )
        
        # Check if private chat
        if not is_private_chat(chat_type):
            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="telegram.update.ignored",
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
                event="telegram.update.denied",
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
        
        # Parse command
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
        
        # Track command received
        command_name = parsed.command.value
        self._metrics.command_received(command_name)
        
        if self._rate_limiter is not None and self._rate_limiter.is_enabled:
            decision = self._rate_limiter.check_and_consume(user_id)
            if not decision.allowed:
                retry_after_seconds = max(1, math.ceil(decision.retry_after_seconds or 0.0))
                log_telegram_event(
                    self._logger,
                    level=logging.WARNING,
                    event="telegram.update.rate_limited",
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
        
        # Route to appropriate handler
        await self._route_command(parsed)
    
    async def _route_command(self, parsed: ParsedCommand) -> None:
        """Route parsed command to handler."""
        handlers: dict[CommandType, Callable[[ParsedCommand], Awaitable[None]]] = {
            CommandType.START: self._handle_start,
            CommandType.HELP: self._handle_help,
            CommandType.TTS: self._handle_tts,
            CommandType.DESIGN: self._handle_design,
            CommandType.CLONE: self._handle_clone,
            CommandType.UNKNOWN: self._handle_unknown,
        }
        
        handler = handlers.get(parsed.command, self._handle_unknown)
        
        # Track command accepted
        self._metrics.command_accepted(parsed.command.value)
        
        await handler(parsed)
    
    async def _handle_start(self, parsed: ParsedCommand) -> None:
        """Handle /start command."""
        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="telegram.command.start",
            message="Handling /start command",
            user_id=parsed.user_id,
            chat_id=parsed.chat_id,
        )
        
        await self._sender.send_text(parsed.chat_id, self.START_MESSAGE)
    
    async def _handle_help(self, parsed: ParsedCommand) -> None:
        """Handle /help command."""
        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="telegram.command.help",
            message="Handling /help command",
            user_id=parsed.user_id,
            chat_id=parsed.chat_id,
        )
        
        await self._sender.send_text(parsed.chat_id, self._get_help_message())
    
    async def _handle_tts(self, parsed: ParsedCommand) -> None:
        """Handle /tts command with async UX."""
        command_name = parsed.command.value
        
        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="telegram.command.tts.received",
            message="TTS command received",
            user_id=parsed.user_id,
            chat_id=parsed.chat_id,
            text_length=len(parsed.args),
        )
        
        # Validate command syntax first
        validation = validate_tts_command(parsed, self._settings.telegram_max_text_length)
        if not validation.is_valid:
            log_telegram_event(
                self._logger,
                level=logging.WARNING,
                event="telegram.command.tts.rejected",
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
        
        # Parse TTS arguments for speaker and speed
        tts_args = parse_tts_args(parsed.args)
        if tts_args is None:
            # This shouldn't happen if validation passed, but handle it
            log_telegram_event(
                self._logger,
                level=logging.WARNING,
                event="telegram.command.tts.rejected",
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
            ),
        )
        
        # Stage 2: Use job model if available
        if self._use_job_model:
            await self._handle_tts_via_job(
                parsed.chat_id,
                parsed.message_id,
                tts_args.text,
                effective_speaker,
                effective_speed,
            )
        else:
            # Fallback to direct synthesis
            await self._process_tts_async(
                parsed.chat_id,
                tts_args.text,
                effective_speaker,
                effective_speed,
            )
    
    async def _handle_tts_via_job(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        speaker: str,
        speed: float,
    ) -> None:
        """
        Handle TTS via job model (Stage 2).
        
        Creates a job for TTS synthesis and delivery metadata.
        Job completion and delivery is handled by TelegramJobPoller.
        """
        from telegram_bot.observability import METRICS
        
        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="telegram.command.tts.queued",
            message="TTS request queued for processing via job model",
            chat_id=chat_id,
            message_id=message_id,
            text_length=len(text),
            speaker=speaker,
            speed=speed,
        )
        
        # Submit job through orchestrator
        result = self._job_orchestrator.submit_tts_job(
            text=text,
            speaker=speaker,
            speed=speed,
            chat_id=chat_id,
            message_id=message_id,
        )
        
        if result.success:
            if result.is_duplicate:
                # Job already exists, just acknowledge
                METRICS.jobs_duplicate()
                log_telegram_event(
                    self._logger,
                    level=logging.INFO,
                    event="telegram.job.duplicate",
                    message="Duplicate job submission detected, reusing existing job",
                    chat_id=chat_id,
                    message_id=message_id,
                    job_id=result.job_id,
                )
            else:
                # Create delivery metadata
                await self._delivery_store.create(
                    chat_id=chat_id,
                    message_id=message_id,
                    job_id=result.job_id,
                )
                METRICS.jobs_submitted()
        else:
            # Submission failed
            METRICS.jobs_submission_failed()
            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.job.submit_failed",
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
    
    async def _process_tts_async(
        self,
        chat_id: int,
        text: str,
        speaker: str,
        speed: float,
    ) -> None:
        """Process TTS synthesis and send result."""
        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="telegram.tts.processing",
            message="Starting TTS synthesis",
            chat_id=chat_id,
            text_length=len(text),
            speaker=speaker,
            speed=speed,
        )
        
        result = await self._synthesizer.synthesize(
            text,
            speaker=speaker,
            speed=speed,
        )
        
        if result.success:
            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="telegram.tts.synthesis_completed",
                message="TTS synthesis completed",
                chat_id=chat_id,
                duration_ms=result.duration_ms,
                audio_size_bytes=len(result.audio_bytes) if result.audio_bytes else 0,
                speaker=speaker,
            )
            
            # Estimate duration from audio size (rough approximation)
            # Assuming ~12KB per second for OGG at typical settings
            est_duration = len(result.audio_bytes) / 12000 if result.audio_bytes else 0
            
            caption = self.SUCCESS_TEMPLATE.format(
                duration=est_duration,
                speaker=speaker,
                text=text[:100] + "..." if len(text) > 100 else text,
            )
            
            # Send voice with retry
            delivery_result = await self._sender.send_voice(
                chat_id,
                result.audio_bytes,
                caption=caption,
            )
            
            if delivery_result.success:
                log_telegram_event(
                    self._logger,
                    level=logging.INFO,
                    event="telegram.voice.send_completed",
                    message="Voice message sent successfully",
                    chat_id=chat_id,
                    attempts=delivery_result.attempts,
                    duration_ms=delivery_result.duration_ms,
                )
            else:
                log_telegram_event(
                    self._logger,
                    level=logging.ERROR,
                    event="telegram.voice.send_failed",
                    message="Voice message delivery failed",
                    chat_id=chat_id,
                    error=delivery_result.error_message,
                    error_class=delivery_result.error_class,
                    attempts=delivery_result.attempts,
                )
        else:
            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.tts.synthesis_failed",
                message="TTS synthesis failed",
                chat_id=chat_id,
                error=result.error_message,
                duration_ms=result.duration_ms,
                speaker=speaker,
            )
            
            await self._sender.send_text(
                chat_id,
                self.ERROR_TEMPLATE.format(error=result.error_message or "Unknown error during synthesis"),
            )
    
    async def _handle_design(self, parsed: ParsedCommand) -> None:
        """
        Handle /design command with async UX (Stage 3 Voice Design).
        
        Creates a job for voice design synthesis and delivery metadata.
        Job completion and delivery is handled by TelegramJobPoller.
        """
        from telegram_bot.handlers.commands import MAX_TEXT_LENGTH, MAX_VOICE_DESCRIPTION_LENGTH
        
        command_name = parsed.command.value
        
        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="telegram.command.design.received",
            message="Voice Design command received",
            user_id=parsed.user_id,
            chat_id=parsed.chat_id,
            text_length=len(parsed.args),
        )
        
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
                event="telegram.command.design.rejected",
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
        
        # Parse design arguments
        design_args = parse_design_args(parsed.args)
        if design_args is None:
            log_telegram_event(
                self._logger,
                level=logging.WARNING,
                event="telegram.command.design.rejected",
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
            ),
        )
        
        # Stage 3: Use job model for design
        if self._use_job_model:
            await self._handle_design_via_job(
                parsed.chat_id,
                parsed.message_id,
                design_args.voice_description,
                design_args.text,
            )
        else:
            # Fallback to direct synthesis
            await self._process_design_async(
                parsed.chat_id,
                design_args.voice_description,
                design_args.text,
            )
    
    async def _handle_design_via_job(
        self,
        chat_id: int,
        message_id: int,
        voice_description: str,
        text: str,
    ) -> None:
        """
        Handle Voice Design via job model (Stage 3).
        
        Creates a job for voice design synthesis and delivery metadata.
        Job completion and delivery is handled by TelegramJobPoller.
        """
        from telegram_bot.observability import METRICS
        
        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="telegram.command.design.queued",
            message="Voice Design request queued for processing via job model",
            chat_id=chat_id,
            message_id=message_id,
            voice_description_length=len(voice_description),
            text_length=len(text),
        )
        
        # Submit job through orchestrator
        result = self._job_orchestrator.submit_design_job(
            voice_description=voice_description,
            text=text,
            chat_id=chat_id,
            message_id=message_id,
        )
        
        if result.success:
            if result.is_duplicate:
                METRICS.jobs_duplicate()
                log_telegram_event(
                    self._logger,
                    level=logging.INFO,
                    event="telegram.job.duplicate",
                    message="Duplicate design job submission detected, reusing existing job",
                    chat_id=chat_id,
                    message_id=message_id,
                    job_id=result.job_id,
                )
            else:
                await self._delivery_store.create(
                    chat_id=chat_id,
                    message_id=message_id,
                    job_id=result.job_id,
                )
                METRICS.jobs_submitted()
        else:
            METRICS.jobs_submission_failed()
            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.job.submit_failed",
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
    
    async def _process_design_async(
        self,
        chat_id: int,
        voice_description: str,
        text: str,
    ) -> None:
        """Process Voice Design synthesis and send result."""
        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="telegram.design.processing",
            message="Starting Voice Design synthesis",
            chat_id=chat_id,
            voice_description_length=len(voice_description),
            text_length=len(text),
        )
        
        result = await self._synthesizer.synthesize_design(
            voice_description=voice_description,
            text=text,
        )
        
        if result.success:
            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="telegram.design.synthesis_completed",
                message="Voice Design synthesis completed",
                chat_id=chat_id,
                duration_ms=result.duration_ms,
                audio_size_bytes=len(result.audio_bytes) if result.audio_bytes else 0,
            )
            
            est_duration = len(result.audio_bytes) / 12000 if result.audio_bytes else 0
            
            if len(text) > 100:
                caption = f"✅ *Done!*\n\nGenerated voice design ({est_duration:.1f}s)\n\n\"{text[:100]}...\""
            else:
                caption = f"✅ *Done!*\n\nGenerated voice design ({est_duration:.1f}s)\n\n\"{text}\""
            
            delivery_result = await self._sender.send_voice(
                chat_id,
                result.audio_bytes,
                caption=caption,
            )
            
            if delivery_result.success:
                log_telegram_event(
                    self._logger,
                    level=logging.INFO,
                    event="telegram.voice.send_completed",
                    message="Voice message sent successfully",
                    chat_id=chat_id,
                    attempts=delivery_result.attempts,
                    duration_ms=delivery_result.duration_ms,
                )
            else:
                log_telegram_event(
                    self._logger,
                    level=logging.ERROR,
                    event="telegram.voice.send_failed",
                    message="Voice message delivery failed",
                    chat_id=chat_id,
                    error=delivery_result.error_message,
                    error_class=delivery_result.error_class,
                    attempts=delivery_result.attempts,
                )
        else:
            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.design.synthesis_failed",
                message="Voice Design synthesis failed",
                chat_id=chat_id,
                error=result.error_message,
                duration_ms=result.duration_ms,
            )
            
            await self._sender.send_text(
                chat_id,
                self.ERROR_TEMPLATE.format(error=result.error_message or "Unknown error during synthesis"),
            )



    async def _handle_clone(self, parsed: ParsedCommand) -> None:
        """
        Handle /clone command with async UX (Stage 4 Voice Cloning).
        
        This command requires a reply to a message with media (voice, audio, or document).
        The reply should contain: /clone [-- ref=<transcript>] -- <text>
        
        Creates a job for voice cloning synthesis and delivery metadata.
        Job completion and delivery is handled by TelegramJobPoller.
        """
        command_name = parsed.command.value
        
        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="telegram.command.clone.received",
            message="Voice Clone command received",
            user_id=parsed.user_id,
            chat_id=parsed.chat_id,
            text_length=len(parsed.args),
        )
        
        # Validate command syntax
        validation = validate_clone_command(
            parsed,
            max_text_length=MAX_CLONE_TEXT_LENGTH,
        )
        if not validation.is_valid:
            log_telegram_event(
                self._logger,
                level=logging.WARNING,
                event="telegram.command.clone.rejected",
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
        
        # Parse clone arguments
        clone_args = parse_clone_args(parsed.args)
        if clone_args is None:
            log_telegram_event(
                self._logger,
                level=logging.WARNING,
                event="telegram.command.clone.rejected",
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
        
        if not self._use_job_model:
            # Clone is only supported via job model
            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.command.clone.no_job_model",
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
            self._metrics.command_rejected(command_name, "clone_media_client_unavailable")
            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(
                    error="Клонирование временно недоступно: не настроен доступ к Telegram media API."
                ),
            )
            return

        try:
            staged_media, _ = await stage_clone_media(
                client=telegram_client,
                message=reply_message,
                settings=self._settings,
            )
        except MediaValidationError as exc:
            self._metrics.command_rejected(command_name, "invalid_reference_media")
            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(error=str(exc)),
            )
            return
        except DownloadError as exc:
            self._metrics.command_rejected(command_name, "reference_media_download_failed")
            await self._sender.send_text(
                parsed.chat_id,
                self.ERROR_TEMPLATE.format(
                    error=f"Не удалось скачать reference audio из reply-сообщения: {exc}"
                ),
            )
            return

        # Track command accepted
        self._metrics.command_accepted(command_name)
        
        # Acknowledge
        await self._sender.send_text(
            parsed.chat_id,
            self.CLONE_ACCEPTED_MESSAGE,
        )

        reply_message_id = reply_message.get("message_id") or parsed.message_id
        ref_audio_path = str(staged_media.get_audio_path())

        await self._handle_clone_via_job(
            parsed.chat_id,
            reply_message_id,
            clone_args.text,
            clone_args.ref_text,
            ref_audio_path,
        )
    
    async def _handle_clone_via_job(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        ref_text: str | None,
        ref_audio_path: str,
    ) -> None:
        """
        Handle Voice Clone via job model (Stage 4).
        
        Creates a job for voice cloning synthesis and delivery metadata.
        Job completion and delivery is handled by TelegramJobPoller.
        """
        from telegram_bot.observability import METRICS
        
        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="telegram.command.clone.queued",
            message="Voice Clone request queued for processing via job model",
            chat_id=chat_id,
            message_id=message_id,
            ref_text_provided=ref_text is not None,
            text_length=len(text),
            ref_audio_path=ref_audio_path,
        )
        
        # Submit job through orchestrator
        result = self._job_orchestrator.submit_clone_job(
            text=text,
            ref_text=ref_text,
            chat_id=chat_id,
            message_id=message_id,
            ref_audio_path=ref_audio_path,
        )
        
        if result.success:
            if result.is_duplicate:
                METRICS.jobs_duplicate()
                log_telegram_event(
                    self._logger,
                    level=logging.INFO,
                    event="telegram.job.clone.duplicate",
                    message="Duplicate clone job submission detected, reusing existing job",
                    chat_id=chat_id,
                    message_id=message_id,
                    job_id=result.job_id,
                )
            else:
                await self._delivery_store.create(
                    chat_id=chat_id,
                    message_id=message_id,
                    job_id=result.job_id,
                )
                METRICS.jobs_submitted()
        else:
            METRICS.jobs_submission_failed()
            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.job.clone.submit_failed",
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

    async def _handle_unknown(self, parsed: ParsedCommand) -> None:
        """Handle unknown commands."""
        log_telegram_event(
            self._logger,
            level=logging.INFO,
            event="telegram.command.unknown",
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
    
    @property
    def _metrics(self):
        """Access metrics singleton."""
        return METRICS
