"""
TTS synthesis handler using core application service with observability.

This module bridges the Telegram transport layer with the existing
core TTS infrastructure, providing async audio synthesis with:
- Structured logging with correlation context
- Synthesis timing metrics
- Error classification
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from core.application import TTSApplicationService
from core.contracts.commands import (
    CustomVoiceCommand,
    VoiceDesignCommand,
    VoiceCloneCommand,
)
from core.contracts.results import GenerationResult
from core.errors import CoreError
from core.observability import Timer, get_logger, log_event

from telegram_bot.observability import (
    METRICS,
    TelegramCorrelationContext,
    log_telegram_event,
)


LOGGER = get_logger(__name__)

# Default speed multiplier
DEFAULT_SPEED = 1.0


@dataclass(frozen=True)
class TTSSynthesisResult:
    """Result of TTS synthesis operation."""

    success: bool
    audio_bytes: Optional[bytes] = None
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    speaker: str = "unknown"


@dataclass(frozen=True)
class VoiceDesignSynthesisResult:
    """Result of Voice Design synthesis operation."""

    success: bool
    audio_bytes: Optional[bytes] = None
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    voice_description: str = ""


@dataclass(frozen=True)
class VoiceCloneSynthesisResult:
    """Result of Voice Clone synthesis operation."""

    success: bool
    audio_bytes: Optional[bytes] = None
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    ref_text: str | None = None


class TTSSynthesizer:
    """
    TTS synthesizer that uses core application service.

    This class provides the bridge between Telegram requests and
    the core TTS infrastructure with full observability.
    """

    def __init__(
        self,
        application_service: TTSApplicationService,
        settings: TelegramSettings,
        logger: logging.Logger | None = None,
        metrics: TelegramMetrics | None = None,
    ):
        """
        Initialize TTS synthesizer.

        Args:
            application_service: Core TTS application service
            settings: Telegram settings including default speaker
            logger: Optional logger instance
            metrics: Optional metrics collector
        """
        self._app = application_service
        self._settings = settings
        self._logger = logger or LOGGER
        self.__metrics = metrics or METRICS

    async def synthesize(
        self,
        text: str,
        speaker: str | None = None,
        speed: float | None = None,
        correlation: Optional[TelegramCorrelationContext] = None,
    ) -> TTSSynthesisResult:
        """
        Synthesize speech from text using core application service.

        This method runs the synthesis in a thread pool to avoid blocking
        the async event loop, since TTS inference is CPU/GPU-bound.

        Args:
            text: Text to synthesize
            speaker: Speaker name (uses default if None)
            speed: Speed multiplier (uses 1.0 if None, valid range: 0.5-2.0)
            correlation: Optional correlation context for observability

        Returns:
            TTSSynthesisResult with audio bytes or error
        """
        timer = Timer()
        effective_speaker = speaker or self._settings.telegram_default_speaker
        effective_speed = speed if speed is not None else DEFAULT_SPEED

        # Create correlation context if not provided
        if correlation:
            correlation.set_operation("tts.synthesis")
            correlation.bind()

        try:
            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="telegram.tts.started",
                message="Starting TTS synthesis",
                text_length=len(text),
                speaker=effective_speaker,
                speed=effective_speed,
            )

            self._metrics.synthesis_started(effective_speaker)

            # Run blocking synthesis in thread pool
            result: GenerationResult = await asyncio.get_event_loop().run_in_executor(
                None,
                self._synthesize_sync,
                text,
                effective_speaker,
                effective_speed,
            )

            duration_ms = timer.elapsed_ms

            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="telegram.tts.completed",
                message="TTS synthesis completed",
                text_length=len(text),
                speaker=effective_speaker,
                speed=effective_speed,
                audio_size_bytes=len(result.audio.bytes_data),
                duration_ms=duration_ms,
                backend=result.backend,
            )

            self._metrics.synthesis_completed(effective_speaker, duration_ms)

            return TTSSynthesisResult(
                success=True,
                audio_bytes=result.audio.bytes_data,
                duration_ms=duration_ms,
                speaker=effective_speaker,
            )

        except CoreError as exc:
            duration_ms = timer.elapsed_ms

            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.tts.core_error",
                message=f"TTS synthesis failed: {exc}",
                text_length=len(text),
                speaker=effective_speaker,
                speed=effective_speed,
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
            )

            self._metrics.synthesis_failed(effective_speaker, type(exc).__name__)

            return TTSSynthesisResult(
                success=False,
                error_message=f"Synthesis error: {exc}",
                duration_ms=duration_ms,
                speaker=effective_speaker,
            )

        except Exception as exc:
            duration_ms = timer.elapsed_ms

            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.tts.failed",
                message=f"TTS synthesis failed unexpectedly: {exc}",
                text_length=len(text),
                speaker=effective_speaker,
                speed=effective_speed,
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
            )

            self._metrics.synthesis_failed(effective_speaker, type(exc).__name__)

            return TTSSynthesisResult(
                success=False,
                error_message="An unexpected error occurred during synthesis",
                duration_ms=duration_ms,
                speaker=effective_speaker,
            )

        finally:
            if correlation:
                correlation.unbind()

    def _synthesize_sync(
        self,
        text: str,
        speaker: str,
        speed: float,
    ) -> GenerationResult:
        """
        Synchronous synthesis method that calls core application service.

        This is called from the thread pool executor.

        Note: The speed parameter is passed through but actual speed control
        depends on the backend implementation. Some backends may not support
        speed modification.
        """
        command = CustomVoiceCommand(
            text=text,
            speaker=speaker,
            speed=speed,
            save_output=False,
        )
        return self._app.synthesize_custom(command)

    async def synthesize_design(
        self,
        voice_description: str,
        text: str,
        correlation: Optional[TelegramCorrelationContext] = None,
    ) -> VoiceDesignSynthesisResult:
        """
        Synthesize speech from text using a custom voice design.

        This method runs the synthesis in a thread pool to avoid blocking
        the async event loop, since TTS inference is CPU/GPU-bound.

        Args:
            voice_description: Natural language description of the voice
            text: Text to synthesize
            correlation: Optional correlation context for observability

        Returns:
            VoiceDesignSynthesisResult with audio bytes or error
        """
        timer = Timer()

        # Create correlation context if not provided
        if correlation:
            correlation.set_operation("voice_design.synthesis")
            correlation.bind()

        try:
            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="telegram.voice_design.started",
                message="Starting Voice Design synthesis",
                text_length=len(text),
                voice_description_length=len(voice_description),
            )

            self._metrics.synthesis_started("design")

            # Run blocking synthesis in thread pool
            result: GenerationResult = await asyncio.get_event_loop().run_in_executor(
                None,
                self._synthesize_design_sync,
                voice_description,
                text,
            )

            duration_ms = timer.elapsed_ms

            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="telegram.voice_design.completed",
                message="Voice Design synthesis completed",
                text_length=len(text),
                voice_description_length=len(voice_description),
                audio_size_bytes=len(result.audio.bytes_data),
                duration_ms=duration_ms,
                backend=result.backend,
            )

            self._metrics.synthesis_completed("design", duration_ms)

            return VoiceDesignSynthesisResult(
                success=True,
                audio_bytes=result.audio.bytes_data,
                duration_ms=duration_ms,
                voice_description=voice_description,
            )

        except CoreError as exc:
            duration_ms = timer.elapsed_ms

            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.voice_design.core_error",
                message=f"Voice Design synthesis failed: {exc}",
                text_length=len(text),
                voice_description_length=len(voice_description),
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
            )

            self._metrics.synthesis_failed("design", type(exc).__name__)

            return VoiceDesignSynthesisResult(
                success=False,
                error_message=f"Synthesis error: {exc}",
                duration_ms=duration_ms,
                voice_description=voice_description,
            )

        except Exception as exc:
            duration_ms = timer.elapsed_ms

            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.voice_design.failed",
                message=f"Voice Design synthesis failed unexpectedly: {exc}",
                text_length=len(text),
                voice_description_length=len(voice_description),
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
            )

            self._metrics.synthesis_failed("design", type(exc).__name__)

            return VoiceDesignSynthesisResult(
                success=False,
                error_message="An unexpected error occurred during synthesis",
                duration_ms=duration_ms,
                voice_description=voice_description,
            )

        finally:
            if correlation:
                correlation.unbind()

    def _synthesize_design_sync(
        self,
        voice_description: str,
        text: str,
    ) -> GenerationResult:
        """
        Synchronous synthesis method for voice design that calls core application service.

        This is called from the thread pool executor.
        """
        command = VoiceDesignCommand(
            text=text,
            voice_description=voice_description,
            save_output=False,
        )
        return self._app.synthesize_design(command)

    async def synthesize_clone(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str | None = None,
        correlation: Optional[TelegramCorrelationContext] = None,
    ) -> VoiceCloneSynthesisResult:
        """
        Synthesize speech from text using voice cloning.

        This method runs the synthesis in a thread pool to avoid blocking
        the async event loop, since TTS inference is CPU/GPU-bound.

        Args:
            text: Text to synthesize
            ref_audio_path: Path to reference audio file
            ref_text: Optional reference text transcript
            correlation: Optional correlation context for observability

        Returns:
            VoiceCloneSynthesisResult with audio bytes or error
        """
        timer = Timer()

        # Create correlation context if not provided
        if correlation:
            correlation.set_operation("voice_clone.synthesis")
            correlation.bind()

        try:
            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="telegram.voice_clone.started",
                message="Starting Voice Clone synthesis",
                text_length=len(text),
                ref_audio_path=ref_audio_path,
                ref_text_provided=ref_text is not None,
            )

            self._metrics.synthesis_started("clone")

            # Run blocking synthesis in thread pool
            result: GenerationResult = await asyncio.get_event_loop().run_in_executor(
                None,
                self._synthesize_clone_sync,
                ref_audio_path,
                text,
                ref_text,
            )

            duration_ms = timer.elapsed_ms

            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="telegram.voice_clone.completed",
                message="Voice Clone synthesis completed",
                text_length=len(text),
                audio_size_bytes=len(result.audio.bytes_data),
                duration_ms=duration_ms,
                backend=result.backend,
            )

            self._metrics.synthesis_completed("clone", duration_ms)

            return VoiceCloneSynthesisResult(
                success=True,
                audio_bytes=result.audio.bytes_data,
                duration_ms=duration_ms,
                ref_text=ref_text,
            )

        except CoreError as exc:
            duration_ms = timer.elapsed_ms

            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.voice_clone.core_error",
                message=f"Voice Clone synthesis failed: {exc}",
                text_length=len(text),
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
            )

            self._metrics.synthesis_failed("clone", type(exc).__name__)

            return VoiceCloneSynthesisResult(
                success=False,
                error_message=f"Synthesis error: {exc}",
                duration_ms=duration_ms,
                ref_text=ref_text,
            )

        except Exception as exc:
            duration_ms = timer.elapsed_ms

            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.voice_clone.failed",
                message=f"Voice Clone synthesis failed unexpectedly: {exc}",
                text_length=len(text),
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
            )

            self._metrics.synthesis_failed("clone", type(exc).__name__)

            return VoiceCloneSynthesisResult(
                success=False,
                error_message="An unexpected error occurred during synthesis",
                duration_ms=duration_ms,
                ref_text=ref_text,
            )

        finally:
            if correlation:
                correlation.unbind()

    def _synthesize_clone_sync(
        self,
        ref_audio_path: str,
        text: str,
        ref_text: str | None,
    ) -> GenerationResult:
        """
        Synchronous synthesis method for voice cloning that calls core application service.

        This is called from the thread pool executor.
        """
        from pathlib import Path as PathLib

        command = VoiceCloneCommand(
            text=text,
            ref_audio_path=PathLib(ref_audio_path),
            ref_text=ref_text,
            save_output=False,
        )
        return self._app.synthesize_clone(command)

    @property
    def _metrics(self):
        """Access metrics singleton."""
        return self.__metrics


if TYPE_CHECKING:
    from telegram_bot.config import TelegramSettings
    from telegram_bot.observability import TelegramMetrics
