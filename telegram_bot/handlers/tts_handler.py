# FILE: telegram_bot/handlers/tts_handler.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Implement /tts, /design, and /clone command handlers.
#   SCOPE: TTS command parsing, validation, job submission, result delivery
#   DEPENDS: M-APPLICATION, M-CONTRACTS, M-ERRORS
#   LINKS: M-TELEGRAM
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LOGGER - Module logger for Telegram synthesis handler events
#   DEFAULT_SPEED - Default speed multiplier for Telegram TTS
#   TTSSynthesisResult - Outcome payload for Telegram custom-voice synthesis
#   VoiceDesignSynthesisResult - Outcome payload for Telegram voice design synthesis
#   VoiceCloneSynthesisResult - Outcome payload for Telegram voice clone synthesis
#   TTSSynthesizer - Telegram adapter for shared TTS application service
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, function contracts, semantic blocks, and migrated log events to block-reference format]
# END_CHANGE_SUMMARY

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


# START_CONTRACT: TTSSynthesisResult
#   PURPOSE: Describe the outcome of a Telegram custom-voice synthesis request.
#   INPUTS: { success: bool - synthesis result flag, audio_bytes: Optional[bytes] - synthesized audio payload, error_message: Optional[str] - failure detail, duration_ms: float - elapsed time, speaker: str - resolved speaker name, language: str - requested language code }
#   OUTPUTS: { TTSSynthesisResult - immutable TTS synthesis outcome }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: TTSSynthesisResult
@dataclass(frozen=True)
class TTSSynthesisResult:
    """Result of TTS synthesis operation."""

    success: bool
    audio_bytes: Optional[bytes] = None
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    speaker: str = "unknown"
    language: str = "auto"


# START_CONTRACT: VoiceDesignSynthesisResult
#   PURPOSE: Describe the outcome of a Telegram voice-design synthesis request.
#   INPUTS: { success: bool - synthesis result flag, audio_bytes: Optional[bytes] - synthesized audio payload, error_message: Optional[str] - failure detail, duration_ms: float - elapsed time, voice_description: str - supplied voice description, language: str - requested language code }
#   OUTPUTS: { VoiceDesignSynthesisResult - immutable design synthesis outcome }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: VoiceDesignSynthesisResult
@dataclass(frozen=True)
class VoiceDesignSynthesisResult:
    """Result of Voice Design synthesis operation."""

    success: bool
    audio_bytes: Optional[bytes] = None
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    voice_description: str = ""
    language: str = "auto"


# START_CONTRACT: VoiceCloneSynthesisResult
#   PURPOSE: Describe the outcome of a Telegram voice-clone synthesis request.
#   INPUTS: { success: bool - synthesis result flag, audio_bytes: Optional[bytes] - synthesized audio payload, error_message: Optional[str] - failure detail, duration_ms: float - elapsed time, ref_text: str | None - optional reference transcript, language: str - requested language code }
#   OUTPUTS: { VoiceCloneSynthesisResult - immutable clone synthesis outcome }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: VoiceCloneSynthesisResult
@dataclass(frozen=True)
class VoiceCloneSynthesisResult:
    """Result of Voice Clone synthesis operation."""

    success: bool
    audio_bytes: Optional[bytes] = None
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    ref_text: str | None = None
    language: str = "auto"


# START_CONTRACT: TTSSynthesizer
#   PURPOSE: Bridge Telegram commands to the shared TTS application service with async orchestration.
#   INPUTS: { application_service: TTSApplicationService - core synthesis service, settings: TelegramSettings - Telegram runtime settings, logger: logging.Logger | None - optional logger, metrics: TelegramMetrics | None - optional metrics collector }
#   OUTPUTS: { TTSSynthesizer - Telegram synthesis adapter }
#   SIDE_EFFECTS: Executes synthesis work in a thread pool and emits logs and metrics.
#   LINKS: M-TELEGRAM
# END_CONTRACT: TTSSynthesizer
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

    # START_CONTRACT: synthesize
    #   PURPOSE: Synthesize custom-voice speech for Telegram from text input.
    #   INPUTS: { text: str - synthesis text, speaker: str | None - optional speaker override, speed: float | None - optional speed override, language: str - requested language code, correlation: Optional[TelegramCorrelationContext] - optional observability context }
    #   OUTPUTS: { TTSSynthesisResult - custom-voice synthesis outcome }
    #   SIDE_EFFECTS: Runs core synthesis work in an executor and emits metrics and logs.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: synthesize
    async def synthesize(
        self,
        text: str,
        speaker: str | None = None,
        speed: float | None = None,
        language: str = "auto",
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
        # START_BLOCK_PREPARE_TTS_REQUEST
        timer = Timer()
        effective_speaker = speaker or self._settings.telegram_default_speaker
        effective_speed = speed if speed is not None else DEFAULT_SPEED
        # END_BLOCK_PREPARE_TTS_REQUEST

        # START_BLOCK_BIND_TTS_CORRELATION
        # Create correlation context if not provided
        if correlation:
            correlation.set_operation("tts.synthesis")
            correlation.bind()
        # END_BLOCK_BIND_TTS_CORRELATION

        try:
            # START_BLOCK_EXECUTE_TTS_SYNTHESIS
            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="[TTSHandler][synthesize][BLOCK_EXECUTE_TTS_SYNTHESIS]",
                message="Starting TTS synthesis",
                text_length=len(text),
                speaker=effective_speaker,
                speed=effective_speed,
                language=language,
            )

            self._metrics.synthesis_started(effective_speaker)

            # Run blocking synthesis in thread pool
            result: GenerationResult = await asyncio.get_event_loop().run_in_executor(
                None,
                self._synthesize_sync,
                text,
                effective_speaker,
                effective_speed,
                language,
            )

            duration_ms = timer.elapsed_ms

            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="[TTSHandler][synthesize][BLOCK_EXECUTE_TTS_SYNTHESIS]",
                message="TTS synthesis completed",
                text_length=len(text),
                speaker=effective_speaker,
                speed=effective_speed,
                audio_size_bytes=len(result.audio.bytes_data),
                duration_ms=duration_ms,
                backend=result.backend,
                language=language,
            )

            self._metrics.synthesis_completed(effective_speaker, duration_ms)

            return TTSSynthesisResult(
                success=True,
                audio_bytes=result.audio.bytes_data,
                duration_ms=duration_ms,
                speaker=effective_speaker,
                language=language,
            )
            # END_BLOCK_EXECUTE_TTS_SYNTHESIS

        except CoreError as exc:
            # START_BLOCK_HANDLE_TTS_CORE_ERROR
            duration_ms = timer.elapsed_ms

            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="[TTSHandler][synthesize][BLOCK_HANDLE_TTS_CORE_ERROR]",
                message=f"TTS synthesis failed: {exc}",
                text_length=len(text),
                speaker=effective_speaker,
                speed=effective_speed,
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
                language=language,
            )

            self._metrics.synthesis_failed(effective_speaker, type(exc).__name__)

            return TTSSynthesisResult(
                success=False,
                error_message=f"Synthesis error: {exc}",
                duration_ms=duration_ms,
                speaker=effective_speaker,
                language=language,
            )
            # END_BLOCK_HANDLE_TTS_CORE_ERROR

        except Exception as exc:
            # START_BLOCK_HANDLE_TTS_UNEXPECTED_ERROR
            duration_ms = timer.elapsed_ms

            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="[TTSHandler][synthesize][BLOCK_HANDLE_TTS_UNEXPECTED_ERROR]",
                message=f"TTS synthesis failed unexpectedly: {exc}",
                text_length=len(text),
                speaker=effective_speaker,
                speed=effective_speed,
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
                language=language,
            )

            self._metrics.synthesis_failed(effective_speaker, type(exc).__name__)

            return TTSSynthesisResult(
                success=False,
                error_message="An unexpected error occurred during synthesis",
                duration_ms=duration_ms,
                speaker=effective_speaker,
                language=language,
            )
            # END_BLOCK_HANDLE_TTS_UNEXPECTED_ERROR

        finally:
            # START_BLOCK_UNBIND_TTS_CORRELATION
            if correlation:
                correlation.unbind()
            # END_BLOCK_UNBIND_TTS_CORRELATION

    def _synthesize_sync(
        self,
        text: str,
        speaker: str,
        speed: float,
        language: str,
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
            language=language,
            save_output=False,
        )
        return self._app.synthesize_custom(command)

    # START_CONTRACT: synthesize_design
    #   PURPOSE: Synthesize Telegram speech from a natural-language voice description.
    #   INPUTS: { voice_description: str - voice design prompt, text: str - synthesis text, language: str - requested language code, correlation: Optional[TelegramCorrelationContext] - optional observability context }
    #   OUTPUTS: { VoiceDesignSynthesisResult - voice-design synthesis outcome }
    #   SIDE_EFFECTS: Runs core synthesis work in an executor and emits metrics and logs.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: synthesize_design
    async def synthesize_design(
        self,
        voice_description: str,
        text: str,
        language: str = "auto",
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
        # START_BLOCK_PREPARE_DESIGN_REQUEST
        timer = Timer()
        # END_BLOCK_PREPARE_DESIGN_REQUEST

        # START_BLOCK_BIND_DESIGN_CORRELATION
        # Create correlation context if not provided
        if correlation:
            correlation.set_operation("voice_design.synthesis")
            correlation.bind()
        # END_BLOCK_BIND_DESIGN_CORRELATION

        try:
            # START_BLOCK_EXECUTE_DESIGN_SYNTHESIS
            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="[TTSHandler][synthesize_design][BLOCK_EXECUTE_DESIGN_SYNTHESIS]",
                message="Starting Voice Design synthesis",
                text_length=len(text),
                voice_description_length=len(voice_description),
                language=language,
            )

            self._metrics.synthesis_started("design")

            # Run blocking synthesis in thread pool
            result: GenerationResult = await asyncio.get_event_loop().run_in_executor(
                None,
                self._synthesize_design_sync,
                voice_description,
                text,
                language,
            )

            duration_ms = timer.elapsed_ms

            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="[TTSHandler][synthesize_design][BLOCK_EXECUTE_DESIGN_SYNTHESIS]",
                message="Voice Design synthesis completed",
                text_length=len(text),
                voice_description_length=len(voice_description),
                audio_size_bytes=len(result.audio.bytes_data),
                duration_ms=duration_ms,
                backend=result.backend,
                language=language,
            )

            self._metrics.synthesis_completed("design", duration_ms)

            return VoiceDesignSynthesisResult(
                success=True,
                audio_bytes=result.audio.bytes_data,
                duration_ms=duration_ms,
                voice_description=voice_description,
                language=language,
            )
            # END_BLOCK_EXECUTE_DESIGN_SYNTHESIS

        except CoreError as exc:
            # START_BLOCK_HANDLE_DESIGN_CORE_ERROR
            duration_ms = timer.elapsed_ms

            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="[TTSHandler][synthesize_design][BLOCK_HANDLE_DESIGN_CORE_ERROR]",
                message=f"Voice Design synthesis failed: {exc}",
                text_length=len(text),
                voice_description_length=len(voice_description),
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
                language=language,
            )

            self._metrics.synthesis_failed("design", type(exc).__name__)

            return VoiceDesignSynthesisResult(
                success=False,
                error_message=f"Synthesis error: {exc}",
                duration_ms=duration_ms,
                voice_description=voice_description,
                language=language,
            )
            # END_BLOCK_HANDLE_DESIGN_CORE_ERROR

        except Exception as exc:
            # START_BLOCK_HANDLE_DESIGN_UNEXPECTED_ERROR
            duration_ms = timer.elapsed_ms

            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="[TTSHandler][synthesize_design][BLOCK_HANDLE_DESIGN_UNEXPECTED_ERROR]",
                message=f"Voice Design synthesis failed unexpectedly: {exc}",
                text_length=len(text),
                voice_description_length=len(voice_description),
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
                language=language,
            )

            self._metrics.synthesis_failed("design", type(exc).__name__)

            return VoiceDesignSynthesisResult(
                success=False,
                error_message="An unexpected error occurred during synthesis",
                duration_ms=duration_ms,
                voice_description=voice_description,
                language=language,
            )
            # END_BLOCK_HANDLE_DESIGN_UNEXPECTED_ERROR

        finally:
            # START_BLOCK_UNBIND_DESIGN_CORRELATION
            if correlation:
                correlation.unbind()
            # END_BLOCK_UNBIND_DESIGN_CORRELATION

    def _synthesize_design_sync(
        self,
        voice_description: str,
        text: str,
        language: str,
    ) -> GenerationResult:
        """
        Synchronous synthesis method for voice design that calls core application service.

        This is called from the thread pool executor.
        """
        command = VoiceDesignCommand(
            text=text,
            voice_description=voice_description,
            language=language,
            save_output=False,
        )
        return self._app.synthesize_design(command)

    # START_CONTRACT: synthesize_clone
    #   PURPOSE: Synthesize Telegram speech using cloned voice reference audio.
    #   INPUTS: { text: str - synthesis text, ref_audio_path: str - path to staged reference audio, ref_text: str | None - optional reference transcript, language: str - requested language code, correlation: Optional[TelegramCorrelationContext] - optional observability context }
    #   OUTPUTS: { VoiceCloneSynthesisResult - voice-clone synthesis outcome }
    #   SIDE_EFFECTS: Runs core synthesis work in an executor and emits metrics and logs.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: synthesize_clone
    async def synthesize_clone(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str | None = None,
        language: str = "auto",
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
        # START_BLOCK_PREPARE_CLONE_REQUEST
        timer = Timer()
        # END_BLOCK_PREPARE_CLONE_REQUEST

        # START_BLOCK_BIND_CLONE_CORRELATION
        # Create correlation context if not provided
        if correlation:
            correlation.set_operation("voice_clone.synthesis")
            correlation.bind()
        # END_BLOCK_BIND_CLONE_CORRELATION

        try:
            # START_BLOCK_EXECUTE_CLONE_SYNTHESIS
            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="[TTSHandler][synthesize_clone][BLOCK_EXECUTE_CLONE_SYNTHESIS]",
                message="Starting Voice Clone synthesis",
                text_length=len(text),
                ref_audio_path=ref_audio_path,
                ref_text_provided=ref_text is not None,
                language=language,
            )

            self._metrics.synthesis_started("clone")

            # Run blocking synthesis in thread pool
            result: GenerationResult = await asyncio.get_event_loop().run_in_executor(
                None,
                self._synthesize_clone_sync,
                ref_audio_path,
                text,
                ref_text,
                language,
            )

            duration_ms = timer.elapsed_ms

            log_telegram_event(
                self._logger,
                level=logging.INFO,
                event="[TTSHandler][synthesize_clone][BLOCK_EXECUTE_CLONE_SYNTHESIS]",
                message="Voice Clone synthesis completed",
                text_length=len(text),
                audio_size_bytes=len(result.audio.bytes_data),
                duration_ms=duration_ms,
                backend=result.backend,
                language=language,
            )

            self._metrics.synthesis_completed("clone", duration_ms)

            return VoiceCloneSynthesisResult(
                success=True,
                audio_bytes=result.audio.bytes_data,
                duration_ms=duration_ms,
                ref_text=ref_text,
                language=language,
            )
            # END_BLOCK_EXECUTE_CLONE_SYNTHESIS

        except CoreError as exc:
            # START_BLOCK_HANDLE_CLONE_CORE_ERROR
            duration_ms = timer.elapsed_ms

            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="[TTSHandler][synthesize_clone][BLOCK_HANDLE_CLONE_CORE_ERROR]",
                message=f"Voice Clone synthesis failed: {exc}",
                text_length=len(text),
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
                language=language,
            )

            self._metrics.synthesis_failed("clone", type(exc).__name__)

            return VoiceCloneSynthesisResult(
                success=False,
                error_message=f"Synthesis error: {exc}",
                duration_ms=duration_ms,
                ref_text=ref_text,
                language=language,
            )
            # END_BLOCK_HANDLE_CLONE_CORE_ERROR

        except Exception as exc:
            # START_BLOCK_HANDLE_CLONE_UNEXPECTED_ERROR
            duration_ms = timer.elapsed_ms

            log_telegram_event(
                self._logger,
                level=logging.ERROR,
                event="[TTSHandler][synthesize_clone][BLOCK_HANDLE_CLONE_UNEXPECTED_ERROR]",
                message=f"Voice Clone synthesis failed unexpectedly: {exc}",
                text_length=len(text),
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
                language=language,
            )

            self._metrics.synthesis_failed("clone", type(exc).__name__)

            return VoiceCloneSynthesisResult(
                success=False,
                error_message="An unexpected error occurred during synthesis",
                duration_ms=duration_ms,
                ref_text=ref_text,
                language=language,
            )
            # END_BLOCK_HANDLE_CLONE_UNEXPECTED_ERROR

        finally:
            # START_BLOCK_UNBIND_CLONE_CORRELATION
            if correlation:
                correlation.unbind()
            # END_BLOCK_UNBIND_CLONE_CORRELATION

    def _synthesize_clone_sync(
        self,
        ref_audio_path: str,
        text: str,
        ref_text: str | None,
        language: str,
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
            language=language,
            save_output=False,
        )
        return self._app.synthesize_clone(command)

    # START_CONTRACT: _metrics
    #   PURPOSE: Expose the metrics collector used by the Telegram synthesis adapter.
    #   INPUTS: {}
    #   OUTPUTS: { TelegramMetrics - Telegram metrics collector }
    #   SIDE_EFFECTS: none
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: _metrics
    @property
    def _metrics(self):
        """Access metrics singleton."""
        return self.__metrics


if TYPE_CHECKING:
    from telegram_bot.config import TelegramSettings
    from telegram_bot.observability import TelegramMetrics

__all__ = [
    "LOGGER",
    "DEFAULT_SPEED",
    "TTSSynthesisResult",
    "VoiceDesignSynthesisResult",
    "VoiceCloneSynthesisResult",
    "TTSSynthesizer",
]
