from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import CoreSettings
from core.contracts.commands import CustomVoiceCommand, VoiceCloneCommand, VoiceDesignCommand
from core.contracts.results import GenerationResult
from core.errors import AudioArtifactNotFoundError, TTSGenerationError
from core.infrastructure.audio_io import (
    convert_audio_to_wav_if_needed,
    persist_output,
    read_generated_wav,
    temporary_output_dir,
)
from core.infrastructure.concurrency import InferenceGuard
from core.models.catalog import ModelSpec
from core.observability import Timer, get_logger, log_event, operation_scope
from core.services.model_registry import ModelRegistry


LOGGER = get_logger(__name__)


def generate_audio(*args, **kwargs):
    backend = kwargs.pop("backend")
    mode = kwargs.pop("mode")
    handle = kwargs.pop("handle")
    output_path = Path(kwargs.pop("output_path"))
    text = kwargs.pop("text")

    if mode == "custom":
        backend.synthesize_custom(
            handle,
            text=text,
            output_dir=output_path,
            speaker=kwargs.pop("voice"),
            instruct=kwargs.pop("instruct"),
            speed=kwargs.pop("speed"),
        )
        return

    if mode == "design":
        backend.synthesize_design(
            handle,
            text=text,
            output_dir=output_path,
            voice_description=kwargs.pop("instruct"),
        )
        return

    if mode == "clone":
        backend.synthesize_clone(
            handle,
            text=text,
            output_dir=output_path,
            ref_audio_path=Path(kwargs.pop("ref_audio")),
            ref_text=kwargs.pop("ref_text", None),
        )
        return

    raise TTSGenerationError(
        f"Unsupported generation mode: {mode}",
        details={"mode": mode, "backend": handle.backend_key},
    )


class TTSService:
    def __init__(self, registry: ModelRegistry, settings: CoreSettings, inference_guard: InferenceGuard | None = None):
        self.registry = registry
        self.settings = settings
        self.inference_guard = inference_guard or InferenceGuard()

    def _backend_key(self, fallback: str | None = None) -> str | None:
        backend = getattr(self.registry, "backend", None)
        if backend is not None:
            return getattr(backend, "key", fallback)
        return fallback

    @staticmethod
    def _handle_backend_key(handle, fallback: str | None = None) -> str | None:
        return getattr(handle, "backend_key", fallback)

    def synthesize_custom(self, command: CustomVoiceCommand) -> GenerationResult:
        with operation_scope("core.tts_service.synthesize_custom"):
            log_event(
                LOGGER,
                level=20,
                event="tts.custom.started",
                message="Starting custom voice synthesis",
                model=command.model,
                mode="custom",
                save_output=command.save_output,
                text_length=len(command.text),
                backend=self._backend_key(),
            )
            spec, handle = self.registry.get_model(model_name=command.model, mode="custom")
            result = self._run_generation(
                spec=spec,
                handle=handle,
                text=command.text,
                save_output=command.save_output,
                generation_kwargs={
                    "voice": command.speaker,
                    "instruct": command.instruct,
                    "speed": command.speed,
                },
            )
            log_event(
                LOGGER,
                level=20,
                event="tts.custom.completed",
                message="Custom voice synthesis finished",
                model=result.model,
                mode=result.mode,
                saved_path=str(result.saved_path) if result.saved_path else None,
                backend=result.backend,
            )
            return result

    def synthesize_design(self, command: VoiceDesignCommand) -> GenerationResult:
        with operation_scope("core.tts_service.synthesize_design"):
            log_event(
                LOGGER,
                level=20,
                event="tts.design.started",
                message="Starting voice design synthesis",
                model=command.model,
                mode="design",
                save_output=command.save_output,
                text_length=len(command.text),
                backend=self._backend_key(),
            )
            spec, handle = self.registry.get_model(model_name=command.model, mode="design")
            result = self._run_generation(
                spec=spec,
                handle=handle,
                text=command.text,
                save_output=command.save_output,
                generation_kwargs={
                    "instruct": command.voice_description,
                },
            )
            log_event(
                LOGGER,
                level=20,
                event="tts.design.completed",
                message="Voice design synthesis finished",
                model=result.model,
                mode=result.mode,
                saved_path=str(result.saved_path) if result.saved_path else None,
                backend=result.backend,
            )
            return result

    def synthesize_clone(self, command: VoiceCloneCommand) -> GenerationResult:
        with operation_scope("core.tts_service.synthesize_clone"):
            if command.ref_audio_path is None:
                raise TTSGenerationError(
                    "Reference audio is required for clone synthesis",
                    details={"mode": "clone", "reference_audio": None, "backend": self._backend_key()},
                )

            log_event(
                LOGGER,
                level=20,
                event="tts.clone.started",
                message="Starting clone synthesis",
                model=command.model,
                mode="clone",
                save_output=command.save_output,
                text_length=len(command.text),
                ref_text_provided=bool(command.ref_text),
                ref_audio_path=str(command.ref_audio_path),
                backend=self._backend_key(),
            )
            spec, handle = self.registry.get_model(model_name=command.model, mode="clone")

            with temporary_output_dir(prefix="qwen3_tts_clone_input_") as temp_dir:
                source_audio = temp_dir / command.ref_audio_path.name
                source_audio.write_bytes(command.ref_audio_path.read_bytes())
                wav_audio, converted = convert_audio_to_wav_if_needed(source_audio, self.settings)
                log_event(
                    LOGGER,
                    level=20,
                    event="tts.clone.reference_audio_prepared",
                    message="Reference audio prepared for clone synthesis",
                    model=spec.api_name,
                    mode=spec.mode,
                    source_audio=str(source_audio),
                    prepared_audio=str(wav_audio),
                    converted=converted,
                    backend=self._handle_backend_key(handle, self._backend_key()),
                )
                try:
                    result = self._run_generation(
                        spec=spec,
                        handle=handle,
                        text=command.text,
                        save_output=command.save_output,
                        generation_kwargs={
                            "ref_audio": str(wav_audio),
                            "ref_text": command.ref_text or ".",
                        },
                    )
                    log_event(
                        LOGGER,
                        level=20,
                        event="tts.clone.completed",
                        message="Clone synthesis finished",
                        model=result.model,
                        mode=result.mode,
                        saved_path=str(result.saved_path) if result.saved_path else None,
                        backend=result.backend,
                    )
                    return result
                finally:
                    if converted and wav_audio.exists():
                        wav_audio.unlink(missing_ok=True)

    def _run_generation(
        self,
        *,
        spec: ModelSpec,
        handle,
        text: str,
        save_output: bool,
        generation_kwargs: dict[str, Any],
    ) -> GenerationResult:
        timer = Timer()
        self.inference_guard.acquire()
        log_event(
            LOGGER,
            level=20,
            event="tts.generation.acquired",
            message="Inference slot acquired",
            model=spec.api_name,
            mode=spec.mode,
            save_output=save_output,
            text_length=len(text),
            backend=self._handle_backend_key(handle, self._backend_key()),
        )
        try:
            with temporary_output_dir(prefix="qwen3_tts_output_") as output_dir:
                try:
                    generate_audio(
                        backend=getattr(self.registry, "backend", None),
                        handle=handle,
                        mode=spec.mode,
                        text=text,
                        output_path=str(output_dir),
                        **generation_kwargs,
                    )
                    audio = read_generated_wav(output_dir)
                except AudioArtifactNotFoundError as exc:
                    log_event(
                        LOGGER,
                        level=40,
                        event="tts.generation.artifact_missing",
                        message="Generation finished without output artifact",
                        model=spec.api_name,
                        mode=spec.mode,
                        duration_ms=timer.elapsed_ms,
                        error=str(exc),
                        backend=self._handle_backend_key(handle, self._backend_key()),
                    )
                    raise TTSGenerationError(
                        str(exc),
                        details={"model": spec.api_name, "mode": spec.mode, "failure_kind": "missing_artifact", "backend": self._handle_backend_key(handle, self._backend_key())},
                    ) from exc
                except TTSGenerationError as exc:
                    log_event(
                        LOGGER,
                        level=40,
                        event="tts.generation.failed",
                        message="Generation failed with controlled error",
                        model=spec.api_name,
                        mode=spec.mode,
                        duration_ms=timer.elapsed_ms,
                        error=str(exc),
                        backend=self._handle_backend_key(handle, self._backend_key()),
                    )
                    raise
                except Exception as exc:  # pragma: no cover
                    log_event(
                        LOGGER,
                        level=40,
                        event="tts.generation.failed",
                        message="Generation failed with unexpected error",
                        model=spec.api_name,
                        mode=spec.mode,
                        duration_ms=timer.elapsed_ms,
                        error=str(exc),
                        backend=self._handle_backend_key(handle, self._backend_key()),
                    )
                    raise TTSGenerationError(
                        str(exc),
                        details={"model": spec.api_name, "mode": spec.mode, "backend": self._handle_backend_key(handle, self._backend_key())},
                    ) from exc

                saved_path = None
                if save_output:
                    saved_path = persist_output(audio, spec.output_subfolder, text, self.settings)

                result = GenerationResult(
                    audio=audio,
                    saved_path=saved_path,
                    model=spec.api_name,
                    mode=spec.mode,
                    backend=self._handle_backend_key(handle, self._backend_key()) or "unknown",
                )
                log_event(
                    LOGGER,
                    level=20,
                    event="tts.generation.completed",
                    message="Generation completed successfully",
                    model=result.model,
                    mode=result.mode,
                    duration_ms=timer.elapsed_ms,
                    saved_path=str(result.saved_path) if result.saved_path else None,
                    audio_path=str(result.audio.path),
                    backend=result.backend,
                )
                return result
        finally:
            self.inference_guard.release()
            log_event(
                LOGGER,
                level=20,
                event="tts.generation.released",
                message="Inference slot released",
                model=spec.api_name,
                mode=spec.mode,
                duration_ms=timer.elapsed_ms,
                backend=self._handle_backend_key(handle, self._backend_key()),
            )
