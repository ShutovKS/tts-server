# FILE: core/services/tts_service.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Coordinate inference for custom, design, and clone synthesis modes.
#   SCOPE: TTSService class with synthesize_custom/design/clone, generate_audio dispatcher
#   DEPENDS: M-MODEL-REGISTRY, M-CONFIG, M-ERRORS, M-OBSERVABILITY, M-INFRASTRUCTURE
#   LINKS: M-TTS-SERVICE
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LOGGER - Module logger for synthesis service events
#   SynthesisCoordinator - Internal coordinator over planning, family preparation, and guarded generation
#   TTSService - Public synthesis facade preserving transport-facing command methods
#   generate_audio - Dispatch family-prepared execution requests to the backend contract
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Tightened coordinator/runtime seams to depend on explicit registry and handle contracts instead of duck-typed fallback access]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from core.config import CoreSettings
from core.backends.base import ExecutionRequest
from core.contracts import RuntimeExecutionRegistry
from core.contracts.commands import (
    CustomVoiceCommand,
    VoiceCloneCommand,
    VoiceDesignCommand,
)
from core.contracts.results import GenerationResult
from core.contracts.synthesis import SynthesisRequest
from core.errors import AudioArtifactNotFoundError, TTSGenerationError
from core.infrastructure.audio_io import (
    convert_audio_to_wav_if_needed,
    persist_output,
    read_generated_wav,
    temporary_output_dir,
)
from core.infrastructure.concurrency import InferenceGuard
from core.model_families import (
    ModelFamilyAdapter,
    OmniVoiceFamilyAdapter,
    PiperFamilyAdapter,
    Qwen3FamilyAdapter,
)
from core.models.catalog import ModelSpec
from core.observability import Timer, get_logger, log_event, operation_scope
from core.planning import SynthesisPlanner


LOGGER = get_logger(__name__)


# START_CONTRACT: generate_audio
#   PURPOSE: Dispatch a family-prepared generation request to the backend execution contract.
#   INPUTS: { args: tuple[object, ...] - Positional passthrough arguments, kwargs: dict[str, Any] - Backend, handle, mode, text, output_path, and mode-specific generation fields }
#   OUTPUTS: { None - Invokes the backend execution contract and writes output artifacts }
#   SIDE_EFFECTS: Triggers backend inference and writes generated audio files into the provided output directory
#   LINKS: M-TTS-SERVICE
# END_CONTRACT: generate_audio
def generate_audio(*args, **kwargs):
    backend = kwargs.pop("backend")
    mode = kwargs.pop("mode")
    handle = kwargs.pop("handle")
    output_path = Path(kwargs.pop("output_path"))
    text = kwargs.pop("text")
    language = kwargs.pop("language")
    backend.execute(
        ExecutionRequest(
            handle=handle,
            text=text,
            output_dir=output_path,
            language=language,
            execution_mode=mode,
            generation_kwargs=dict(kwargs),
        )
    )


class SynthesisCoordinator:
    def __init__(
        self,
        registry: RuntimeExecutionRegistry,
        settings: CoreSettings,
        inference_guard: InferenceGuard,
        planner: SynthesisPlanner,
        family_adapters: dict[str, ModelFamilyAdapter],
    ):
        self.registry = registry
        self.settings = settings
        self.inference_guard = inference_guard
        self.planner = planner
        self._family_adapters = family_adapters

    def _selected_backend_key(self) -> str:
        return self.registry.backend.key

    @staticmethod
    def _handle_backend_key(handle) -> str:
        return handle.backend_key

    def synthesize_custom(self, command: CustomVoiceCommand) -> GenerationResult:
        plan = self.planner.plan_command(command)
        spec, handle = self.registry.get_model(
            model_name=plan.model_spec.model_id,
            mode=plan.execution_mode,
        )
        prepared = self._prepare_execution(plan)
        return self._run_generation(
            spec=spec,
            handle=handle,
            text=command.text,
            save_output=command.save_output,
            generation_kwargs=prepared,
        )

    def synthesize_design(self, command: VoiceDesignCommand) -> GenerationResult:
        plan = self.planner.plan_command(command)
        spec, handle = self.registry.get_model(
            model_name=plan.model_spec.model_id,
            mode=plan.execution_mode,
        )
        prepared = self._prepare_execution(plan)
        return self._run_generation(
            spec=spec,
            handle=handle,
            text=command.text,
            save_output=command.save_output,
            generation_kwargs=prepared,
        )

    def synthesize_clone(self, command: VoiceCloneCommand) -> GenerationResult:
        plan = self.planner.plan_command(command)
        spec, handle = self.registry.get_model(
            model_name=plan.model_spec.model_id,
            mode=plan.execution_mode,
        )

        with temporary_output_dir(prefix="qwen3_tts_clone_input_") as temp_dir:
            source_audio = temp_dir / command.ref_audio_path.name
            source_audio.write_bytes(command.ref_audio_path.read_bytes())
            wav_audio, converted = convert_audio_to_wav_if_needed(
                source_audio, self.settings
            )
            log_event(
                LOGGER,
                level=20,
                event="[TTSService][synthesize_clone][BLOCK_PREPARE_REFERENCE_AUDIO]",
                message="Reference audio prepared for clone synthesis",
                model=spec.api_name,
                mode=spec.mode,
                source_audio=str(source_audio),
                prepared_audio=str(wav_audio),
                converted=converted,
                backend=self._handle_backend_key(handle),
            )
            try:
                prepared_request = SynthesisRequest.from_command(
                    VoiceCloneCommand(
                        text=command.text,
                        model=command.model,
                        save_output=command.save_output,
                        language=command.language,
                        ref_audio_path=wav_audio,
                        ref_text=command.ref_text,
                    )
                )
                prepared_plan = replace(plan, request=prepared_request)
                prepared_generation = self._prepare_execution(prepared_plan)
                result = self._run_generation(
                    spec=spec,
                    handle=handle,
                    text=command.text,
                    save_output=command.save_output,
                    generation_kwargs=prepared_generation,
                )
                log_event(
                    LOGGER,
                    level=20,
                    event="[TTSService][synthesize_clone][BLOCK_EXECUTE_CLONE]",
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

    def _prepare_execution(self, plan) -> dict[str, Any]:
        adapter = self._family_adapters.get(plan.family_key)
        if adapter is None:
            raise TTSGenerationError(
                "No family adapter is registered for the execution plan",
                details={
                    "family": plan.family_key,
                    "model": plan.model_spec.api_name,
                    "capability": plan.request.capability,
                },
            )
        return dict(adapter.prepare_execution(plan).generation_kwargs)

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
        generation_kwargs = dict(generation_kwargs)
        language = generation_kwargs.pop("language", "auto")
        backend = self.registry.backend_for_spec(spec)
        self.inference_guard.acquire()
        log_event(
            LOGGER,
            level=20,
            event="[TTSService][_run_generation][BLOCK_ACQUIRE_INFERENCE]",
            message="Inference slot acquired",
            model=spec.api_name,
            mode=spec.mode,
            save_output=save_output,
            text_length=len(text),
            language=language,
            backend=backend.key,
        )
        try:
            with temporary_output_dir(prefix="qwen3_tts_output_") as output_dir:
                try:
                    generate_audio(
                        backend=backend,
                        handle=handle,
                        mode=spec.mode,
                        text=text,
                        language=language,
                        output_path=str(output_dir),
                        **generation_kwargs,
                    )
                    audio = read_generated_wav(output_dir)
                except AudioArtifactNotFoundError as exc:
                    log_event(
                        LOGGER,
                        level=40,
                        event="[TTSService][_run_generation][BLOCK_HANDLE_GENERATION_ERRORS]",
                        message="Generation finished without output artifact",
                        model=spec.api_name,
                        mode=spec.mode,
                        duration_ms=timer.elapsed_ms,
                        language=language,
                        error=str(exc),
                        backend=backend.key,
                    )
                    raise TTSGenerationError(
                        str(exc),
                        details={
                            "model": spec.api_name,
                            "mode": spec.mode,
                            "failure_kind": "missing_artifact",
                            "backend": backend.key,
                        },
                    ) from exc
                except TTSGenerationError as exc:
                    log_event(
                        LOGGER,
                        level=40,
                        event="[TTSService][_run_generation][BLOCK_HANDLE_GENERATION_ERRORS]",
                        message="Generation failed with controlled error",
                        model=spec.api_name,
                        mode=spec.mode,
                        language=language,
                        duration_ms=timer.elapsed_ms,
                        error=str(exc),
                        backend=backend.key,
                    )
                    raise
                except Exception as exc:  # pragma: no cover
                    log_event(
                        LOGGER,
                        level=40,
                        event="[TTSService][_run_generation][BLOCK_HANDLE_GENERATION_ERRORS]",
                        message="Generation failed with unexpected error",
                        model=spec.api_name,
                        mode=spec.mode,
                        language=language,
                        duration_ms=timer.elapsed_ms,
                        error=str(exc),
                        backend=backend.key,
                    )
                    raise TTSGenerationError(
                        str(exc),
                        details={
                            "model": spec.api_name,
                            "mode": spec.mode,
                            "backend": backend.key,
                        },
                    ) from exc

                saved_path = None
                if save_output:
                    saved_path = persist_output(
                        audio, spec.output_subfolder, text, self.settings
                    )

                result = GenerationResult(
                    audio=audio,
                    saved_path=saved_path,
                    model=spec.model_id,
                    mode=spec.mode,
                    backend=backend.key,
                )
                log_event(
                    LOGGER,
                    level=20,
                    event="[TTSService][_run_generation][BLOCK_PERSIST_OUTPUT]",
                    message="Generation completed successfully",
                    model=result.model,
                    mode=result.mode,
                    duration_ms=timer.elapsed_ms,
                    language=language,
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
                event="[TTSService][_run_generation][BLOCK_RELEASE_INFERENCE]",
                message="Inference slot released",
                model=spec.api_name,
                mode=spec.mode,
                duration_ms=timer.elapsed_ms,
                language=language,
                backend=backend.key,
            )


# START_CONTRACT: TTSService
#   PURPOSE: Coordinate model resolution, guarded inference execution, and output persistence for TTS requests.
#   INPUTS: { registry: ModelRegistry - Model registry used to resolve and load models, settings: CoreSettings - Shared runtime settings controlling audio handling and persistence, inference_guard: InferenceGuard | None - Optional shared inference concurrency guard }
#   OUTPUTS: { instance - TTS synthesis service for custom, design, and clone modes }
#   SIDE_EFFECTS: none
#   LINKS: M-TTS-SERVICE
# END_CONTRACT: TTSService
class TTSService:
    def __init__(
        self,
        registry: RuntimeExecutionRegistry,
        settings: CoreSettings,
        inference_guard: InferenceGuard | None = None,
    ):
        self.registry = registry
        self.settings = settings
        self.inference_guard = inference_guard or InferenceGuard()
        self.planner = SynthesisPlanner(registry, settings)
        self._family_adapters = {
            "qwen3_tts": Qwen3FamilyAdapter(),
            "omnivoice": OmniVoiceFamilyAdapter(),
            "piper": PiperFamilyAdapter(),
        }
        self.coordinator = SynthesisCoordinator(
            registry=registry,
            settings=settings,
            inference_guard=self.inference_guard,
            planner=self.planner,
            family_adapters=self._family_adapters,
        )

    def _selected_backend_key(self) -> str:
        return self.registry.backend.key

    # START_CONTRACT: synthesize_custom
    #   PURPOSE: Run a guarded custom-voice synthesis workflow from a validated command.
    #   INPUTS: { command: CustomVoiceCommand - Custom voice synthesis request }
    #   OUTPUTS: { GenerationResult - Generated audio result and persistence metadata }
    #   SIDE_EFFECTS: Loads model state, emits structured logs, performs inference, and may persist generated audio
    #   LINKS: M-TTS-SERVICE
    # END_CONTRACT: synthesize_custom
    def synthesize_custom(self, command: CustomVoiceCommand) -> GenerationResult:
        with operation_scope("core.tts_service.synthesize_custom"):
            log_event(
                LOGGER,
                level=20,
                event="[TTSService][synthesize_custom][SYNTHESIZE_CUSTOM]",
                message="Starting custom voice synthesis",
                model=command.model,
                mode="custom",
                save_output=command.save_output,
                text_length=len(command.text),
                language=command.language,
                backend=self._selected_backend_key(),
            )
            result = self.coordinator.synthesize_custom(command)
            log_event(
                LOGGER,
                level=20,
                event="[TTSService][synthesize_custom][SYNTHESIZE_CUSTOM]",
                message="Custom voice synthesis finished",
                model=result.model,
                mode=result.mode,
                saved_path=str(result.saved_path) if result.saved_path else None,
                backend=result.backend,
            )
            return result

    # START_CONTRACT: synthesize_design
    #   PURPOSE: Run a guarded voice-design synthesis workflow from a validated command.
    #   INPUTS: { command: VoiceDesignCommand - Voice design synthesis request }
    #   OUTPUTS: { GenerationResult - Generated audio result and persistence metadata }
    #   SIDE_EFFECTS: Loads model state, emits structured logs, performs inference, and may persist generated audio
    #   LINKS: M-TTS-SERVICE
    # END_CONTRACT: synthesize_design
    def synthesize_design(self, command: VoiceDesignCommand) -> GenerationResult:
        with operation_scope("core.tts_service.synthesize_design"):
            log_event(
                LOGGER,
                level=20,
                event="[TTSService][synthesize_design][SYNTHESIZE_DESIGN]",
                message="Starting voice design synthesis",
                model=command.model,
                mode="design",
                save_output=command.save_output,
                text_length=len(command.text),
                language=command.language,
                backend=self._selected_backend_key(),
            )
            result = self.coordinator.synthesize_design(command)
            log_event(
                LOGGER,
                level=20,
                event="[TTSService][synthesize_design][SYNTHESIZE_DESIGN]",
                message="Voice design synthesis finished",
                model=result.model,
                mode=result.mode,
                saved_path=str(result.saved_path) if result.saved_path else None,
                backend=result.backend,
            )
            return result

    # START_CONTRACT: synthesize_clone
    #   PURPOSE: Run a guarded voice-clone synthesis workflow including reference audio preparation.
    #   INPUTS: { command: VoiceCloneCommand - Voice clone synthesis request with reference audio metadata }
    #   OUTPUTS: { GenerationResult - Generated audio result and persistence metadata }
    #   SIDE_EFFECTS: Copies and may convert reference audio, loads model state, emits structured logs, performs inference, and may persist generated audio
    #   LINKS: M-TTS-SERVICE
    # END_CONTRACT: synthesize_clone
    def synthesize_clone(self, command: VoiceCloneCommand) -> GenerationResult:
        with operation_scope("core.tts_service.synthesize_clone"):
            # START_BLOCK_VALIDATE_CLONE_INPUT
            if command.ref_audio_path is None:
                raise TTSGenerationError(
                    "Reference audio is required for clone synthesis",
                    details={
                        "mode": "clone",
                        "reference_audio": None,
                        "backend": self._selected_backend_key(),
                    },
                )
            # END_BLOCK_VALIDATE_CLONE_INPUT

            log_event(
                LOGGER,
                level=20,
                event="[TTSService][synthesize_clone][SYNTHESIZE_CLONE]",
                message="Starting clone synthesis",
                model=command.model,
                mode="clone",
                save_output=command.save_output,
                text_length=len(command.text),
                language=command.language,
                ref_text_provided=bool(command.ref_text),
                ref_audio_path=str(command.ref_audio_path),
                backend=self._selected_backend_key(),
            )
            result = self.coordinator.synthesize_clone(command)
            return result


__all__ = [
    "LOGGER",
    "SynthesisCoordinator",
    "generate_audio",
    "TTSService",
]
