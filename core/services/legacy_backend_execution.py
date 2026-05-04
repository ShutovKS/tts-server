# FILE: core/services/legacy_backend_execution.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Execute the explicit legacy backend compatibility path outside SynthesisCoordinator.
#   SCOPE: scheduler-gated backend.execute dispatch, legacy output persistence, and controlled error mapping for unmigrated runtime lanes
#   DEPENDS: M-CONTRACTS, M-ERRORS, M-OBSERVABILITY, M-INFRASTRUCTURE, M-AUDIO-PERSISTENCE
#   LINKS: M-TTS-COORDINATOR, M-BACKENDS-V2
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LegacyBackendExecutionService - Thin compatibility adapter for backend.execute-based synthesis lanes.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Extracted the legacy backend execution lane from SynthesisCoordinator so engine-first runtime routing can fail closed for migrated families]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from core.backends.base import ExecutionRequest
from core.contracts.results import GenerationResult
from core.contracts.runtime import RuntimeExecutionRegistry
from core.engines.scheduler import EngineScheduler
from core.errors import AudioArtifactNotFoundError, TTSGenerationError
from core.models.catalog import ModelSpec
from core.observability import Timer, get_logger, log_event
from core.services.audio_persistence import AudioPersistenceService

LOGGER = get_logger(__name__)


# START_CONTRACT: LegacyBackendExecutionService
#   PURPOSE: Isolate the remaining backend.execute compatibility lane behind a dedicated service outside TTSService orchestration.
#   INPUTS: { registry: RuntimeExecutionRegistry - Runtime registry used to resolve the legacy backend for a spec, audio_persistence: AudioPersistenceService - Persistence helper for backend output directories }
#   OUTPUTS: { instance - Legacy compatibility executor }
#   SIDE_EFFECTS: none on construction
#   LINKS: M-LEGACY-BACKEND-EXECUTION, M-TTS-COORDINATOR
# END_CONTRACT: LegacyBackendExecutionService
class LegacyBackendExecutionService:
    def __init__(
        self,
        *,
        registry: RuntimeExecutionRegistry,
        audio_persistence: AudioPersistenceService,
    ) -> None:
        self._registry = registry
        self._audio_persistence = audio_persistence

    # START_CONTRACT: execute
    #   PURPOSE: Run one legacy backend.execute synthesis call through the shared scheduler gateway and materialize the resulting GenerationResult.
    #   INPUTS: { spec: ModelSpec - Model metadata, handle: Any - Loaded legacy backend handle, text: str - Source text, save_output: bool - Whether durable persistence is requested, generation_kwargs: dict[str, Any] - Legacy execution payload, scheduler_submit: Callable[..., GenerationResult] - Scheduler gateway callback, scheduler_engine_key: str - Scheduler key for the legacy lane }
    #   OUTPUTS: { GenerationResult - Materialized legacy execution result }
    #   SIDE_EFFECTS: Creates a temporary output directory, invokes backend.execute, reads/persists generated WAV output, emits structured logs, and may raise TTSGenerationError.
    #   LINKS: M-LEGACY-BACKEND-EXECUTION, M-TTS-COORDINATOR
    # END_CONTRACT: execute
    def execute(
        self,
        *,
        spec: ModelSpec,
        handle,
        text: str,
        save_output: bool,
        generation_kwargs: dict[str, Any],
        scheduler_submit: Callable[..., GenerationResult],
        scheduler_engine_key: str,
    ) -> GenerationResult:
        timer = Timer()
        generation_kwargs = dict(generation_kwargs)
        language = generation_kwargs.pop("language", "auto")
        backend = self._registry.legacy_backend_for_spec(spec)

        def execute_generation() -> GenerationResult:
            log_event(
                LOGGER,
                level=20,
                event="[LegacyBackendExecutionService][execute][BLOCK_ACQUIRE_INFERENCE]",
                message="Inference slot acquired for legacy backend execution",
                model=spec.api_name,
                mode=spec.mode,
                save_output=save_output,
                text_length=len(text),
                language=language,
                backend=backend.key,
            )
            try:
                from core.infrastructure.audio_io import temporary_output_dir

                with temporary_output_dir(prefix="legacy_tts_output_") as output_dir:
                    try:
                        backend.execute(
                            ExecutionRequest(
                                handle=handle,
                                text=text,
                                output_dir=Path(output_dir),
                                language=language,
                                execution_mode=spec.mode,
                                generation_kwargs=dict(generation_kwargs),
                            )
                        )
                        persisted = self._audio_persistence.persist_backend_output(
                            output_dir=Path(output_dir),
                            spec=spec,
                            text=text,
                            save_output=save_output,
                        )
                    except AudioArtifactNotFoundError as exc:
                        log_event(
                            LOGGER,
                            level=40,
                            event="[LegacyBackendExecutionService][execute][BLOCK_HANDLE_GENERATION_ERRORS]",
                            message="Legacy backend execution finished without output artifact",
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
                                "execution_lane": "legacy_backend",
                            },
                        ) from exc
                    except TTSGenerationError as exc:
                        log_event(
                            LOGGER,
                            level=40,
                            event="[LegacyBackendExecutionService][execute][BLOCK_HANDLE_GENERATION_ERRORS]",
                            message="Legacy backend execution failed with controlled error",
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
                            event="[LegacyBackendExecutionService][execute][BLOCK_HANDLE_GENERATION_ERRORS]",
                            message="Legacy backend execution failed with unexpected error",
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
                                "execution_lane": "legacy_backend",
                            },
                        ) from exc

                    result = GenerationResult(
                        audio=persisted.audio,
                        saved_path=persisted.saved_path,
                        model=spec.model_id,
                        mode=spec.mode,
                        backend=backend.key,
                    )
                    log_event(
                        LOGGER,
                        level=20,
                        event="[LegacyBackendExecutionService][execute][BLOCK_PERSIST_OUTPUT]",
                        message="Legacy backend execution completed successfully",
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
                log_event(
                    LOGGER,
                    level=20,
                    event="[LegacyBackendExecutionService][execute][BLOCK_RELEASE_INFERENCE]",
                    message="Inference slot released after legacy backend execution",
                    model=spec.api_name,
                    mode=spec.mode,
                    duration_ms=timer.elapsed_ms,
                    language=language,
                    backend=backend.key,
                )

        return scheduler_submit(
            spec=spec,
            backend_key=backend.key,
            call=execute_generation,
            engine_key=scheduler_engine_key,
        )


__all__ = ["LegacyBackendExecutionService"]
