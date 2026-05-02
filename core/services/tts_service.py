# FILE: core/services/tts_service.py
# VERSION: 1.5.0
# START_MODULE_CONTRACT
#   PURPOSE: Coordinate inference for custom, design, and clone synthesis modes via the SynthesisRouter unified seam, while preserving the transport-facing TTSService.synthesize_X(...) facade for backwards compatibility and allowing a guarded Piper engine route.
#   SCOPE: TTSService class with synthesize_custom/design/clone delegating through SynthesisRouter, SynthesisCoordinator (kept as the per-mode worker; now invokes backend.execute directly by default and can opt Piper custom synthesis into a TTSEngine registry path).
#   DEPENDS: M-MODEL-REGISTRY, M-CONFIG, M-DISCOVERY, M-ERRORS, M-OBSERVABILITY, M-INFRASTRUCTURE, M-MODEL-FAMILY, M-ENGINE-REGISTRY, M-ENGINE-CONTRACTS
#   LINKS: M-TTS-SERVICE
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LOGGER - Module logger for synthesis service events
#   SynthesisCoordinator - Internal coordinator over planning, family preparation, and guarded generation; the per-mode worker invoked by SynthesisRouter; invokes backend.execute(ExecutionRequest(...)) by default and can route Piper custom synthesis through EngineRegistry when explicitly enabled.
#   _build_family_adapter_map - Instantiate a deterministic family-keyed adapter map from discovery results while rejecting duplicate keys
#   _build_engine_registry - Build the explicit process-local engine registry used by the guarded Piper engine seam
#   TTSService - Public synthesis facade preserving transport-facing command methods; delegates each call through SynthesisRouter to keep the public pipeline at three layers (TTSService -> SynthesisRouter -> backend)
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.5.0 - Task 10: added an explicit Piper-only TTSEngine route behind a runtime flag while keeping the legacy backend execution path as the default fallback]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from core.infrastructure.concurrency import InferenceGuard
    from core.services.result_cache import ResultCache

from core.backends.base import ExecutionRequest
from core.config import CoreSettings
from core.contracts import RuntimeExecutionRegistry
from core.contracts.results import AudioResult, GenerationResult
from core.contracts.commands import (
    CustomVoiceCommand,
    VoiceCloneCommand,
    VoiceDesignCommand,
)
from core.contracts.synthesis import SynthesisRequest
from core.discovery import discover_family_adapter_classes
from core.engines import EngineRegistry, EngineRegistryError, SynthesisJob, load_engine_registry
from core.engines.piper import PiperOnnxEngine
from core.errors import AudioArtifactNotFoundError, TTSGenerationError
from core.model_families import ModelFamilyAdapter
from core.models.catalog import ModelSpec
from core.observability import Timer, get_logger, log_event, operation_scope
from core.planning import SynthesisPlanner

LOGGER = get_logger(__name__)


# START_CONTRACT: _build_family_adapter_map
#   PURPOSE: Build the runtime family-adapter registry from discovered adapter classes while preserving deterministic startup behavior.
#   INPUTS: { adapter_classes: tuple[type[ModelFamilyAdapter], ...] | None - Optional pre-resolved adapter classes for tests or alternate wiring }
#   OUTPUTS: { dict[str, ModelFamilyAdapter] - Family-keyed adapter instance map used by TTSService and SynthesisCoordinator }
#   SIDE_EFFECTS: imports built-in family adapter modules indirectly through discovery and raises ValueError when duplicate adapter keys are discovered
#   LINKS: M-TTS-SERVICE, M-DISCOVERY
# END_CONTRACT: _build_family_adapter_map
def _build_family_adapter_map(
    adapter_classes: tuple[type[ModelFamilyAdapter], ...] | None = None,
) -> dict[str, ModelFamilyAdapter]:
    resolved_classes = adapter_classes or discover_family_adapter_classes()
    adapter_map: dict[str, ModelFamilyAdapter] = {}
    for adapter_class in resolved_classes:
        adapter = adapter_class()
        adapter_key = getattr(adapter, "key", "")
        if not isinstance(adapter_key, str) or not adapter_key.strip():
            raise ValueError(
                f"Family adapter class {adapter_class.__module__}.{adapter_class.__qualname__} must declare a non-empty key"
            )
        existing = adapter_map.get(adapter_key)
        if existing is not None:
            raise ValueError(
                "Duplicate family adapter key discovered: "
                f"{adapter_key} ({existing.__class__.__module__}.{existing.__class__.__qualname__}, "
                f"{adapter_class.__module__}.{adapter_class.__qualname__})"
            )
        adapter_map[adapter_key] = adapter
    return adapter_map


# START_CONTRACT: _build_engine_registry
#   PURPOSE: Build the local runtime engine registry with the first production Piper engine while keeping the rest of the runtime on legacy execution seams.
#   INPUTS: { settings: CoreSettings - Runtime settings containing explicit engine-route toggles }
#   OUTPUTS: { EngineRegistry | None - Process-local engine registry when any engine route is enabled }
#   SIDE_EFFECTS: none
#   LINKS: M-TTS-SERVICE, M-ENGINE-REGISTRY
# END_CONTRACT: _build_engine_registry
def _build_engine_registry(settings: CoreSettings) -> EngineRegistry | None:
    if not settings.piper_engine_enabled:
        return None
    return load_engine_registry(
        explicit_engines=(PiperOnnxEngine(),),
        include_entry_points=False,
    )


class SynthesisCoordinator:
    def __init__(
        self,
        registry: RuntimeExecutionRegistry,
        settings: CoreSettings,
        inference_guard: InferenceGuard,
        planner: SynthesisPlanner,
        family_adapters: dict[str, ModelFamilyAdapter],
        engine_registry: EngineRegistry | None = None,
    ):
        self.registry = registry
        self.settings = settings
        self.inference_guard = inference_guard
        self.planner = planner
        self._family_adapters = family_adapters
        self._engine_registry = engine_registry

    def _selected_backend_key(self) -> str:
        return self.registry.backend.key

    @staticmethod
    def _handle_backend_key(handle) -> str:
        return handle.backend_key

    def synthesize_custom(self, command: CustomVoiceCommand) -> GenerationResult:
        plan = self.planner.plan_command(command)
        prepared = self._prepare_execution(plan)
        engine = self._resolve_runtime_engine(
            family_key=plan.family_key,
            capability=plan.request.capability,
            backend_key=plan.backend_key,
        )
        if engine is not None:
            return self._run_engine_generation(
                spec=plan.model_spec,
                text=command.text,
                save_output=command.save_output,
                language=plan.request.language,
                execution_mode=plan.execution_mode,
                capability=plan.request.capability,
                generation_kwargs=prepared,
                engine_key=engine.key,
            )
        spec, handle = self.registry.get_model(
            model_name=plan.model_spec.model_id,
            mode=plan.execution_mode,
        )
        return self._run_generation(
            spec=spec,
            handle=handle,
            text=command.text,
            save_output=command.save_output,
            generation_kwargs=prepared,
        )

    def _resolve_runtime_engine(
        self,
        *,
        family_key: str,
        capability: str,
        backend_key: str,
    ):
        if self._engine_registry is None:
            return None
        try:
            return self._engine_registry.resolve_engine(
                capability=capability,
                family=family_key,
                backend_key=backend_key,
            )
        except EngineRegistryError:
            return None

    def _run_engine_generation(
        self,
        *,
        spec: ModelSpec,
        text: str,
        save_output: bool,
        language: str,
        execution_mode: str,
        capability: str,
        generation_kwargs: dict[str, Any],
        engine_key: str,
    ) -> GenerationResult:
        timer = Timer()
        backend = self.registry.backend_for_spec(spec)
        self.inference_guard.acquire()
        log_event(
            LOGGER,
            level=20,
            event="[TTSService][_run_engine_generation][BLOCK_ACQUIRE_INFERENCE]",
            message="Inference slot acquired for engine synthesis",
            model=spec.api_name,
            mode=spec.mode,
            save_output=save_output,
            text_length=len(text),
            language=language,
            backend=backend.key,
            engine=engine_key,
        )
        try:
            from core.infrastructure.audio_io import persist_output, temporary_output_dir

            engine = self._resolve_runtime_engine(
                family_key=spec.family_key,
                capability=capability,
                backend_key=backend.key,
            )
            if engine is None:
                raise TTSGenerationError(
                    "No runtime engine is registered for the requested execution path",
                    details={
                        "model": spec.api_name,
                        "family": spec.family_key,
                        "capability": capability,
                        "backend": backend.key,
                    },
                )

            with temporary_output_dir(prefix="qwen3_tts_output_") as output_dir:
                model_path = backend.resolve_model_path(spec.folder)
                handle = engine.load_model(spec=spec, backend_key=backend.key, model_path=model_path)
                audio_buffer = engine.synthesize(
                    handle,
                    SynthesisJob(
                        capability=capability,
                        execution_mode=execution_mode,
                        text=text,
                        language=language,
                        output_dir=Path(output_dir),
                        payload=dict(generation_kwargs),
                    ),
                )
                output_path = Path(output_dir) / "audio_0001.wav"
                output_path.write_bytes(bytes(audio_buffer.waveform))
                audio = AudioResult(path=output_path, bytes_data=bytes(audio_buffer.waveform))
                saved_path = None
                if save_output:
                    saved_path = persist_output(audio, spec.output_subfolder, text, self.settings)
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
                    event="[TTSService][_run_engine_generation][BLOCK_PERSIST_OUTPUT]",
                    message="Engine generation completed successfully",
                    model=result.model,
                    mode=result.mode,
                    duration_ms=timer.elapsed_ms,
                    language=language,
                    saved_path=str(result.saved_path) if result.saved_path else None,
                    audio_path=str(result.audio.path),
                    backend=result.backend,
                    engine=engine.key,
                )
                return result
        finally:
            self.inference_guard.release()
            log_event(
                LOGGER,
                level=20,
                event="[TTSService][_run_engine_generation][BLOCK_RELEASE_INFERENCE]",
                message="Inference slot released after engine synthesis",
                model=spec.api_name,
                mode=spec.mode,
                duration_ms=timer.elapsed_ms,
                language=language,
                backend=backend.key,
                engine=engine_key,
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
        from core.infrastructure.audio_io import convert_audio_to_wav_if_needed, temporary_output_dir

        plan = self.planner.plan_command(command)
        spec, handle = self.registry.get_model(
            model_name=plan.model_spec.model_id,
            mode=plan.execution_mode,
        )
        ref_audio_path = command.ref_audio_path
        if ref_audio_path is None:
            raise TTSGenerationError(
                "Reference audio is required for clone synthesis",
                details={
                    "mode": "clone",
                    "reference_audio": None,
                    "backend": self._handle_backend_key(handle),
                },
            )

        with temporary_output_dir(prefix="qwen3_tts_clone_input_") as temp_dir:
            source_audio = temp_dir / ref_audio_path.name
            source_audio.write_bytes(ref_audio_path.read_bytes())
            wav_audio, converted = convert_audio_to_wav_if_needed(source_audio, self.settings)
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
            from core.infrastructure.audio_io import persist_output, read_generated_wav, temporary_output_dir

            with temporary_output_dir(prefix="qwen3_tts_output_") as output_dir:
                try:
                    # START_BLOCK_DISPATCH_TO_BACKEND
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
                    # END_BLOCK_DISPATCH_TO_BACKEND
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
                    saved_path = persist_output(audio, spec.output_subfolder, text, self.settings)

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
#   INPUTS: { registry: ModelRegistry - Model registry used to resolve and load models, settings: CoreSettings - Shared runtime settings controlling audio handling and persistence, inference_guard: InferenceGuard | None - Optional shared inference concurrency guard, result_cache: ResultCache | None - Optional cache for repeat-result short-circuiting }
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
        result_cache: ResultCache | None = None,
    ):
        from core.services.result_cache import NullResultCache
        from core.services.synthesis_router import SynthesisRouter
        from core.infrastructure.concurrency import InferenceGuard as _InferenceGuard

        self.registry = registry
        self.settings = settings
        self.inference_guard = inference_guard or _InferenceGuard()
        self.planner = SynthesisPlanner(registry, settings)
        self._family_adapters = _build_family_adapter_map()
        self._engine_registry = _build_engine_registry(settings)
        self.coordinator = SynthesisCoordinator(
            registry=registry,
            settings=settings,
            inference_guard=self.inference_guard,
            planner=self.planner,
            family_adapters=self._family_adapters,
            engine_registry=self._engine_registry,
        )
        self._result_cache = result_cache or NullResultCache()
        self.router = SynthesisRouter(
            coordinator=self.coordinator,
            result_cache=self._result_cache,
        )

    @property
    def result_cache(self) -> ResultCache:
        return self._result_cache

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
        with cast(Any, operation_scope("core.tts_service.synthesize_custom")):
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
            result = self.router.route_custom(command)
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
        with cast(Any, operation_scope("core.tts_service.synthesize_design")):
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
            result = self.router.route_design(command)
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
        with cast(Any, operation_scope("core.tts_service.synthesize_clone")):
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
            result = self.router.route_clone(command)
            return result


__all__ = [
    "LOGGER",
    "SynthesisCoordinator",
    "TTSService",
    "_build_family_adapter_map",
]
