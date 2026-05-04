# FILE: core/services/tts_service.py
# VERSION: 2.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Coordinate inference for custom, design, and clone synthesis modes via the SynthesisRouter unified seam, while preserving the transport-facing TTSService.synthesize_X(...) facade for backwards compatibility and routing runtime execution through the scheduler gateway.
#   SCOPE: TTSService class with synthesize_custom/design/clone delegating through SynthesisRouter, SynthesisCoordinator (kept as the per-mode worker; routes legacy backend plus Piper, Qwen3, and OmniVoice engine execution through injected EngineRegistry, EngineScheduler, persistence, and clone-preprocessing services).
#   DEPENDS: M-MODEL-REGISTRY, M-CONFIG, M-DISCOVERY, M-ERRORS, M-OBSERVABILITY, M-INFRASTRUCTURE, M-MODEL-FAMILY, M-ENGINE-REGISTRY, M-ENGINE-CONTRACTS, M-ENGINE-SCHEDULER, M-AUDIO-PERSISTENCE, M-CLONE-PREPROCESSING, M-ENGINE-AUDIO-PIPELINE, M-LEGACY-BACKEND-EXECUTION, M-TTS-COORDINATOR
#   LINKS: M-TTS-SERVICE
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LOGGER - Module logger for synthesis service events
#   SynthesisCoordinator - Internal coordinator over planning, family preparation, and scheduler-gated generation; the per-mode worker invoked by SynthesisRouter.
#   _build_family_adapter_map - Instantiate a deterministic family-keyed adapter map from discovery results while rejecting duplicate keys
#   TTSService - Public synthesis facade preserving transport-facing command methods; delegates each call through SynthesisRouter to keep the public pipeline at three layers (TTSService -> SynthesisRouter -> scheduler-gated runtime execution)
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v2.1.0 - Removed InferenceGuard constructor/runtime wiring; EngineScheduler is the sole bounded execution seam]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from core.services.result_cache import ResultCache

from core.config import CoreSettings
from core.contracts import RuntimeExecutionRegistry
from core.contracts.results import GenerationResult
from core.contracts.commands import (
    CustomVoiceCommand,
    VoiceCloneCommand,
    VoiceDesignCommand,
)
from core.contracts.synthesis import SynthesisRequest
from core.discovery import discover_family_adapter_classes
from core.engines.contracts import SynthesisJob
from core.engines.contracts import TTSEngine
from core.engines.audio_pipeline import AudioPipeline
from core.engines.config import DisabledEngineConfig
from core.engines.registry import EngineRegistry, EngineRegistryError
from core.engines.scheduler import EngineScheduler
from core.errors import TTSGenerationError
from core.model_families import ModelFamilyAdapter
from core.models.catalog import ModelSpec
from core.observability import Timer, get_logger, log_event, operation_scope
from core.planning import SynthesisPlanner
from core.services.audio_persistence import AudioPersistenceService
from core.services.clone_preprocessing import CloneReferenceAudioPreprocessor
from core.services.legacy_backend_execution import LegacyBackendExecutionService

LOGGER = get_logger(__name__)
_COMPAT_SCHEDULER_ENGINE_KEY = "tts-service-compat"
_ENGINE_REQUIRED_FAMILIES = frozenset({"qwen3_tts", "omnivoice", "piper"})


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


class SynthesisCoordinator:
    def __init__(
        self,
        registry: RuntimeExecutionRegistry,
        settings: CoreSettings,
        scheduler: EngineScheduler,
        planner: SynthesisPlanner,
        family_adapters: dict[str, ModelFamilyAdapter],
        engine_registry: EngineRegistry | None = None,
        audio_persistence: AudioPersistenceService | None = None,
        clone_preprocessor: CloneReferenceAudioPreprocessor | None = None,
        audio_pipeline: AudioPipeline | None = None,
        legacy_backend_execution: LegacyBackendExecutionService | None = None,
    ):
        self.registry = registry
        self.settings = settings
        self.scheduler = scheduler
        self.planner = planner
        self._family_adapters = family_adapters
        self._engine_registry = engine_registry
        self._audio_persistence = audio_persistence or AudioPersistenceService(settings)
        self._clone_preprocessor = clone_preprocessor or CloneReferenceAudioPreprocessor(settings)
        self._audio_pipeline = audio_pipeline or AudioPipeline(target_sample_rate=settings.sample_rate)
        self._legacy_backend_execution = legacy_backend_execution or LegacyBackendExecutionService(
            registry=registry,
            audio_persistence=self._audio_persistence,
        )

    def _scheduler_submit(self, *, spec: ModelSpec, backend_key: str, call, engine_key: str | None = None):
        resolved_engine_key = engine_key or _COMPAT_SCHEDULER_ENGINE_KEY
        device_key = None if engine_key is None else self._device_key_for_engine(engine_key)
        return self.scheduler.submit_engine_task(
            engine_key=resolved_engine_key,
            device_key=device_key,
            call=call,
        )

    def _selected_backend_key(self) -> str:
        return self.registry.backend.key

    @staticmethod
    def _handle_backend_key(handle) -> str:
        return handle.backend_key

    def synthesize_custom(self, command: CustomVoiceCommand) -> GenerationResult:
        plan = self.planner.plan_command(command)
        prepared = self._prepare_execution(plan)
        engine_required = self._engine_required_for_family(plan.family_key)
        engine = self._resolve_runtime_engine(
            family_key=plan.family_key,
            capability=plan.request.capability,
            backend_key=plan.backend_key,
            spec=plan.model_spec,
            required=engine_required,
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
                engine=engine,
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
        spec: ModelSpec,
        required: bool = False,
    ) -> TTSEngine | None:
        if self._engine_registry is None:
            if required:
                raise TTSGenerationError(
                    "Runtime engine registry is required for the requested execution path",
                    details={
                        "model": spec.api_name,
                        "family": family_key,
                        "capability": capability,
                        "backend": backend_key,
                        "engine_required": True,
                    },
                )
            return None
        try:
            return self._engine_registry.resolve_engine(
                capability=capability,
                family=family_key,
                backend_key=backend_key,
            )
        except EngineRegistryError as exc:
            if required:
                raise TTSGenerationError(
                    "No runtime engine is registered for the requested execution path",
                    details={
                        "model": spec.api_name,
                        "family": family_key,
                        "capability": capability,
                        "backend": backend_key,
                        "engine_required": True,
                        "resolution_error": str(exc),
                    },
                ) from exc
            return None

    def _engine_required_for_family(self, family_key: str) -> bool:
        return family_key in _ENGINE_REQUIRED_FAMILIES

    def _device_key_for_engine(self, engine_key: str) -> str | None:
        if self._engine_registry is None:
            return None
        config = self._engine_registry.get_config(engine_key)
        if config is None or isinstance(config, DisabledEngineConfig):
            return None
        return config.device

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
        engine: TTSEngine,
    ) -> GenerationResult:
        timer = Timer()
        backend = self.registry.backend_for_spec(spec)

        def execute_generation() -> GenerationResult:
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
                engine=engine.key,
            )
            try:
                def generate_waveform(output_dir: Path) -> bytes:
                    model_path = backend.resolve_model_path(spec.folder)
                    handle = engine.load_model(spec=spec, backend_key=backend.key, model_path=model_path)
                    audio_buffer = engine.synthesize(
                        handle,
                        SynthesisJob(
                            capability=capability,
                            execution_mode=execution_mode,
                            text=text,
                            language=language,
                            output_dir=output_dir,
                            payload=dict(generation_kwargs),
                        ),
                    )
                    processed_audio = self._audio_pipeline.process(audio_buffer)
                    return bytes(processed_audio.waveform)

                result = self._audio_persistence.materialize_engine_generation(
                    spec=spec,
                    text=text,
                    save_output=save_output,
                    backend_key=backend.key,
                    generate_waveform=generate_waveform,
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
                    engine=engine.key,
                )

        return self._scheduler_submit(
            spec=spec,
            backend_key=backend.key,
            call=execute_generation,
            engine_key=engine.key,
        )

    def synthesize_design(self, command: VoiceDesignCommand) -> GenerationResult:
        plan = self.planner.plan_command(command)
        prepared = self._prepare_execution(plan)
        engine_required = self._engine_required_for_family(plan.family_key)
        engine = self._resolve_runtime_engine(
            family_key=plan.family_key,
            capability=plan.request.capability,
            backend_key=plan.backend_key,
            spec=plan.model_spec,
            required=engine_required,
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
                engine=engine,
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

    def synthesize_clone(self, command: VoiceCloneCommand) -> GenerationResult:
        plan = self.planner.plan_command(command)
        spec = plan.model_spec
        with self._clone_preprocessor.prepare(command) as prepared_reference:
            log_event(
                LOGGER,
                level=20,
                event="[TTSService][synthesize_clone][BLOCK_PREPARE_REFERENCE_AUDIO]",
                message="Reference audio prepared for clone synthesis",
                model=spec.api_name,
                mode=spec.mode,
                source_audio=str(prepared_reference.source_audio),
                prepared_audio=str(prepared_reference.prepared_audio),
                converted=prepared_reference.converted,
                backend=plan.backend_key,
            )
            prepared_request = SynthesisRequest.from_command(
                VoiceCloneCommand(
                    text=command.text,
                    model=command.model,
                    save_output=command.save_output,
                    language=command.language,
                    ref_audio_path=prepared_reference.prepared_audio,
                    ref_text=command.ref_text,
                )
            )
            prepared_plan = replace(plan, request=prepared_request)
            prepared_generation = self._prepare_execution(prepared_plan)
            engine_required = self._engine_required_for_family(prepared_plan.family_key)
            engine = self._resolve_runtime_engine(
                family_key=prepared_plan.family_key,
                capability=prepared_plan.request.capability,
                backend_key=prepared_plan.backend_key,
                spec=prepared_plan.model_spec,
                required=engine_required,
            )
            if engine is not None:
                result = self._run_engine_generation(
                    spec=prepared_plan.model_spec,
                    text=command.text,
                    save_output=command.save_output,
                    language=prepared_plan.request.language,
                    execution_mode=prepared_plan.execution_mode,
                    capability=prepared_plan.request.capability,
                    generation_kwargs=prepared_generation,
                    engine=engine,
                )
            else:
                spec, handle = self.registry.get_model(
                    model_name=plan.model_spec.model_id,
                    mode=plan.execution_mode,
                )
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
        return self._legacy_backend_execution.execute(
            spec=spec,
            handle=handle,
            text=text,
            save_output=save_output,
            generation_kwargs=generation_kwargs,
            scheduler_submit=self._scheduler_submit,
            scheduler_engine_key=_COMPAT_SCHEDULER_ENGINE_KEY,
        )


# START_CONTRACT: TTSService
#   PURPOSE: Coordinate model resolution, scheduler-gated inference execution, and output persistence for TTS requests.
#   INPUTS: { registry: ModelRegistry - Model registry used to resolve and load models, settings: CoreSettings - Shared runtime settings controlling audio handling and persistence, scheduler: EngineScheduler | None - Optional shared engine scheduler gateway, engine_registry: EngineRegistry | None - Runtime engine registry supplied by bootstrap, result_cache: ResultCache | None - Optional cache for repeat-result short-circuiting }
#   OUTPUTS: { instance - TTS synthesis service for custom, design, and clone modes }
#   SIDE_EFFECTS: none
#   LINKS: M-TTS-SERVICE
# END_CONTRACT: TTSService
class TTSService:
    def __init__(
        self,
        registry: RuntimeExecutionRegistry,
        settings: CoreSettings,
        scheduler: EngineScheduler | None = None,
        engine_registry: EngineRegistry | None = None,
        result_cache: ResultCache | None = None,
    ):
        from core.services.result_cache import NullResultCache
        from core.services.synthesis_router import SynthesisRouter

        self.registry = registry
        self.settings = settings
        self.scheduler = scheduler or EngineScheduler()
        self.planner = SynthesisPlanner(registry, settings)
        self._family_adapters = _build_family_adapter_map()
        self._engine_registry = engine_registry
        self.coordinator = SynthesisCoordinator(
            registry=registry,
            settings=settings,
            scheduler=self.scheduler,
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
    #   SIDE_EFFECTS: Delegates reference-audio staging and conversion, loads model state, emits structured logs, performs inference, and may persist generated audio
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
