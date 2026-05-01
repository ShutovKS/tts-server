# FILE: core/services/synthesis_router.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide the SynthesisRouter unified seam that collapses the public synthesis pipeline (Command -> Router -> backend execution) by dispatching every GenerationCommand variant through a single entry-point and, when configured, transparently short-circuiting identical requests against a swappable ResultCache.
#   SCOPE: SynthesisRouter class with a unified route(command) entry-point, explicit per-mode helpers route_custom/_design/_clone, and an optional result_cache (NullResultCache by default) that lets identical commands skip the underlying coordinator.
#   DEPENDS: M-CONFIG, M-MODEL-FAMILY, M-MODEL-REGISTRY, M-INFRASTRUCTURE, M-OBSERVABILITY, M-TTS-SERVICE, M-RESULT-CACHE
#   LINKS: M-TTS-SERVICE, M-RESULT-CACHE
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LOGGER - Module logger for synthesis routing events
#   SynthesisRouter - Unified routing seam between transport-facing commands and the synthesis coordinator
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Phase 4.14: accept an optional ResultCache (default NullResultCache) and consult it on each route_* path, so identical custom/design/clone requests can short-circuit the underlying coordinator while existing call-sites remain unchanged]
# END_CHANGE_SUMMARY

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from core.config import CoreSettings
from core.contracts import RuntimeExecutionRegistry
from core.contracts.commands import (
    CustomVoiceCommand,
    GenerationCommand,
    VoiceCloneCommand,
    VoiceDesignCommand,
)
from core.contracts.results import AudioResult, GenerationResult
from core.errors import TTSGenerationError
from core.infrastructure.concurrency import InferenceGuard
from core.model_families import ModelFamilyAdapter
from core.observability import get_logger, log_event
from core.planning import SynthesisPlanner
from core.services.result_cache import (
    CachedResult,
    NullResultCache,
    ResultCache,
    build_cache_key,
)

if TYPE_CHECKING:
    from core.services.tts_service import SynthesisCoordinator

LOGGER = get_logger(__name__)


# START_CONTRACT: SynthesisRouter
#   PURPOSE: Route any GenerationCommand variant through a single entry-point to the synthesis coordinator so callers (TTSService and tests) do not have to know about per-mode coordinator helpers.
#   INPUTS: { coordinator: SynthesisCoordinator | None - Optional pre-built coordinator (test injection); when omitted, the router builds one from the remaining wiring fields, registry: RuntimeExecutionRegistry - Runtime registry used to load models and resolve backends, settings: CoreSettings - Shared runtime settings, inference_guard: InferenceGuard - Concurrency guard, planner: SynthesisPlanner - Planner reused by the coordinator, family_adapters: dict[str, ModelFamilyAdapter] - Family-keyed adapters used to prepare execution payloads }
#   OUTPUTS: { instance - Router ready to dispatch route(command) calls }
#   SIDE_EFFECTS: none on construction; route(...) triggers planning, family preparation, guarded inference, and audio persistence inside the underlying coordinator
#   LINKS: M-TTS-SERVICE
# END_CONTRACT: SynthesisRouter
class SynthesisRouter:
    def __init__(
        self,
        *,
        coordinator: SynthesisCoordinator | None = None,
        registry: RuntimeExecutionRegistry | None = None,
        settings: CoreSettings | None = None,
        inference_guard: InferenceGuard | None = None,
        planner: SynthesisPlanner | None = None,
        family_adapters: dict[str, ModelFamilyAdapter] | None = None,
        result_cache: ResultCache | None = None,
    ) -> None:
        self._result_cache: ResultCache = result_cache or NullResultCache()
        if coordinator is None:
            if (
                registry is None
                or settings is None
                or inference_guard is None
                or planner is None
                or family_adapters is None
            ):
                raise ValueError(
                    "SynthesisRouter requires either a pre-built coordinator or all of "
                    "(registry, settings, inference_guard, planner, family_adapters)."
                )
            from core.services.tts_service import SynthesisCoordinator as _Coord

            coordinator = _Coord(
                registry=registry,
                settings=settings,
                inference_guard=inference_guard,
                planner=planner,
                family_adapters=family_adapters,
            )
        self._coordinator = coordinator

    @property
    def coordinator(self) -> SynthesisCoordinator:
        return self._coordinator

    @property
    def result_cache(self) -> ResultCache:
        return self._result_cache

    def _command_cache_key(self, command: GenerationCommand) -> tuple[str, str] | None:
        # START_BLOCK_BUILD_CACHE_KEY
        if isinstance(self._result_cache, NullResultCache):
            return None
        if isinstance(command, CustomVoiceCommand):
            kind = "custom"
            params = {
                "text": command.text,
                "model": command.model,
                "language": command.language,
                "speaker": command.speaker,
                "instruct": command.instruct,
                "speed": command.speed,
                "save_output": command.save_output,
            }
        elif isinstance(command, VoiceDesignCommand):
            kind = "design"
            params = {
                "text": command.text,
                "model": command.model,
                "language": command.language,
                "voice_description": command.voice_description,
                "save_output": command.save_output,
            }
        elif isinstance(command, VoiceCloneCommand):
            kind = "clone"
            params = {
                "text": command.text,
                "model": command.model,
                "language": command.language,
                "ref_audio_path": str(command.ref_audio_path) if command.ref_audio_path else None,
                "ref_text": command.ref_text,
                "save_output": command.save_output,
            }
        else:
            return None
        return kind, build_cache_key(kind, params)
        # END_BLOCK_BUILD_CACHE_KEY

    def _result_to_cache_entry(self, result: GenerationResult) -> CachedResult:
        return CachedResult(
            payload=result.audio.bytes_data,
            media_type=result.audio.media_type,
            model=result.model,
            mode=result.mode,
            backend=result.backend,
            saved_path=str(result.saved_path) if result.saved_path is not None else None,
            created_at=time.time(),
        )

    def _cache_entry_to_result(self, entry: CachedResult) -> GenerationResult:
        path = Path(entry.saved_path) if entry.saved_path else Path("")
        return GenerationResult(
            audio=AudioResult(
                path=path,
                bytes_data=entry.payload,
                media_type=entry.media_type,
            ),
            saved_path=Path(entry.saved_path) if entry.saved_path else None,
            model=entry.model,
            mode=entry.mode,
            backend=entry.backend,
        )

    # START_CONTRACT: route
    #   PURPOSE: Dispatch a generic GenerationCommand to the matching synthesis flow without exposing per-mode entry points to callers.
    #   INPUTS: { command: GenerationCommand - Concrete subclass selected by the transport adapter (CustomVoiceCommand, VoiceDesignCommand, or VoiceCloneCommand) }
    #   OUTPUTS: { GenerationResult - Generated audio together with persistence metadata }
    #   SIDE_EFFECTS: Triggers planning, model loading, guarded inference, structured logging, and (when configured) artifact persistence via the underlying coordinator
    #   LINKS: M-TTS-SERVICE
    # END_CONTRACT: route
    def route(self, command: GenerationCommand) -> GenerationResult:
        # START_BLOCK_DISPATCH_BY_COMMAND_TYPE
        if isinstance(command, CustomVoiceCommand):
            mode = "custom"
            target = self._coordinator.synthesize_custom
        elif isinstance(command, VoiceDesignCommand):
            mode = "design"
            target = self._coordinator.synthesize_design
        elif isinstance(command, VoiceCloneCommand):
            mode = "clone"
            target = self._coordinator.synthesize_clone
        else:
            raise TTSGenerationError(
                "Unsupported synthesis command type",
                details={
                    "command_type": type(command).__name__,
                    "expected": [
                        "CustomVoiceCommand",
                        "VoiceDesignCommand",
                        "VoiceCloneCommand",
                    ],
                },
            )
        # END_BLOCK_DISPATCH_BY_COMMAND_TYPE
        log_event(
            LOGGER,
            level=20,
            event="[SynthesisRouter][route][DISPATCH]",
            message="Routing synthesis command",
            mode=mode,
            command_type=type(command).__name__,
            requested_model=command.model,
            text_length=len(command.text),
            language=command.language,
        )
        return target(command)

    # START_CONTRACT: route_custom
    #   PURPOSE: Route an explicit CustomVoiceCommand without dispatching by isinstance.
    #   INPUTS: { command: CustomVoiceCommand - Pre-typed custom voice command }
    #   OUTPUTS: { GenerationResult - Generated audio }
    #   SIDE_EFFECTS: same as route()
    #   LINKS: M-TTS-SERVICE
    # END_CONTRACT: route_custom
    def route_custom(self, command: CustomVoiceCommand) -> GenerationResult:
        # START_BLOCK_ROUTE_CUSTOM_WITH_CACHE
        return self._dispatch_with_cache(command, self._coordinator.synthesize_custom)
        # END_BLOCK_ROUTE_CUSTOM_WITH_CACHE

    # START_CONTRACT: route_design
    #   PURPOSE: Route an explicit VoiceDesignCommand without dispatching by isinstance.
    #   INPUTS: { command: VoiceDesignCommand - Pre-typed voice design command }
    #   OUTPUTS: { GenerationResult - Generated audio }
    #   SIDE_EFFECTS: same as route()
    #   LINKS: M-TTS-SERVICE
    # END_CONTRACT: route_design
    def route_design(self, command: VoiceDesignCommand) -> GenerationResult:
        # START_BLOCK_ROUTE_DESIGN_WITH_CACHE
        return self._dispatch_with_cache(command, self._coordinator.synthesize_design)
        # END_BLOCK_ROUTE_DESIGN_WITH_CACHE

    # START_CONTRACT: route_clone
    #   PURPOSE: Route an explicit VoiceCloneCommand without dispatching by isinstance.
    #   INPUTS: { command: VoiceCloneCommand - Pre-typed voice clone command (reference audio is required upstream) }
    #   OUTPUTS: { GenerationResult - Generated audio }
    #   SIDE_EFFECTS: same as route()
    #   LINKS: M-TTS-SERVICE
    # END_CONTRACT: route_clone
    def route_clone(self, command: VoiceCloneCommand) -> GenerationResult:
        # START_BLOCK_ROUTE_CLONE_WITH_CACHE
        return self._dispatch_with_cache(command, self._coordinator.synthesize_clone)
        # END_BLOCK_ROUTE_CLONE_WITH_CACHE

    def _dispatch_with_cache(self, command, target):
        # START_BLOCK_DISPATCH_WITH_CACHE
        cache_key = self._command_cache_key(command)
        if cache_key is not None:
            kind, key = cache_key
            cached = self._result_cache.get(key)
            if cached is not None:
                log_event(
                    LOGGER,
                    level=20,
                    event="[SynthesisRouter][_dispatch_with_cache][BLOCK_CACHE_HIT]",
                    message="Returning cached synthesis result",
                    mode=kind,
                    cache_key=key,
                    cached_model=cached.model,
                    cached_backend=cached.backend,
                )
                return self._cache_entry_to_result(cached)
        result = target(command)
        if cache_key is not None:
            _, key = cache_key
            try:
                self._result_cache.put(key, self._result_to_cache_entry(result))
            except Exception as exc:
                log_event(
                    LOGGER,
                    level=30,
                    event="[SynthesisRouter][_dispatch_with_cache][BLOCK_CACHE_PUT_FAILED]",
                    message="Failed to write synthesis result to cache",
                    cache_key=key,
                    error=str(exc),
                )
        return result
        # END_BLOCK_DISPATCH_WITH_CACHE


__all__ = ["LOGGER", "SynthesisRouter"]
