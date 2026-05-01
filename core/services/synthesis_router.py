# FILE: core/services/synthesis_router.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide the SynthesisRouter unified seam that collapses the public synthesis pipeline (Command -> Router -> backend execution) by dispatching every GenerationCommand variant through a single entry-point so transports and TTSService no longer need to know about per-mode coordinator layers.
#   SCOPE: SynthesisRouter class with a unified route(command) entry-point and explicit per-mode helpers route_custom, route_design, and route_clone that internally delegate to the same coordinator instance used by TTSService.
#   DEPENDS: M-CONFIG, M-MODEL-FAMILY, M-MODEL-REGISTRY, M-INFRASTRUCTURE, M-OBSERVABILITY, M-TTS-SERVICE
#   LINKS: M-TTS-SERVICE
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
#   LAST_CHANGE: [v1.0.0 - Phase 3.9: introduced SynthesisRouter as the unified Command -> backend seam in front of the existing SynthesisCoordinator; collapses six perceived public layers (TTSService, Coordinator, planner, adapter, generate_audio, backend) into three (TTSService, Router, backend) without changing the internal plumbing yet]
# END_CHANGE_SUMMARY

from __future__ import annotations

from typing import TYPE_CHECKING

from core.config import CoreSettings
from core.contracts import RuntimeExecutionRegistry
from core.contracts.commands import (
    CustomVoiceCommand,
    GenerationCommand,
    VoiceCloneCommand,
    VoiceDesignCommand,
)
from core.contracts.results import GenerationResult
from core.errors import TTSGenerationError
from core.infrastructure.concurrency import InferenceGuard
from core.model_families import ModelFamilyAdapter
from core.observability import get_logger, log_event
from core.planning import SynthesisPlanner

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
    ) -> None:
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
        return self._coordinator.synthesize_custom(command)

    # START_CONTRACT: route_design
    #   PURPOSE: Route an explicit VoiceDesignCommand without dispatching by isinstance.
    #   INPUTS: { command: VoiceDesignCommand - Pre-typed voice design command }
    #   OUTPUTS: { GenerationResult - Generated audio }
    #   SIDE_EFFECTS: same as route()
    #   LINKS: M-TTS-SERVICE
    # END_CONTRACT: route_design
    def route_design(self, command: VoiceDesignCommand) -> GenerationResult:
        return self._coordinator.synthesize_design(command)

    # START_CONTRACT: route_clone
    #   PURPOSE: Route an explicit VoiceCloneCommand without dispatching by isinstance.
    #   INPUTS: { command: VoiceCloneCommand - Pre-typed voice clone command (reference audio is required upstream) }
    #   OUTPUTS: { GenerationResult - Generated audio }
    #   SIDE_EFFECTS: same as route()
    #   LINKS: M-TTS-SERVICE
    # END_CONTRACT: route_clone
    def route_clone(self, command: VoiceCloneCommand) -> GenerationResult:
        return self._coordinator.synthesize_clone(command)


__all__ = ["LOGGER", "SynthesisRouter"]
