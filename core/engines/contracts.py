# FILE: core/engines/contracts.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define the shared engine contracts and immutable DTOs that future engine registry and scheduler layers will consume.
#   SCOPE: TTSEngine abstract contract, model/audio/job DTOs, capability flags, and availability metadata
#   DEPENDS: M-MODELS
#   LINKS: M-ENGINE-CONTRACTS, M-BACKENDS, M-MODEL-FAMILY
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   ModelHandle - Loaded engine model handle plus backend/family metadata.
#   AudioBuffer - In-memory audio payload returned by engine synthesis.
#   SynthesisJob - Normalized synthesis request owned by a TTSEngine.
#   EngineCapabilities - Typed capability summary for engine selection.
#   EngineAvailability - Structured availability/enabled-state report for engine filtering.
#   TTSEngine - Abstract engine contract separating model loading from synthesis execution.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Phase 2 engine wave: introduced contract-only TTSEngine DTOs with explicit load_model(...) and synthesize(...) separation while leaving current runtime paths untouched]
# END_CHANGE_SUMMARY

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.models.manifest import ModelSpec


# START_CONTRACT: ModelHandle
#   PURPOSE: Carry a loaded runtime model together with the engine, backend, and family metadata needed for later synthesis calls.
#   INPUTS: { spec: ModelSpec - Resolved model specification, runtime_model: Any - Loaded engine-specific runtime object, resolved_path: Path | None - Filesystem path used to load the model, engine_key: str - Engine identifier that produced the handle, backend_key: str - Backend lane the handle was prepared for, family_key: str - Family key the handle belongs to }
#   OUTPUTS: { instance - Immutable model handle suitable for future synthesize(handle, job) calls }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-CONTRACTS, M-BACKENDS, M-MODEL-FAMILY
# END_CONTRACT: ModelHandle
@dataclass(frozen=True)
class ModelHandle:
    spec: ModelSpec
    runtime_model: Any
    resolved_path: Path | None
    engine_key: str
    backend_key: str
    family_key: str


# START_CONTRACT: AudioBuffer
#   PURPOSE: Represent the in-memory audio payload returned by engine synthesis without forcing an on-disk artifact contract.
#   INPUTS: { waveform: Any - Engine-produced waveform or audio payload, sample_rate: int - Sample rate for the audio, audio_format: str - Logical audio format label }
#   OUTPUTS: { instance - Immutable audio payload container }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-CONTRACTS
# END_CONTRACT: AudioBuffer
@dataclass(frozen=True)
class AudioBuffer:
    waveform: Any
    sample_rate: int
    audio_format: str = "wav"


# START_CONTRACT: SynthesisJob
#   PURPOSE: Carry the normalized synthesis inputs a TTSEngine needs after planning has already resolved the model/backend path.
#   INPUTS: { capability: str - Synthesis capability identifier, execution_mode: str - Resolved execution mode, text: str - Prompt text, language: str - Resolved language code, output_dir: Path - Directory available to future engine-owned persistence logic, payload: dict[str, Any] - Engine or family-specific synthesis fields }
#   OUTPUTS: { instance - Immutable synthesis request handed to TTSEngine.synthesize }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-CONTRACTS, M-EXECUTION-PLAN
# END_CONTRACT: SynthesisJob
@dataclass(frozen=True)
class SynthesisJob:
    capability: str
    execution_mode: str
    text: str
    language: str
    output_dir: Path
    payload: dict[str, Any] = field(default_factory=dict)


# START_CONTRACT: EngineCapabilities
#   PURPOSE: Describe what a TTSEngine can execute across families, backends, and synthesis capabilities.
#   INPUTS: { families: tuple[str, ...] - Supported family keys, backends: tuple[str, ...] - Supported backend keys, capabilities: tuple[str, ...] - Supported synthesis capabilities, supports_streaming: bool - Whether streaming output is supported, supports_batching: bool - Whether batched execution is supported }
#   OUTPUTS: { instance - Immutable capability summary for engine selection }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-CONTRACTS
# END_CONTRACT: EngineCapabilities
@dataclass(frozen=True)
class EngineCapabilities:
    families: tuple[str, ...] = ()
    backends: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    supports_streaming: bool = False
    supports_batching: bool = False


# START_CONTRACT: EngineAvailability
#   PURPOSE: Expose whether an engine is both configured as enabled and available in the current host/runtime environment.
#   INPUTS: { engine_key: str - Engine identifier, is_available: bool - Runtime dependency readiness result, is_enabled: bool - Configured enablement flag, reason: str | None - Human-readable explanation for unavailable or disabled states, missing_dependencies: tuple[str, ...] - Missing runtime dependencies if known }
#   OUTPUTS: { instance - Immutable availability report }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-CONTRACTS
# END_CONTRACT: EngineAvailability
@dataclass(frozen=True)
class EngineAvailability:
    engine_key: str
    is_available: bool
    is_enabled: bool = True
    reason: str | None = None
    missing_dependencies: tuple[str, ...] = ()


# START_CONTRACT: TTSEngine
#   PURPOSE: Define the future engine-facing execution contract while keeping model loading explicitly separate from synthesis calls.
#   INPUTS: { subclass - Concrete engine class declaring key and label }
#   OUTPUTS: { instance - Abstract engine contract for registry layers }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-CONTRACTS, M-ENGINE-REGISTRY
# END_CONTRACT: TTSEngine
class TTSEngine(ABC):
    key: str
    label: str

    @abstractmethod
    # START_CONTRACT: capabilities
    #   PURPOSE: Describe the family/backend/capability envelope supported by the engine implementation.
    #   INPUTS: {}
    #   OUTPUTS: { EngineCapabilities - Typed capability summary for engine selection }
    #   SIDE_EFFECTS: none
    #   LINKS: M-ENGINE-CONTRACTS
    # END_CONTRACT: capabilities
    def capabilities(self) -> EngineCapabilities:
        raise NotImplementedError

    @abstractmethod
    # START_CONTRACT: availability
    #   PURPOSE: Report whether the engine is enabled and whether its runtime dependencies are currently available.
    #   INPUTS: {}
    #   OUTPUTS: { EngineAvailability - Structured enabled/available state }
    #   SIDE_EFFECTS: none
    #   LINKS: M-ENGINE-CONTRACTS
    # END_CONTRACT: availability
    def availability(self) -> EngineAvailability:
        raise NotImplementedError

    @abstractmethod
    # START_CONTRACT: load_model
    #   PURPOSE: Load the runtime model for an already-resolved spec/backend path and return a reusable handle for later synthesis.
    #   INPUTS: { spec: ModelSpec - Manifest model specification to load, backend_key: str - Backend lane the engine should prepare for, model_path: Path | None - Resolved on-disk model path }
    #   OUTPUTS: { ModelHandle - Loaded reusable model handle }
    #   SIDE_EFFECTS: May allocate runtime resources and cache loaded state
    #   LINKS: M-ENGINE-CONTRACTS, M-MODELS
    # END_CONTRACT: load_model
    def load_model(
        self,
        *,
        spec: ModelSpec,
        backend_key: str,
        model_path: Path | None,
    ) -> ModelHandle:
        raise NotImplementedError

    @abstractmethod
    # START_CONTRACT: synthesize
    #   PURPOSE: Execute synthesis against an existing ModelHandle without repeating model-loading responsibilities.
    #   INPUTS: { handle: ModelHandle - Previously loaded engine handle, job: SynthesisJob - Normalized synthesis request }
    #   OUTPUTS: { AudioBuffer - In-memory audio payload produced by the engine }
    #   SIDE_EFFECTS: Performs inference and may allocate transient runtime resources
    #   LINKS: M-ENGINE-CONTRACTS
    # END_CONTRACT: synthesize
    def synthesize(self, handle: ModelHandle, job: SynthesisJob) -> AudioBuffer:
        raise NotImplementedError


__all__ = [
    "AudioBuffer",
    "EngineAvailability",
    "EngineCapabilities",
    "ModelHandle",
    "SynthesisJob",
    "TTSEngine",
]
