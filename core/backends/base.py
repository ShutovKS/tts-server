# FILE: core/backends/base.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define abstract base interface for TTS inference backends.
#   SCOPE: TTSBackend abstract class, LoadedModelHandle dataclass
#   DEPENDS: M-ERRORS
#   LINKS: M-BACKENDS
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   TTSBackend - Abstract backend interface
#   LoadedModelHandle - Handle to a loaded model with metadata
#   ExecutionRequest - Runtime-oriented execution request carrying family-prepared inputs
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.backends.capabilities import BackendCapabilitySet, BackendDiagnostics
from core.models.catalog import ModelSpec


# START_CONTRACT: LoadedModelHandle
#   PURPOSE: Carry a loaded runtime model together with its resolved spec and backend metadata.
#   INPUTS: { spec: ModelSpec - Resolved model specification, runtime_model: Any - Loaded backend-specific runtime object, resolved_path: Path | None - Filesystem path used for loading, backend_key: str - Backend key that produced the handle }
#   OUTPUTS: { instance - Immutable loaded model handle }
#   SIDE_EFFECTS: none
#   LINKS: M-BACKENDS
# END_CONTRACT: LoadedModelHandle
@dataclass(frozen=True)
class LoadedModelHandle:
    spec: ModelSpec
    runtime_model: Any
    resolved_path: Path | None
    backend_key: str


@dataclass(frozen=True)
class ExecutionRequest:
    handle: LoadedModelHandle
    text: str
    output_dir: Path
    language: str
    legacy_mode: str
    generation_kwargs: dict[str, Any]


# START_CONTRACT: TTSBackend
#   PURPOSE: Define the abstract interface that all concrete TTS inference backends must implement.
#   INPUTS: {}
#   OUTPUTS: { instance - Abstract backend contract for model loading, readiness, and synthesis }
#   SIDE_EFFECTS: none
#   LINKS: M-BACKENDS
# END_CONTRACT: TTSBackend
class TTSBackend(ABC):
    key: str
    label: str

    @abstractmethod
    # START_CONTRACT: capabilities
    #   PURPOSE: Describe the feature capabilities supported by the backend implementation.
    #   INPUTS: {}
    #   OUTPUTS: { BackendCapabilitySet - Capability flags for the backend }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: capabilities
    def capabilities(self) -> BackendCapabilitySet:
        raise NotImplementedError

    @abstractmethod
    # START_CONTRACT: is_available
    #   PURPOSE: Report whether the backend runtime dependencies are available in the current environment.
    #   INPUTS: {}
    #   OUTPUTS: { bool - True when runtime dependencies are available }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: is_available
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    # START_CONTRACT: supports_platform
    #   PURPOSE: Report whether the backend supports the current operating platform.
    #   INPUTS: {}
    #   OUTPUTS: { bool - True when the backend is compatible with the active platform }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: supports_platform
    def supports_platform(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    # START_CONTRACT: resolve_model_path
    #   PURPOSE: Resolve the on-disk path for a configured model folder.
    #   INPUTS: { folder_name: str - Model directory name from the manifest }
    #   OUTPUTS: { Path | None - Resolved model path when present }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: resolve_model_path
    def resolve_model_path(self, folder_name: str) -> Path | None:
        raise NotImplementedError

    @abstractmethod
    # START_CONTRACT: load_model
    #   PURPOSE: Load a model runtime for the provided specification and return a reusable handle.
    #   INPUTS: { spec: ModelSpec - Model specification to load }
    #   OUTPUTS: { LoadedModelHandle - Loaded backend model handle }
    #   SIDE_EFFECTS: May allocate backend runtime resources and cache loaded models
    #   LINKS: M-BACKENDS
    # END_CONTRACT: load_model
    def load_model(self, spec: ModelSpec) -> LoadedModelHandle:
        raise NotImplementedError

    @abstractmethod
    # START_CONTRACT: inspect_model
    #   PURPOSE: Inspect model availability, artifact completeness, and runtime readiness for the backend.
    #   INPUTS: { spec: ModelSpec - Model specification to inspect }
    #   OUTPUTS: { dict[str, Any] - Structured inspection result for the model }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: inspect_model
    def inspect_model(self, spec: ModelSpec) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    # START_CONTRACT: readiness_diagnostics
    #   PURPOSE: Report backend availability and readiness diagnostics for selection and health checks.
    #   INPUTS: {}
    #   OUTPUTS: { BackendDiagnostics - Structured backend readiness diagnostics }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: readiness_diagnostics
    def readiness_diagnostics(self) -> BackendDiagnostics:
        raise NotImplementedError

    # START_CONTRACT: cache_diagnostics
    #   PURPOSE: Report cache state details for loaded model runtimes.
    #   INPUTS: {}
    #   OUTPUTS: { dict[str, Any] - Structured cache diagnostics }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: cache_diagnostics
    def cache_diagnostics(self) -> dict[str, Any]:
        return {
            "cached_model_count": 0,
            "cached_model_ids": [],
            "cache_policy": {
                "eviction": "not_configured",
            },
            "loaded_models": [],
        }

    # START_CONTRACT: metrics_summary
    #   PURPOSE: Summarize backend-specific model loading and cache metrics.
    #   INPUTS: {}
    #   OUTPUTS: { dict[str, Any] - Backend metrics summary }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: metrics_summary
    def metrics_summary(self) -> dict[str, Any]:
        return {
            "cache": {
                "hit": {},
                "miss": {},
            },
            "load": {
                "failures": {},
                "duration_ms": {},
            },
        }

    # START_CONTRACT: preload_models
    #   PURPOSE: Preload a collection of model specifications into backend runtime memory.
    #   INPUTS: { specs: tuple[ModelSpec, ...] - Model specifications to preload }
    #   OUTPUTS: { dict[str, Any] - Structured preload outcome summary }
    #   SIDE_EFFECTS: May allocate and cache backend runtime resources
    #   LINKS: M-BACKENDS
    # END_CONTRACT: preload_models
    def preload_models(self, specs: tuple[ModelSpec, ...]) -> dict[str, Any]:
        return {
            "requested": 0,
            "attempted": 0,
            "loaded": 0,
            "failed": 0,
            "loaded_model_ids": [],
            "failed_model_ids": [],
            "errors": [],
        }

    def execute(self, request: ExecutionRequest) -> None:
        payload = dict(request.generation_kwargs)
        if request.legacy_mode == "custom":
            self.synthesize_custom(
                request.handle,
                text=request.text,
                output_dir=request.output_dir,
                language=request.language,
                speaker=str(payload.pop("voice")),
                instruct=str(payload.pop("instruct")),
                speed=float(payload.pop("speed")),
            )
            return

        if request.legacy_mode == "design":
            self.synthesize_design(
                request.handle,
                text=request.text,
                output_dir=request.output_dir,
                language=request.language,
                voice_description=str(payload.pop("instruct")),
            )
            return

        if request.legacy_mode == "clone":
            ref_audio = payload.pop("ref_audio")
            self.synthesize_clone(
                request.handle,
                text=request.text,
                output_dir=request.output_dir,
                language=request.language,
                ref_audio_path=Path(str(ref_audio)),
                ref_text=(
                    None
                    if payload.get("ref_text") is None
                    else str(payload.pop("ref_text"))
                ),
            )
            return

        raise NotImplementedError(
            f"Unsupported legacy mode '{request.legacy_mode}' for backend '{self.key}'"
        )

    @abstractmethod
    # START_CONTRACT: synthesize_custom
    #   PURPOSE: Generate custom-voice audio for the provided text and speaker inputs.
    #   INPUTS: { handle: LoadedModelHandle - Loaded backend model handle, text: str - Input text to synthesize, output_dir: Path - Directory for generated artifacts, language: str - Requested language code, speaker: str - Speaker preset or identifier, instruct: str - Additional generation instruction, speed: float - Playback speed modifier }
    #   OUTPUTS: { None - Writes generated audio into the output directory }
    #   SIDE_EFFECTS: Performs backend inference and writes audio artifacts to disk
    #   LINKS: M-BACKENDS
    # END_CONTRACT: synthesize_custom
    def synthesize_custom(
        self,
        handle: LoadedModelHandle,
        *,
        text: str,
        output_dir: Path,
        language: str,
        speaker: str,
        instruct: str,
        speed: float,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    # START_CONTRACT: synthesize_design
    #   PURPOSE: Generate voice-design audio from text and a voice description prompt.
    #   INPUTS: { handle: LoadedModelHandle - Loaded backend model handle, text: str - Input text to synthesize, output_dir: Path - Directory for generated artifacts, language: str - Requested language code, voice_description: str - Natural language description of the target voice }
    #   OUTPUTS: { None - Writes generated audio into the output directory }
    #   SIDE_EFFECTS: Performs backend inference and writes audio artifacts to disk
    #   LINKS: M-BACKENDS
    # END_CONTRACT: synthesize_design
    def synthesize_design(
        self,
        handle: LoadedModelHandle,
        *,
        text: str,
        output_dir: Path,
        language: str,
        voice_description: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    # START_CONTRACT: synthesize_clone
    #   PURPOSE: Generate cloned-voice audio from text and prepared reference audio.
    #   INPUTS: { handle: LoadedModelHandle - Loaded backend model handle, text: str - Input text to synthesize, output_dir: Path - Directory for generated artifacts, language: str - Requested language code, ref_audio_path: Path - Prepared reference audio path, ref_text: str | None - Optional transcription for the reference audio }
    #   OUTPUTS: { None - Writes generated audio into the output directory }
    #   SIDE_EFFECTS: Performs backend inference and writes audio artifacts to disk
    #   LINKS: M-BACKENDS
    # END_CONTRACT: synthesize_clone
    def synthesize_clone(
        self,
        handle: LoadedModelHandle,
        *,
        text: str,
        output_dir: Path,
        language: str,
        ref_audio_path: Path,
        ref_text: str | None,
    ) -> None:
        raise NotImplementedError


__all__ = [
    "ExecutionRequest",
    "LoadedModelHandle",
    "TTSBackend",
]
