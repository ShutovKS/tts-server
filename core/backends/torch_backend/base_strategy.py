# FILE: core/backends/torch_backend/base_strategy.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define the abstract contract every Torch family-specific execution strategy must implement.
#   SCOPE: TorchFamilyStrategy ABC, lifecycle hooks for runtime-class loading, and per-mode generation entry points
#   DEPENDS: M-MODELS
#   LINKS: M-BACKENDS
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   TorchFamilyStrategy - Abstract strategy contract. Each Torch family (Qwen3, OmniVoice, ...) implements one.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Extracted from monolithic torch_backend.py during Phase 1.4 strategy split]
# END_CHANGE_SUMMARY

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


# START_CONTRACT: TorchFamilyStrategy
#   PURPOSE: Family-specific contract owned by a single model family. Encapsulates the runtime-class loader, optional load-time kwargs, and the three generation entry points (custom / design / clone).
#   INPUTS: { subclass - Concrete strategy class declaring family_key and runtime_dependency }
#   OUTPUTS: { instance - Strategy ready to be plugged into TorchBackend's dispatcher }
#   SIDE_EFFECTS: none
#   LINKS: M-BACKENDS
# END_CONTRACT: TorchFamilyStrategy
class TorchFamilyStrategy(ABC):
    family_key: str = ""
    runtime_dependency: str = "torch_runtime"

    # START_CONTRACT: load_model_class
    #   PURPOSE: Resolve the runtime model class for this family or return None when unavailable.
    #   INPUTS: {}
    #   OUTPUTS: { Any | None - Model class with a `from_pretrained` constructor, or None when imports failed }
    #   SIDE_EFFECTS: May import the family's runtime package on first call and cache it
    #   LINKS: M-BACKENDS
    # END_CONTRACT: load_model_class
    @abstractmethod
    def load_model_class(self) -> Any | None: ...

    # START_CONTRACT: import_error
    #   PURPOSE: Surface the captured ImportError, if any, encountered while loading this family's runtime.
    #   INPUTS: {}
    #   OUTPUTS: { Exception | None - Captured import error, or None when imports succeeded or never ran }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: import_error
    @abstractmethod
    def import_error(self) -> Exception | None: ...

    # START_CONTRACT: load_kwargs
    #   PURPOSE: Build the keyword arguments passed to model_cls.from_pretrained for this family.
    #   INPUTS: { device_map: str - Resolved device_map string (e.g. "cuda:0" or "cpu"), dtype: Any | None - Resolved torch dtype handle, or None }
    #   OUTPUTS: { dict[str, Any] - Keyword arguments forwarded to the runtime model loader }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: load_kwargs
    def load_kwargs(self, *, device_map: str, dtype: Any) -> dict[str, Any]:
        return {"device_map": device_map, "dtype": dtype}

    # START_CONTRACT: synthesize_custom
    #   PURPOSE: Run custom-voice (preset speaker) synthesis through the family's runtime.
    #   INPUTS: { runtime_model: Any - Loaded family runtime, text: str - Input text, language: str - Resolved language code, speaker: str - Preset speaker identifier, instruct: str - Generation instruction string, speed: float - Playback speed modifier }
    #   OUTPUTS: { tuple[list[Any], int] - (waveforms, sample_rate) }
    #   SIDE_EFFECTS: Performs Torch inference on the runtime model
    #   LINKS: M-BACKENDS
    # END_CONTRACT: synthesize_custom
    @abstractmethod
    def synthesize_custom(
        self,
        runtime_model: Any,
        *,
        text: str,
        language: str,
        speaker: str,
        instruct: str,
        speed: float,
    ) -> tuple[list[Any], int]: ...

    # START_CONTRACT: synthesize_design
    #   PURPOSE: Run voice-design synthesis from a free-form voice description prompt.
    #   INPUTS: { runtime_model: Any - Loaded family runtime, text: str - Input text, language: str - Resolved language code, voice_description: str - Natural language description of the target voice }
    #   OUTPUTS: { tuple[list[Any], int] - (waveforms, sample_rate) }
    #   SIDE_EFFECTS: Performs Torch inference on the runtime model
    #   LINKS: M-BACKENDS
    # END_CONTRACT: synthesize_design
    @abstractmethod
    def synthesize_design(
        self,
        runtime_model: Any,
        *,
        text: str,
        language: str,
        voice_description: str,
    ) -> tuple[list[Any], int]: ...

    # START_CONTRACT: synthesize_clone
    #   PURPOSE: Run voice-clone synthesis from prepared reference audio.
    #   INPUTS: { runtime_model: Any - Loaded family runtime, text: str - Input text, language: str - Resolved language code, ref_audio: str - Path to the prepared reference audio file, ref_text: str | None - Optional reference transcript }
    #   OUTPUTS: { tuple[list[Any], int] - (waveforms, sample_rate) }
    #   SIDE_EFFECTS: Performs Torch inference on the runtime model
    #   LINKS: M-BACKENDS
    # END_CONTRACT: synthesize_clone
    @abstractmethod
    def synthesize_clone(
        self,
        runtime_model: Any,
        *,
        text: str,
        language: str,
        ref_audio: str,
        ref_text: str | None,
    ) -> tuple[list[Any], int]: ...


__all__ = [
    "TorchFamilyStrategy",
]
