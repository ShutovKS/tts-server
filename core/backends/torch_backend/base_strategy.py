# FILE: core/backends/torch_backend/base_strategy.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Define the abstract contract every Torch family-specific execution strategy must implement.
#   SCOPE: TorchFamilyStrategy ABC, lifecycle hooks for runtime-class loading, per-mode generation entry points, and deterministic built-in strategy registration helpers
#   DEPENDS: M-MODELS
#   LINKS: M-BACKENDS
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   TorchFamilyStrategy - Abstract strategy contract. Each Torch family (Qwen3, OmniVoice, ...) implements one.
#   built_in_torch_family_strategies - Instantiates the built-in Qwen3 and OmniVoice Torch family strategies in deterministic order.
#   build_torch_strategy_map - Builds the active {family_key -> strategy} map and rejects duplicate keys.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Task 4 compatibility wiring: added deterministic built-in strategy registration helpers and duplicate-key validation for TorchBackend]
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


# START_CONTRACT: built_in_torch_family_strategies
#   PURPOSE: Instantiate the built-in Torch family strategies in the canonical compatibility order.
#   INPUTS: {}
#   OUTPUTS: { tuple[TorchFamilyStrategy, ...] - Built-in strategy instances for the supported Torch families }
#   SIDE_EFFECTS: none
#   LINKS: M-BACKENDS
# END_CONTRACT: built_in_torch_family_strategies
def built_in_torch_family_strategies() -> tuple[TorchFamilyStrategy, ...]:
    from core.backends.torch_backend.omnivoice_strategy import OmniVoiceStrategy
    from core.backends.torch_backend.qwen3_strategy import Qwen3TTSStrategy

    return (Qwen3TTSStrategy(), OmniVoiceStrategy())


# START_CONTRACT: build_torch_strategy_map
#   PURPOSE: Build the active Torch family strategy map from built-ins plus optional injected strategies, rejecting duplicate family keys deterministically.
#   INPUTS: { strategies: tuple[TorchFamilyStrategy, ...] | None - Optional additional strategies to register alongside the built-in families }
#   OUTPUTS: { dict[str, TorchFamilyStrategy] - Active strategy registry keyed by family_key }
#   SIDE_EFFECTS: none
#   LINKS: M-BACKENDS
# END_CONTRACT: build_torch_strategy_map
def build_torch_strategy_map(
    strategies: tuple[TorchFamilyStrategy, ...] | None = None,
) -> dict[str, TorchFamilyStrategy]:
    registry: dict[str, TorchFamilyStrategy] = {}

    # START_BLOCK_REGISTER_BUILTINS
    for strategy in built_in_torch_family_strategies():
        registry[strategy.family_key] = strategy
    # END_BLOCK_REGISTER_BUILTINS

    # START_BLOCK_REGISTER_INJECTED
    for strategy in strategies or ():
        duplicate_key = strategy.family_key
        if duplicate_key in registry:
            raise ValueError(
                f"Duplicate Torch family strategy registration for family '{duplicate_key}'"
            )
        registry[duplicate_key] = strategy
    # END_BLOCK_REGISTER_INJECTED

    return registry


__all__ = [
    "TorchFamilyStrategy",
    "build_torch_strategy_map",
    "built_in_torch_family_strategies",
]
