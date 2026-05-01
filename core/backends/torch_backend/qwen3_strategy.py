# FILE: core/backends/torch_backend/qwen3_strategy.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide the Qwen3-specific Torch family strategy: lazy runtime loader and per-mode generation calls.
#   SCOPE: Qwen3TTSStrategy implementation, lazy `qwen_tts.Qwen3TTSModel` import, module-level cache for back-compatibility
#   DEPENDS: M-BACKENDS, M-MODELS
#   LINKS: M-BACKENDS
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   Qwen3TTSStrategy - Torch family strategy that drives qwen_tts.Qwen3TTSModel for custom / design / clone modes.
#   Qwen3TTSModel - Module-level cache of the resolved qwen_tts model class (None until loaded).
#   QWEN_MODEL_IMPORT_ERROR - Module-level cache of the captured ImportError, if any.
#   load_qwen_tts_model_cls - Lazy loader returning Qwen3TTSModel or None.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Extracted from monolithic torch_backend.py during Phase 1.4 strategy split]
# END_CHANGE_SUMMARY

from __future__ import annotations

import importlib
from typing import Any

from core.backends.torch_backend.base_strategy import TorchFamilyStrategy

Qwen3TTSModel: Any | None = None
QWEN_MODEL_IMPORT_ERROR: Exception | None = None


# START_CONTRACT: load_qwen_tts_model_cls
#   PURPOSE: Lazily import and cache the Qwen3 runtime model class.
#   INPUTS: {}
#   OUTPUTS: { Any | None - qwen_tts.Qwen3TTSModel class, or None when the import failed }
#   SIDE_EFFECTS: Caches the resolved class (or import error) at module scope on first call
#   LINKS: M-BACKENDS
# END_CONTRACT: load_qwen_tts_model_cls
def load_qwen_tts_model_cls() -> Any | None:
    global Qwen3TTSModel, QWEN_MODEL_IMPORT_ERROR
    if Qwen3TTSModel is not None:
        return Qwen3TTSModel
    if QWEN_MODEL_IMPORT_ERROR is not None:
        return None
    try:
        module = importlib.import_module("qwen_tts")
    except Exception as exc:  # pragma: no cover
        QWEN_MODEL_IMPORT_ERROR = exc
        return None
    model_cls = getattr(module, "Qwen3TTSModel", None)
    if model_cls is None:
        QWEN_MODEL_IMPORT_ERROR = ImportError("qwen_tts does not expose Qwen3TTSModel")
        return None
    Qwen3TTSModel = model_cls
    QWEN_MODEL_IMPORT_ERROR = None
    return Qwen3TTSModel


# START_CONTRACT: Qwen3TTSStrategy
#   PURPOSE: Family strategy backed by qwen_tts.Qwen3TTSModel. Owns load and inference for the Qwen3 family.
#   INPUTS: {}
#   OUTPUTS: { instance - Strategy ready to be registered with TorchBackend }
#   SIDE_EFFECTS: none (model imports happen lazily on first load)
#   LINKS: M-BACKENDS
# END_CONTRACT: Qwen3TTSStrategy
class Qwen3TTSStrategy(TorchFamilyStrategy):
    family_key = "qwen3_tts"
    runtime_dependency = "qwen_tts.Qwen3TTSModel"

    def load_model_class(self) -> Any | None:
        return load_qwen_tts_model_cls()

    def import_error(self) -> Exception | None:
        return QWEN_MODEL_IMPORT_ERROR

    def synthesize_custom(
        self,
        runtime_model: Any,
        *,
        text: str,
        language: str,
        speaker: str,
        instruct: str,
        speed: float,
    ) -> tuple[list[Any], int]:
        return runtime_model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            speed=speed,
        )

    def synthesize_design(
        self,
        runtime_model: Any,
        *,
        text: str,
        language: str,
        voice_description: str,
    ) -> tuple[list[Any], int]:
        return runtime_model.generate_voice_design(
            text=text,
            language=language,
            instruct=voice_description,
        )

    def synthesize_clone(
        self,
        runtime_model: Any,
        *,
        text: str,
        language: str,
        ref_audio: str,
        ref_text: str | None,
    ) -> tuple[list[Any], int]:
        return runtime_model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )


__all__ = [
    "QWEN_MODEL_IMPORT_ERROR",
    "Qwen3TTSModel",
    "Qwen3TTSStrategy",
    "load_qwen_tts_model_cls",
]
