# FILE: core/backends/torch_backend/__init__.py
# VERSION: 2.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Public surface of the Torch backend; re-exports the dispatcher and the runtime/strategy symbols that earlier callers imported from the monolithic torch_backend module.
#   SCOPE: barrel re-exports, public TorchBackend dispatcher, family-strategy classes, lazy runtime loaders for back-compatibility
#   DEPENDS: M-BACKENDS
#   LINKS: M-BACKENDS
#   ROLE: BARREL
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   TorchBackend - Thin Torch dispatcher (defined in `dispatcher`).
#   TorchFamilyStrategy - Abstract strategy contract (defined in `base_strategy`).
#   Qwen3TTSStrategy / OmniVoiceStrategy - Concrete family strategies.
#   load_qwen_tts_model_cls / load_omnivoice_model_cls - Lazy runtime-class loaders (re-exported for back-compatibility).
#   Qwen3TTSModel / OmniVoiceModel - Module-level caches of resolved runtime classes (None until first load).
#   TORCH_IMPORT_ERROR / QWEN_MODEL_IMPORT_ERROR / OMNIVOICE_IMPORT_ERROR - Captured ImportError sentinels.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v2.0.0 - Phase 1.4 strategy split: torch_backend.py replaced with a package; family generation logic now lives in TorchFamilyStrategy implementations.]
# END_CHANGE_SUMMARY

from __future__ import annotations

from core.backends.torch_backend._torch_runtime import (
    TORCH_IMPORT_ERROR,
    torch,
)
from core.backends.torch_backend.base_strategy import TorchFamilyStrategy
from core.backends.torch_backend.dispatcher import TorchBackend
from core.backends.torch_backend.omnivoice_strategy import (
    OMNIVOICE_IMPORT_ERROR,
    OmniVoiceModel,
    OmniVoiceStrategy,
    load_omnivoice_model_cls,
)
from core.backends.torch_backend.qwen3_strategy import (
    QWEN_MODEL_IMPORT_ERROR,
    Qwen3TTSModel,
    Qwen3TTSStrategy,
    load_qwen_tts_model_cls,
)

# Back-compat aliases for the underscore-prefixed names used in the old module.
_load_qwen_tts_model_cls = load_qwen_tts_model_cls
_load_omnivoice_model_cls = load_omnivoice_model_cls

__all__ = [
    "OMNIVOICE_IMPORT_ERROR",
    "OmniVoiceModel",
    "OmniVoiceStrategy",
    "QWEN_MODEL_IMPORT_ERROR",
    "Qwen3TTSModel",
    "Qwen3TTSStrategy",
    "TORCH_IMPORT_ERROR",
    "TorchBackend",
    "TorchFamilyStrategy",
    "load_omnivoice_model_cls",
    "load_qwen_tts_model_cls",
    "torch",
]
