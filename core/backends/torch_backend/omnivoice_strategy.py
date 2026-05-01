# FILE: core/backends/torch_backend/omnivoice_strategy.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide the OmniVoice-specific Torch family strategy: lazy runtime loader, language/instruction normalization, and per-mode generation calls.
#   SCOPE: OmniVoiceStrategy implementation, lazy `omnivoice.OmniVoice` import, module-level cache for back-compatibility
#   DEPENDS: M-BACKENDS, M-ERRORS, M-MODELS
#   LINKS: M-BACKENDS
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   OmniVoiceStrategy - Torch family strategy that drives omnivoice.OmniVoice for custom / design / clone modes.
#   OmniVoiceModel - Module-level cache of the resolved omnivoice model class (None until loaded).
#   OMNIVOICE_IMPORT_ERROR - Module-level cache of the captured ImportError, if any.
#   load_omnivoice_model_cls - Lazy loader returning OmniVoiceModel or None.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Extracted from monolithic torch_backend.py during Phase 1.4 strategy split]
# END_CHANGE_SUMMARY

from __future__ import annotations

import importlib
from typing import Any

from core.backends.torch_backend.base_strategy import TorchFamilyStrategy
from core.errors import TTSGenerationError

OmniVoiceModel: Any | None = None
OMNIVOICE_IMPORT_ERROR: Exception | None = None


# START_CONTRACT: load_omnivoice_model_cls
#   PURPOSE: Lazily import and cache the OmniVoice runtime model class.
#   INPUTS: {}
#   OUTPUTS: { Any | None - omnivoice.OmniVoice class, or None when the import failed }
#   SIDE_EFFECTS: Caches the resolved class (or import error) at module scope on first call
#   LINKS: M-BACKENDS
# END_CONTRACT: load_omnivoice_model_cls
def load_omnivoice_model_cls() -> Any | None:
    global OmniVoiceModel, OMNIVOICE_IMPORT_ERROR
    if OmniVoiceModel is not None:
        return OmniVoiceModel
    if OMNIVOICE_IMPORT_ERROR is not None:
        return None
    try:
        module = importlib.import_module("omnivoice")
    except Exception as exc:  # pragma: no cover
        OMNIVOICE_IMPORT_ERROR = exc
        return None
    model_cls = getattr(module, "OmniVoice", None)
    if model_cls is None:
        OMNIVOICE_IMPORT_ERROR = ImportError("omnivoice does not expose OmniVoice")
        return None
    OmniVoiceModel = model_cls
    OMNIVOICE_IMPORT_ERROR = None
    return OmniVoiceModel


# START_CONTRACT: OmniVoiceStrategy
#   PURPOSE: Family strategy backed by omnivoice.OmniVoice. Owns load and inference for the OmniVoice family.
#   INPUTS: {}
#   OUTPUTS: { instance - Strategy ready to be registered with TorchBackend }
#   SIDE_EFFECTS: none (model imports happen lazily on first load)
#   LINKS: M-BACKENDS
# END_CONTRACT: OmniVoiceStrategy
class OmniVoiceStrategy(TorchFamilyStrategy):
    family_key = "omnivoice"
    runtime_dependency = "omnivoice.OmniVoice"

    def load_model_class(self) -> Any | None:
        return load_omnivoice_model_cls()

    def import_error(self) -> Exception | None:
        return OMNIVOICE_IMPORT_ERROR

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
        # OmniVoice ignores `speaker`; it is selected via instruction text.
        kwargs: dict[str, Any] = {"text": text, "speed": speed}
        normalized_language = self._resolve_language(language)
        normalized_instruct = self._normalize_instruct(instruct)
        if normalized_language is not None:
            kwargs["language"] = normalized_language
        if normalized_instruct is not None:
            kwargs["instruct"] = normalized_instruct
        return self._run(runtime_model=runtime_model, **kwargs)

    def synthesize_design(
        self,
        runtime_model: Any,
        *,
        text: str,
        language: str,
        voice_description: str,
    ) -> tuple[list[Any], int]:
        kwargs: dict[str, Any] = {"text": text}
        normalized_language = self._resolve_language(language)
        normalized_instruct = self._normalize_instruct(voice_description)
        if normalized_language is not None:
            kwargs["language"] = normalized_language
        if normalized_instruct is not None:
            kwargs["instruct"] = normalized_instruct
        return self._run(runtime_model=runtime_model, **kwargs)

    def synthesize_clone(
        self,
        runtime_model: Any,
        *,
        text: str,
        language: str,
        ref_audio: str,
        ref_text: str | None,
    ) -> tuple[list[Any], int]:
        kwargs: dict[str, Any] = {"text": text, "ref_audio": ref_audio}
        normalized_language = self._resolve_language(language)
        if normalized_language is not None:
            kwargs["language"] = normalized_language
        if ref_text is not None:
            kwargs["ref_text"] = ref_text
        return self._run(runtime_model=runtime_model, **kwargs)

    # START_BLOCK_OMNIVOICE_HELPERS
    @staticmethod
    def _resolve_language(language: str) -> str | None:
        normalized = language.strip().lower()
        if normalized == "auto":
            return None
        return language

    @staticmethod
    def _normalize_instruct(instruct: str) -> str | None:
        normalized = instruct.strip()
        if not normalized:
            return None
        if normalized.lower() == "normal tone":
            return None
        return normalized

    def _run(self, *, runtime_model: Any, **kwargs: Any) -> tuple[list[Any], int]:
        wavs = runtime_model.generate(**kwargs)
        sample_rate = self._resolve_sample_rate(runtime_model)
        return list(wavs), sample_rate

    @staticmethod
    def _resolve_sample_rate(runtime_model: Any) -> int:
        audio_tokenizer = getattr(runtime_model, "audio_tokenizer", None)
        tokenizer_config = getattr(audio_tokenizer, "config", None)
        sample_rate = getattr(tokenizer_config, "sample_rate", None)
        if sample_rate is None:
            raise TTSGenerationError(
                "OmniVoice runtime did not expose a sample rate",
                details={
                    "backend": "torch",
                    "family": "omnivoice",
                    "failure_kind": "missing_sample_rate",
                },
            )
        return int(sample_rate)

    # END_BLOCK_OMNIVOICE_HELPERS


__all__ = [
    "OMNIVOICE_IMPORT_ERROR",
    "OmniVoiceModel",
    "OmniVoiceStrategy",
    "load_omnivoice_model_cls",
]
