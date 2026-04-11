# FILE: core/backends/capabilities.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define backend capability flags and readiness diagnostics.
#   SCOPE: BackendCapabilities dataclass, ReadinessDiagnostics
#   DEPENDS: none
#   LINKS: M-BACKENDS
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   BackendCapabilitySet - Backend feature flags and supported modes
#   BackendDiagnostics - Backend readiness and availability diagnostics
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# START_CONTRACT: BackendCapabilitySet
#   PURPOSE: Describe the functional capabilities and platform support advertised by a backend.
#   INPUTS: { supports_custom: bool - Whether custom voice synthesis is supported, supports_design: bool - Whether voice design synthesis is supported, supports_clone: bool - Whether voice cloning is supported, supports_streaming: bool - Whether streaming output is supported, supports_local_models: bool - Whether local model directories are supported, supports_voice_prompt_cache: bool - Whether reusable prompt caches are supported, supports_reference_transcription: bool - Whether reference audio transcription is supported, supports_preset_speaker_tts: bool - Whether preset-speaker capability is supported, supports_voice_description_tts: bool - Whether voice-description capability is supported, supports_reference_voice_clone: bool - Whether reference-audio cloning capability is supported, preferred_formats: tuple[str, ...] - Preferred audio output formats, platforms: tuple[str, ...] - Supported platform identifiers }
#   OUTPUTS: { instance - Immutable backend capability descriptor }
#   SIDE_EFFECTS: none
#   LINKS: M-BACKENDS
# END_CONTRACT: BackendCapabilitySet
@dataclass(frozen=True)
class BackendCapabilitySet:
    supports_custom: bool
    supports_design: bool
    supports_clone: bool
    supports_streaming: bool = False
    supports_local_models: bool = True
    supports_voice_prompt_cache: bool = False
    supports_reference_transcription: bool = False
    supports_preset_speaker_tts: bool = True
    supports_voice_description_tts: bool = True
    supports_reference_voice_clone: bool = True
    preferred_formats: tuple[str, ...] = ("wav",)
    platforms: tuple[str, ...] = ()

    def supports_mode(self, mode: str) -> bool:
        mapping = {
            "custom": self.supports_custom,
            "design": self.supports_design,
            "clone": self.supports_clone,
        }
        return mapping.get(mode, False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "supports_custom": self.supports_custom,
            "supports_design": self.supports_design,
            "supports_clone": self.supports_clone,
            "supports_streaming": self.supports_streaming,
            "supports_local_models": self.supports_local_models,
            "supports_voice_prompt_cache": self.supports_voice_prompt_cache,
            "supports_reference_transcription": self.supports_reference_transcription,
            "supports_preset_speaker_tts": self.supports_preset_speaker_tts,
            "supports_voice_description_tts": self.supports_voice_description_tts,
            "supports_reference_voice_clone": self.supports_reference_voice_clone,
            "preferred_formats": list(self.preferred_formats),
            "platforms": list(self.platforms),
        }


@dataclass(frozen=True)
class BackendDiagnostics:
    backend_key: str
    backend_label: str
    available: bool
    ready: bool
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend_key,
            "label": self.backend_label,
            "available": self.available,
            "ready": self.ready,
            "reason": self.reason,
            "details": self.details,
        }


__all__ = [
    "BackendCapabilitySet",
    "BackendDiagnostics",
]
