from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BackendCapabilitySet:
    supports_custom: bool
    supports_design: bool
    supports_clone: bool
    supports_streaming: bool = False
    supports_local_models: bool = True
    supports_voice_prompt_cache: bool = False
    supports_reference_transcription: bool = False
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
