from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class CoreError(Exception):
    """Base class for reusable core errors."""


@dataclass(frozen=True)
class ErrorContext:
    reason: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {"reason": self.reason}
        if self.details:
            payload.update(self.details)
        return payload


class ModelNotAvailableError(CoreError):
    def __init__(self, model_name: str):
        super().__init__(f"Requested model is not available: {model_name}")
        self.model_name = model_name


class BackendError(CoreError):
    def __init__(self, reason: str, *, details: dict[str, Any] | None = None):
        super().__init__(reason)
        self.context = ErrorContext(reason=reason, details=details)


class BackendNotAvailableError(BackendError):
    pass


class BackendCapabilityError(BackendError):
    pass


class ModelLoadError(CoreError):
    def __init__(self, reason: str, *, details: dict[str, Any] | None = None):
        super().__init__(reason)
        self.context = ErrorContext(reason=reason, details=details)


class TTSGenerationError(CoreError):
    def __init__(self, reason: str, *, details: dict[str, Any] | None = None):
        super().__init__(reason)
        self.context = ErrorContext(reason=reason, details=details)


class InferenceBusyError(CoreError):
    def __init__(self, reason: str = "Inference is already in progress", *, details: dict[str, Any] | None = None):
        super().__init__(reason)
        self.context = ErrorContext(reason=reason, details=details)


class AudioConversionError(CoreError):
    def __init__(self, reason: str, *, details: dict[str, Any] | None = None):
        super().__init__(reason)
        self.context = ErrorContext(reason=reason, details=details)


class AudioArtifactNotFoundError(CoreError):
    def __init__(self, reason: str, *, details: dict[str, Any] | None = None):
        super().__init__(reason)
        self.context = ErrorContext(reason=reason, details=details)
