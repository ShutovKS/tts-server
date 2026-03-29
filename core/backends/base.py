from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.backends.capabilities import BackendCapabilitySet, BackendDiagnostics
from core.models.catalog import ModelSpec


@dataclass(frozen=True)
class LoadedModelHandle:
    spec: ModelSpec
    runtime_model: Any
    resolved_path: Path | None
    backend_key: str


class TTSBackend(ABC):
    key: str
    label: str

    @abstractmethod
    def capabilities(self) -> BackendCapabilitySet:
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def supports_platform(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def resolve_model_path(self, folder_name: str) -> Path | None:
        raise NotImplementedError

    @abstractmethod
    def load_model(self, spec: ModelSpec) -> LoadedModelHandle:
        raise NotImplementedError

    @abstractmethod
    def inspect_model(self, spec: ModelSpec) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def readiness_diagnostics(self) -> BackendDiagnostics:
        raise NotImplementedError

    @abstractmethod
    def synthesize_custom(
        self,
        handle: LoadedModelHandle,
        *,
        text: str,
        output_dir: Path,
        speaker: str,
        instruct: str,
        speed: float,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def synthesize_design(
        self,
        handle: LoadedModelHandle,
        *,
        text: str,
        output_dir: Path,
        voice_description: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def synthesize_clone(
        self,
        handle: LoadedModelHandle,
        *,
        text: str,
        output_dir: Path,
        ref_audio_path: Path,
        ref_text: str | None,
    ) -> None:
        raise NotImplementedError
