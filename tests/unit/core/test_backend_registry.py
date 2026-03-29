from __future__ import annotations

from pathlib import Path

import pytest

from core.backends.base import LoadedModelHandle, TTSBackend
from core.backends.capabilities import BackendCapabilitySet, BackendDiagnostics
from core.backends.registry import BackendRegistry
from core.errors import BackendCapabilityError, BackendNotAvailableError
from core.models.catalog import MODEL_SPECS


pytestmark = pytest.mark.unit


class StubBackend(TTSBackend):
    def __init__(self, *, key: str, available: bool, platform_supported: bool):
        self.key = key
        self.label = key.upper()
        self._available = available
        self._platform_supported = platform_supported

    def capabilities(self) -> BackendCapabilitySet:
        return BackendCapabilitySet(
            supports_custom=True,
            supports_design=self.key != "clone-only",
            supports_clone=True,
            platforms=("darwin", "linux", "windows"),
        )

    def is_available(self) -> bool:
        return self._available

    def supports_platform(self) -> bool:
        return self._platform_supported

    def resolve_model_path(self, folder_name: str) -> Path | None:
        return Path(".models") / folder_name

    def load_model(self, spec):
        return LoadedModelHandle(spec=spec, runtime_model=object(), resolved_path=Path(".models") / spec.folder, backend_key=self.key)

    def inspect_model(self, spec):
        return {
            "key": spec.key,
            "id": spec.api_name,
            "name": spec.public_name,
            "mode": spec.mode,
            "folder": spec.folder,
            "backend": self.key,
            "configured": True,
            "available": True,
            "loadable": True,
            "runtime_ready": self._available,
            "cached": False,
            "resolved_path": str(Path(".models") / spec.folder),
            "missing_artifacts": [],
            "required_artifacts": ["config.json"],
            "capabilities": self.capabilities().to_dict(),
        }

    def readiness_diagnostics(self) -> BackendDiagnostics:
        return BackendDiagnostics(
            backend_key=self.key,
            backend_label=self.label,
            available=self._available,
            ready=self._available and self._platform_supported,
            reason=None,
            details={},
        )

    def synthesize_custom(self, handle, *, text: str, output_dir: Path, speaker: str, instruct: str, speed: float) -> None:
        return None

    def synthesize_design(self, handle, *, text: str, output_dir: Path, voice_description: str) -> None:
        return None

    def synthesize_clone(self, handle, *, text: str, output_dir: Path, ref_audio_path: Path, ref_text: str | None) -> None:
        return None


def test_backend_registry_prefers_explicit_backend():
    registry = BackendRegistry(
        [
            StubBackend(key="mlx", available=True, platform_supported=True),
            StubBackend(key="torch", available=True, platform_supported=True),
        ],
        requested_backend="torch",
        autoselect=True,
    )

    assert registry.selected_backend.key == "torch"
    assert registry.selection.selection_reason == "explicit_config"


def test_backend_registry_raises_for_unknown_backend():
    with pytest.raises(BackendNotAvailableError):
        BackendRegistry([StubBackend(key="mlx", available=True, platform_supported=True)], requested_backend="unknown", autoselect=True)


def test_backend_registry_rejects_unsupported_mode():
    registry = BackendRegistry([StubBackend(key="clone-only", available=True, platform_supported=True)], autoselect=True)

    with pytest.raises(BackendCapabilityError):
        registry.ensure_mode_supported("design")


def test_backend_registry_lists_selected_backend_metadata():
    registry = BackendRegistry(
        [
            StubBackend(key="mlx", available=False, platform_supported=True),
            StubBackend(key="torch", available=True, platform_supported=True),
        ],
        requested_backend="torch",
        autoselect=True,
    )

    payload = registry.list_backends()

    assert any(item["key"] == "torch" and item["selected"] is True for item in payload)
    assert any(item["key"] == "mlx" and item["selected"] is False for item in payload)


def test_backend_registry_resolves_model_spec_for_mode():
    registry = BackendRegistry([StubBackend(key="torch", available=True, platform_supported=True)], requested_backend="torch", autoselect=True)

    spec = registry.get_model_spec(mode="custom")

    assert spec == MODEL_SPECS["1"]
