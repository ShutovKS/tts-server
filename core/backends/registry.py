from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Iterable, Sequence

from core.backends.base import TTSBackend
from core.errors import BackendCapabilityError, BackendNotAvailableError
from core.models.catalog import MODEL_SPECS, ModelSpec


@dataclass(frozen=True)
class BackendSelection:
    backend: TTSBackend
    requested_backend: str | None
    auto_selected: bool
    selection_reason: str


class BackendRegistry:
    def __init__(self, backends: Sequence[TTSBackend], *, requested_backend: str | None = None, autoselect: bool = True):
        if not backends:
            raise ValueError("At least one backend must be registered")
        self._backends = {backend.key: backend for backend in backends}
        self._requested_backend = requested_backend
        self._autoselect = autoselect
        self._selection = self._select_backend()

    @property
    def selected_backend(self) -> TTSBackend:
        return self._selection.backend

    @property
    def selection(self) -> BackendSelection:
        return self._selection

    def list_backends(self) -> list[dict[str, object]]:
        selected_key = self.selected_backend.key
        return [
            {
                "key": backend.key,
                "label": backend.label,
                "selected": backend.key == selected_key,
                "platform_supported": backend.supports_platform(),
                "available": backend.is_available(),
                "capabilities": backend.capabilities().to_dict(),
                "diagnostics": backend.readiness_diagnostics().to_dict(),
            }
            for backend in self._backends.values()
        ]

    def get_model_spec(self, model_name: str | None = None, mode: str | None = None) -> ModelSpec:
        if model_name:
            for spec in MODEL_SPECS.values():
                if model_name in {spec.api_name, spec.folder, spec.key}:
                    self.ensure_mode_supported(spec.mode)
                    return spec
            raise BackendNotAvailableError(
                f"Requested model is not available: {model_name}",
                details={"model": model_name, "backend": self.selected_backend.key},
            )

        if mode:
            self.ensure_mode_supported(mode)
            for spec in MODEL_SPECS.values():
                if spec.mode == mode and self.selected_backend.resolve_model_path(spec.folder):
                    return spec
            for spec in MODEL_SPECS.values():
                if spec.mode == mode:
                    return spec
            raise BackendNotAvailableError(
                f"Requested mode is not available: {mode}",
                details={"mode": mode, "backend": self.selected_backend.key},
            )

        raise BackendNotAvailableError(
            "No model or mode was specified",
            details={"backend": self.selected_backend.key},
        )

    def ensure_mode_supported(self, mode: str) -> None:
        if self.selected_backend.capabilities().supports_mode(mode):
            return
        raise BackendCapabilityError(
            f"Backend '{self.selected_backend.key}' does not support mode '{mode}'",
            details={"backend": self.selected_backend.key, "mode": mode},
        )

    def _select_backend(self) -> BackendSelection:
        if self._requested_backend:
            backend = self._backends.get(self._requested_backend)
            if backend is None:
                raise BackendNotAvailableError(
                    f"Configured backend is unknown: {self._requested_backend}",
                    details={"backend": self._requested_backend, "known_backends": sorted(self._backends)},
                )
            return BackendSelection(
                backend=backend,
                requested_backend=self._requested_backend,
                auto_selected=False,
                selection_reason="explicit_config",
            )

        if not self._autoselect:
            first = next(iter(self._backends.values()))
            return BackendSelection(
                backend=first,
                requested_backend=None,
                auto_selected=False,
                selection_reason="first_registered",
            )

        ordered = list(self._prefer_platform_backend(self._backends.values()))
        for backend in ordered:
            if backend.supports_platform() and backend.is_available():
                return BackendSelection(
                    backend=backend,
                    requested_backend=None,
                    auto_selected=True,
                    selection_reason="platform_and_runtime_match",
                )
        for backend in ordered:
            if backend.supports_platform():
                return BackendSelection(
                    backend=backend,
                    requested_backend=None,
                    auto_selected=True,
                    selection_reason="platform_match_runtime_missing",
                )
        first = ordered[0]
        return BackendSelection(
            backend=first,
            requested_backend=None,
            auto_selected=True,
            selection_reason="fallback_first_backend",
        )

    @staticmethod
    def _prefer_platform_backend(backends: Iterable[TTSBackend]) -> Iterable[TTSBackend]:
        current = platform.system().lower()
        if current == "darwin":
            preferred = ["mlx", "torch"]
        else:
            preferred = ["torch", "mlx"]
        ranked = sorted(backends, key=lambda backend: preferred.index(backend.key) if backend.key in preferred else len(preferred))
        return ranked
