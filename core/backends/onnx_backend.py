# FILE: core/backends/onnx_backend.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Implement local Piper voice synthesis using onnxruntime through the supported piper-tts Python API.
#   SCOPE: ONNXBackend class with Piper model loading, inspection, caching, and synthesis
#   DEPENDS: M-ERRORS, M-METRICS, M-BACKENDS
#   LINKS: M-BACKENDS
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   ONNXBackend - ONNXRuntime backend for Piper local voice models
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added ONNX backend for Piper local voice synthesis]
# END_CHANGE_SUMMARY

from __future__ import annotations

import platform
import wave
from pathlib import Path
from threading import Lock
from typing import Any

from core.backends.base import ExecutionRequest, LoadedModelHandle, TTSBackend
from core.backends.capabilities import BackendCapabilitySet, BackendDiagnostics
from core.errors import ModelLoadError, TTSGenerationError
from core.metrics import OperationalMetricsRegistry
from core.models.catalog import ModelSpec

try:
    from piper import PiperVoice
except ImportError as exc:  # pragma: no cover
    PiperVoice = None
    PIPER_IMPORT_ERROR = exc
else:
    PIPER_IMPORT_ERROR = None


class ONNXBackend(TTSBackend):
    key = "onnx"
    label = "ONNX Runtime"

    def __init__(self, models_dir: Path, *, metrics: OperationalMetricsRegistry | None = None):
        self.models_dir = models_dir
        self._cache: dict[str, Any] = {}
        self._lock = Lock()
        self._metrics = metrics or OperationalMetricsRegistry()

    def capabilities(self) -> BackendCapabilitySet:
        return BackendCapabilitySet(
            supports_custom=True,
            supports_design=False,
            supports_clone=False,
            supports_streaming=False,
            supports_local_models=True,
            supports_voice_prompt_cache=False,
            supports_reference_transcription=False,
            supports_preset_speaker_tts=True,
            supports_voice_description_tts=False,
            supports_reference_voice_clone=False,
            preferred_formats=("wav",),
            platforms=("darwin", "linux", "windows"),
        )

    def is_available(self) -> bool:
        return PiperVoice is not None

    def supports_platform(self) -> bool:
        return platform.system().lower() in {"darwin", "linux", "windows"}

    def resolve_model_path(self, folder_name: str) -> Path | None:
        full_path = self.models_dir / folder_name
        if not full_path.exists():
            return None
        return full_path

    def load_model(self, spec: ModelSpec) -> LoadedModelHandle:
        model_path = self.resolve_model_path(spec.folder)
        if model_path is None:
            raise ModelLoadError(
                f"ONNX model path is unavailable: {spec.folder}",
                details={"model": spec.model_id, "backend": self.key},
            )
        if PiperVoice is None:
            raise ModelLoadError(
                str(PIPER_IMPORT_ERROR),
                details={
                    "model": spec.model_id,
                    "runtime_dependency": "piper-tts",
                    "backend": self.key,
                },
            )

        model_key = spec.folder
        with self._lock:
            voice = self._cache.get(model_key)
            if voice is None:
                self._metrics.collector.increment("models.cache.miss", tags={"backend": self.key})
                try:
                    voice = PiperVoice.load(
                        model_path / "model.onnx",
                        config_path=model_path / "model.onnx.json",
                        use_cuda=False,
                    )
                except Exception as exc:  # pragma: no cover
                    self._metrics.collector.increment(
                        "models.load.failed", tags={"backend": self.key}
                    )
                    raise ModelLoadError(
                        str(exc),
                        details={
                            "model": spec.model_id,
                            "backend": self.key,
                            "model_path": str(model_path),
                        },
                    ) from exc
                self._cache[model_key] = voice
            else:
                self._metrics.collector.increment("models.cache.hit", tags={"backend": self.key})

        return LoadedModelHandle(
            spec=spec,
            runtime_model=voice,
            resolved_path=model_path,
            backend_key=self.key,
        )

    def inspect_model(self, spec: ModelSpec) -> dict[str, Any]:
        resolved_path = self.resolve_model_path(spec.folder)
        available = resolved_path is not None
        required_files = ["model.onnx", "model.onnx.json"]
        missing_artifacts: list[str] = []
        if resolved_path is not None:
            for filename in required_files:
                if not (resolved_path / filename).exists():
                    missing_artifacts.append(filename)
        else:
            missing_artifacts = list(required_files)

        runtime_ready = available and not missing_artifacts and self.is_available()
        cached = spec.folder in self._cache
        return {
            "key": spec.key,
            "id": spec.model_id,
            "name": spec.public_name,
            "mode": spec.mode,
            "folder": spec.folder,
            "backend": self.key,
            "configured": True,
            "available": available,
            "loadable": not missing_artifacts,
            "runtime_ready": runtime_ready,
            "cached": cached,
            "resolved_path": str(resolved_path) if resolved_path else None,
            "runtime_path": str(resolved_path) if resolved_path else None,
            "cache": {
                "loaded": cached,
                "cache_key": spec.folder,
                "backend": self.key,
                "normalized_runtime": False,
                "runtime_path": str(resolved_path) if resolved_path else None,
                "eviction_policy": "not_configured",
            },
            "missing_artifacts": missing_artifacts,
            "required_artifacts": required_files,
            "capabilities": self.capabilities().to_dict(),
        }

    def readiness_diagnostics(self) -> BackendDiagnostics:
        ready = self.supports_platform() and self.is_available()
        reason = None
        if not self.supports_platform():
            reason = "unsupported_platform"
        elif not self.is_available():
            reason = "runtime_dependency_missing"
        return BackendDiagnostics(
            backend_key=self.key,
            backend_label=self.label,
            available=self.is_available(),
            ready=ready,
            reason=reason,
            details={
                "platform_supported": self.supports_platform(),
                "piper_available": PiperVoice is not None,
                "piper_error": None if PIPER_IMPORT_ERROR is None else str(PIPER_IMPORT_ERROR),
            },
        )

    def cache_diagnostics(self) -> dict[str, Any]:
        loaded_models = []
        for folder in sorted(self._cache):
            resolved_path = self.resolve_model_path(folder)
            loaded_models.append(
                {
                    "cache_key": folder,
                    "model_id": folder,
                    "backend": self.key,
                    "loaded": True,
                    "resolved_path": str(resolved_path) if resolved_path else None,
                    "runtime_path": str(resolved_path) if resolved_path else None,
                    "normalized_runtime": False,
                }
            )
        return {
            "cached_model_count": len(loaded_models),
            "cached_model_ids": [item["model_id"] for item in loaded_models],
            "cache_policy": {
                "cache_scope": "process_local",
                "eviction": "not_configured",
            },
            "loaded_models": loaded_models,
        }

    def metrics_summary(self) -> dict[str, Any]:
        return self._metrics.model_summary()

    def preload_models(self, specs: tuple[ModelSpec, ...]) -> dict[str, Any]:
        loaded_model_ids: list[str] = []
        failed_model_ids: list[str] = []
        errors: list[dict[str, Any]] = []
        for spec in specs:
            try:
                self.load_model(spec)
            except ModelLoadError as exc:
                failed_model_ids.append(spec.model_id)
                errors.append(
                    {
                        "model": spec.model_id,
                        "reason": str(exc),
                        "details": exc.context.to_dict(),
                    }
                )
            else:
                loaded_model_ids.append(spec.model_id)
        return {
            "requested": len(specs),
            "attempted": len(specs),
            "loaded": len(loaded_model_ids),
            "failed": len(failed_model_ids),
            "loaded_model_ids": loaded_model_ids,
            "failed_model_ids": failed_model_ids,
            "errors": errors,
        }

    def execute(self, request: ExecutionRequest) -> None:
        payload = dict(request.generation_kwargs)
        if request.execution_mode == "custom":
            self._execute_custom(
                request.handle,
                text=request.text,
                output_dir=request.output_dir,
                language=request.language,
                speaker=str(payload.pop("voice")),
                instruct=str(payload.pop("instruct")),
                speed=float(payload.pop("speed")),
            )
            return
        if request.execution_mode == "design":
            self._execute_design(request.handle)
            return
        if request.execution_mode == "clone":
            self._execute_clone(request.handle)
            return
        raise TTSGenerationError(
            f"Unsupported execution mode '{request.execution_mode}' for backend '{self.key}'",
            details={
                "backend": self.key,
                "mode": request.execution_mode,
                "model": request.handle.spec.model_id,
            },
        )

    def _execute_custom(
        self,
        handle: LoadedModelHandle,
        *,
        text: str,
        output_dir: Path,
        language: str,
        speaker: str,
        instruct: str,
        speed: float,
    ) -> None:
        if not handle.spec.family_key == "piper":
            raise TTSGenerationError(
                "ONNX backend only supports Piper family models for custom synthesis",
                details={"backend": self.key, "model": handle.spec.model_id},
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "audio_0001.wav"
        try:
            with wave.open(str(target), "wb") as wav_file:
                handle.runtime_model.synthesize_wav(text, wav_file)
        except Exception as exc:  # pragma: no cover
            raise TTSGenerationError(
                str(exc),
                details={"backend": self.key, "model": handle.spec.model_id},
            ) from exc

    def _execute_design(self, handle: LoadedModelHandle) -> None:
        raise TTSGenerationError(
            "Piper models do not support voice design synthesis",
            details={"backend": self.key, "model": handle.spec.model_id},
        )

    def _execute_clone(self, handle: LoadedModelHandle) -> None:
        raise TTSGenerationError(
            "Piper models do not support voice cloning synthesis",
            details={"backend": self.key, "model": handle.spec.model_id},
        )


__all__ = ["ONNXBackend"]
