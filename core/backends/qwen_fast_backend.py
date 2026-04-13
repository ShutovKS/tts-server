# FILE: core/backends/qwen_fast_backend.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide an additive accelerated Qwen custom-only backend with explicit CUDA/runtime eligibility diagnostics.
#   SCOPE: QwenFastBackend class with dependency checks, model inspection, cache handling, and custom synthesis execution
#   DEPENDS: M-CONFIG, M-ERRORS, M-OBSERVABILITY, M-METRICS
#   LINKS: M-BACKENDS
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   QwenFastBackend - Optional accelerated Qwen custom-only backend with safe readiness gating
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added additive accelerated Qwen custom-only backend with explicit fallback-safe diagnostics]
# END_CHANGE_SUMMARY

from __future__ import annotations

import os
import platform
import inspect
from pathlib import Path
from threading import Lock
from typing import Any

from core.backends.base import LoadedModelHandle, TTSBackend
from core.backends.capabilities import BackendCapabilitySet, BackendDiagnostics
from core.errors import ModelLoadError, TTSGenerationError
from core.metrics import OperationalMetricsRegistry
from core.models.catalog import ModelSpec

try:
    import torch
except ImportError as exc:  # pragma: no cover
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    import faster_qwen3_tts as faster_qwen3_tts_runtime
except ImportError as exc:  # pragma: no cover
    faster_qwen3_tts_runtime = None
    FASTER_QWEN_IMPORT_ERROR = exc
else:
    FASTER_QWEN_IMPORT_ERROR = None


class QwenFastBackend(TTSBackend):
    key = "qwen_fast"
    label = "Qwen Fast CUDA"
    _MIN_TORCH_VERSION = (2, 5, 1)

    def __init__(
        self,
        models_dir: Path,
        *,
        enabled: bool = True,
        metrics: OperationalMetricsRegistry | None = None,
    ):
        self.models_dir = models_dir
        self.enabled = enabled
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
            supports_voice_prompt_cache=True,
            supports_reference_transcription=False,
            supports_preset_speaker_tts=True,
            supports_voice_description_tts=False,
            supports_reference_voice_clone=False,
            preferred_formats=("wav",),
            platforms=("linux", "windows"),
        )

    def is_available(self) -> bool:
        test_mode = self._test_mode()
        if test_mode in {"eligible", "cuda_missing"}:
            return True
        if test_mode == "dependency_missing":
            return False
        return torch is not None and faster_qwen3_tts_runtime is not None

    def supports_platform(self) -> bool:
        if self._test_mode() in {"eligible", "cuda_missing", "dependency_missing"}:
            return True
        return platform.system().lower() in {"linux", "windows"}

    def resolve_model_path(self, folder_name: str) -> Path | None:
        full_path = self.models_dir / folder_name
        if not full_path.exists():
            return None

        snapshots_dir = full_path / "snapshots"
        if snapshots_dir.exists():
            subfolders = sorted(
                path
                for path in snapshots_dir.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            )
            if subfolders:
                return subfolders[0]

        return full_path

    def load_model(self, spec: ModelSpec) -> LoadedModelHandle:
        model_path = self.resolve_model_path(spec.folder)
        if model_path is None:
            raise ModelLoadError(
                f"Fast Qwen model path is unavailable: {spec.folder}",
                details={"model": spec.api_name, "backend": self.key},
            )
        diagnostics = self.readiness_diagnostics()
        if not diagnostics.ready:
            raise ModelLoadError(
                "Fast Qwen backend is not ready",
                details={
                    "model": spec.api_name,
                    "model_path": str(model_path),
                    "backend": self.key,
                    "readiness_reason": diagnostics.reason,
                    "readiness_details": diagnostics.details,
                },
            )

        with self._lock:
            runtime_model = self._cache.get(spec.folder)
            if runtime_model is None:
                self._metrics.collector.increment(
                    "models.cache.miss", tags={"backend": self.key}
                )
                try:
                    runtime_model = self._load_runtime_model(model_path)
                except ModelLoadError:
                    self._metrics.collector.increment(
                        "models.load.failed", tags={"backend": self.key}
                    )
                    raise
                except Exception as exc:  # pragma: no cover
                    self._metrics.collector.increment(
                        "models.load.failed", tags={"backend": self.key}
                    )
                    raise ModelLoadError(
                        str(exc),
                        details={
                            "model": spec.api_name,
                            "model_path": str(model_path),
                            "backend": self.key,
                        },
                    ) from exc
                self._cache[spec.folder] = runtime_model
                self._metrics.collector.observe_timing(
                    "models.load.duration_ms", 0.0, tags={"backend": self.key}
                )
            else:
                self._metrics.collector.increment(
                    "models.cache.hit", tags={"backend": self.key}
                )

        return LoadedModelHandle(
            spec=spec,
            runtime_model=runtime_model,
            resolved_path=model_path,
            backend_key=self.key,
        )

    def inspect_model(self, spec: ModelSpec) -> dict[str, Any]:
        resolved_path = self.resolve_model_path(spec.folder)
        available = resolved_path is not None
        artifact_check = (
            spec.artifact_validation_for_backend(self.key).validate(resolved_path)
            if resolved_path
            else {
                "loadable": False,
                "required_artifacts": [
                    rule.describe()
                    for rule in spec.artifact_validation_for_backend(
                        self.key
                    ).required_rules
                ],
                "missing_artifacts": ["model_directory"],
            }
        )
        diagnostics = self.readiness_diagnostics()
        runtime_ready = bool(
            available and artifact_check["loadable"] and diagnostics.ready
        )
        cached = spec.folder in self._cache
        return {
            "key": spec.key,
            "id": spec.api_name,
            "name": spec.public_name,
            "mode": spec.mode,
            "folder": spec.folder,
            "backend": self.key,
            "configured": True,
            "available": available,
            "loadable": artifact_check["loadable"],
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
            "missing_artifacts": artifact_check["missing_artifacts"],
            "required_artifacts": artifact_check["required_artifacts"],
            "capabilities": self.capabilities().to_dict(),
        }

    def readiness_diagnostics(self) -> BackendDiagnostics:
        platform_supported = self.supports_platform()
        dependency_available = self.enabled and self.is_available()
        cuda_available = self._cuda_available()
        torch_version = self._torch_version_text()
        version_supported = self._torch_version_supported()
        ready = bool(
            self.enabled
            and platform_supported
            and dependency_available
            and cuda_available
            and version_supported
        )
        reason = None
        if not self.enabled:
            reason = "disabled_by_config"
        elif not platform_supported:
            reason = "platform_unsupported"
        elif not dependency_available:
            reason = "runtime_dependency_missing"
        elif not cuda_available:
            reason = "cuda_required"
        elif not version_supported:
            reason = "torch_version_unsupported"
        return BackendDiagnostics(
            backend_key=self.key,
            backend_label=self.label,
            available=dependency_available,
            ready=ready,
            reason=reason,
            details={
                "enabled": self.enabled,
                "test_mode": self._test_mode(),
                "platform_supported": platform_supported,
                "torch_available": torch is not None,
                "faster_qwen3_tts_available": faster_qwen3_tts_runtime is not None,
                "cuda_available": cuda_available,
                "torch_version": torch_version,
                "minimum_torch_version": self._version_text(self._MIN_TORCH_VERSION),
                "torch_version_supported": version_supported,
                "torch_error": None
                if TORCH_IMPORT_ERROR is None
                else str(TORCH_IMPORT_ERROR),
                "faster_qwen3_tts_error": None
                if FASTER_QWEN_IMPORT_ERROR is None
                else str(FASTER_QWEN_IMPORT_ERROR),
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
                "normalized_runtime_dirs": 0,
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
                failed_model_ids.append(spec.api_name)
                errors.append(
                    {
                        "model": spec.api_name,
                        "reason": str(exc),
                        "details": exc.context.to_dict(),
                    }
                )
            else:
                loaded_model_ids.append(spec.api_name)
        return {
            "requested": len(specs),
            "attempted": len(specs),
            "loaded": len(loaded_model_ids),
            "failed": len(failed_model_ids),
            "loaded_model_ids": loaded_model_ids,
            "failed_model_ids": failed_model_ids,
            "errors": errors,
        }

    def synthesize_custom(
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
        runtime_model = handle.runtime_model
        if not hasattr(runtime_model, "generate_custom_voice"):
            raise TTSGenerationError(
                "Fast Qwen runtime does not expose generate_custom_voice",
                details={
                    "backend": self.key,
                    "model": handle.spec.api_name,
                    "runtime_type": type(runtime_model).__name__,
                },
            )
        generate_custom_voice = runtime_model.generate_custom_voice
        call_kwargs = {
            "text": text,
            "language": self._resolve_language(language),
            "speaker": speaker,
            "instruct": instruct,
        }
        try:
            signature = inspect.signature(generate_custom_voice)
        except (TypeError, ValueError):
            signature = None
        if signature is None or "speed" in signature.parameters:
            call_kwargs["speed"] = speed
        wavs, sample_rate = generate_custom_voice(**call_kwargs)
        self._persist_first_wav(output_dir, wavs, sample_rate)

    def synthesize_design(
        self,
        handle: LoadedModelHandle,
        *,
        text: str,
        output_dir: Path,
        language: str,
        voice_description: str,
    ) -> None:
        raise TTSGenerationError(
            "Fast Qwen backend supports custom synthesis only in MVP",
            details={
                "backend": self.key,
                "mode": "design",
                "model": handle.spec.api_name,
            },
        )

    def synthesize_clone(
        self,
        handle: LoadedModelHandle,
        *,
        text: str,
        output_dir: Path,
        language: str,
        ref_audio_path: Path,
        ref_text: str | None,
    ) -> None:
        raise TTSGenerationError(
            "Fast Qwen backend supports custom synthesis only in MVP",
            details={
                "backend": self.key,
                "mode": "clone",
                "model": handle.spec.api_name,
            },
        )

    @classmethod
    def _torch_version_supported(cls) -> bool:
        if cls._test_mode() in {"eligible", "cuda_missing", "dependency_missing"}:
            return True
        if torch is None:
            return False
        version_text = getattr(torch, "__version__", "")
        parsed = cls._parse_version(version_text)
        return parsed >= cls._MIN_TORCH_VERSION

    @staticmethod
    def _parse_version(version_text: str) -> tuple[int, int, int]:
        core = version_text.split("+", maxsplit=1)[0]
        parts = []
        for raw_part in core.split(".")[:3]:
            digits = "".join(character for character in raw_part if character.isdigit())
            parts.append(int(digits or "0"))
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts[:3])

    @classmethod
    def _version_text(cls, version: tuple[int, int, int]) -> str:
        return ".".join(str(part) for part in version)

    @staticmethod
    def _torch_version_text() -> str | None:
        if torch is None:
            return None
        version = getattr(torch, "__version__", None)
        return None if version is None else str(version)

    @staticmethod
    def _cuda_available() -> bool:
        test_mode = QwenFastBackend._test_mode()
        if test_mode == "eligible":
            return True
        if test_mode == "cuda_missing":
            return False
        if torch is None:
            return False
        try:
            return bool(torch.cuda.is_available())
        except Exception:  # pragma: no cover
            return False

    @staticmethod
    def _resolve_language(language: str) -> str:
        return "Auto" if language == "auto" else language

    def _load_runtime_model(self, model_path: Path) -> Any:
        if self._test_mode() == "eligible":
            return self._build_test_runtime_model(model_path)
        if faster_qwen3_tts_runtime is None:
            raise ModelLoadError(
                str(FASTER_QWEN_IMPORT_ERROR or "faster_qwen3_tts is not available"),
                details={
                    "backend": self.key,
                    "model_path": str(model_path),
                    "runtime_dependency": "faster_qwen3_tts",
                },
            )
        faster_runtime_cls = getattr(
            faster_qwen3_tts_runtime, "FasterQwen3TTS", None
        )
        if faster_runtime_cls is not None and hasattr(
            faster_runtime_cls, "from_pretrained"
        ):
            return faster_runtime_cls.from_pretrained(str(model_path))
        runtime_cls = getattr(faster_qwen3_tts_runtime, "Qwen3TTSModel", None)
        if runtime_cls is not None and hasattr(runtime_cls, "from_pretrained"):
            return runtime_cls.from_pretrained(str(model_path))
        runtime_factory = getattr(faster_qwen3_tts_runtime, "from_pretrained", None)
        if callable(runtime_factory):
            return runtime_factory(str(model_path))
        raise ModelLoadError(
            "faster_qwen3_tts runtime does not expose a supported from_pretrained loader",
            details={
                "backend": self.key,
                "model_path": str(model_path),
                "runtime_dependency": "faster_qwen3_tts",
            },
        )

    @staticmethod
    def _test_mode() -> str | None:
        raw = os.getenv("QWEN_TTS_QWEN_FAST_TEST_MODE", "").strip().lower()
        return raw or None

    @staticmethod
    def _build_test_runtime_model(model_path: Path) -> Any:
        class _TestRuntimeModel:
            def __init__(self, path: Path):
                self.model_path = str(path)

            @staticmethod
            def generate_custom_voice(**kwargs):
                return [[0.0] * 480], 24000

        return _TestRuntimeModel(model_path)

    def _persist_first_wav(
        self, output_dir: Path, wavs: list[Any], sample_rate: int
    ) -> None:
        if not wavs:
            raise TTSGenerationError(
                "Fast Qwen backend returned empty audio result",
                details={"backend": self.key, "failure_kind": "empty_audio"},
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "audio_0001.wav"
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover
            raise TTSGenerationError(
                str(exc),
                details={
                    "backend": self.key,
                    "failure_kind": "audio_write_dependency_missing",
                    "output_path": str(target),
                    "runtime_dependency": "soundfile",
                },
            ) from exc

        try:
            sf.write(target, wavs[0], sample_rate)
        except Exception as exc:  # pragma: no cover
            raise TTSGenerationError(
                str(exc),
                details={
                    "backend": self.key,
                    "failure_kind": "audio_write_failed",
                    "output_path": str(target),
                },
            ) from exc


__all__ = ["QwenFastBackend"]
