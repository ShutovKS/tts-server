from __future__ import annotations

import platform
from pathlib import Path
from threading import Lock
from typing import Any

from core.backends.base import LoadedModelHandle, TTSBackend
from core.backends.capabilities import BackendCapabilitySet, BackendDiagnostics
from core.errors import ModelLoadError, TTSGenerationError
from core.models.catalog import ModelSpec

try:
    import torch
except ImportError as exc:  # pragma: no cover
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    from qwen_tts import Qwen3TTSModel
except ImportError as exc:  # pragma: no cover
    Qwen3TTSModel = None
    QWEN_TTS_IMPORT_ERROR = exc
else:
    QWEN_TTS_IMPORT_ERROR = None


class TorchBackend(TTSBackend):
    key = "torch"
    label = "PyTorch + Transformers"

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._cache: dict[str, Any] = {}
        self._lock = Lock()

    def capabilities(self) -> BackendCapabilitySet:
        return BackendCapabilitySet(
            supports_custom=True,
            supports_design=True,
            supports_clone=True,
            supports_streaming=False,
            supports_local_models=True,
            supports_voice_prompt_cache=True,
            supports_reference_transcription=False,
            preferred_formats=("wav",),
            platforms=("linux", "windows", "darwin"),
        )

    def is_available(self) -> bool:
        return torch is not None and Qwen3TTSModel is not None

    def supports_platform(self) -> bool:
        return platform.system().lower() in {"linux", "windows", "darwin"}

    def resolve_model_path(self, folder_name: str) -> Path | None:
        full_path = self.models_dir / folder_name
        if not full_path.exists():
            return None

        snapshots_dir = full_path / "snapshots"
        if snapshots_dir.exists():
            subfolders = sorted(path for path in snapshots_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
            if subfolders:
                return subfolders[0]

        return full_path

    def load_model(self, spec: ModelSpec) -> LoadedModelHandle:
        model_path = self.resolve_model_path(spec.folder)
        if model_path is None:
            raise ModelLoadError(
                f"Torch model path is unavailable: {spec.folder}",
                details={"model": spec.api_name, "backend": self.key},
            )
        if Qwen3TTSModel is None or torch is None:
            raise ModelLoadError(
                str(QWEN_TTS_IMPORT_ERROR or TORCH_IMPORT_ERROR),
                details={
                    "model": spec.api_name,
                    "model_path": str(model_path),
                    "runtime_dependency": "qwen_tts.Qwen3TTSModel",
                    "backend": self.key,
                },
            )

        with self._lock:
            runtime_model = self._cache.get(spec.folder)
            if runtime_model is None:
                try:
                    runtime_model = Qwen3TTSModel.from_pretrained(
                        str(model_path),
                        device_map=self._resolve_device_map(),
                        dtype=self._resolve_dtype(),
                    )
                except Exception as exc:  # pragma: no cover
                    raise ModelLoadError(
                        str(exc),
                        details={"model": spec.api_name, "model_path": str(model_path), "backend": self.key},
                    ) from exc
                self._cache[spec.folder] = runtime_model

        return LoadedModelHandle(
            spec=spec,
            runtime_model=runtime_model,
            resolved_path=model_path,
            backend_key=self.key,
        )

    def inspect_model(self, spec: ModelSpec) -> dict[str, Any]:
        resolved_path = self.resolve_model_path(spec.folder)
        available = resolved_path is not None
        artifact_check = self._check_model_artifacts(resolved_path) if resolved_path else {
            "loadable": False,
            "required_artifacts": ["config.json", "model.safetensors|model.safetensors.index.json", "preprocessor_config.json", "tokenizer_config.json|vocab.json"],
            "missing_artifacts": ["model_directory"],
        }
        runtime_ready = bool(available and artifact_check["loadable"] and self.is_available())
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
            "cached": spec.folder in self._cache,
            "resolved_path": str(resolved_path) if resolved_path else None,
            "missing_artifacts": artifact_check["missing_artifacts"],
            "required_artifacts": artifact_check["required_artifacts"],
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
                "torch_available": torch is not None,
                "qwen_tts_available": Qwen3TTSModel is not None,
                "torch_error": None if TORCH_IMPORT_ERROR is None else str(TORCH_IMPORT_ERROR),
                "qwen_tts_error": None if QWEN_TTS_IMPORT_ERROR is None else str(QWEN_TTS_IMPORT_ERROR),
                "device_map": self._resolve_device_map_name(),
                "dtype": self._resolve_dtype_name(),
            },
        )

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
        wavs, sr = handle.runtime_model.generate_custom_voice(
            text=text,
            language="Auto",
            speaker=speaker,
            instruct=instruct,
            speed=speed,
        )
        self._persist_first_wav(output_dir, wavs, sr)

    def synthesize_design(
        self,
        handle: LoadedModelHandle,
        *,
        text: str,
        output_dir: Path,
        voice_description: str,
    ) -> None:
        wavs, sr = handle.runtime_model.generate_voice_design(
            text=text,
            language="Auto",
            instruct=voice_description,
        )
        self._persist_first_wav(output_dir, wavs, sr)

    def synthesize_clone(
        self,
        handle: LoadedModelHandle,
        *,
        text: str,
        output_dir: Path,
        ref_audio_path: Path,
        ref_text: str | None,
    ) -> None:
        wavs, sr = handle.runtime_model.generate_voice_clone(
            text=text,
            language="Auto",
            ref_audio=str(ref_audio_path),
            ref_text=ref_text,
        )
        self._persist_first_wav(output_dir, wavs, sr)

    def _persist_first_wav(self, output_dir: Path, wavs: list[Any], sample_rate: int) -> None:
        if not wavs:
            raise TTSGenerationError(
                "Torch backend returned empty audio result",
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
                details={"backend": self.key, "failure_kind": "audio_write_failed", "output_path": str(target)},
            ) from exc

    @staticmethod
    def _check_model_artifacts(model_path: Path) -> dict[str, Any]:
        requirements = {
            "config.json": model_path / "config.json",
            "model.safetensors|model.safetensors.index.json": [
                model_path / "model.safetensors",
                model_path / "model.safetensors.index.json",
            ],
            "preprocessor_config.json": model_path / "preprocessor_config.json",
            "tokenizer_config.json|vocab.json": [
                model_path / "tokenizer_config.json",
                model_path / "vocab.json",
            ],
        }
        missing: list[str] = []
        for name, requirement in requirements.items():
            if isinstance(requirement, list):
                if not any(path.exists() for path in requirement):
                    missing.append(name)
            elif not requirement.exists():
                missing.append(name)
        return {
            "loadable": not missing,
            "required_artifacts": list(requirements.keys()),
            "missing_artifacts": missing,
        }

    @staticmethod
    def _resolve_device_map() -> str:
        if torch is None:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    @classmethod
    def _resolve_device_map_name(cls) -> str:
        return cls._resolve_device_map()

    @staticmethod
    def _resolve_dtype():
        if torch is None:
            return None
        if torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32

    @classmethod
    def _resolve_dtype_name(cls) -> str | None:
        dtype = cls._resolve_dtype()
        return None if dtype is None else str(dtype).replace("torch.", "")
