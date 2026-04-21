# FILE: core/backends/torch_backend.py
# VERSION: 1.1.4
# START_MODULE_CONTRACT
#   PURPOSE: Implement TTS inference using PyTorch framework.
#   SCOPE: TorchBackend execution, model-path resolution, runtime loading, inspection, diagnostics, preload management, and family-aware generation helpers
#   DEPENDS: M-BACKENDS, M-ERRORS, M-METRICS, M-MODELS
#   LINKS: M-BACKENDS
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   TorchBackend - PyTorch CPU/CUDA inference backend
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.4 - Refreshed GRACE module dependencies and added contracts for the backend entry and initialization surfaces]
# END_CHANGE_SUMMARY

from __future__ import annotations

import importlib
import platform
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from core.backends.base import ExecutionRequest, LoadedModelHandle, TTSBackend
from core.backends.capabilities import BackendCapabilitySet, BackendDiagnostics
from core.errors import ModelLoadError, TTSGenerationError
from core.metrics import OperationalMetricsRegistry
from core.models.catalog import ModelSpec

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

Qwen3TTSModel = None
QWEN_MODEL_IMPORT_ERROR = None
OmniVoiceModel = None
OMNIVOICE_IMPORT_ERROR = None
_MIN_CLONE_AUDIO_SECONDS = 1.0


def _load_qwen_tts_model_cls():
    global Qwen3TTSModel, QWEN_MODEL_IMPORT_ERROR
    if Qwen3TTSModel is not None:
        return Qwen3TTSModel
    if QWEN_MODEL_IMPORT_ERROR is not None:
        return None
    try:
        module = importlib.import_module("qwen_tts")
    except Exception as exc:  # pragma: no cover
        QWEN_MODEL_IMPORT_ERROR = exc
        return None
    model_cls = getattr(module, "Qwen3TTSModel", None)
    if model_cls is None:
        QWEN_MODEL_IMPORT_ERROR = ImportError(
            "qwen_tts does not expose Qwen3TTSModel"
        )
        return None
    Qwen3TTSModel = model_cls
    QWEN_MODEL_IMPORT_ERROR = None
    return Qwen3TTSModel


def _load_omnivoice_model_cls():
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
        OMNIVOICE_IMPORT_ERROR = ImportError(
            "omnivoice does not expose OmniVoice"
        )
        return None
    OmniVoiceModel = model_cls
    OMNIVOICE_IMPORT_ERROR = None
    return OmniVoiceModel


# START_CONTRACT: TorchBackend
#   PURPOSE: Provide the PyTorch implementation of the shared TTS backend contract.
#   INPUTS: { models_dir: Path - Root directory containing Torch model folders, metrics: OperationalMetricsRegistry | None - Optional metrics facade for cache and load observations }
#   OUTPUTS: { instance - Torch backend with process-local model cache }
#   SIDE_EFFECTS: none
#   LINKS: M-BACKENDS
# END_CONTRACT: TorchBackend
class TorchBackend(TTSBackend):
    key = "torch"
    label = "PyTorch + Transformers"

    # START_CONTRACT: execute
    #   PURPOSE: Dispatch a prepared execution request into the matching Torch synthesis flow.
    #   INPUTS: { request: ExecutionRequest - Prepared execution request with loaded model handle and generation kwargs }
    #   OUTPUTS: { None - Writes generated audio into the provided output directory }
    #   SIDE_EFFECTS: Performs Torch inference and writes output artifacts to disk
    #   LINKS: M-BACKENDS
    # END_CONTRACT: execute
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
            self._execute_design(
                request.handle,
                text=request.text,
                output_dir=request.output_dir,
                language=request.language,
                voice_description=str(payload.pop("instruct")),
            )
            return
        if request.execution_mode == "clone":
            ref_audio = payload.pop("ref_audio")
            self._execute_clone(
                request.handle,
                text=request.text,
                output_dir=request.output_dir,
                language=request.language,
                ref_audio_path=Path(str(ref_audio)),
                ref_text=None if payload.get("ref_text") is None else str(payload.pop("ref_text")),
            )
            return
        raise TTSGenerationError(
            f"Unsupported execution mode '{request.execution_mode}' for backend '{self.key}'",
            details={"backend": self.key, "mode": request.execution_mode, "model": request.handle.spec.api_name},
        )

    # START_CONTRACT: __init__
    #   PURPOSE: Initialize a process-local Torch backend with model root, cache state, and metrics collection.
    #   INPUTS: { models_dir: Path - Root directory containing backend-loadable model folders, metrics: OperationalMetricsRegistry | None - Optional metrics facade for cache and load observations }
    #   OUTPUTS: { TorchBackend - Torch backend instance ready for readiness checks and model loading }
    #   SIDE_EFFECTS: Allocates in-memory cache and synchronization primitives for backend use
    #   LINKS: M-BACKENDS
    # END_CONTRACT: __init__
    def __init__(
        self, models_dir: Path, *, metrics: OperationalMetricsRegistry | None = None
    ):
        self.models_dir = models_dir
        self._cache: dict[str, Any] = {}
        self._lock = Lock()
        self._metrics = metrics or OperationalMetricsRegistry()

    # START_CONTRACT: capabilities
    #   PURPOSE: Describe the synthesis features and platform coverage supported by the Torch backend.
    #   INPUTS: {}
    #   OUTPUTS: { BackendCapabilitySet - Capability descriptor for the Torch backend }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: capabilities
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

    # START_CONTRACT: is_available
    #   PURPOSE: Report whether Torch runtime dependencies are importable in the current environment.
    #   INPUTS: {}
    #   OUTPUTS: { bool - True when Torch runtime dependencies are available }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: is_available
    def is_available(self) -> bool:
        return torch is not None and any(
            loader() is not None
            for loader in (
                _load_qwen_tts_model_cls,
                _load_omnivoice_model_cls,
            )
        )

    # START_CONTRACT: supports_platform
    #   PURPOSE: Report whether the current platform is supported by the Torch backend.
    #   INPUTS: {}
    #   OUTPUTS: { bool - True on supported Linux, Windows, or Darwin environments }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: supports_platform
    def supports_platform(self) -> bool:
        return platform.system().lower() in {"linux", "windows", "darwin"}

    # START_CONTRACT: resolve_model_path
    #   PURPOSE: Resolve the effective Torch model directory, including Hugging Face snapshot layouts.
    #   INPUTS: { folder_name: str - Model directory name from the manifest }
    #   OUTPUTS: { Path | None - Resolved model directory when present }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: resolve_model_path
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

    # START_CONTRACT: load_model
    #   PURPOSE: Load or reuse a cached Torch runtime model for the provided specification.
    #   INPUTS: { spec: ModelSpec - Model specification to load }
    #   OUTPUTS: { LoadedModelHandle - Loaded Torch model handle }
    #   SIDE_EFFECTS: Allocates runtime resources, updates in-memory cache, and records load metrics
    #   LINKS: M-BACKENDS
    # END_CONTRACT: load_model
    def load_model(self, spec: ModelSpec) -> LoadedModelHandle:
        # START_BLOCK_CHECK_CACHE
        model_path = self.resolve_model_path(spec.folder)
        if model_path is None:
            raise ModelLoadError(
                f"Torch model path is unavailable: {spec.folder}",
                details={"model": spec.api_name, "backend": self.key},
            )
        model_cls = self._model_class_for_spec(spec)
        runtime_dependency = self._runtime_dependency_for_spec(spec)
        if model_cls is None or torch is None:
            raise ModelLoadError(
                str(self._runtime_import_error_for_spec(spec) or TORCH_IMPORT_ERROR),
                details={
                    "model": spec.api_name,
                    "model_path": str(model_path),
                    "runtime_dependency": runtime_dependency,
                    "backend": self.key,
                    "family": spec.family_key,
                },
            )
        with self._lock:
            runtime_model = self._cache.get(spec.folder)
            if runtime_model is None:
                self._metrics.collector.increment(
                    "models.cache.miss", tags={"backend": self.key}
                )
                try:
                    runtime_model = model_cls.from_pretrained(
                        str(model_path),
                        **self._load_kwargs_for_spec(spec),
                    )
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
                            "family": spec.family_key,
                            "runtime_dependency": runtime_dependency,
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
        # END_BLOCK_CHECK_CACHE

        # START_BLOCK_LOAD_FROM_DISK
        return LoadedModelHandle(
            spec=spec,
            runtime_model=runtime_model,
            resolved_path=model_path,
            backend_key=self.key,
        )
        # END_BLOCK_LOAD_FROM_DISK

    # START_CONTRACT: inspect_model
    #   PURPOSE: Inspect Torch model availability, artifact completeness, cache state, and runtime readiness.
    #   INPUTS: { spec: ModelSpec - Model specification to inspect }
    #   OUTPUTS: { dict[str, Any] - Structured model inspection details }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: inspect_model
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
        runtime_ready = bool(
            available
            and artifact_check["loadable"]
            and self.is_available()
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
            "runtime_blockers": [],
        }

    # START_CONTRACT: readiness_diagnostics
    #   PURPOSE: Report Torch backend availability and readiness diagnostics for selection and health checks.
    #   INPUTS: {}
    #   OUTPUTS: { BackendDiagnostics - Structured Torch readiness diagnostics }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: readiness_diagnostics
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
                "qwen_tts_available": _load_qwen_tts_model_cls() is not None,
                "omnivoice_available": _load_omnivoice_model_cls() is not None,
                "torch_error": None
                if TORCH_IMPORT_ERROR is None
                else str(TORCH_IMPORT_ERROR),
                "qwen_tts_error": None
                if QWEN_MODEL_IMPORT_ERROR is None
                else str(QWEN_MODEL_IMPORT_ERROR),
                "omnivoice_error": None
                if OMNIVOICE_IMPORT_ERROR is None
                else str(OMNIVOICE_IMPORT_ERROR),
                "device_map": self._resolve_device_map_name(),
                "dtype": self._resolve_dtype_name(),
            },
        )

    # START_CONTRACT: cache_diagnostics
    #   PURPOSE: Report cached Torch model handles held by the backend.
    #   INPUTS: {}
    #   OUTPUTS: { dict[str, Any] - Structured cache diagnostics for Torch models }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: cache_diagnostics
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

    # START_CONTRACT: metrics_summary
    #   PURPOSE: Summarize Torch backend cache and model loading metrics.
    #   INPUTS: {}
    #   OUTPUTS: { dict[str, Any] - Torch backend metrics summary }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: metrics_summary
    def metrics_summary(self) -> dict[str, Any]:
        return self._metrics.model_summary()

    # START_CONTRACT: preload_models
    #   PURPOSE: Preload a set of Torch model specifications into the backend cache.
    #   INPUTS: { specs: tuple[ModelSpec, ...] - Model specifications to preload }
    #   OUTPUTS: { dict[str, Any] - Structured preload outcome summary }
    #   SIDE_EFFECTS: Loads model runtimes, updates in-memory cache, and records load outcomes
    #   LINKS: M-BACKENDS
    # END_CONTRACT: preload_models
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

    # START_CONTRACT: synthesize_custom
    #   PURPOSE: Generate custom-voice audio through the Torch runtime using speaker and instruction inputs.
    #   INPUTS: { handle: LoadedModelHandle - Loaded Torch model handle, text: str - Input text to synthesize, output_dir: Path - Directory for generated artifacts, language: str - Requested language code, speaker: str - Speaker preset or identifier, instruct: str - Additional generation instruction, speed: float - Playback speed modifier }
    #   OUTPUTS: { None - Writes generated audio into the output directory }
    #   SIDE_EFFECTS: Performs Torch inference and writes audio artifacts to disk
    #   LINKS: M-BACKENDS
    # END_CONTRACT: synthesize_custom
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
        # START_BLOCK_RESOLVE_MODEL
        runtime_model = handle.runtime_model
        family = handle.spec.family_key
        resolved_language = self._resolve_language(language)
        # END_BLOCK_RESOLVE_MODEL
        # START_BLOCK_RUN_TORCH_INFERENCE
        wavs, sr = self._run_custom_generation(
            runtime_model=runtime_model,
            family=family,
            text=text,
            language=resolved_language,
            speaker=speaker,
            instruct=instruct,
            speed=speed,
        )
        # END_BLOCK_RUN_TORCH_INFERENCE
        # START_BLOCK_WRITE_OUTPUT
        self._persist_first_wav(output_dir, wavs, sr)
        # END_BLOCK_WRITE_OUTPUT

    # START_CONTRACT: synthesize_design
    #   PURPOSE: Generate voice-design audio through the Torch runtime from a voice description prompt.
    #   INPUTS: { handle: LoadedModelHandle - Loaded Torch model handle, text: str - Input text to synthesize, output_dir: Path - Directory for generated artifacts, language: str - Requested language code, voice_description: str - Natural language description of the target voice }
    #   OUTPUTS: { None - Writes generated audio into the output directory }
    #   SIDE_EFFECTS: Performs Torch inference and writes audio artifacts to disk
    #   LINKS: M-BACKENDS
    # END_CONTRACT: synthesize_design
    def _execute_design(
        self,
        handle: LoadedModelHandle,
        *,
        text: str,
        output_dir: Path,
        language: str,
        voice_description: str,
    ) -> None:
        wavs, sr = self._run_design_generation(
            runtime_model=handle.runtime_model,
            family=handle.spec.family_key,
            text=text,
            language=self._resolve_language(language),
            voice_description=voice_description,
        )
        self._persist_first_wav(output_dir, wavs, sr)

    # START_CONTRACT: synthesize_clone
    #   PURPOSE: Generate cloned-voice audio through the Torch runtime using prepared reference audio.
    #   INPUTS: { handle: LoadedModelHandle - Loaded Torch model handle, text: str - Input text to synthesize, output_dir: Path - Directory for generated artifacts, language: str - Requested language code, ref_audio_path: Path - Prepared reference audio path, ref_text: str | None - Optional transcription for the reference audio }
    #   OUTPUTS: { None - Writes generated audio into the output directory }
    #   SIDE_EFFECTS: Performs Torch inference and writes audio artifacts to disk
    #   LINKS: M-BACKENDS
    # END_CONTRACT: synthesize_clone
    def _execute_clone(
        self,
        handle: LoadedModelHandle,
        *,
        text: str,
        output_dir: Path,
        language: str,
        ref_audio_path: Path,
        ref_text: str | None,
    ) -> None:
        # START_BLOCK_PREPARE_CLONE_INPUT
        resolved_language = self._resolve_language(language)
        prepared_ref_audio = str(ref_audio_path)
        # END_BLOCK_PREPARE_CLONE_INPUT
        # START_BLOCK_RUN_CLONE_INFERENCE
        wavs, sr = self._run_clone_generation(
            runtime_model=handle.runtime_model,
            family=handle.spec.family_key,
            text=text,
            language=resolved_language,
            ref_audio=prepared_ref_audio,
            ref_text=ref_text,
        )
        self._assert_clone_audio_duration(
            wavs,
            sr,
            text=text,
            family=handle.spec.family_key,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
        )
        # END_BLOCK_RUN_CLONE_INFERENCE
        self._persist_first_wav(output_dir, wavs, sr)

    @staticmethod
    def _model_class_for_spec(spec: ModelSpec):
        mapping = {
            "qwen3_tts": _load_qwen_tts_model_cls,
            "omnivoice": _load_omnivoice_model_cls,
        }
        loader = mapping.get(spec.family_key)
        if loader is None:
            return None
        return loader()

    @staticmethod
    def _runtime_import_error_for_spec(spec: ModelSpec):
        mapping = {
            "qwen3_tts": QWEN_MODEL_IMPORT_ERROR,
            "omnivoice": OMNIVOICE_IMPORT_ERROR,
        }
        return mapping.get(spec.family_key)

    @staticmethod
    def _runtime_dependency_for_spec(spec: ModelSpec) -> str:
        mapping = {
            "qwen3_tts": "qwen_tts.Qwen3TTSModel",
            "omnivoice": "omnivoice.OmniVoice",
        }
        return mapping.get(spec.family_key, "torch_runtime")

    @staticmethod
    def _extra_load_kwargs_for_spec(spec: ModelSpec) -> dict[str, Any]:
        return {}

    def _load_kwargs_for_spec(self, spec: ModelSpec) -> dict[str, Any]:
        kwargs = dict(self._extra_load_kwargs_for_spec(spec))
        if spec.family_key in {"qwen3_tts", "omnivoice"}:
            kwargs["device_map"] = self._resolve_device_map()
            kwargs["dtype"] = self._resolve_dtype()
        return kwargs

    def _run_custom_generation(
        self,
        *,
        runtime_model: Any,
        family: str,
        text: str,
        language: str,
        speaker: str,
        instruct: str,
        speed: float,
    ) -> tuple[list[Any], int]:
        if family == "qwen3_tts":
            return runtime_model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
                speed=speed,
            )
        if family == "omnivoice":
            kwargs: dict[str, Any] = {
                "text": text,
                "speed": speed,
            }
            normalized_language = self._resolve_omnivoice_language(language)
            normalized_instruct = self._normalize_omnivoice_instruct(instruct)
            if normalized_language is not None:
                kwargs["language"] = normalized_language
            if normalized_instruct is not None:
                kwargs["instruct"] = normalized_instruct
            return self._run_omnivoice_generation(runtime_model=runtime_model, **kwargs)
        raise TTSGenerationError(
            "Torch backend does not support custom synthesis for the requested family",
            details={"backend": self.key, "family": family, "mode": "custom"},
        )

    def _run_design_generation(
        self,
        *,
        runtime_model: Any,
        family: str,
        text: str,
        language: str,
        voice_description: str,
    ) -> tuple[list[Any], int]:
        if family == "qwen3_tts":
            return runtime_model.generate_voice_design(
                text=text,
                language=language,
                instruct=voice_description,
            )
        if family == "omnivoice":
            kwargs: dict[str, Any] = {"text": text}
            normalized_language = self._resolve_omnivoice_language(language)
            normalized_instruct = self._normalize_omnivoice_instruct(voice_description)
            if normalized_language is not None:
                kwargs["language"] = normalized_language
            if normalized_instruct is not None:
                kwargs["instruct"] = normalized_instruct
            return self._run_omnivoice_generation(runtime_model=runtime_model, **kwargs)
        raise TTSGenerationError(
            "Torch backend does not support voice design for the requested family",
            details={"backend": self.key, "family": family, "mode": "design"},
        )

    def _run_clone_generation(
        self,
        *,
        runtime_model: Any,
        family: str,
        text: str,
        language: str,
        ref_audio: str,
        ref_text: str | None,
    ) -> tuple[list[Any], int]:
        if family == "qwen3_tts":
            return runtime_model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
        if family == "omnivoice":
            kwargs = {
                "text": text,
                "ref_audio": ref_audio,
            }
            normalized_language = self._resolve_omnivoice_language(language)
            if normalized_language is not None:
                kwargs["language"] = normalized_language
            if ref_text is not None:
                kwargs["ref_text"] = ref_text
            return self._run_omnivoice_generation(runtime_model=runtime_model, **kwargs)
        raise TTSGenerationError(
            "Torch backend does not support voice cloning for the requested family",
            details={"backend": self.key, "family": family, "mode": "clone"},
        )

    def _run_omnivoice_generation(
        self,
        *,
        runtime_model: Any,
        **kwargs: Any,
    ) -> tuple[list[Any], int]:
        wavs = runtime_model.generate(**kwargs)
        sample_rate = self._resolve_omnivoice_sample_rate(runtime_model)
        return list(wavs), sample_rate

    def _resolve_omnivoice_sample_rate(self, runtime_model: Any) -> int:
        audio_tokenizer = getattr(runtime_model, "audio_tokenizer", None)
        tokenizer_config = getattr(audio_tokenizer, "config", None)
        sample_rate = getattr(tokenizer_config, "sample_rate", None)
        if sample_rate is None:
            raise TTSGenerationError(
                "OmniVoice runtime did not expose a sample rate",
                details={
                    "backend": self.key,
                    "family": "omnivoice",
                    "failure_kind": "missing_sample_rate",
                },
            )
        return int(sample_rate)

    @staticmethod
    def _resolve_omnivoice_language(language: str) -> str | None:
        normalized = language.strip().lower()
        if normalized == "auto":
            return None
        return language

    @staticmethod
    def _normalize_omnivoice_instruct(instruct: str) -> str | None:
        normalized = instruct.strip()
        if not normalized:
            return None
        if normalized.lower() == "normal tone":
            return None
        return normalized

    @staticmethod
    def _resolve_language(language: str) -> str:
        return "Auto" if language == "auto" else language

    def _persist_first_wav(
        self, output_dir: Path, wavs: list[Any], sample_rate: int
    ) -> None:
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
                details={
                    "backend": self.key,
                    "failure_kind": "audio_write_failed",
                    "output_path": str(target),
                },
            ) from exc

    def _assert_clone_audio_duration(
        self,
        wavs: list[Any],
        sample_rate: int,
        *,
        text: str,
        family: str,
        ref_audio_path: Path,
        ref_text: str | None,
    ) -> None:
        if not wavs:
            return
        first_wav = np.asarray(wavs[0])
        frame_count = 0 if first_wav.ndim == 0 else int(first_wav.shape[0])
        duration_seconds = frame_count / float(sample_rate) if sample_rate > 0 else 0.0
        if duration_seconds >= _MIN_CLONE_AUDIO_SECONDS:
            return
        raise TTSGenerationError(
            "Voice clone synthesis produced implausibly short audio",
            details={
                "backend": self.key,
                "family": family,
                "failure_kind": "clone_audio_too_short",
                "duration_seconds": round(duration_seconds, 3),
                "minimum_expected_seconds": _MIN_CLONE_AUDIO_SECONDS,
                "text_length": len(text),
                "ref_audio_path": str(ref_audio_path),
                "ref_text_provided": ref_text is not None,
                "hint": "Use a clean spoken reference clip and provide ref_text only when it exactly matches the reference audio transcript.",
            },
        )

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

__all__ = [
    "TorchBackend",
]
