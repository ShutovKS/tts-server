# FILE: core/backends/torch_backend/dispatcher.py
# VERSION: 2.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Implement the thin TorchBackend dispatcher that routes requests to family-specific strategies.
#   SCOPE: TorchBackend execution, model-path resolution, runtime loading, inspection, diagnostics, preload management; family generation logic now lives in TorchFamilyStrategy implementations.
#   DEPENDS: M-BACKENDS, M-ERRORS, M-METRICS, M-MODELS
#   LINKS: M-BACKENDS
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   TorchBackend - Thin Torch dispatcher; selects a TorchFamilyStrategy by family key and forwards execution.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v2.0.0 - Phase 1.4 strategy split: TorchBackend reduced to a dispatcher; family generation logic moved into TorchFamilyStrategy implementations under core/backends/torch_backend/]
# END_CHANGE_SUMMARY

from __future__ import annotations

import platform
from pathlib import Path
from threading import Lock
from typing import Any

from core.backends.base import ExecutionRequest, LoadedModelHandle, TTSBackend
from core.backends.capabilities import BackendCapabilitySet, BackendDiagnostics
from core.backends.torch_backend._torch_runtime import (
    TORCH_IMPORT_ERROR,
    resolve_device_map,
    resolve_device_map_name,
    resolve_dtype,
    resolve_dtype_name,
    torch,
)
from core.backends.torch_backend.audio_io import (
    assert_clone_audio_duration,
    persist_first_wav,
)
from core.backends.torch_backend.base_strategy import TorchFamilyStrategy
from core.backends.torch_backend.omnivoice_strategy import (
    OMNIVOICE_IMPORT_ERROR,
    OmniVoiceStrategy,
    load_omnivoice_model_cls,
)
from core.backends.torch_backend.qwen3_strategy import (
    QWEN_MODEL_IMPORT_ERROR,
    Qwen3TTSStrategy,
    load_qwen_tts_model_cls,
)
from core.errors import ModelLoadError, TTSGenerationError
from core.metrics import OperationalMetricsRegistry
from core.models.catalog import ModelSpec


# START_CONTRACT: TorchBackend
#   PURPOSE: Provide the PyTorch implementation of the shared TTS backend contract by dispatching to per-family strategies.
#   INPUTS: { models_dir: Path - Root directory containing Torch model folders, metrics: OperationalMetricsRegistry | None - Optional metrics facade, strategies: tuple[TorchFamilyStrategy, ...] | None - Optional override of registered family strategies (defaults to Qwen3 + OmniVoice) }
#   OUTPUTS: { instance - Torch backend with process-local model cache and family strategy registry }
#   SIDE_EFFECTS: none
#   LINKS: M-BACKENDS
# END_CONTRACT: TorchBackend
class TorchBackend(TTSBackend):
    key = "torch"
    label = "PyTorch + Transformers"

    # START_CONTRACT: __init__
    #   PURPOSE: Initialize the dispatcher with models root, metrics, and the registry of family strategies.
    #   INPUTS: { models_dir: Path - Root directory containing backend-loadable model folders, metrics: OperationalMetricsRegistry | None - Optional metrics facade, strategies: tuple[TorchFamilyStrategy, ...] | None - Optional strategy override }
    #   OUTPUTS: { TorchBackend - Dispatcher ready for readiness checks, model loading, and execution }
    #   SIDE_EFFECTS: Allocates in-memory cache, lock primitives, and the {family_key -> strategy} mapping
    #   LINKS: M-BACKENDS
    # END_CONTRACT: __init__
    def __init__(
        self,
        models_dir: Path,
        *,
        metrics: OperationalMetricsRegistry | None = None,
        strategies: tuple[TorchFamilyStrategy, ...] | None = None,
    ):
        self.models_dir = models_dir
        self._cache: dict[str, Any] = {}
        self._lock = Lock()
        self._metrics = metrics or OperationalMetricsRegistry()
        active_strategies = strategies or (Qwen3TTSStrategy(), OmniVoiceStrategy())
        self._strategies: dict[str, TorchFamilyStrategy] = {
            strategy.family_key: strategy for strategy in active_strategies
        }

    # START_CONTRACT: execute
    #   PURPOSE: Dispatch a prepared execution request to the matching family strategy.
    #   INPUTS: { request: ExecutionRequest - Prepared execution request with loaded model handle and generation kwargs }
    #   OUTPUTS: { None - Writes generated audio into the provided output directory }
    #   SIDE_EFFECTS: Performs Torch inference and writes output artifacts to disk
    #   LINKS: M-BACKENDS
    # END_CONTRACT: execute
    def execute(self, request: ExecutionRequest) -> None:
        strategy = self._strategy_for_handle(request.handle, mode=request.execution_mode)
        payload = dict(request.generation_kwargs)
        resolved_language = self._resolve_language(request.language)
        if request.execution_mode == "custom":
            wavs, sample_rate = strategy.synthesize_custom(
                request.handle.runtime_model,
                text=request.text,
                language=resolved_language,
                speaker=str(payload.pop("voice")),
                instruct=str(payload.pop("instruct")),
                speed=float(payload.pop("speed")),
            )
            persist_first_wav(
                backend_key=self.key,
                output_dir=request.output_dir,
                wavs=wavs,
                sample_rate=sample_rate,
            )
            return
        if request.execution_mode == "design":
            wavs, sample_rate = strategy.synthesize_design(
                request.handle.runtime_model,
                text=request.text,
                language=resolved_language,
                voice_description=str(payload.pop("instruct")),
            )
            persist_first_wav(
                backend_key=self.key,
                output_dir=request.output_dir,
                wavs=wavs,
                sample_rate=sample_rate,
            )
            return
        if request.execution_mode == "clone":
            ref_audio = payload.pop("ref_audio")
            ref_audio_path = Path(str(ref_audio))
            ref_text = None if payload.get("ref_text") is None else str(payload.pop("ref_text"))
            wavs, sample_rate = strategy.synthesize_clone(
                request.handle.runtime_model,
                text=request.text,
                language=resolved_language,
                ref_audio=str(ref_audio_path),
                ref_text=ref_text,
            )
            assert_clone_audio_duration(
                backend_key=self.key,
                wavs=wavs,
                sample_rate=sample_rate,
                text=request.text,
                family=request.handle.spec.family_key,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
            )
            persist_first_wav(
                backend_key=self.key,
                output_dir=request.output_dir,
                wavs=wavs,
                sample_rate=sample_rate,
            )
            return
        raise TTSGenerationError(
            f"Unsupported execution mode '{request.execution_mode}' for backend '{self.key}'",
            details={
                "backend": self.key,
                "mode": request.execution_mode,
                "model": request.handle.spec.api_name,
            },
        )

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
    #   PURPOSE: Report whether Torch and at least one registered family runtime are importable.
    #   INPUTS: {}
    #   OUTPUTS: { bool - True when Torch is loaded and at least one strategy reports its runtime as importable }
    #   SIDE_EFFECTS: none
    #   LINKS: M-BACKENDS
    # END_CONTRACT: is_available
    def is_available(self) -> bool:
        return torch is not None and any(
            strategy.load_model_class() is not None for strategy in self._strategies.values()
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
    #   PURPOSE: Load or reuse a cached Torch runtime model for the provided specification using the matching family strategy.
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
        strategy = self._strategy_for_spec(spec)
        model_cls = strategy.load_model_class() if strategy is not None else None
        runtime_dependency = (
            strategy.runtime_dependency if strategy is not None else "torch_runtime"
        )
        if strategy is None or model_cls is None or torch is None:
            import_error = strategy.import_error() if strategy is not None else None
            raise ModelLoadError(
                str(import_error or TORCH_IMPORT_ERROR),
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
                self._metrics.collector.increment("models.cache.miss", tags={"backend": self.key})
                try:
                    runtime_model = model_cls.from_pretrained(
                        str(model_path),
                        **strategy.load_kwargs(
                            device_map=resolve_device_map(),
                            dtype=resolve_dtype(),
                        ),
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
                self._metrics.collector.increment("models.cache.hit", tags={"backend": self.key})
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
                    for rule in spec.artifact_validation_for_backend(self.key).required_rules
                ],
                "missing_artifacts": ["model_directory"],
            }
        )
        runtime_ready = bool(available and artifact_check["loadable"] and self.is_available())
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
                "qwen_tts_available": load_qwen_tts_model_cls() is not None,
                "omnivoice_available": load_omnivoice_model_cls() is not None,
                "torch_error": None if TORCH_IMPORT_ERROR is None else str(TORCH_IMPORT_ERROR),
                "qwen_tts_error": None
                if QWEN_MODEL_IMPORT_ERROR is None
                else str(QWEN_MODEL_IMPORT_ERROR),
                "omnivoice_error": None
                if OMNIVOICE_IMPORT_ERROR is None
                else str(OMNIVOICE_IMPORT_ERROR),
                "device_map": resolve_device_map_name(),
                "dtype": resolve_dtype_name(),
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

    # START_BLOCK_BACKWARD_COMPAT_HELPERS
    # The following private hooks are kept on the dispatcher for back-compatibility
    # with existing tests and external callers that referenced them on the old
    # monolithic TorchBackend (e.g. tests/unit/core/test_torch_backend_clone_guard.py).
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
        assert_clone_audio_duration(
            backend_key=self.key,
            wavs=wavs,
            sample_rate=sample_rate,
            text=text,
            family=family,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
        )

    # END_BLOCK_BACKWARD_COMPAT_HELPERS

    # START_BLOCK_STRATEGY_RESOLUTION
    def _strategy_for_spec(self, spec: ModelSpec) -> TorchFamilyStrategy | None:
        return self._strategies.get(spec.family_key)

    def _strategy_for_handle(self, handle: LoadedModelHandle, *, mode: str) -> TorchFamilyStrategy:
        strategy = self._strategies.get(handle.spec.family_key)
        if strategy is None:
            raise TTSGenerationError(
                f"Torch backend has no strategy for family '{handle.spec.family_key}'",
                details={
                    "backend": self.key,
                    "family": handle.spec.family_key,
                    "mode": mode,
                },
            )
        return strategy

    @staticmethod
    def _resolve_language(language: str) -> str:
        return "Auto" if language == "auto" else language

    # END_BLOCK_STRATEGY_RESOLUTION


__all__ = [
    "TorchBackend",
]
