from __future__ import annotations

import json
import platform
import shutil
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any

from core.backends.base import LoadedModelHandle, TTSBackend
from core.backends.capabilities import BackendCapabilitySet, BackendDiagnostics
from core.errors import ModelLoadError, TTSGenerationError
from core.metrics import OperationalMetricsRegistry
from core.models.catalog import ModelSpec
from core.observability import Timer, get_logger, log_event, operation_scope

try:
    from mlx_audio.tts.generate import generate_audio
except ImportError as exc:  # pragma: no cover
    generate_audio = None
    GENERATE_AUDIO_IMPORT_ERROR = exc
else:
    GENERATE_AUDIO_IMPORT_ERROR = None

try:
    from mlx_audio.tts.utils import load_model
except ImportError as exc:  # pragma: no cover
    load_model = None
    LOAD_MODEL_IMPORT_ERROR = exc
else:
    LOAD_MODEL_IMPORT_ERROR = None


LOGGER = get_logger(__name__)
REQUIRED_QWEN3_TALKER_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "intermediate_size",
    "num_attention_heads",
    "rms_norm_eps",
    "vocab_size",
    "num_key_value_heads",
    "max_position_embeddings",
    "rope_theta",
    "head_dim",
)


class MLXBackend(TTSBackend):
    key = "mlx"
    label = "MLX Apple Silicon"

    def __init__(
        self, models_dir: Path, *, metrics: OperationalMetricsRegistry | None = None
    ):
        self.models_dir = models_dir
        self._cache: dict[str, Any] = {}
        self._lock = Lock()
        self._normalized_runtime_dirs: dict[str, Path] = {}
        self._normalized_runtime_tempdirs: dict[
            str, tempfile.TemporaryDirectory[str]
        ] = {}
        self._metrics = metrics or OperationalMetricsRegistry()

    def capabilities(self) -> BackendCapabilitySet:
        return BackendCapabilitySet(
            supports_custom=True,
            supports_design=True,
            supports_clone=True,
            supports_streaming=False,
            supports_local_models=True,
            supports_voice_prompt_cache=False,
            supports_reference_transcription=False,
            preferred_formats=("wav",),
            platforms=("darwin",),
        )

    def is_available(self) -> bool:
        return load_model is not None and generate_audio is not None

    def supports_platform(self) -> bool:
        return platform.system().lower() == "darwin"

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
                f"MLX model path is unavailable: {spec.folder}",
                details={"model": spec.api_name, "backend": self.key},
            )
        if load_model is None:
            raise ModelLoadError(
                str(LOAD_MODEL_IMPORT_ERROR),
                details={
                    "model": spec.api_name,
                    "model_path": str(model_path),
                    "runtime_dependency": "mlx_audio.tts.utils.load_model",
                    "backend": self.key,
                },
            )

        with self._lock:
            runtime_model = self._cache.get(spec.folder)
            if runtime_model is None:
                self._metrics.collector.increment(
                    "models.cache.miss", tags={"backend": self.key}
                )
                runtime_model, runtime_path, used_normalized_runtime = (
                    self._load_runtime_model(spec=spec, model_path=model_path)
                )
                self._cache[spec.folder] = runtime_model
                log_event(
                    LOGGER,
                    level=20,
                    event="mlx_backend.load.completed",
                    message="MLX model runtime loaded",
                    model=spec.api_name,
                    mode=spec.mode,
                    backend=self.key,
                    model_path=str(model_path),
                    runtime_model_path=str(runtime_path),
                    normalized_runtime=used_normalized_runtime,
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
        runtime_ready = bool(
            available and artifact_check["loadable"] and self.is_available()
        )
        cached = spec.folder in self._cache
        normalized_runtime_path = self._normalized_runtime_dirs.get(spec.folder)
        active_runtime_path = normalized_runtime_path or resolved_path
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
            "runtime_path": str(active_runtime_path) if active_runtime_path else None,
            "cache": {
                "loaded": cached,
                "cache_key": spec.folder,
                "backend": self.key,
                "normalized_runtime": normalized_runtime_path is not None,
                "runtime_path": str(active_runtime_path)
                if active_runtime_path
                else None,
                "eviction_policy": "not_configured",
            },
            "missing_artifacts": artifact_check["missing_artifacts"],
            "required_artifacts": artifact_check["required_artifacts"],
            "capabilities": self.capabilities().to_dict(),
        }

    def readiness_diagnostics(self) -> BackendDiagnostics:
        reason = None
        if not self.supports_platform():
            reason = "unsupported_platform"
        elif not self.is_available():
            reason = "runtime_dependency_missing"
        return BackendDiagnostics(
            backend_key=self.key,
            backend_label=self.label,
            available=self.is_available(),
            ready=self.supports_platform() and self.is_available(),
            reason=reason,
            details={
                "platform_supported": self.supports_platform(),
                "load_model_available": load_model is not None,
                "generate_audio_available": generate_audio is not None,
                "load_model_error": None
                if LOAD_MODEL_IMPORT_ERROR is None
                else str(LOAD_MODEL_IMPORT_ERROR),
                "generate_audio_error": None
                if GENERATE_AUDIO_IMPORT_ERROR is None
                else str(GENERATE_AUDIO_IMPORT_ERROR),
            },
        )

    def cache_diagnostics(self) -> dict[str, Any]:
        loaded_models = []
        for folder in sorted(self._cache):
            resolved_path = self.resolve_model_path(folder)
            runtime_path = self._normalized_runtime_dirs.get(folder) or resolved_path
            loaded_models.append(
                {
                    "cache_key": folder,
                    "model_id": folder,
                    "backend": self.key,
                    "loaded": True,
                    "resolved_path": str(resolved_path) if resolved_path else None,
                    "runtime_path": str(runtime_path) if runtime_path else None,
                    "normalized_runtime": folder in self._normalized_runtime_dirs,
                }
            )
        return {
            "cached_model_count": len(loaded_models),
            "cached_model_ids": [item["model_id"] for item in loaded_models],
            "cache_policy": {
                "cache_scope": "process_local",
                "eviction": "not_configured",
                "normalized_runtime_dirs": len(self._normalized_runtime_dirs),
            },
            "loaded_models": loaded_models,
        }

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
        speaker: str,
        instruct: str,
        speed: float,
    ) -> None:
        self._generate(
            handle,
            text=text,
            output_dir=output_dir,
            voice=speaker,
            instruct=instruct,
            speed=speed,
        )

    def synthesize_design(
        self,
        handle: LoadedModelHandle,
        *,
        text: str,
        output_dir: Path,
        voice_description: str,
    ) -> None:
        self._generate(
            handle,
            text=text,
            output_dir=output_dir,
            instruct=voice_description,
        )

    def synthesize_clone(
        self,
        handle: LoadedModelHandle,
        *,
        text: str,
        output_dir: Path,
        ref_audio_path: Path,
        ref_text: str | None,
    ) -> None:
        self._generate(
            handle,
            text=text,
            output_dir=output_dir,
            ref_audio=str(ref_audio_path),
            ref_text=ref_text or ".",
        )

    def _generate(
        self,
        handle: LoadedModelHandle,
        *,
        text: str,
        output_dir: Path,
        **generation_kwargs: Any,
    ) -> None:
        if generate_audio is None:
            raise TTSGenerationError(
                str(GENERATE_AUDIO_IMPORT_ERROR),
                details={
                    "runtime_dependency": "mlx_audio.tts.generate.generate_audio",
                    "model": handle.spec.api_name,
                    "mode": handle.spec.mode,
                    "backend": self.key,
                },
            )
        try:
            generate_audio(
                model=handle.runtime_model,
                text=text,
                output_path=str(output_dir),
                **generation_kwargs,
            )
        except TTSGenerationError:
            raise
        except Exception as exc:  # pragma: no cover
            raise TTSGenerationError(
                str(exc),
                details={
                    "model": handle.spec.api_name,
                    "mode": handle.spec.mode,
                    "backend": self.key,
                },
            ) from exc

    def _load_runtime_model(
        self, *, spec: ModelSpec, model_path: Path
    ) -> tuple[Any, Path, bool]:
        timer = Timer()
        direct_error: Exception | None = None
        try:
            runtime_model = self._invoke_runtime_loader(
                spec=spec,
                model_path=model_path,
                runtime_path=model_path,
                normalized_runtime=False,
            )
            self._observe_load_duration(timer.elapsed_ms)
            return runtime_model, model_path, False
        except Exception as exc:
            direct_error = exc
            if not self._should_retry_with_normalized_runtime(exc):
                self._observe_load_failure()
                raise self._wrap_runtime_load_error(
                    spec=spec,
                    model_path=model_path,
                    runtime_path=model_path,
                    normalized_runtime=False,
                    exc=exc,
                ) from exc
            log_event(
                LOGGER,
                level=20,
                event="mlx_backend.load.normalization_retry",
                message="Retrying MLX model load with normalized Qwen3-TTS config",
                model=spec.api_name,
                mode=spec.mode,
                backend=self.key,
                model_path=str(model_path),
                error=str(exc),
            )

        runtime_path = self._prepare_runtime_model_path(
            spec=spec, model_path=model_path
        )
        try:
            runtime_model = self._invoke_runtime_loader(
                spec=spec,
                model_path=model_path,
                runtime_path=runtime_path,
                normalized_runtime=True,
            )
        except Exception as exc:
            self._observe_load_failure()
            raise self._wrap_runtime_load_error(
                spec=spec,
                model_path=model_path,
                runtime_path=runtime_path,
                normalized_runtime=True,
                exc=exc,
                fallback_reason=None if direct_error is None else str(direct_error),
            ) from exc
        self._observe_load_duration(timer.elapsed_ms)
        return runtime_model, runtime_path, True

    def _invoke_runtime_loader(
        self,
        *,
        spec: ModelSpec,
        model_path: Path,
        runtime_path: Path,
        normalized_runtime: bool,
    ) -> Any:
        log_event(
            LOGGER,
            level=20,
            event="mlx_backend.load.started",
            message="Loading MLX model runtime",
            model=spec.api_name,
            mode=spec.mode,
            backend=self.key,
            model_path=str(model_path),
            runtime_model_path=str(runtime_path),
            normalized_runtime=normalized_runtime,
        )
        with operation_scope("core.backends.mlx.load_model"):
            runtime_model = load_model(str(runtime_path))

        if normalized_runtime:
            runtime_model = self._rebind_runtime_resources(
                runtime_model=runtime_model,
                model_path=model_path,
                runtime_path=runtime_path,
                spec=spec,
            )

        return runtime_model

    def _rebind_runtime_resources(
        self,
        *,
        runtime_model: Any,
        model_path: Path,
        runtime_path: Path,
        spec: ModelSpec,
    ) -> Any:
        model_class = type(runtime_model)
        post_load_hook = getattr(model_class, "post_load_hook", None)
        if post_load_hook is None:
            return runtime_model

        try:
            rebound_model = post_load_hook(runtime_model, model_path)
        except Exception as exc:
            raise ModelLoadError(
                "MLX runtime loaded but failed to bind original model resources",
                details={
                    "model": spec.api_name,
                    "mode": spec.mode,
                    "model_path": str(model_path),
                    "runtime_model_path": str(runtime_path),
                    "normalized_runtime": True,
                    "backend": self.key,
                    "reason": str(exc),
                },
            ) from exc

        log_event(
            LOGGER,
            level=20,
            event="mlx_backend.load.resources_rebound",
            message="Rebound MLX runtime resources to original model path",
            model=spec.api_name,
            mode=spec.mode,
            backend=self.key,
            model_path=str(model_path),
            runtime_model_path=str(runtime_path),
        )
        return rebound_model

    def _wrap_runtime_load_error(
        self,
        *,
        spec: ModelSpec,
        model_path: Path,
        runtime_path: Path,
        normalized_runtime: bool,
        exc: Exception,
        fallback_reason: str | None = None,
    ) -> ModelLoadError:
        log_event(
            LOGGER,
            level=40,
            event="mlx_backend.load.failed",
            message="MLX model runtime load failed",
            model=spec.api_name,
            mode=spec.mode,
            backend=self.key,
            model_path=str(model_path),
            runtime_model_path=str(runtime_path),
            normalized_runtime=normalized_runtime,
            error=str(exc),
            fallback_reason=fallback_reason,
        )
        details = {
            "model": spec.api_name,
            "mode": spec.mode,
            "model_path": str(model_path),
            "runtime_model_path": str(runtime_path),
            "normalized_runtime": normalized_runtime,
            "backend": self.key,
            "loader": "mlx_audio.tts.utils.load_model",
            "reason": str(exc),
        }
        if fallback_reason is not None:
            details["fallback_reason"] = fallback_reason
        return ModelLoadError("MLX runtime failed to load model", details=details)

    def metrics_summary(self) -> dict[str, Any]:
        return self._metrics.model_summary()

    def _observe_load_duration(self, duration_ms: float) -> None:
        self._metrics.collector.observe_timing(
            "models.load.duration_ms", duration_ms, tags={"backend": self.key}
        )

    def _observe_load_failure(self) -> None:
        self._metrics.collector.increment(
            "models.load.failed", tags={"backend": self.key}
        )

    @staticmethod
    def _should_retry_with_normalized_runtime(exc: Exception) -> bool:
        message = str(exc)
        return (
            isinstance(exc, TypeError) and "ModelConfig.__init__() missing" in message
        )

    def _prepare_runtime_model_path(self, *, spec: ModelSpec, model_path: Path) -> Path:
        config = self._read_model_config(spec=spec, model_path=model_path)
        normalized_config = self._normalize_qwen3_tts_config(
            spec=spec, model_path=model_path, config=config
        )
        if normalized_config is None:
            return model_path
        return self._build_normalized_runtime_dir(
            spec=spec, model_path=model_path, normalized_config=normalized_config
        )

    def _read_model_config(
        self, *, spec: ModelSpec, model_path: Path
    ) -> dict[str, Any]:
        config_path = model_path / "config.json"
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ModelLoadError(
                "MLX model config.json is missing",
                details={
                    "model": spec.api_name,
                    "mode": spec.mode,
                    "model_path": str(model_path),
                    "config_path": str(config_path),
                    "backend": self.key,
                },
            ) from exc
        except json.JSONDecodeError as exc:
            raise ModelLoadError(
                "MLX model config.json is not valid JSON",
                details={
                    "model": spec.api_name,
                    "mode": spec.mode,
                    "model_path": str(model_path),
                    "config_path": str(config_path),
                    "backend": self.key,
                    "reason": str(exc),
                },
            ) from exc

    def _normalize_qwen3_tts_config(
        self, *, spec: ModelSpec, model_path: Path, config: dict[str, Any]
    ) -> dict[str, Any] | None:
        if config.get("model_type") != "qwen3_tts":
            return None

        talker_config = config.get("talker_config")
        if talker_config is None:
            raise ModelLoadError(
                "Qwen3-TTS MLX config is missing talker_config",
                details={
                    "model": spec.api_name,
                    "mode": spec.mode,
                    "model_path": str(model_path),
                    "backend": self.key,
                },
            )
        if not isinstance(talker_config, dict):
            raise ModelLoadError(
                "Qwen3-TTS MLX talker_config must be an object",
                details={
                    "model": spec.api_name,
                    "mode": spec.mode,
                    "model_path": str(model_path),
                    "backend": self.key,
                },
            )

        missing_fields = [
            field
            for field in REQUIRED_QWEN3_TALKER_FIELDS
            if field not in talker_config or talker_config[field] is None
        ]
        if missing_fields:
            raise ModelLoadError(
                "Qwen3-TTS MLX talker_config is incomplete",
                details={
                    "model": spec.api_name,
                    "mode": spec.mode,
                    "model_path": str(model_path),
                    "backend": self.key,
                    "missing_fields": missing_fields,
                    "required_fields": list(REQUIRED_QWEN3_TALKER_FIELDS),
                },
            )

        normalized = dict(config)
        normalized.update(talker_config)
        normalized["model_type"] = config.get("model_type")
        normalized["tie_word_embeddings"] = bool(
            talker_config.get("tie_word_embeddings", False)
        )
        normalized["talker_config"] = talker_config
        return normalized

    def _build_normalized_runtime_dir(
        self, *, spec: ModelSpec, model_path: Path, normalized_config: dict[str, Any]
    ) -> Path:
        runtime_dir = self._normalized_runtime_dirs.get(spec.folder)
        if runtime_dir is not None:
            self._write_normalized_config(
                runtime_dir=runtime_dir, normalized_config=normalized_config
            )
            return runtime_dir

        temp_dir = tempfile.TemporaryDirectory(prefix=f"qwen3-tts-mlx-{spec.mode}-")
        runtime_dir = Path(temp_dir.name)
        self._mirror_model_directory(source=model_path, destination=runtime_dir)
        self._write_normalized_config(
            runtime_dir=runtime_dir, normalized_config=normalized_config
        )
        self._normalized_runtime_dirs[spec.folder] = runtime_dir
        self._normalized_runtime_tempdirs[spec.folder] = temp_dir
        log_event(
            LOGGER,
            level=20,
            event="mlx_backend.config.normalized",
            message="Created normalized MLX runtime directory for nested Qwen3-TTS config",
            model=spec.api_name,
            mode=spec.mode,
            backend=self.key,
            model_path=str(model_path),
            runtime_model_path=str(runtime_dir),
        )
        return runtime_dir

    def _mirror_model_directory(self, *, source: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            if child.name == "config.json":
                continue
            target = destination / child.name
            self._link_or_copy(child, target)

    def _write_normalized_config(
        self, *, runtime_dir: Path, normalized_config: dict[str, Any]
    ) -> None:
        config_path = runtime_dir / "config.json"
        config_path.write_text(
            json.dumps(normalized_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _link_or_copy(self, source: Path, destination: Path) -> None:
        if destination.exists() or destination.is_symlink():
            return
        try:
            destination.symlink_to(source, target_is_directory=source.is_dir())
            return
        except OSError:
            pass

        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)

    @staticmethod
    def _check_model_artifacts(model_path: Path) -> dict[str, Any]:
        requirements = {
            "config.json": model_path / "config.json",
            "model.safetensors|model.safetensors.index.json": [
                model_path / "model.safetensors",
                model_path / "model.safetensors.index.json",
            ],
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
