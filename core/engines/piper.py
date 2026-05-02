# FILE: core/engines/piper.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide the first production TTSEngine implementation for Piper custom synthesis through ONNX runtime while preserving current manifest and artifact compatibility.
#   SCOPE: PiperOnnxEngine availability, model loading, synthesis, cache handling, and ONNX/Piper artifact validation
#   DEPENDS: M-ENGINE-CONTRACTS, M-ERRORS, M-METRICS, M-MODELS
#   LINKS: M-ENGINE-CONTRACTS, M-ENGINE-REGISTRY, M-BACKENDS
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   PiperOnnxEngine - Production TTSEngine for Piper custom synthesis on the ONNX lane
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Task 10: added the first production TTSEngine for Piper ONNX synthesis behind an explicit runtime opt-in]
# END_CHANGE_SUMMARY

from __future__ import annotations

import io
import platform
import wave
from pathlib import Path
from threading import Lock
from typing import Any

from core.engines.contracts import (
    AudioBuffer,
    EngineAvailability,
    EngineCapabilities,
    ModelHandle,
    SynthesisJob,
    TTSEngine,
)
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


class PiperOnnxEngine(TTSEngine):
    key = "piper-onnx"
    label = "Piper ONNX Engine"
    aliases = ("piper", "piper-engine", "piper-onnx-engine")

    def __init__(self, *, metrics: OperationalMetricsRegistry | None = None) -> None:
        self._cache: dict[str, Any] = {}
        self._lock = Lock()
        self._metrics = metrics or OperationalMetricsRegistry()

    # START_CONTRACT: capabilities
    #   PURPOSE: Describe the Piper-only custom synthesis envelope supported by this engine.
    #   INPUTS: {}
    #   OUTPUTS: { EngineCapabilities - Piper engine capability summary }
    #   SIDE_EFFECTS: none
    #   LINKS: M-ENGINE-CONTRACTS
    # END_CONTRACT: capabilities
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            families=("piper",),
            backends=("onnx",),
            capabilities=("preset_speaker_tts",),
            supports_streaming=False,
            supports_batching=False,
        )

    # START_CONTRACT: availability
    #   PURPOSE: Report whether the Piper runtime dependency is importable on a supported host platform.
    #   INPUTS: {}
    #   OUTPUTS: { EngineAvailability - Structured enabled/available state }
    #   SIDE_EFFECTS: none
    #   LINKS: M-ENGINE-CONTRACTS
    # END_CONTRACT: availability
    def availability(self) -> EngineAvailability:
        platform_supported = platform.system().lower() in {"darwin", "linux", "windows"}
        runtime_available = PiperVoice is not None
        reason = None
        missing_dependencies: tuple[str, ...] = ()
        if not platform_supported:
            reason = "unsupported_platform"
        elif not runtime_available:
            reason = "runtime_dependency_missing"
            missing_dependencies = ("piper-tts",)
        return EngineAvailability(
            engine_key=self.key,
            is_available=platform_supported and runtime_available,
            is_enabled=True,
            reason=reason,
            missing_dependencies=missing_dependencies,
        )

    # START_CONTRACT: load_model
    #   PURPOSE: Load and cache a Piper voice runtime from the manifest-compatible model.onnx layout.
    #   INPUTS: { spec: ModelSpec - Piper model specification, backend_key: str - Requested backend lane, model_path: Path | None - Resolved local model directory }
    #   OUTPUTS: { ModelHandle - Loaded reusable Piper voice handle }
    #   SIDE_EFFECTS: May allocate and cache Piper runtime state
    #   LINKS: M-ENGINE-CONTRACTS, M-BACKENDS
    # END_CONTRACT: load_model
    def load_model(
        self,
        *,
        spec: ModelSpec,
        backend_key: str,
        model_path: Path | None,
    ) -> ModelHandle:
        # START_BLOCK_VALIDATE_MODEL_LOAD_REQUEST
        if spec.family_key != "piper":
            raise ModelLoadError(
                "PiperOnnxEngine only supports Piper family models",
                details={
                    "model": spec.model_id,
                    "family": spec.family_key,
                    "engine": self.key,
                    "backend": backend_key,
                },
            )
        if backend_key != "onnx":
            raise ModelLoadError(
                "PiperOnnxEngine requires the ONNX backend lane",
                details={
                    "model": spec.model_id,
                    "engine": self.key,
                    "backend": backend_key,
                },
            )
        if model_path is None:
            raise ModelLoadError(
                f"ONNX model path is unavailable: {spec.folder}",
                details={"model": spec.model_id, "backend": backend_key, "engine": self.key},
            )
        if PiperVoice is None:
            raise ModelLoadError(
                str(PIPER_IMPORT_ERROR),
                details={
                    "model": spec.model_id,
                    "runtime_dependency": "piper-tts",
                    "backend": backend_key,
                    "engine": self.key,
                },
            )
        # END_BLOCK_VALIDATE_MODEL_LOAD_REQUEST

        # START_BLOCK_VALIDATE_MODEL_ARTIFACTS
        required_files = ("model.onnx", "model.onnx.json")
        missing_artifacts = [filename for filename in required_files if not (model_path / filename).exists()]
        if missing_artifacts:
            raise ModelLoadError(
                "Missing required Piper model artifacts",
                details={
                    "model": spec.model_id,
                    "backend": backend_key,
                    "engine": self.key,
                    "model_path": str(model_path),
                    "missing_artifacts": missing_artifacts,
                    "required_artifacts": list(required_files),
                },
            )
        # END_BLOCK_VALIDATE_MODEL_ARTIFACTS

        model_key = spec.folder
        with self._lock:
            voice = self._cache.get(model_key)
            if voice is None:
                self._metrics.collector.increment("models.cache.miss", tags={"backend": backend_key})
                try:
                    voice = getattr(PiperVoice, "load")(
                        model_path / "model.onnx",
                        config_path=model_path / "model.onnx.json",
                        use_cuda=False,
                    )
                except Exception as exc:  # pragma: no cover
                    self._metrics.collector.increment(
                        "models.load.failed", tags={"backend": backend_key}
                    )
                    raise ModelLoadError(
                        str(exc),
                        details={
                            "model": spec.model_id,
                            "backend": backend_key,
                            "engine": self.key,
                            "model_path": str(model_path),
                        },
                    ) from exc
                self._cache[model_key] = voice
            else:
                self._metrics.collector.increment("models.cache.hit", tags={"backend": backend_key})

        return ModelHandle(
            spec=spec,
            runtime_model=voice,
            resolved_path=model_path,
            engine_key=self.key,
            backend_key=backend_key,
            family_key=spec.family_key,
        )

    # START_CONTRACT: synthesize
    #   PURPOSE: Execute deterministic in-memory Piper synthesis and return a WAV audio buffer.
    #   INPUTS: { handle: ModelHandle - Loaded Piper voice handle, job: SynthesisJob - Normalized custom synthesis request }
    #   OUTPUTS: { AudioBuffer - WAV audio bytes and sample-rate metadata }
    #   SIDE_EFFECTS: Performs Piper inference in-memory
    #   LINKS: M-ENGINE-CONTRACTS, M-BACKENDS
    # END_CONTRACT: synthesize
    def synthesize(self, handle: ModelHandle, job: SynthesisJob) -> AudioBuffer:
        # START_BLOCK_VALIDATE_SYNTHESIS_JOB
        if handle.family_key != "piper":
            raise TTSGenerationError(
                "PiperOnnxEngine received a non-Piper model handle",
                details={
                    "engine": self.key,
                    "model": handle.spec.model_id,
                    "family": handle.family_key,
                },
            )
        if job.execution_mode != "custom" or job.capability != "preset_speaker_tts":
            raise TTSGenerationError(
                "PiperOnnxEngine only supports custom preset-speaker synthesis",
                details={
                    "engine": self.key,
                    "model": handle.spec.model_id,
                    "mode": job.execution_mode,
                    "capability": job.capability,
                },
            )
        # END_BLOCK_VALIDATE_SYNTHESIS_JOB

        # START_BLOCK_EXECUTE_PIPER_SYNTHESIS
        audio_buffer = io.BytesIO()
        try:
            with wave.open(audio_buffer, "wb") as wav_file:
                handle.runtime_model.synthesize_wav(job.text, wav_file)
        except Exception as exc:  # pragma: no cover
            raise TTSGenerationError(
                str(exc),
                details={
                    "engine": self.key,
                    "backend": handle.backend_key,
                    "model": handle.spec.model_id,
                },
            ) from exc
        waveform = audio_buffer.getvalue()
        # END_BLOCK_EXECUTE_PIPER_SYNTHESIS

        return AudioBuffer(
            waveform=waveform,
            sample_rate=_sample_rate_from_wave_bytes(waveform),
            audio_format="wav",
        )


def _sample_rate_from_wave_bytes(wave_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wave_bytes), "rb") as wav_file:
        return int(wav_file.getframerate())


__all__ = ["PiperOnnxEngine"]
