# FILE: core/engines/omnivoice.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide a production TTSEngine for OmniVoice Torch synthesis while preserving current custom, design, and clone runtime semantics.
#   SCOPE: OmniVoiceTorchEngine availability, model loading, in-memory synthesis, sample-rate validation, and omnivoice runtime compatibility
#   DEPENDS: M-ENGINE-CONTRACTS, M-ERRORS, M-METRICS, M-MODELS, M-BACKENDS, M-OMNIVOICE-FAMILY
#   LINKS: M-ENGINE-CONTRACTS, M-ENGINE-REGISTRY, M-BACKENDS, M-OMNIVOICE-FAMILY
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   OmniVoiceTorchEngine - Production TTSEngine for OmniVoice custom, design, and clone synthesis on the Torch lane.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Task 16: migrated OmniVoice Torch execution into the engine seam while preserving family-prepared payload semantics]
# END_CHANGE_SUMMARY

from __future__ import annotations

import io
import platform
from pathlib import Path
from threading import Lock
from typing import Any

from core.backends.torch_backend._torch_runtime import TORCH_IMPORT_ERROR, torch
from core.backends.torch_backend.omnivoice_strategy import (
    OMNIVOICE_IMPORT_ERROR,
    OmniVoiceStrategy,
    load_omnivoice_model_cls,
)
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


class OmniVoiceTorchEngine(TTSEngine):
    key = "omnivoice-torch"
    label = "OmniVoice Torch Engine"
    aliases = ("omnivoice", "omnivoice-engine", "omnivoice-torch-engine")

    def __init__(self, *, metrics: OperationalMetricsRegistry | None = None) -> None:
        self._cache: dict[str, Any] = {}
        self._lock = Lock()
        self._metrics = metrics or OperationalMetricsRegistry()
        self._strategy = OmniVoiceStrategy()

    # START_CONTRACT: capabilities
    #   PURPOSE: Describe the OmniVoice Torch synthesis envelope supported by this engine.
    #   INPUTS: {}
    #   OUTPUTS: { EngineCapabilities - OmniVoice engine capability summary }
    #   SIDE_EFFECTS: none
    #   LINKS: M-ENGINE-CONTRACTS
    # END_CONTRACT: capabilities
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            families=("omnivoice",),
            backends=("torch",),
            capabilities=(
                "preset_speaker_tts",
                "voice_description_tts",
                "reference_voice_clone",
            ),
            supports_streaming=False,
            supports_batching=False,
        )

    # START_CONTRACT: availability
    #   PURPOSE: Report whether the OmniVoice Torch runtime dependencies are importable on a supported host platform.
    #   INPUTS: {}
    #   OUTPUTS: { EngineAvailability - Structured enabled/available state }
    #   SIDE_EFFECTS: none
    #   LINKS: M-ENGINE-CONTRACTS
    # END_CONTRACT: availability
    def availability(self) -> EngineAvailability:
        platform_supported = platform.system().lower() in {"darwin", "linux", "windows"}
        torch_available = torch is not None
        runtime_available = load_omnivoice_model_cls() is not None
        reason = None
        missing_dependencies: list[str] = []
        if not platform_supported:
            reason = "unsupported_platform"
        elif not torch_available:
            reason = "runtime_dependency_missing"
            missing_dependencies.append("torch")
        elif not runtime_available:
            reason = "runtime_dependency_missing"
            missing_dependencies.append("omnivoice")
        return EngineAvailability(
            engine_key=self.key,
            is_available=platform_supported and torch_available and runtime_available,
            is_enabled=True,
            reason=reason,
            missing_dependencies=tuple(missing_dependencies),
        )

    # START_CONTRACT: load_model
    #   PURPOSE: Load and cache an OmniVoice Torch runtime for an already-resolved manifest model path.
    #   INPUTS: { spec: ModelSpec - OmniVoice model specification, backend_key: str - Requested backend lane, model_path: Path | None - Resolved local model directory }
    #   OUTPUTS: { ModelHandle - Loaded reusable OmniVoice runtime handle }
    #   SIDE_EFFECTS: May allocate and cache OmniVoice runtime state
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
        if spec.family_key != "omnivoice":
            raise ModelLoadError(
                "OmniVoiceTorchEngine only supports OmniVoice family models",
                details={
                    "model": spec.model_id,
                    "family": spec.family_key,
                    "engine": self.key,
                    "backend": backend_key,
                },
            )
        if backend_key != "torch":
            raise ModelLoadError(
                "OmniVoiceTorchEngine requires the Torch backend lane",
                details={
                    "model": spec.model_id,
                    "engine": self.key,
                    "backend": backend_key,
                },
            )
        if model_path is None:
            raise ModelLoadError(
                f"Torch model path is unavailable: {spec.folder}",
                details={"model": spec.model_id, "backend": backend_key, "engine": self.key},
            )
        model_cls = load_omnivoice_model_cls()
        if torch is None or model_cls is None:
            raise ModelLoadError(
                str(OMNIVOICE_IMPORT_ERROR or TORCH_IMPORT_ERROR),
                details={
                    "model": spec.model_id,
                    "model_path": str(model_path),
                    "runtime_dependency": "omnivoice.OmniVoice",
                    "backend": backend_key,
                    "engine": self.key,
                    "family": spec.family_key,
                },
            )
        # END_BLOCK_VALIDATE_MODEL_LOAD_REQUEST

        model_key = spec.folder
        with self._lock:
            runtime_model = self._cache.get(model_key)
            if runtime_model is None:
                self._metrics.collector.increment("models.cache.miss", tags={"backend": backend_key})
                try:
                    runtime_model = model_cls(str(model_path))
                except Exception as exc:  # pragma: no cover
                    self._metrics.collector.increment(
                        "models.load.failed", tags={"backend": backend_key}
                    )
                    raise ModelLoadError(
                        str(exc),
                        details={
                            "model": spec.model_id,
                            "model_path": str(model_path),
                            "backend": backend_key,
                            "engine": self.key,
                            "family": spec.family_key,
                            "runtime_dependency": "omnivoice.OmniVoice",
                        },
                    ) from exc
                self._cache[model_key] = runtime_model
                self._metrics.collector.observe_timing(
                    "models.load.duration_ms", 0.0, tags={"backend": backend_key}
                )
            else:
                self._metrics.collector.increment("models.cache.hit", tags={"backend": backend_key})

        return ModelHandle(
            spec=spec,
            runtime_model=runtime_model,
            resolved_path=model_path,
            engine_key=self.key,
            backend_key=backend_key,
            family_key=spec.family_key,
        )

    # START_CONTRACT: synthesize
    #   PURPOSE: Execute OmniVoice custom, design, or clone synthesis and return in-memory WAV bytes.
    #   INPUTS: { handle: ModelHandle - Loaded OmniVoice runtime handle, job: SynthesisJob - Normalized synthesis request }
    #   OUTPUTS: { AudioBuffer - WAV audio bytes and sample-rate metadata }
    #   SIDE_EFFECTS: Performs OmniVoice Torch inference in-memory
    #   LINKS: M-ENGINE-CONTRACTS, M-BACKENDS
    # END_CONTRACT: synthesize
    def synthesize(self, handle: ModelHandle, job: SynthesisJob) -> AudioBuffer:
        # START_BLOCK_VALIDATE_SYNTHESIS_JOB
        if handle.family_key != "omnivoice":
            raise TTSGenerationError(
                "OmniVoiceTorchEngine received a non-OmniVoice model handle",
                details={
                    "engine": self.key,
                    "model": handle.spec.model_id,
                    "family": handle.family_key,
                },
            )
        # END_BLOCK_VALIDATE_SYNTHESIS_JOB

        payload = dict(job.payload)
        try:
            if job.execution_mode == "custom" and job.capability == "preset_speaker_tts":
                wavs, sample_rate = self._strategy.synthesize_custom(
                    handle.runtime_model,
                    text=job.text,
                    language=job.language,
                    speaker=str(payload.pop("voice")),
                    instruct=str(payload.pop("instruct")),
                    speed=float(payload.pop("speed")),
                )
            elif job.execution_mode == "design" and job.capability == "voice_description_tts":
                wavs, sample_rate = self._strategy.synthesize_design(
                    handle.runtime_model,
                    text=job.text,
                    language=job.language,
                    voice_description=str(payload.pop("instruct")),
                )
            elif job.execution_mode == "clone" and job.capability == "reference_voice_clone":
                ref_audio = payload.pop("ref_audio")
                wavs, sample_rate = self._strategy.synthesize_clone(
                    handle.runtime_model,
                    text=job.text,
                    language=job.language,
                    ref_audio="" if ref_audio is None else str(ref_audio),
                    ref_text=None if payload.get("ref_text") is None else str(payload.pop("ref_text")),
                )
            else:
                raise TTSGenerationError(
                    "OmniVoiceTorchEngine received an unsupported synthesis mode",
                    details={
                        "engine": self.key,
                        "mode": job.execution_mode,
                        "capability": job.capability,
                        "model": handle.spec.model_id,
                    },
                )
        except TTSGenerationError:
            raise
        except Exception as exc:  # pragma: no cover
            raise TTSGenerationError(
                str(exc),
                details={
                    "engine": self.key,
                    "backend": handle.backend_key,
                    "model": handle.spec.model_id,
                    "mode": job.execution_mode,
                },
            ) from exc

        return AudioBuffer(
            waveform=_serialize_first_wav(
                wavs=wavs,
                sample_rate=sample_rate,
                backend_key=handle.backend_key,
                engine_key=self.key,
            ),
            sample_rate=sample_rate,
            audio_format="wav",
        )


def _serialize_first_wav(
    *,
    wavs: list[Any],
    sample_rate: int,
    backend_key: str,
    engine_key: str,
) -> bytes:
    if not wavs:
        raise TTSGenerationError(
            "Torch backend returned empty audio result",
            details={
                "backend": backend_key,
                "engine": engine_key,
                "failure_kind": "empty_audio",
            },
        )
    try:
        import soundfile as sf  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover
        raise TTSGenerationError(
            str(exc),
            details={
                "backend": backend_key,
                "engine": engine_key,
                "failure_kind": "audio_write_dependency_missing",
                "runtime_dependency": "soundfile",
            },
        ) from exc

    wav_buffer = io.BytesIO()
    try:
        sf.write(wav_buffer, wavs[0], sample_rate, format="WAV")
    except Exception as exc:  # pragma: no cover
        raise TTSGenerationError(
            str(exc),
            details={
                "backend": backend_key,
                "engine": engine_key,
                "failure_kind": "audio_write_failed",
            },
        ) from exc
    return wav_buffer.getvalue()


__all__ = ["OmniVoiceTorchEngine"]
