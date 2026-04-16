# FILE: tests/support/api_fakes.py
# VERSION: 1.2.0
# START_MODULE_CONTRACT
#   PURPOSE: Test doubles and helper builders for API, readiness, and runtime-validation tests.
#   SCOPE: Fake registries, stub TTS services, subprocess doubles, readiness payload helpers, WAV fixture helpers
#   DEPENDS: M-SERVER, M-CORE
#   LINKS: none
#   ROLE: TEST
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   DummyRegistry - Fake model registry for API and readiness tests
#   DummyTTSService - Fake synthesis service returning deterministic generation results
#   FailingTTSService - Fake synthesis service that raises model load failures
#   BusyTTSService - Fake synthesis service that raises inference busy errors
#   MissingModelTTSService - Fake synthesis service that raises unknown-model errors
#   MissingModeTTSService - Fake synthesis service that raises missing-mode errors
#   SlowTTSService - Fake synthesis service that sleeps to exercise timeout paths
#   WorkerFailingTTSService - Fake synthesis service that simulates worker execution failures
#   DegradedRegistry - Fake registry that reports degraded readiness state
#   ManagedProcessDouble - Deterministic subprocess double for start/stop orchestration tests
#   make_validation_model_entry - Build a minimal readiness item for validation harness tests
#   make_validation_self_check_payload - Build a minimal self-check payload for validation harness tests
#   make_wav_bytes - Deterministic WAV fixture helper
#   extract_json_logs - Structured log extraction helper for assertions
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.2.0 - Expanded validation payload doubles for representative-model preflight and routing semantics]
# END_CHANGE_SUMMARY

from __future__ import annotations

import io
import json
import time
import wave
from pathlib import Path
from typing import cast

import pytest

from core.backends.base import TTSBackend
from core.contracts.results import AudioResult, GenerationResult
from core.errors import (
    InferenceBusyError,
    ModelLoadError,
    ModelNotAvailableError,
    TTSGenerationError,
)
from core.models.catalog import MODEL_SPECS
from profiles import ProfileResolver
from core.services.model_registry import ModelRegistry
from server.bootstrap import ServerSettings


class DummyRegistry(ModelRegistry):
    def __init__(self, settings: ServerSettings):
        self.settings = settings
        resolver = ProfileResolver(Path(__file__).resolve().parents[2])
        self._qwen_profile = resolver.get_family_profile("qwen").to_dict()
        self._piper_profile = resolver.get_family_profile("piper").to_dict()

    @property
    def backend(self) -> TTSBackend:
        return cast(
            TTSBackend,
            type("BackendStub", (), {"key": "mlx", "label": "MLX Apple Silicon"})(),
        )

    def list_models(self) -> list[dict]:
        return [
            {
                "key": "1",
                "id": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                "name": "Custom Voice",
                "mode": "custom",
                "family": "Qwen3-TTS",
                "family_key": "qwen3_tts",
                "capabilities_supported": ["preset_speaker_tts"],
                "backend_support": ["mlx", "qwen_fast", "torch"],
                "profile": self._qwen_profile,
                "folder": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                "available": True,
                "loadable": True,
                "runtime_ready": True,
                "backend": "mlx",
                "selected_backend": "mlx",
                "selected_backend_label": "MLX Apple Silicon",
                "execution_backend": "mlx",
                "execution_backend_label": "MLX Apple Silicon",
                "capabilities": {
                    "supports_custom": True,
                    "supports_design": True,
                    "supports_clone": True,
                },
                "route": {
                    "selected_backend": "mlx",
                    "selected_backend_label": "MLX Apple Silicon",
                    "selected_backend_compatible_with_model": True,
                    "selected_backend_ready_for_model": True,
                    "execution_backend": "mlx",
                    "execution_backend_label": "MLX Apple Silicon",
                    "routing_mode": "selected_backend",
                    "route_reason": "selected_backend_supports_model",
                    "candidates": [
                        {
                            "key": "qwen_fast",
                            "label": "Qwen Fast CUDA",
                            "selected": False,
                            "compatible_with_model": True,
                            "supports_mode": True,
                            "platform_supported": False,
                            "available": False,
                            "ready": False,
                            "host_reason": "platform_unsupported",
                            "selection_score": 0,
                            "route_reason": "platform_unsupported",
                            "diagnostics": {
                                "backend": "qwen_fast",
                                "label": "Qwen Fast CUDA",
                                "available": False,
                                "ready": False,
                                "reason": "platform_unsupported",
                                "details": {
                                    "enabled": True,
                                    "test_mode": None,
                                },
                            },
                        }
                    ],
                },
                "missing_artifacts": [],
                "required_artifacts": [
                    "config.json",
                    "model.safetensors|model.safetensors.index.json",
                    "tokenizer_config.json|vocab.json",
                ],
            },
            {
                "key": "3",
                "id": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
                "name": "Voice Cloning",
                "mode": "clone",
                "family": "Qwen3-TTS",
                "family_key": "qwen3_tts",
                "capabilities_supported": ["reference_voice_clone"],
                "backend_support": ["mlx", "torch"],
                "profile": self._qwen_profile,
                "folder": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
                "available": True,
                "loadable": True,
                "runtime_ready": True,
                "backend": "mlx",
                "selected_backend": "mlx",
                "selected_backend_label": "MLX Apple Silicon",
                "execution_backend": "mlx",
                "execution_backend_label": "MLX Apple Silicon",
                "capabilities": {
                    "supports_custom": True,
                    "supports_design": True,
                    "supports_clone": True,
                },
                "route": {
                    "selected_backend": "mlx",
                    "selected_backend_label": "MLX Apple Silicon",
                    "selected_backend_compatible_with_model": True,
                    "selected_backend_ready_for_model": True,
                    "execution_backend": "mlx",
                    "execution_backend_label": "MLX Apple Silicon",
                    "routing_mode": "selected_backend",
                    "route_reason": "selected_backend_supports_model",
                    "candidates": [],
                },
                "missing_artifacts": [],
                "required_artifacts": [
                    "config.json",
                    "model.safetensors|model.safetensors.index.json",
                    "tokenizer_config.json|vocab.json",
                ],
            },
            {
                "key": "piper-1",
                "id": "Piper-en_US-lessac-medium",
                "name": "Piper Lessac",
                "mode": "custom",
                "family": "Piper",
                "family_key": "piper",
                "capabilities_supported": ["preset_speaker_tts"],
                "profile": self._piper_profile,
                "backend_support": ["onnx"],
                "folder": "Piper-en_US-lessac-medium",
                "available": False,
                "loadable": False,
                "runtime_ready": False,
                "backend": "onnx",
                "selected_backend": "mlx",
                "selected_backend_label": "MLX Apple Silicon",
                "execution_backend": "onnx",
                "execution_backend_label": "ONNX Runtime",
                "capabilities": {
                    "supports_custom": True,
                    "supports_design": False,
                    "supports_clone": False,
                    "supports_preset_speaker_tts": True,
                    "supports_voice_description_tts": False,
                    "supports_reference_voice_clone": False,
                },
                "route": {
                    "selected_backend": "mlx",
                    "selected_backend_label": "MLX Apple Silicon",
                    "selected_backend_compatible_with_model": False,
                    "selected_backend_ready_for_model": False,
                    "execution_backend": "onnx",
                    "execution_backend_label": "ONNX Runtime",
                    "routing_mode": "per_model_backend_override",
                    "route_reason": "selected_backend_incompatible_with_model",
                    "candidates": [],
                },
                "missing_artifacts": ["model.onnx", "model.onnx.json"],
                "required_artifacts": ["model.onnx", "model.onnx.json"],
            },
        ]

    def get_model_spec(self, model_name=None, mode=None):
        if model_name is not None:
            for spec in MODEL_SPECS.values():
                if model_name in {spec.model_id, spec.api_name, spec.folder, spec.key}:
                    return spec
            raise ModelNotAvailableError(
                model_name=model_name,
                details={"model": model_name, "backend": "mlx"},
            )
        if mode is not None:
            for spec in MODEL_SPECS.values():
                if spec.mode == mode:
                    return spec
            raise ModelNotAvailableError(
                reason=f"No local model is available for mode: {mode}",
                details={"mode": mode, "backend": "mlx"},
            )
        raise ModelNotAvailableError(reason="No model or mode was specified")

    def readiness_report(self) -> dict:
        return {
            "configured_models": 2,
            "available_models": 2,
            "loadable_models": 2,
            "runtime_ready_models": 2,
            "loaded_models": 1,
            "selected_backend": "mlx",
            "selected_backend_label": "MLX Apple Silicon",
            "backend_selection": {
                "requested_backend": None,
                "auto_selected": True,
                "selection_reason": "platform_and_runtime_match",
            },
            "backend_capabilities": {
                "supports_custom": True,
                "supports_design": True,
                "supports_clone": True,
                "supports_streaming": False,
                "supports_local_models": True,
                "supports_voice_prompt_cache": False,
                "supports_reference_transcription": False,
                "preferred_formats": ["wav"],
                "platforms": ["darwin"],
            },
            "backend_diagnostics": {
                "backend": "mlx",
                "label": "MLX Apple Silicon",
                "available": True,
                "ready": True,
                "reason": None,
                "details": {
                    "platform_supported": True,
                    "load_model_available": True,
                    "generate_audio_available": True,
                },
            },
            "host": {
                "platform_system": "darwin",
                "platform_release": "test",
                "architecture": "arm64",
                "python_version": "3.11.0",
                "ffmpeg_available": True,
                "mlx_runtime_available": True,
                "torch_runtime_available": True,
                "cuda_available": False,
            },
            "routing": {
                "mixed_backend_routing": True,
                "per_model_backend_overrides": 1,
                "degraded_routes": 0,
            },
            "family_summary": {
                "qwen3_tts": {
                    "family": "Qwen3-TTS",
                    "configured_models": 2,
                    "available_models": 2,
                    "runtime_ready_models": 2,
                },
                "piper": {
                    "family": "Piper",
                    "configured_models": 1,
                    "available_models": 0,
                    "runtime_ready_models": 0,
                },
            },
            "family_profiles": {
                "qwen3_tts": self._qwen_profile,
                "piper": self._piper_profile,
            },
            "cache_diagnostics": {
                "cached_model_count": 1,
                "cached_model_ids": ["Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"],
                "cache_policy": {
                    "cache_scope": "process_local",
                    "eviction": "not_configured",
                    "normalized_runtime_dirs": 1,
                },
                "loaded_models": [
                    {
                        "cache_key": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                        "model_id": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                        "backend": "mlx",
                        "loaded": True,
                        "resolved_path": "/tmp/custom",
                        "runtime_path": "/tmp/custom-runtime",
                        "normalized_runtime": True,
                    }
                ],
            },
            "metrics": {
                "selected_backend": {
                    "cache": {
                        "hit": {"mlx": 1},
                        "miss": {"mlx": 1},
                    },
                    "load": {
                        "failures": {},
                        "duration_ms": {
                            "mlx": {
                                "count": 1,
                                "avg_ms": 1.0,
                                "max_ms": 1.0,
                                "last_ms": 1.0,
                            }
                        },
                    },
                },
                "operational": {
                    "execution": {
                        "submitted": 1,
                        "started": 1,
                        "completed": 1,
                        "failed": 0,
                        "timeout": 0,
                        "cancelled": 0,
                        "queue_depth": {"current": 0, "peak": 1},
                    },
                    "models": {
                        "cache": {"hit": {"mlx": 1}, "miss": {"mlx": 1}},
                        "load": {
                            "failures": {},
                            "duration_ms": {
                                "mlx": {
                                    "count": 1,
                                    "avg_ms": 1.0,
                                    "max_ms": 1.0,
                                    "last_ms": 1.0,
                                }
                            },
                        },
                    },
                },
            },
            "preload": {
                "policy": "listed",
                "configured_model_ids": ["Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"],
                "requested_model_ids": ["Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"],
                "resolved_model_ids": ["Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"],
                "status": "completed",
                "policy_reason": "configured",
                "attempted": 1,
                "loaded": 1,
                "failed": 0,
                "loaded_model_ids": ["Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"],
                "failed_model_ids": [],
                "errors": [],
            },
            "available_backends": [
                {
                    "key": "mlx",
                    "label": "MLX Apple Silicon",
                    "selected": True,
                    "platform_supported": True,
                    "available": True,
                    "capabilities": {
                        "supports_custom": True,
                        "supports_design": True,
                        "supports_clone": True,
                    },
                    "diagnostics": {
                        "backend": "mlx",
                        "label": "MLX Apple Silicon",
                        "available": True,
                        "ready": True,
                        "reason": None,
                        "details": {},
                    },
                },
                {
                    "key": "qwen_fast",
                    "label": "Qwen Fast CUDA",
                    "selected": False,
                    "platform_supported": False,
                    "available": False,
                    "capabilities": {
                        "supports_custom": True,
                        "supports_design": False,
                        "supports_clone": False,
                    },
                    "diagnostics": {
                        "backend": "qwen_fast",
                        "label": "Qwen Fast CUDA",
                        "available": False,
                        "ready": False,
                        "reason": "platform_unsupported",
                        "details": {
                            "enabled": True,
                            "test_mode": None,
                        },
                    },
                },
            ],
            "items": [
                {
                    "key": "1",
                    "id": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                    "name": "Custom Voice",
                    "mode": "custom",
                    "family": "Qwen3-TTS",
                    "family_key": "qwen3_tts",
                    "profile": self._qwen_profile,
                    "capabilities_supported": ["preset_speaker_tts"],
                    "backend_support": ["mlx", "qwen_fast", "torch"],
                    "folder": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                    "backend": "mlx",
                    "selected_backend": "mlx",
                    "selected_backend_label": "MLX Apple Silicon",
                    "execution_backend": "mlx",
                    "execution_backend_label": "MLX Apple Silicon",
                    "route": {
                        "selected_backend": "mlx",
                        "selected_backend_label": "MLX Apple Silicon",
                        "selected_backend_compatible_with_model": True,
                        "selected_backend_ready_for_model": True,
                        "execution_backend": "mlx",
                        "execution_backend_label": "MLX Apple Silicon",
                        "routing_mode": "selected_backend",
                        "route_reason": "selected_backend_supports_model",
                        "candidates": [
                            {
                                "key": "qwen_fast",
                                "label": "Qwen Fast CUDA",
                                "selected": False,
                                "compatible_with_model": True,
                                "supports_mode": True,
                                "platform_supported": False,
                                "available": False,
                                "ready": False,
                                "host_reason": "platform_unsupported",
                                "selection_score": 0,
                                "route_reason": "platform_unsupported",
                                "diagnostics": {
                                    "backend": "qwen_fast",
                                    "label": "Qwen Fast CUDA",
                                    "available": False,
                                    "ready": False,
                                    "reason": "platform_unsupported",
                                    "details": {
                                        "enabled": True,
                                        "test_mode": None,
                                    },
                                },
                            }
                        ],
                    },
                    "configured": True,
                    "available": True,
                    "loadable": True,
                    "runtime_ready": True,
                    "cached": True,
                    "resolved_path": "/tmp/custom",
                    "runtime_path": "/tmp/custom-runtime",
                    "cache": {
                        "loaded": True,
                        "cache_key": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                        "backend": "mlx",
                        "normalized_runtime": True,
                        "runtime_path": "/tmp/custom-runtime",
                        "eviction_policy": "not_configured",
                    },
                    "preload": {
                        "policy": "listed",
                        "requested": True,
                        "status": "loaded",
                    },
                    "missing_artifacts": [],
                    "required_artifacts": [
                        "config.json",
                        "model.safetensors|model.safetensors.index.json",
                        "tokenizer_config.json|vocab.json",
                    ],
                    "capabilities": {
                        "supports_custom": True,
                        "supports_design": True,
                        "supports_clone": True,
                    },
                },
                {
                    "key": "3",
                    "id": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
                    "name": "Voice Cloning",
                    "mode": "clone",
                    "family": "Qwen3-TTS",
                    "family_key": "qwen3_tts",
                    "profile": self._qwen_profile,
                    "capabilities_supported": ["reference_voice_clone"],
                    "backend_support": ["mlx", "torch"],
                    "folder": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
                    "backend": "mlx",
                    "selected_backend": "mlx",
                    "selected_backend_label": "MLX Apple Silicon",
                    "execution_backend": "mlx",
                    "execution_backend_label": "MLX Apple Silicon",
                    "route": {
                        "selected_backend": "mlx",
                        "selected_backend_label": "MLX Apple Silicon",
                        "selected_backend_compatible_with_model": True,
                        "selected_backend_ready_for_model": True,
                        "execution_backend": "mlx",
                        "execution_backend_label": "MLX Apple Silicon",
                        "routing_mode": "selected_backend",
                        "route_reason": "selected_backend_supports_model",
                        "candidates": [
                            {
                                "key": "qwen_fast",
                                "label": "Qwen Fast CUDA",
                                "selected": False,
                                "compatible_with_model": False,
                                "supports_mode": False,
                                "platform_supported": False,
                                "available": False,
                                "ready": False,
                                "host_reason": "platform_unsupported",
                                "selection_score": 0,
                                "route_reason": "model_backend_affinity_mismatch",
                                "diagnostics": {
                                    "backend": "qwen_fast",
                                    "label": "Qwen Fast CUDA",
                                    "available": False,
                                    "ready": False,
                                    "reason": "platform_unsupported",
                                    "details": {
                                        "enabled": True,
                                        "test_mode": None,
                                    },
                                },
                            }
                        ],
                    },
                    "configured": True,
                    "available": True,
                    "loadable": True,
                    "runtime_ready": True,
                    "cached": False,
                    "resolved_path": "/tmp/clone",
                    "runtime_path": "/tmp/clone",
                    "cache": {
                        "loaded": False,
                        "cache_key": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
                        "backend": "mlx",
                        "normalized_runtime": False,
                        "runtime_path": "/tmp/clone",
                        "eviction_policy": "not_configured",
                    },
                    "preload": {
                        "policy": "listed",
                        "requested": False,
                        "status": "not_requested",
                    },
                    "missing_artifacts": [],
                    "required_artifacts": [
                        "config.json",
                        "model.safetensors|model.safetensors.index.json",
                        "tokenizer_config.json|vocab.json",
                    ],
                    "capabilities": {
                        "supports_custom": True,
                        "supports_design": True,
                        "supports_clone": True,
                    },
                },
                {
                    "key": "piper-1",
                    "id": "Piper-en_US-lessac-medium",
                    "name": "Piper Lessac",
                    "mode": "custom",
                    "family": "Piper",
                    "family_key": "piper",
                    "profile": self._piper_profile,
                    "capabilities_supported": ["preset_speaker_tts"],
                    "folder": "Piper-en_US-lessac-medium",
                    "backend": "onnx",
                    "configured": True,
                    "available": False,
                    "loadable": False,
                    "runtime_ready": False,
                    "cached": False,
                    "resolved_path": None,
                    "runtime_path": None,
                    "cache": {
                        "loaded": False,
                        "cache_key": "Piper-en_US-lessac-medium",
                        "backend": "onnx",
                        "normalized_runtime": False,
                        "runtime_path": None,
                        "eviction_policy": "not_configured",
                    },
                    "preload": {
                        "policy": "listed",
                        "requested": False,
                        "status": "not_requested",
                    },
                    "missing_artifacts": ["model.onnx", "model.onnx.json"],
                    "required_artifacts": ["model.onnx", "model.onnx.json"],
                    "capabilities": {
                        "supports_custom": True,
                        "supports_design": False,
                        "supports_clone": False,
                        "supports_preset_speaker_tts": True,
                        "supports_voice_description_tts": False,
                        "supports_reference_voice_clone": False,
                    },
                },
            ],
            "registry_ready": True,
        }

    def is_ready(self):
        report = self.readiness_report()
        return True, report


class DummyTTSService:
    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self.last_clone_request = None
        self.last_custom_request = None
        self.last_design_request = None

    def synthesize_custom(self, request):
        self.last_custom_request = request
        return GenerationResult(
            audio=_audio_result(self.settings.outputs_dir / "dummy_custom.wav"),
            saved_path=(self.settings.outputs_dir / "saved_custom.wav")
            if request.save_output
            else None,
            model=request.model or "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            mode="custom",
            backend="mlx",
        )

    def synthesize_design(self, request):
        self.last_design_request = request
        return GenerationResult(
            audio=_audio_result(self.settings.outputs_dir / "dummy_design.wav"),
            saved_path=(self.settings.outputs_dir / "saved_design.wav")
            if request.save_output
            else None,
            model=request.model or "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
            mode="design",
            backend="mlx",
        )

    def synthesize_clone(self, request):
        self.last_clone_request = request
        return GenerationResult(
            audio=_audio_result(self.settings.outputs_dir / "dummy_clone.wav"),
            saved_path=(self.settings.outputs_dir / "saved_clone.wav")
            if request.save_output
            else None,
            model=request.model or "Qwen3-TTS-12Hz-1.7B-Base-8bit",
            mode="clone",
            backend="mlx",
        )


class FailingTTSService(DummyTTSService):
    def synthesize_custom(self, request):
        raise ModelLoadError(
            "mlx runtime failed",
            details={"model": request.model, "runtime_dependency": "mlx_audio"},
        )


class BusyTTSService(DummyTTSService):
    def synthesize_custom(self, request):
        raise InferenceBusyError(
            "Inference is already in progress", details={"queue_depth": 1}
        )


class MissingModelTTSService(DummyTTSService):
    def synthesize_custom(self, request):
        raise ModelNotAvailableError(
            model_name=request.model,
            details={"model": request.model, "backend": "mlx"},
        )


class MissingModeTTSService(DummyTTSService):
    def synthesize_design(self, request):
        raise ModelNotAvailableError(
            reason="No local model is available for mode: design",
            details={"mode": "design", "backend": "mlx"},
        )


class SlowTTSService(DummyTTSService):
    def __init__(self, settings: ServerSettings, *, sleep_seconds: float):
        super().__init__(settings)
        self.sleep_seconds = sleep_seconds

    def synthesize_custom(self, request):
        time.sleep(self.sleep_seconds)
        return super().synthesize_custom(request)


class WorkerFailingTTSService(DummyTTSService):
    def synthesize_custom(self, request):
        raise TTSGenerationError(
            "worker execution failed", details={"failure_kind": "worker_exception"}
        )


class DegradedRegistry(DummyRegistry):
    def readiness_report(self) -> dict:
        report = super().readiness_report()
        report["runtime_ready_models"] = 0
        report["registry_ready"] = False
        report["backend_diagnostics"]["ready"] = False
        report["routing"]["degraded_routes"] = 1
        report["preload"]["status"] = "failed"
        report["preload"]["failed"] = 1
        report["preload"]["failed_model_ids"] = ["Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"]
        report["preload"]["errors"] = [
            {
                "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                "reason": "missing_artifacts",
            }
        ]
        report["items"][0]["runtime_ready"] = False
        report["items"][0]["preload"]["status"] = "failed"
        report["items"][0]["missing_artifacts"] = ["config.json"]
        return report


class ManagedProcessDouble:
    def __init__(self, *, poll_result: int | None = None, wait_result: int = 0):
        self._poll_result = poll_result
        self.wait_result = wait_result
        self.terminated = False
        self.killed = False
        self.wait_calls: list[float | int | None] = []

    def poll(self) -> int | None:
        return self._poll_result

    def terminate(self) -> None:
        self.terminated = True
        self._poll_result = self.wait_result

    def wait(self, timeout: float | int | None = None) -> int:
        self.wait_calls.append(timeout)
        return self.wait_result

    def kill(self) -> None:
        self.killed = True
        self._poll_result = self.wait_result


def make_validation_model_entry(
    *,
    model_id: str,
    folder: str,
    runtime_ready: bool = True,
    execution_backend: str = "mlx",
    available: bool | None = None,
    loadable: bool | None = None,
    selected_backend: str | None = None,
    missing_artifacts: list[str] | None = None,
    required_artifacts: list[str] | None = None,
    route_reason: str | None = None,
    selected_backend_compatible_with_model: bool | None = None,
    candidate_diagnostics: list[dict] | None = None,
) -> dict:
    resolved_available = runtime_ready if available is None else available
    resolved_loadable = runtime_ready if loadable is None else loadable
    resolved_selected_backend = selected_backend or execution_backend
    return {
        "id": model_id,
        "folder": folder,
        "runtime_ready": runtime_ready,
        "execution_backend": execution_backend,
        "available": resolved_available,
        "loadable": resolved_loadable,
        "selected_backend": resolved_selected_backend,
        "missing_artifacts": list(missing_artifacts or []),
        "required_artifacts": list(required_artifacts or []),
        "route": {
            "selected_backend": resolved_selected_backend,
            "execution_backend": execution_backend,
            "selected_backend_compatible_with_model": (
                runtime_ready
                if selected_backend_compatible_with_model is None
                else selected_backend_compatible_with_model
            ),
            "route_reason": route_reason or ("selected_backend_supports_model" if runtime_ready else "runtime_not_ready"),
            "candidates": [
                {"diagnostics": item} if "diagnostics" not in item else item
                for item in list(candidate_diagnostics or [])
            ],
        },
    }


def make_validation_self_check_payload(
    *,
    items: list[dict] | None = None,
    ffmpeg_available: bool = True,
    models_missing_assets: list[str] | None = None,
    selected_backend: str = "mlx",
) -> dict:
    return {
        "readiness": {
            "host": {
                "ffmpeg_available": ffmpeg_available,
            },
            "backend_diagnostics": {
                "backend": selected_backend,
            },
            "items": list(items or []),
        },
        "assets": {
            "models_missing_assets": list(models_missing_assets or []),
        },
        "representative_models": {
            "targets": [],
            "ready_targets": [],
            "skipped_targets": [],
            "failed_targets": [],
        },
    }


def make_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    silence_frame = bytes([0, 0])
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(silence_frame * 240)
    return buffer.getvalue()


def _audio_result(path: Path) -> AudioResult:
    return AudioResult(path=path, bytes_data=make_wav_bytes())


def extract_json_logs(caplog: pytest.LogCaptureFixture, event_name: str) -> list[dict]:
    matched: list[dict] = []
    for record in caplog.records:
        try:
            payload = json.loads(record.getMessage())
        except json.JSONDecodeError:
            continue
        if payload.get("event") == event_name:
            matched.append(payload)
    return matched


__all__ = [
    "DummyRegistry",
    "DummyTTSService",
    "FailingTTSService",
    "BusyTTSService",
    "MissingModelTTSService",
    "MissingModeTTSService",
    "SlowTTSService",
    "WorkerFailingTTSService",
    "DegradedRegistry",
    "ManagedProcessDouble",
    "make_validation_model_entry",
    "make_validation_self_check_payload",
    "make_wav_bytes",
    "extract_json_logs",
]
