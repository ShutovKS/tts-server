# FILE: tests/unit/core/test_engine_runtime_seams.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for engine runtime seams that sit below TTSService orchestration.
#   SCOPE: typed engine config runtime fields, ModelCache reuse/eviction and wrapping, AudioPipeline normalization behavior, and runtime factory policy parsing
#   DEPENDS: M-ENGINE-CONFIG, M-ENGINE-CONTRACTS, M-ENGINE-RUNTIME-FACTORY, M-ENGINE-MODEL-CACHE, M-ENGINE-AUDIO-PIPELINE
#   LINKS: V-M-ENGINE-CONFIG, V-M-ENGINE-SCHEDULER, V-M-ENGINE-RUNTIME-SEAMS
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_parse_engine_config_accepts_runtime_policy_fields - Verifies typed engine runtime policy fields parse and validate.
#   test_model_cache_reuses_handles_and_evicts_oldest - Verifies thread-safe cache reuse and bounded eviction semantics.
#   test_audio_pipeline_normalizes_to_target_sample_rate_and_loudness - Verifies AudioPipeline centralizes WAV conversion, sample-rate normalization, and loudness-style RMS normalization.
#   test_build_engine_settings_preserves_runtime_configs_without_hidden_disables - Verifies runtime factory preserves operator configs without injecting hidden disabled-engine fallbacks.
#   test_build_engine_registry_wraps_built_in_engines_with_cache - Verifies built-in runtime engines are cache-decorated by the factory.
#   test_load_engine_registry_wraps_entry_point_engines_with_cache - Verifies optional entry-point engines can share the same cache wrapper seam.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Added registry cache-wrapper coverage for built-in and entry-point engines]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import CoreSettings
from core.engines.audio_pipeline import AudioPipeline
from core.engines.config import DisabledEngineConfig, TorchEngineConfig, parse_engine_config, parse_engine_settings
from core.engines.contracts import AudioBuffer, EngineAvailability, EngineCapabilities, ModelHandle, SynthesisJob, TTSEngine
from core.engines.model_cache import ModelCache, ModelCacheKey
from core.engines.registry import load_engine_registry
from core.engines.runtime_factory import CachedEngine, build_engine_registry, build_engine_settings
from core.models.catalog import MODEL_SPECS

pytestmark = pytest.mark.unit


def test_parse_engine_config_accepts_runtime_policy_fields() -> None:
    config = parse_engine_config(
        {
            "kind": "torch",
            "name": "qwen3-torch",
            "family": "qwen3_tts",
            "capabilities": ["preset_speaker_tts"],
            "max_active": 2,
            "max_queued": 3,
            "submit_timeout_seconds": 0.5,
            "inference_timeout_seconds": 30,
            "device": "cuda:0",
            "model_cache_size": 4,
        }
    )

    assert isinstance(config, TorchEngineConfig)
    assert config.max_active == 2
    assert config.max_queued == 3
    assert config.submit_timeout_seconds == 0.5
    assert config.inference_timeout_seconds == 30
    assert config.device == "cuda:0"
    assert config.model_cache_size == 4


def test_model_cache_reuses_handles_and_evicts_oldest() -> None:
    spec = next(iter(MODEL_SPECS.values()))
    cache = ModelCache(max_entries=1)
    calls = {"count": 0}
    key_one = ModelCacheKey.from_parts(
        engine_key="qwen3-torch",
        model_id=spec.model_id,
        backend_key="torch",
        model_path=Path("one"),
    )
    key_two = ModelCacheKey.from_parts(
        engine_key="qwen3-torch",
        model_id=spec.model_id,
        backend_key="torch",
        model_path=Path("two"),
    )

    def load_handle() -> ModelHandle:
        calls["count"] += 1
        return ModelHandle(
            spec=spec,
            runtime_model=object(),
            resolved_path=None,
            engine_key="qwen3-torch",
            backend_key="torch",
            family_key=spec.family_key,
        )

    first = cache.get_or_load(key_one, load_handle)
    again = cache.get_or_load(key_one, load_handle)
    second = cache.get_or_load(key_two, load_handle)
    reloaded = cache.get_or_load(key_one, load_handle)

    assert first is again
    assert second is not first
    assert reloaded is not first
    assert calls["count"] == 3
    assert len(cache) == 1


def test_audio_pipeline_normalizes_to_target_sample_rate_and_loudness() -> None:
    import io
    import wave

    low_level_buffer = io.BytesIO()
    loud_buffer = io.BytesIO()
    with wave.open(low_level_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x04\x00\x08\x00\x0c\x00\x10\x00")
    with wave.open(loud_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x20\x00\x40\x00\x60\x00\x7f\x00")
    audio = AudioBuffer(waveform=low_level_buffer.getvalue(), sample_rate=16000, audio_format="wav")
    louder_audio = AudioBuffer(waveform=loud_buffer.getvalue(), sample_rate=16000, audio_format="wav")

    processed = AudioPipeline(target_sample_rate=24000).process(audio)
    processed_louder = AudioPipeline(target_sample_rate=24000).process(louder_audio)

    assert processed.audio_format == "wav"
    assert processed.sample_rate == 24000
    assert isinstance(processed.waveform, bytes)
    assert processed.waveform.startswith(b"RIFF")
    with wave.open(io.BytesIO(processed.waveform), "rb") as wav_file:
        quiet_frames = wav_file.readframes(wav_file.getnframes())
    with wave.open(io.BytesIO(processed_louder.waveform), "rb") as wav_file:
        loud_frames = wav_file.readframes(wav_file.getnframes())
    import audioop

    quiet_rms = audioop.rms(quiet_frames, 2)
    loud_rms = audioop.rms(loud_frames, 2)
    assert abs(quiet_rms - loud_rms) < max(quiet_rms, loud_rms) * 0.15


def test_build_engine_settings_preserves_runtime_configs_without_hidden_disables(tmp_path: Path) -> None:
    settings = CoreSettings(
        models_dir=tmp_path / "models",
        outputs_dir=tmp_path / "outputs",
        voices_dir=tmp_path / "voices",
        engine_configs=(
            {
                "kind": "torch",
                "name": "qwen3-torch",
                "family": "qwen3_tts",
                "capabilities": ["preset_speaker_tts"],
                "max_active": 2,
            },
        ),
    )

    engine_settings = build_engine_settings(settings)

    assert any(
        isinstance(config, TorchEngineConfig) and config.name == "qwen3-torch" and config.max_active == 2
        for config in engine_settings.engines
    )
    assert not any(isinstance(config, DisabledEngineConfig) for config in engine_settings.engines)


def test_build_engine_registry_wraps_built_in_engines_with_cache(tmp_path: Path) -> None:
    settings = CoreSettings(
        models_dir=tmp_path / "models",
        outputs_dir=tmp_path / "outputs",
        voices_dir=tmp_path / "voices",
    )

    registry = build_engine_registry(settings)

    assert registry.keys() == ("qwen3-torch", "omnivoice-torch", "piper-onnx")
    assert all(isinstance(engine, CachedEngine) for engine in registry.registered_engines)


def test_load_engine_registry_wraps_entry_point_engines_with_cache() -> None:
    wrapped: list[tuple[str, str | None]] = []

    def wrapper(engine: TTSEngine, config) -> TTSEngine:
        wrapped.append((engine.key, None if config is None else config.name))
        return CachedEngine(engine, ModelCache(max_entries=1))

    registry = load_engine_registry(
        built_in_engines=(_TinyEngine,),
        explicit_engines=(_TinyEngine(key="explicit-engine"),),
        settings=parse_engine_settings(
            {
                "engines": [
                    {
                        "kind": "torch",
                        "name": "tiny-engine",
                        "family": "tiny",
                        "capabilities": ["preset_speaker_tts"],
                    }
                ]
            }
        ),
        include_entry_points=False,
        engine_wrapper=wrapper,
    )

    assert registry.keys() == ("tiny-engine", "explicit-engine")
    assert all(isinstance(engine, CachedEngine) for engine in registry.registered_engines)
    assert wrapped == [("tiny-engine", "tiny-engine"), ("explicit-engine", None)]


class _TinyEngine(TTSEngine):
    key = "tiny-engine"
    label = "Tiny Engine"

    def __init__(self, *, key: str | None = None) -> None:
        if key is not None:
            self.key = key

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            families=("tiny",),
            backends=("torch",),
            capabilities=("preset_speaker_tts",),
        )

    def availability(self) -> EngineAvailability:
        return EngineAvailability(engine_key=self.key, is_available=True)

    def load_model(self, *, spec, backend_key: str, model_path) -> ModelHandle:
        return ModelHandle(
            spec=spec,
            runtime_model=object(),
            resolved_path=model_path,
            engine_key=self.key,
            backend_key=backend_key,
            family_key=spec.family_key,
        )

    def synthesize(self, handle: ModelHandle, job: SynthesisJob) -> AudioBuffer:
        return AudioBuffer(waveform=b"RIFF", sample_rate=24000)
