# FILE: tests/unit/core/test_engine_contracts.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for the engine contract layer introduced in core/engines/contracts.py.
#   SCOPE: TTSEngine load/synthesize split, immutable DTOs, capability/availability reporting
#   DEPENDS: M-ENGINE-CONTRACTS
#   LINKS: V-M-ENGINE-CONTRACTS
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _StubEngine - Minimal TTSEngine implementation used to exercise contract behavior.
#   test_contract_dtos_are_immutable - Verifies ModelHandle and SynthesisJob stay frozen.
#   test_tts_engine_separates_load_model_from_synthesize - Verifies TTSEngine keeps model loading separate from synthesis execution.
#   test_engine_capabilities_and_availability_shapes_are_explicit - Verifies typed capability and availability metadata stay explicit.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added focused unit coverage for the new TTSEngine contract surface]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from core.engines import (
    AudioBuffer,
    EngineAvailability,
    EngineCapabilities,
    ModelHandle,
    SynthesisJob,
    TTSEngine,
)
from core.models.catalog import MODEL_SPECS

pytestmark = pytest.mark.unit


class _StubEngine(TTSEngine):
    key = "stub"
    label = "Stub Engine"

    def __init__(self) -> None:
        self.loaded_calls: list[tuple[str, str, str | None]] = []
        self.synthesized_calls: list[tuple[str, str]] = []

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            families=("qwen3_tts",),
            backends=("torch",),
            capabilities=("preset_speaker_tts", "voice_description_tts"),
            supports_streaming=False,
            supports_batching=False,
        )

    def availability(self) -> EngineAvailability:
        return EngineAvailability(
            engine_key=self.key,
            is_available=True,
            is_enabled=True,
        )

    def load_model(
        self,
        *,
        spec: Any,
        backend_key: str,
        model_path: Path | None,
    ) -> ModelHandle:
        self.loaded_calls.append((spec.key, backend_key, None if model_path is None else str(model_path)))
        return ModelHandle(
            spec=spec,
            runtime_model={"loaded": spec.key, "backend": backend_key},
            resolved_path=model_path,
            engine_key=self.key,
            backend_key=backend_key,
            family_key=spec.family_key,
        )

    def synthesize(self, handle: ModelHandle, job: SynthesisJob) -> AudioBuffer:
        self.synthesized_calls.append((handle.spec.key, job.text))
        return AudioBuffer(
            waveform=b"audio:" + job.text.encode("utf-8"),
            sample_rate=24000,
            audio_format="wav",
        )


def test_contract_dtos_are_immutable(tmp_path: Path) -> None:
    spec = next(iter(MODEL_SPECS.values()))
    handle = ModelHandle(
        spec=spec,
        runtime_model=object(),
        resolved_path=tmp_path,
        engine_key="stub",
        backend_key="torch",
        family_key=spec.family_key,
    )
    job = SynthesisJob(
        capability="preset_speaker_tts",
        execution_mode="custom",
        text="hello",
        language="en",
        output_dir=tmp_path,
        payload={"speaker": "Ryan"},
    )

    with pytest.raises(FrozenInstanceError):
        handle.engine_key = "mutated"  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        job.text = "mutated"  # type: ignore[misc]


def test_tts_engine_separates_load_model_from_synthesize(tmp_path: Path) -> None:
    spec = next(iter(MODEL_SPECS.values()))
    engine = _StubEngine()

    handle = engine.load_model(spec=spec, backend_key="torch", model_path=tmp_path)
    job = SynthesisJob(
        capability="preset_speaker_tts",
        execution_mode="custom",
        text="separate lifecycle",
        language="en",
        output_dir=tmp_path,
        payload={"speaker": "Ryan", "speed": 1.0},
    )
    audio = engine.synthesize(handle, job)

    assert engine.loaded_calls == [(spec.key, "torch", str(tmp_path))]
    assert engine.synthesized_calls == [(spec.key, "separate lifecycle")]
    assert handle.runtime_model == {"loaded": spec.key, "backend": "torch"}
    assert handle.engine_key == "stub"
    assert audio.waveform == b"audio:separate lifecycle"
    assert audio.sample_rate == 24000
    assert audio.audio_format == "wav"


def test_engine_capabilities_and_availability_shapes_are_explicit() -> None:
    engine = _StubEngine()

    capabilities = engine.capabilities()
    availability = engine.availability()

    assert capabilities.families == ("qwen3_tts",)
    assert capabilities.backends == ("torch",)
    assert capabilities.capabilities == (
        "preset_speaker_tts",
        "voice_description_tts",
    )
    assert capabilities.supports_streaming is False
    assert availability == EngineAvailability(
        engine_key="stub",
        is_available=True,
        is_enabled=True,
    )
