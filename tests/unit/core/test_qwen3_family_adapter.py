# FILE: tests/unit/core/test_qwen3_family_adapter.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for the explicit Qwen3 family adapter compatibility bridge.
#   SCOPE: Family capability support, execution preparation, metadata exports
#   DEPENDS: M-CORE
#   LINKS: V-M-QWEN3-FAMILY
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _make_plan - Build execution plans for adapter tests
#   test_qwen3_family_adapter_prepares_custom_execution_kwargs - Verifies custom voice request translation
#   test_qwen3_family_adapter_prepares_design_execution_kwargs - Verifies voice design request translation
#   test_qwen3_family_adapter_prepares_clone_execution_kwargs - Verifies clone request translation
#   test_catalog_re_exports_qwen3_family_metadata - Verifies catalog compatibility exports route through Qwen3 family metadata
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added unit coverage for Qwen3 family adapter extraction]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path

import pytest

from core.contracts.commands import (
    CustomVoiceCommand,
    VoiceCloneCommand,
    VoiceDesignCommand,
)
from core.contracts.synthesis import ExecutionPlan, SynthesisRequest
from core.model_families import PiperFamilyAdapter, Qwen3FamilyAdapter
from core.models.catalog import EMOTION_EXAMPLES, MODEL_SPECS, SPEAKER_MAP


pytestmark = pytest.mark.unit


def _make_plan(command, model_key: str) -> ExecutionPlan:
    request = SynthesisRequest.from_command(command)
    return ExecutionPlan(
        request=request,
        model_spec=MODEL_SPECS[model_key],
        backend_key="torch",
        backend_label="PyTorch + Transformers",
        family_key="qwen3_tts",
        family_label="Qwen3-TTS",
        selection_reason="test",
        legacy_mode=request.legacy_mode,
    )


def test_qwen3_family_adapter_prepares_custom_execution_kwargs():
    adapter = Qwen3FamilyAdapter()
    plan = _make_plan(
        CustomVoiceCommand(
            text="Hello",
            speaker="Ryan",
            instruct="Friendly",
            speed=1.15,
        ),
        "1",
    )

    prepared = adapter.prepare_execution(plan)

    assert prepared.legacy_mode == "custom"
    assert prepared.generation_kwargs == {
        "language": "auto",
        "voice": "Ryan",
        "instruct": "Friendly",
        "speed": 1.15,
    }


def test_qwen3_family_adapter_prepares_design_execution_kwargs():
    adapter = Qwen3FamilyAdapter()
    plan = _make_plan(
        VoiceDesignCommand(
            text="Hello",
            voice_description="Warm radio host",
        ),
        "2",
    )

    prepared = adapter.prepare_execution(plan)

    assert prepared.legacy_mode == "design"
    assert prepared.generation_kwargs == {
        "language": "auto",
        "instruct": "Warm radio host",
    }


def test_qwen3_family_adapter_prepares_clone_execution_kwargs(tmp_path: Path):
    adapter = Qwen3FamilyAdapter()
    ref_audio_path = tmp_path / "reference.wav"
    ref_audio_path.write_bytes(b"wav")
    plan = _make_plan(
        VoiceCloneCommand(
            text="Clone this",
            ref_audio_path=ref_audio_path,
            ref_text="Clone this",
            language="en",
        ),
        "3",
    )

    prepared = adapter.prepare_execution(plan)

    assert prepared.legacy_mode == "clone"
    assert prepared.generation_kwargs == {
        "language": "en",
        "ref_audio": str(ref_audio_path),
        "ref_text": "Clone this",
    }


def test_catalog_re_exports_qwen3_family_metadata():
    assert "English" in SPEAKER_MAP
    assert "Ryan" in SPEAKER_MAP["English"]
    assert any("Excited and happy" in item for item in EMOTION_EXAMPLES)


def test_piper_family_adapter_prepares_preset_speaker_execution():
    adapter = PiperFamilyAdapter()
    plan = ExecutionPlan(
        request=SynthesisRequest.from_command(CustomVoiceCommand(text="Hello")),
        model_spec=MODEL_SPECS["piper-1"],
        backend_key="onnx",
        backend_label="ONNX Runtime",
        family_key="piper",
        family_label="Piper",
        selection_reason="test",
        legacy_mode="custom",
    )

    prepared = adapter.prepare_execution(plan)

    assert prepared.legacy_mode == "custom"
    assert prepared.generation_kwargs["piper_model"] is True
    assert prepared.generation_kwargs["voice"] == "Piper-en_US-lessac-medium"
