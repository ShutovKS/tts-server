# FILE: tests/unit/core/test_model_families.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for non-Qwen model family adapters added to the planner/runtime pipeline.
#   SCOPE: OmniVoice family capability matching and execution payload preparation
#   DEPENDS: M-MODEL-FAMILY, M-EXECUTION-PLAN
#   LINKS: V-M-MODEL-FAMILY
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   build_plan - Helper that assembles execution plans for family adapter tests
#   test_omnivoice_family_adapter_prepares_voice_design_execution - Verifies OmniVoice design payload preparation
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.1 - Focused family adapter coverage on the supported OmniVoice surface]
# END_CHANGE_SUMMARY

from __future__ import annotations

import pytest

from core.contracts.synthesis import (
    ExecutionPlan,
    PresetSpeakerPayload,
    SynthesisRequest,
    VoiceDesignPayload,
)
from core.model_families import OmniVoiceFamilyAdapter
from core.models.catalog import MODEL_SPECS


pytestmark = pytest.mark.unit


def build_plan(*, model_key: str, payload, capability: str, execution_mode: str) -> ExecutionPlan:
    spec = MODEL_SPECS[model_key]
    return ExecutionPlan(
        request=SynthesisRequest(
            capability=capability,
            text="Hello from tests",
            payload=payload,
            requested_model=spec.model_id,
            language="en",
        ),
        model_spec=spec,
        backend_key="torch",
        backend_label="PyTorch + Transformers",
        family_key=spec.family_key,
        family_label=spec.family,
        selection_reason="unit_test",
        execution_mode=execution_mode,
    )


def test_omnivoice_family_adapter_prepares_voice_design_execution():
    adapter = OmniVoiceFamilyAdapter()
    plan = build_plan(
        model_key="omnivoice-design-1",
        payload=VoiceDesignPayload(voice_description="Calm documentary narrator"),
        capability="voice_description_tts",
        execution_mode="design",
    )

    prepared = adapter.prepare_execution(plan)

    assert prepared.execution_mode == "design"
    assert prepared.generation_kwargs == {
        "language": "en",
        "family": "omnivoice",
        "instruct": "Calm documentary narrator",
    }
