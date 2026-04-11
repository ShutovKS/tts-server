# FILE: tests/unit/core/test_synthesis_planner.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for normalized synthesis requests and planner-backed execution plan resolution.
#   SCOPE: Request normalization, capability mapping, planner resolution, family key normalization
#   DEPENDS: M-CORE
#   LINKS: V-M-EXECUTION-PLAN, V-M-SYNTHESIS-PLANNER
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   PlannerRegistryStub - Minimal registry stub returning model specs for planner tests
#   test_synthesis_request_from_custom_command_normalizes_to_preset_speaker_capability - Verifies custom command normalization
#   test_synthesis_request_from_design_command_normalizes_to_voice_description_capability - Verifies design command normalization
#   test_synthesis_request_from_clone_command_normalizes_to_reference_clone_capability - Verifies clone command normalization
#   test_synthesis_planner_resolves_execution_plan_for_current_backend_registry - Verifies planner output fields for current compatibility bridge
#   test_synthesis_planner_honors_explicit_requested_model_identifier - Verifies planner passes explicit model selection through registry resolution
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added unit coverage for synthesis normalization and planner seams]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path

import pytest

from core.contracts.commands import (
    CustomVoiceCommand,
    VoiceCloneCommand,
    VoiceDesignCommand,
)
from core.contracts.synthesis import (
    PresetSpeakerPayload,
    SynthesisRequest,
    VoiceClonePayload,
    VoiceDesignPayload,
)
from core.errors import ModelCapabilityError
from core.models.catalog import MODEL_SPECS
from core.planning import SynthesisPlanner


pytestmark = pytest.mark.unit


class PlannerRegistryStub:
    def __init__(self):
        self.backend = type(
            "BackendStub",
            (),
            {"key": "torch", "label": "PyTorch + Transformers"},
        )()
        self.last_model_name: str | None = None
        self.last_mode: str | None = None
        self.route_backend_key = "torch"
        self.route_backend_label = "PyTorch + Transformers"
        self.route_reason = "registry_model_resolution"

    def get_model_spec(self, model_name: str | None = None, mode: str | None = None):
        self.last_model_name = model_name
        self.last_mode = mode
        if model_name is not None:
            return next(
                spec
                for spec in MODEL_SPECS.values()
                if model_name in {spec.api_name, spec.folder, spec.key}
            )
        return next(spec for spec in MODEL_SPECS.values() if spec.mode == mode)

    def backend_for_spec(self, spec):
        return type(
            "BackendStub",
            (),
            {"key": self.route_backend_key, "label": self.route_backend_label},
        )()

    def backend_route_for_spec(self, spec):
        return {
            "route_reason": self.route_reason,
            "execution_backend": self.route_backend_key,
        }


def test_synthesis_request_from_custom_command_normalizes_to_preset_speaker_capability():
    request = SynthesisRequest.from_command(
        CustomVoiceCommand(
            text="Hello",
            model="1",
            save_output=True,
            language="En",
            speaker="Ryan",
            instruct="Friendly",
            speed=1.25,
        )
    )

    assert request.capability == "preset_speaker_tts"
    assert request.legacy_mode == "custom"
    assert request.language == "en"
    assert request.requested_model == "1"
    assert request.save_output is True
    assert request.source_command == "CustomVoiceCommand"
    assert request.payload == PresetSpeakerPayload(
        speaker="Ryan",
        instruct="Friendly",
        speed=1.25,
    )


def test_synthesis_request_from_design_command_normalizes_to_voice_description_capability():
    request = SynthesisRequest.from_command(
        VoiceDesignCommand(
            text="Hello",
            voice_description="Warm radio host",
        )
    )

    assert request.capability == "voice_description_tts"
    assert request.legacy_mode == "design"
    assert request.payload == VoiceDesignPayload(
        voice_description="Warm radio host",
    )


def test_synthesis_request_from_clone_command_normalizes_to_reference_clone_capability(
    tmp_path: Path,
):
    ref_audio_path = tmp_path / "reference.wav"
    ref_audio_path.write_bytes(b"wav")

    request = SynthesisRequest.from_command(
        VoiceCloneCommand(
            text="Clone this",
            ref_audio_path=ref_audio_path,
            ref_text="Clone this",
            language="Ru ",
        )
    )

    assert request.capability == "reference_voice_clone"
    assert request.legacy_mode == "clone"
    assert request.language == "ru"
    assert request.payload == VoiceClonePayload(
        ref_audio_path=ref_audio_path,
        ref_text="Clone this",
    )


def test_synthesis_planner_resolves_execution_plan_for_current_backend_registry():
    registry = PlannerRegistryStub()
    planner = SynthesisPlanner(registry=registry)  # type: ignore[arg-type]

    plan = planner.plan(
        SynthesisRequest.from_command(CustomVoiceCommand(text="Hello", speaker="Ryan"))
    )

    assert registry.last_mode == "custom"
    assert plan.backend_key == "torch"
    assert plan.backend_label == "PyTorch + Transformers"
    assert plan.family_key == "qwen3_tts"
    assert plan.family_label == "Qwen3-TTS"
    assert plan.legacy_mode == "custom"
    assert plan.model_spec == MODEL_SPECS["1"]
    assert plan.selection_reason == "registry_model_resolution"


def test_synthesis_planner_honors_explicit_requested_model_identifier():
    registry = PlannerRegistryStub()
    planner = SynthesisPlanner(registry=registry)  # type: ignore[arg-type]

    plan = planner.plan_command(
        VoiceCloneCommand(
            text="Clone this",
            model="6",
            ref_audio_path=Path("reference.wav"),
        )
    )

    assert registry.last_model_name == "6"
    assert registry.last_mode == "clone"
    assert plan.model_spec == MODEL_SPECS["6"]
    assert plan.request.requested_model == "6"


def test_synthesis_planner_rejects_unsupported_family_capability():
    registry = PlannerRegistryStub()
    planner = SynthesisPlanner(registry=registry)  # type: ignore[arg-type]

    with pytest.raises(ModelCapabilityError) as exc_info:
        planner.plan_command(
            VoiceDesignCommand(
                text="Hello",
                model="Piper-en_US-lessac-medium",
                voice_description="Warm narrator",
            )
        )

    assert exc_info.value.context.to_dict() == {
        "reason": "Model 'Piper-en_US-lessac-medium' does not support capability 'voice_description_tts'",
        "model": "Piper-en_US-lessac-medium",
        "capability": "voice_description_tts",
        "supported_capabilities": ["preset_speaker_tts"],
        "family": "Piper",
    }


def test_synthesis_planner_surfaces_fast_backend_selection_reason():
    registry = PlannerRegistryStub()
    registry.route_backend_key = "qwen_fast"
    registry.route_backend_label = "Qwen Fast CUDA"
    registry.route_reason = "selected_backend_supports_model"
    planner = SynthesisPlanner(registry=registry)  # type: ignore[arg-type]

    plan = planner.plan_command(CustomVoiceCommand(text="Hello", speaker="Ryan"))

    assert plan.backend_key == "qwen_fast"
    assert plan.backend_label == "Qwen Fast CUDA"
    assert plan.selection_reason == "selected_backend_supports_model"
