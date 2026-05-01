# FILE: tests/unit/core/test_synthesis_router.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Verify the Phase 3.9 SynthesisRouter unified seam dispatches every GenerationCommand variant to the correct coordinator entry-point and rejects unsupported command types.
#   SCOPE: Unit tests for SynthesisRouter.route, route_custom, route_design, route_clone, and the construction guard.
#   DEPENDS: M-TTS-SERVICE
#   LINKS: V-M-TTS-SERVICE
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _StubCoordinator - Lightweight stub that records which coordinator method was invoked.
#   test_route_dispatches_custom_command - route(CustomVoiceCommand) hits synthesize_custom.
#   test_route_dispatches_design_command - route(VoiceDesignCommand) hits synthesize_design.
#   test_route_dispatches_clone_command - route(VoiceCloneCommand) hits synthesize_clone.
#   test_route_rejects_unknown_command_type - route(<other>) raises TTSGenerationError with details.
#   test_router_requires_either_coordinator_or_full_wiring - constructor guard rejects partial wiring.
#   test_route_explicit_helpers_match_isinstance_dispatch - route_custom/_design/_clone delegate to coordinator directly.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Phase 3.9: introduced unit coverage for SynthesisRouter]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path

import pytest

from core.contracts.commands import (
    CustomVoiceCommand,
    GenerationCommand,
    VoiceCloneCommand,
    VoiceDesignCommand,
)
from core.errors import TTSGenerationError
from core.services.synthesis_router import SynthesisRouter

pytestmark = pytest.mark.unit


class _StubCoordinator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, GenerationCommand]] = []

    def synthesize_custom(self, command: CustomVoiceCommand) -> str:
        self.calls.append(("custom", command))
        return "custom-result"

    def synthesize_design(self, command: VoiceDesignCommand) -> str:
        self.calls.append(("design", command))
        return "design-result"

    def synthesize_clone(self, command: VoiceCloneCommand) -> str:
        self.calls.append(("clone", command))
        return "clone-result"


def test_route_dispatches_custom_command() -> None:
    coord = _StubCoordinator()
    router = SynthesisRouter(coordinator=coord)
    command = CustomVoiceCommand(text="hello", model="m", speaker="Vivian")

    result = router.route(command)

    assert result == "custom-result"
    assert coord.calls == [("custom", command)]


def test_route_dispatches_design_command() -> None:
    coord = _StubCoordinator()
    router = SynthesisRouter(coordinator=coord)
    command = VoiceDesignCommand(text="hello", model="m", voice_description="warm narrator")

    result = router.route(command)

    assert result == "design-result"
    assert coord.calls == [("design", command)]


def test_route_dispatches_clone_command(tmp_path: Path) -> None:
    coord = _StubCoordinator()
    router = SynthesisRouter(coordinator=coord)
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"")
    command = VoiceCloneCommand(text="hello", model="m", ref_audio_path=ref_audio)

    result = router.route(command)

    assert result == "clone-result"
    assert coord.calls == [("clone", command)]


def test_route_rejects_unknown_command_type() -> None:
    coord = _StubCoordinator()
    router = SynthesisRouter(coordinator=coord)
    bare = GenerationCommand(text="hello")

    with pytest.raises(TTSGenerationError) as excinfo:
        router.route(bare)

    assert excinfo.value.context.details["command_type"] == "GenerationCommand"
    assert "CustomVoiceCommand" in excinfo.value.context.details["expected"]
    assert coord.calls == []


def test_router_requires_either_coordinator_or_full_wiring() -> None:
    with pytest.raises(ValueError, match="coordinator"):
        SynthesisRouter()


def test_route_explicit_helpers_match_isinstance_dispatch(tmp_path: Path) -> None:
    coord = _StubCoordinator()
    router = SynthesisRouter(coordinator=coord)

    custom = CustomVoiceCommand(text="hi", model="m")
    design = VoiceDesignCommand(text="hi", model="m", voice_description="x")
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"")
    clone = VoiceCloneCommand(text="hi", model="m", ref_audio_path=ref_audio)

    assert router.route_custom(custom) == "custom-result"
    assert router.route_design(design) == "design-result"
    assert router.route_clone(clone) == "clone-result"
    assert [call[0] for call in coord.calls] == ["custom", "design", "clone"]
