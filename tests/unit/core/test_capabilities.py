# FILE: tests/unit/core/test_capabilities.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Verify the Phase 3.10 CapabilityRegistry seam: default capabilities are registered, lookups round-trip, custom capabilities can be registered and queried, and collisions are rejected.
#   SCOPE: Unit tests for CapabilityRegistry, DEFAULT_CAPABILITY_REGISTRY, and the synthesis.py helpers that defer to it.
#   DEPENDS: M-CAPABILITIES, M-EXECUTION-PLAN
#   LINKS: V-M-CAPABILITIES
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_default_registry_contains_builtin_capabilities - DEFAULT_CAPABILITY_REGISTRY exposes the three built-in capabilities and modes.
#   test_default_registry_round_trips_capability_to_mode - Forward and reverse helpers are inverses.
#   test_synthesis_helpers_defer_to_default_registry - synthesis.capability_to_execution_mode/execution_mode_to_capability use the default registry.
#   test_register_custom_capability_extends_lookup - Custom capability registration is visible to the helpers.
#   test_register_collision_is_rejected - Re-registering a name or mode raises.
#   test_unknown_capability_lookup_raises - Helpers raise on unknown names/modes.
#   test_aliases_resolve_to_canonical_spec - Aliases resolve through get/is_supported but are excluded from names().
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Phase 3.10: introduced unit coverage for CapabilityRegistry]
# END_CHANGE_SUMMARY

from __future__ import annotations

import pytest

from core.contracts.capabilities import (
    DEFAULT_CAPABILITY_REGISTRY,
    CapabilityRegistry,
    CapabilitySpec,
)
from core.contracts.synthesis import (
    capability_to_execution_mode,
    execution_mode_to_capability,
)

pytestmark = pytest.mark.unit


def test_default_registry_contains_builtin_capabilities() -> None:
    assert set(DEFAULT_CAPABILITY_REGISTRY.names()) == {
        "preset_speaker_tts",
        "voice_description_tts",
        "reference_voice_clone",
    }
    assert set(DEFAULT_CAPABILITY_REGISTRY.execution_modes()) == {
        "custom",
        "design",
        "clone",
    }


def test_default_registry_round_trips_capability_to_mode() -> None:
    for capability, expected_mode in [
        ("preset_speaker_tts", "custom"),
        ("voice_description_tts", "design"),
        ("reference_voice_clone", "clone"),
    ]:
        assert DEFAULT_CAPABILITY_REGISTRY.execution_mode_for(capability) == expected_mode
        assert DEFAULT_CAPABILITY_REGISTRY.capability_for_mode(expected_mode) == capability


def test_synthesis_helpers_defer_to_default_registry() -> None:
    assert capability_to_execution_mode("preset_speaker_tts") == "custom"
    assert capability_to_execution_mode("voice_description_tts") == "design"
    assert capability_to_execution_mode("reference_voice_clone") == "clone"
    assert execution_mode_to_capability("custom") == "preset_speaker_tts"
    assert execution_mode_to_capability("design") == "voice_description_tts"
    assert execution_mode_to_capability("clone") == "reference_voice_clone"


def test_register_custom_capability_extends_lookup() -> None:
    registry = CapabilityRegistry(specs=DEFAULT_CAPABILITY_REGISTRY.specs())
    registry.register(
        CapabilitySpec(
            name="emotion_transfer",
            execution_mode="emotion",
            description="Transfer an emotional contour onto a base voice.",
        )
    )

    assert registry.is_supported("emotion_transfer")
    assert registry.execution_mode_for("emotion_transfer") == "emotion"
    assert registry.capability_for_mode("emotion") == "emotion_transfer"
    assert "emotion_transfer" in registry.names()
    assert "emotion" in registry.execution_modes()


def test_register_collision_is_rejected() -> None:
    registry = CapabilityRegistry(specs=DEFAULT_CAPABILITY_REGISTRY.specs())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(CapabilitySpec(name="preset_speaker_tts", execution_mode="other"))

    with pytest.raises(ValueError, match="already mapped"):
        registry.register(CapabilitySpec(name="another", execution_mode="custom"))


def test_unknown_capability_lookup_raises() -> None:
    with pytest.raises(KeyError):
        DEFAULT_CAPABILITY_REGISTRY.execution_mode_for("nonexistent_capability")

    with pytest.raises(ValueError, match="Unsupported execution mode"):
        DEFAULT_CAPABILITY_REGISTRY.capability_for_mode("nonexistent_mode")


def test_aliases_resolve_to_canonical_spec() -> None:
    registry = CapabilityRegistry(
        specs=(
            CapabilitySpec(
                name="canonical_capability",
                execution_mode="canonical_mode",
                aliases=("legacy_name",),
            ),
        )
    )

    canonical = registry.get("canonical_capability")
    legacy = registry.get("legacy_name")
    assert canonical is not None
    assert legacy is canonical
    assert registry.is_supported("legacy_name")
    assert registry.names() == ("canonical_capability",)
