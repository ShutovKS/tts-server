# FILE: core/model_families/__init__.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Re-export public model-family adapter contracts and the unified ModelFamilyPlugin extension surface.
#   SCOPE: barrel re-exports for family adapter surfaces and the plugin extension contract
#   DEPENDS: M-MODEL-FAMILY
#   LINKS: M-MODEL-FAMILY
#   ROLE: BARREL
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   Family adapter surface - Re-export family adapter base contract, prepared execution DTO, and concrete family adapters.
#   Family plugin surface - Re-export the unified ModelFamilyPlugin contract, its DTOs, and the FamilyPluginRegistry.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Phase 2.5: re-exported the unified ModelFamilyPlugin extension surface alongside the existing adapter contracts.]
# END_CHANGE_SUMMARY

from core.model_families.base import FamilyPreparedExecution, ModelFamilyAdapter
from core.model_families.omnivoice import OmniVoiceFamilyAdapter
from core.model_families.piper import PiperFamilyAdapter
from core.model_families.plugin import (
    FamilyExecutionRequest,
    FamilyExecutionResult,
    FamilyPluginRegistry,
    ModelFamilyPlugin,
)
from core.model_families.qwen3 import EMOTION_EXAMPLES, SPEAKER_MAP, Qwen3FamilyAdapter

__all__ = [
    "EMOTION_EXAMPLES",
    "FamilyExecutionRequest",
    "FamilyExecutionResult",
    "FamilyPluginRegistry",
    "FamilyPreparedExecution",
    "ModelFamilyAdapter",
    "ModelFamilyPlugin",
    "OmniVoiceFamilyAdapter",
    "PiperFamilyAdapter",
    "Qwen3FamilyAdapter",
    "SPEAKER_MAP",
]
