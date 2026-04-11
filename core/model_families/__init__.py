# FILE: core/model_families/__init__.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Re-export public model-family adapter contracts.
#   SCOPE: barrel re-exports for family adapter surfaces
#   DEPENDS: M-MODEL-FAMILY
#   LINKS: M-MODEL-FAMILY
#   ROLE: BARREL
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   Family adapter surface - Re-export family adapter base contract, prepared execution DTO, and concrete family adapters
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added model-family barrel exports for migration seams]
# END_CHANGE_SUMMARY

from core.model_families.base import FamilyPreparedExecution, ModelFamilyAdapter
from core.model_families.piper import PiperFamilyAdapter
from core.model_families.qwen3 import EMOTION_EXAMPLES, Qwen3FamilyAdapter, SPEAKER_MAP

__all__ = [
    "EMOTION_EXAMPLES",
    "FamilyPreparedExecution",
    "ModelFamilyAdapter",
    "PiperFamilyAdapter",
    "Qwen3FamilyAdapter",
    "SPEAKER_MAP",
]
