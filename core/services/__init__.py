# FILE: core/services/__init__.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Re-export public service and planning-adjacent types used by the shared runtime.
#   SCOPE: barrel re-exports
#   DEPENDS: M-MODEL-REGISTRY, M-TTS-SERVICE, M-SYNTHESIS-PLANNER
#   LINKS: M-MODEL-REGISTRY, M-TTS-SERVICE, M-SYNTHESIS-PLANNER
#   ROLE: BARREL
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   ModelRegistry - Re-export model discovery and readiness service
#   TTSService - Re-export core synthesis orchestration service
#   SynthesisPlanner - Re-export synthesis planner compatibility bridge
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from core.services.model_registry import ModelRegistry
from core.services.tts_service import TTSService
from core.planning import SynthesisPlanner

__all__ = ["ModelRegistry", "SynthesisPlanner", "TTSService"]
