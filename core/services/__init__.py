#WN|# FILE: core/services/__init__.py
#BV|# VERSION: 1.0.1
#TR|# START_MODULE_CONTRACT
#TR|#   PURPOSE: Re-export public service and planning-adjacent types used by the shared runtime.
#MZ|#   SCOPE: barrel re-exports with lazy resolution
#RK|#   DEPENDS: M-MODEL-REGISTRY, M-TTS-SERVICE, M-SYNTHESIS-PLANNER
#SY|#   LINKS: M-MODEL-REGISTRY, M-TTS-SERVICE, M-SYNTHESIS-PLANNER
#ZH|#   ROLE: BARREL
#XM|#   MAP_MODE: SUMMARY
#QS|# END_MODULE_CONTRACT
#HB|#
#NT|# START_MODULE_MAP
#MV|#   ModelRegistry - Re-export model discovery and readiness service
#KM|#   TTSService - Re-export core synthesis orchestration service
#MQ|#   SynthesisPlanner - Re-export synthesis planner
#PQ|# END_MODULE_MAP
#BZ|#
#ZR|# START_CHANGE_SUMMARY
#XV|#   LAST_CHANGE: [v1.0.1 - Switched the services barrel to lazy export resolution so lightweight imports do not pull the runtime graph eagerly]
#JV|# END_CHANGE_SUMMARY
#RJ|
#SK|from __future__ import annotations
#QN|
#MK|from typing import TYPE_CHECKING, Any
#XW|
#WX|if TYPE_CHECKING:
#WR|    from core.planning import SynthesisPlanner
#WS|    from core.services.model_registry import ModelRegistry
#WT|    from core.services.tts_service import TTSService
#WU|
#WV|
#WW|def __getattr__(name: str) -> Any:
#WX|    if name == "ModelRegistry":
#WY|        from core.services.model_registry import ModelRegistry
#WZ|
#XA|        return ModelRegistry
#XB|    if name == "TTSService":
#XC|        from core.services.tts_service import TTSService
#XD|
#XE|        return TTSService
#XF|    if name == "SynthesisPlanner":
#XG|        from core.planning import SynthesisPlanner
#XH|
#XI|        return SynthesisPlanner
#XJ|    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
#XK|
#XL|__all__ = ["ModelRegistry", "SynthesisPlanner", "TTSService"]
