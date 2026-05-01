# WN|# FILE: core/services/__init__.py
# BV|# VERSION: 1.1.0
# TR|# START_MODULE_CONTRACT
# TR|#   PURPOSE: Re-export public service and planning-adjacent types used by the shared runtime.
# MZ|#   SCOPE: barrel re-exports with lazy resolution
# RK|#   DEPENDS: M-MODEL-REGISTRY, M-TTS-SERVICE, M-SYNTHESIS-PLANNER
# SY|#   LINKS: M-MODEL-REGISTRY, M-TTS-SERVICE, M-SYNTHESIS-PLANNER
# ZH|#   ROLE: BARREL
# XM|#   MAP_MODE: SUMMARY
# QS|# END_MODULE_CONTRACT
# HB|#
# NT|# START_MODULE_MAP
# MV|#   ModelRegistry - Re-export model discovery and readiness service
# KM|#   TTSService - Re-export core synthesis orchestration service
# MQ|#   SynthesisPlanner - Re-export synthesis planner
# YR|#   SynthesisRouter - Re-export the unified Command -> backend routing seam
# PQ|# END_MODULE_MAP
# BZ|#
# ZR|# START_CHANGE_SUMMARY
# XV|#   LAST_CHANGE: [v1.1.0 - Phase 3.9: re-exported SynthesisRouter so transports can depend on the unified routing seam without reaching into core.services.synthesis_router]
# JV|# END_CHANGE_SUMMARY
# RJ|
# SK|from __future__ import annotations
# QN|
# MK|from typing import TYPE_CHECKING, Any
# XW|
# WX|if TYPE_CHECKING:
# WR|    from core.planning import SynthesisPlanner
# WS|    from core.services.model_registry import ModelRegistry
# WT|    from core.services.synthesis_router import SynthesisRouter
# WU|    from core.services.tts_service import TTSService
# WV|
# WW|
# WX|def __getattr__(name: str) -> Any:
# WY|    if name == "ModelRegistry":
# WZ|        from core.services.model_registry import ModelRegistry
# XA|
# XB|        return ModelRegistry
# XC|    if name == "TTSService":
# XD|        from core.services.tts_service import TTSService
# XE|
# XF|        return TTSService
# XG|    if name == "SynthesisPlanner":
# XH|        from core.planning import SynthesisPlanner
# XI|
# XJ|        return SynthesisPlanner
# YS|    if name == "SynthesisRouter":
# YT|        from core.services.synthesis_router import SynthesisRouter
# YU|
# YV|        return SynthesisRouter
# XK|    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
# XL|
# XM|__all__ = ["ModelRegistry", "SynthesisPlanner", "SynthesisRouter", "TTSService"]
