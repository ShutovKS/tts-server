# WN|# FILE: core/services/__init__.py
# BV|# VERSION: 1.2.0
# TR|# START_MODULE_CONTRACT
# TR|#   PURPOSE: Re-export public service and planning-adjacent types used by the shared runtime.
# MZ|#   SCOPE: barrel re-exports with lazy resolution
# RK|#   DEPENDS: M-MODEL-REGISTRY, M-TTS-SERVICE, M-SYNTHESIS-PLANNER, M-STREAMING
# SY|#   LINKS: M-MODEL-REGISTRY, M-TTS-SERVICE, M-SYNTHESIS-PLANNER, M-STREAMING
# ZH|#   ROLE: BARREL
# XM|#   MAP_MODE: SUMMARY
# QS|# END_MODULE_CONTRACT
# HB|#
# NT|# START_MODULE_MAP
# MV|#   ModelRegistry - Re-export model discovery and readiness service
# KM|#   TTSService - Re-export core synthesis orchestration service
# MQ|#   SynthesisPlanner - Re-export synthesis planner
# YR|#   SynthesisRouter - Re-export the unified Command -> backend routing seam
# YS|#   AudioStreamChunk - Re-export the streaming chunk descriptor (Phase 4.12)
# YT|#   stream_generation_result - Re-export the GenerationResult chunker (Phase 4.12)
# PQ|# END_MODULE_MAP
# BZ|#
# ZR|# START_CHANGE_SUMMARY
# XV|#   LAST_CHANGE: [v1.2.0 - Phase 4.12: re-exported AudioStreamChunk and stream_generation_result so transports can stream completed GenerationResult payloads without reaching into core.services.streaming directly]
# JV|# END_CHANGE_SUMMARY
# RJ|
# SK|from __future__ import annotations
# QN|
# MK|from typing import TYPE_CHECKING, Any
# XW|
# WX|if TYPE_CHECKING:
# WR|    from core.planning import SynthesisPlanner
# WS|    from core.services.model_registry import ModelRegistry
# YA|    from core.services.streaming import AudioStreamChunk, stream_generation_result
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
# YW|    if name == "AudioStreamChunk":
# YX|        from core.services.streaming import AudioStreamChunk
# YY|
# YZ|        return AudioStreamChunk
# ZA|    if name == "stream_generation_result":
# ZB|        from core.services.streaming import stream_generation_result
# ZC|
# ZD|        return stream_generation_result
# XK|    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
# XL|
# XM|__all__ = [
# XN|    "AudioStreamChunk",
# XO|    "ModelRegistry",
# XP|    "SynthesisPlanner",
# XQ|    "SynthesisRouter",
# XR|    "TTSService",
# XS|    "stream_generation_result",
# XT|]
