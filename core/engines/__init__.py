# FILE: core/engines/__init__.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Re-export the public engine contract and typed configuration surfaces.
#   SCOPE: barrel re-exports for engine DTOs, TTSEngine, and discriminated engine config models
#   DEPENDS: M-ENGINE-CONTRACTS, M-ENGINE-CONFIG
#   LINKS: M-ENGINE-CONTRACTS, M-ENGINE-CONFIG
#   ROLE: BARREL
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   Contract surface - Re-export TTSEngine, model/audio/job DTOs, and availability/capability types.
#   Config surface - Re-export discriminated engine config models, parsing helpers, and collection settings.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Phase 2 engine wave: introduced the public engine contracts/config barrel without wiring runtime execution to engines yet]
# END_CHANGE_SUMMARY

from core.engines.config import (
    DisabledEngineConfig,
    EngineConfig,
    EngineSettings,
    MlxEngineConfig,
    OnnxEngineConfig,
    QwenFastEngineConfig,
    TorchEngineConfig,
    parse_engine_config,
    parse_engine_settings,
)
from core.engines.contracts import (
    AudioBuffer,
    EngineAvailability,
    EngineCapabilities,
    ModelHandle,
    SynthesisJob,
    TTSEngine,
)

__all__ = [
    "AudioBuffer",
    "DisabledEngineConfig",
    "EngineAvailability",
    "EngineCapabilities",
    "EngineConfig",
    "EngineSettings",
    "MlxEngineConfig",
    "ModelHandle",
    "OnnxEngineConfig",
    "QwenFastEngineConfig",
    "SynthesisJob",
    "TTSEngine",
    "TorchEngineConfig",
    "parse_engine_config",
    "parse_engine_settings",
]
