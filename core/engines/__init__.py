# FILE: core/engines/__init__.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Re-export the public engine contract, typed configuration, and registry/discovery surfaces.
#   SCOPE: barrel re-exports for engine DTOs, TTSEngine, discriminated engine config models, and registry loader helpers
#   DEPENDS: M-ENGINE-CONTRACTS, M-ENGINE-CONFIG, M-ENGINE-REGISTRY
#   LINKS: M-ENGINE-CONTRACTS, M-ENGINE-CONFIG, M-ENGINE-REGISTRY
#   ROLE: BARREL
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   Contract surface - Re-export TTSEngine, model/audio/job DTOs, and availability/capability types.
#   Config surface - Re-export discriminated engine config models, parsing helpers, and collection settings.
#   Registry surface - Re-export EngineRegistry, its typed error, and the loader/entry-point helpers.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Task 7 engine wave: expanded the public barrel with EngineRegistry and loader exports while still leaving runtime execution wiring for later tasks]
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
from core.engines.registry import (
    ENGINE_ENTRY_POINT_GROUP,
    EngineRegistry,
    EngineRegistryError,
    load_engine_registry,
)

__all__ = [
    "AudioBuffer",
    "DisabledEngineConfig",
    "EngineAvailability",
    "EngineCapabilities",
    "EngineConfig",
    "ENGINE_ENTRY_POINT_GROUP",
    "EngineRegistry",
    "EngineRegistryError",
    "EngineSettings",
    "MlxEngineConfig",
    "ModelHandle",
    "OnnxEngineConfig",
    "QwenFastEngineConfig",
    "SynthesisJob",
    "TTSEngine",
    "TorchEngineConfig",
    "load_engine_registry",
    "parse_engine_config",
    "parse_engine_settings",
]
