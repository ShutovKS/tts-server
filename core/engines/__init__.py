# FILE: core/engines/__init__.py
# VERSION: 1.2.0
# START_MODULE_CONTRACT
#   PURPOSE: Re-export the public engine contract, typed configuration, registry/discovery surfaces, and the temporary legacy compatibility bridge.
#   SCOPE: barrel re-exports for engine DTOs, TTSEngine, discriminated engine config models, registry loader helpers, and temporary bridge helpers
#   DEPENDS: M-ENGINE-CONTRACTS, M-ENGINE-CONFIG, M-ENGINE-REGISTRY, M-ENGINE-BRIDGE
#   LINKS: M-ENGINE-CONTRACTS, M-ENGINE-CONFIG, M-ENGINE-REGISTRY, M-ENGINE-BRIDGE
#   ROLE: BARREL
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   Contract surface - Re-export TTSEngine, model/audio/job DTOs, and availability/capability types.
#   Config surface - Re-export discriminated engine config models, parsing helpers, and collection settings.
#   Registry surface - Re-export EngineRegistry, its typed error, and the loader/entry-point helpers.
#   Compatibility surface - Re-export the temporary legacy compatibility bridge and registry builder.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.2.0 - Task 8 compatibility bridge: re-exported the temporary legacy bridge helpers so coexistence wiring can remain explicit and easy to delete after engine migration]
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
from core.engines.compatibility import (
    EngineCompatibilityBridge,
    LegacyEngineRecord,
    build_legacy_engine_registry,
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
    "EngineCompatibilityBridge",
    "EngineCapabilities",
    "EngineConfig",
    "ENGINE_ENTRY_POINT_GROUP",
    "EngineRegistry",
    "EngineRegistryError",
    "EngineSettings",
    "LegacyEngineRecord",
    "MlxEngineConfig",
    "ModelHandle",
    "OnnxEngineConfig",
    "QwenFastEngineConfig",
    "SynthesisJob",
    "TTSEngine",
    "TorchEngineConfig",
    "build_legacy_engine_registry",
    "load_engine_registry",
    "parse_engine_config",
    "parse_engine_settings",
]
