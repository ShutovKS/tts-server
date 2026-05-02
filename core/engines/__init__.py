# FILE: core/engines/__init__.py
# VERSION: 1.5.0
# START_MODULE_CONTRACT
#   PURPOSE: Re-export the public engine contract, typed configuration, production engine implementations, scheduler surface, registry/discovery surfaces, and the temporary legacy compatibility bridge.
#   SCOPE: barrel re-exports for engine DTOs, TTSEngine, discriminated engine config models, production engines, scheduler helpers, registry loader helpers, and temporary bridge helpers
#   DEPENDS: M-ENGINE-CONTRACTS, M-ENGINE-CONFIG, M-ENGINE-SCHEDULER, M-ENGINE-REGISTRY, M-ENGINE-BRIDGE, M-BACKENDS
#   LINKS: M-ENGINE-CONTRACTS, M-ENGINE-CONFIG, M-ENGINE-SCHEDULER, M-ENGINE-REGISTRY, M-ENGINE-BRIDGE, M-BACKENDS
#   ROLE: BARREL
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   Contract surface - Re-export TTSEngine, model/audio/job DTOs, and availability/capability types.
#   Config surface - Re-export discriminated engine config models, parsing helpers, and collection settings.
#   Production engine surface - Re-export the Piper ONNX engine and the Qwen3 Torch engine implementations.
#   Scheduler surface - Re-export the worker-pool key/policy DTOs, scheduler facade, and shutdown error.
#   Registry surface - Re-export EngineRegistry, its typed error, and the loader/entry-point helpers.
#   Compatibility surface - Re-export the temporary legacy compatibility bridge and registry builder.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.5.0 - Task 15: re-exported the Qwen3 Torch engine alongside the existing Piper engine for generic runtime engine routing]
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
from core.engines.piper import PiperOnnxEngine
from core.engines.qwen3 import Qwen3TorchEngine
from core.engines.registry import (
    ENGINE_ENTRY_POINT_GROUP,
    EngineRegistry,
    EngineRegistryError,
    load_engine_registry,
)
from core.engines.scheduler import (
    EngineScheduler,
    EngineSchedulerStoppedError,
    EngineWorkerPoolKey,
    EngineWorkerPoolPolicy,
    WorkerPool,
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
    "EngineScheduler",
    "EngineSchedulerStoppedError",
    "EngineSettings",
    "EngineWorkerPoolKey",
    "EngineWorkerPoolPolicy",
    "LegacyEngineRecord",
    "MlxEngineConfig",
    "ModelHandle",
    "OnnxEngineConfig",
    "PiperOnnxEngine",
    "Qwen3TorchEngine",
    "QwenFastEngineConfig",
    "SynthesisJob",
    "TTSEngine",
    "TorchEngineConfig",
    "WorkerPool",
    "build_legacy_engine_registry",
    "load_engine_registry",
    "parse_engine_config",
    "parse_engine_settings",
]
