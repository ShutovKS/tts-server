# FILE: core/engines/__init__.py
# VERSION: 1.6.0
# START_MODULE_CONTRACT
#   PURPOSE: Re-export the public engine contract, typed configuration, production engine implementations, scheduler surface, registry/discovery helpers.
#   SCOPE: barrel re-exports for engine DTOs, TTSEngine, discriminated engine config models, production engines, scheduler helpers, and registry loader helpers
#   DEPENDS: M-ENGINE-CONTRACTS, M-ENGINE-CONFIG, M-ENGINE-SCHEDULER, M-ENGINE-REGISTRY, M-BACKENDS
#   LINKS: M-ENGINE-CONTRACTS, M-ENGINE-CONFIG, M-ENGINE-SCHEDULER, M-ENGINE-REGISTRY, M-BACKENDS
#   ROLE: BARREL
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   Contract surface - Re-export TTSEngine, model/audio/job DTOs, and availability/capability types.
#   Config surface - Re-export discriminated engine config models, parsing helpers, and collection settings.
#   Production engine surface - Re-export the Piper ONNX engine, the Qwen3 Torch engine, and the OmniVoice Torch engine implementations.
#   Scheduler surface - Re-export the worker-pool key/policy DTOs, scheduler facade, and shutdown error.
#   Registry surface - Re-export EngineRegistry, its typed error, and the loader/entry-point helpers.
#   Registry loader surface - Re-export registry loader helpers.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.7.0 - Task 17: removed temporary legacy registry-adapter exports after engine runtime migration made them obsolete]
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
from core.engines.omnivoice import OmniVoiceTorchEngine
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
    "MlxEngineConfig",
    "ModelHandle",
    "OmniVoiceTorchEngine",
    "OnnxEngineConfig",
    "PiperOnnxEngine",
    "Qwen3TorchEngine",
    "QwenFastEngineConfig",
    "SynthesisJob",
    "TTSEngine",
    "TorchEngineConfig",
    "WorkerPool",
    "load_engine_registry",
    "parse_engine_config",
    "parse_engine_settings",
]
