from __future__ import annotations

from dataclasses import dataclass

from core.application import TTSApplicationService
from core.backends import BackendRegistry, MLXBackend, TorchBackend
from core.config import CoreSettings
from core.infrastructure.concurrency import InferenceGuard
from core.services.model_registry import ModelRegistry
from core.services.tts_service import TTSService


@dataclass(frozen=True)
class CoreRuntime:
    settings: CoreSettings
    backend_registry: BackendRegistry
    registry: ModelRegistry
    tts_service: TTSService
    application: TTSApplicationService
    inference_guard: InferenceGuard



def build_runtime(settings: CoreSettings) -> CoreRuntime:
    settings.ensure_directories()
    inference_guard = InferenceGuard()
    backend_registry = BackendRegistry(
        [
            MLXBackend(settings.models_dir),
            TorchBackend(settings.models_dir),
        ],
        requested_backend=settings.backend,
        autoselect=settings.backend_autoselect,
    )
    registry = ModelRegistry(backend_registry=backend_registry)
    tts_service = TTSService(registry=registry, settings=settings, inference_guard=inference_guard)
    application = TTSApplicationService(tts_service=tts_service)
    return CoreRuntime(
        settings=settings,
        backend_registry=backend_registry,
        registry=registry,
        tts_service=tts_service,
        application=application,
        inference_guard=inference_guard,
    )
