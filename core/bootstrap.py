# FILE: core/bootstrap.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Assemble the full CoreRuntime from settings by wiring all components together.
#   SCOPE: CoreRuntime dataclass, build_runtime factory, sub-component factory helpers
#   DEPENDS: M-CONFIG, M-BACKENDS, M-MODEL-REGISTRY, M-TTS-SERVICE, M-APPLICATION, M-INFRASTRUCTURE, M-METRICS
#   LINKS: M-BOOTSTRAP
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   CoreRuntime - Frozen dataclass holding all wired runtime components
#   build_runtime - Factory function that assembles CoreRuntime from settings
#   build_job_artifact_store - Factory for job artifact store
#   build_job_metadata_store - Factory for job metadata store
#   build_job_execution_backend - Factory for job execution backend
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import dataclass

from core.application import (
    InMemoryJobExecutor,
    JobArtifactStore,
    JobExecutionBackend,
    JobExecutionGateway,
    JobMetadataStore,
    QuotaGuard,
    RateLimiter,
    TTSApplicationService,
)
from core.backends.mlx_backend import MLXBackend
from core.backends.onnx_backend import ONNXBackend
from core.backends.qwen_fast_backend import QwenFastBackend
from core.backends.registry import BackendRegistry
from core.backends.torch_backend import TorchBackend
from core.config import CoreSettings
from core.infrastructure import (
    InferenceGuard,
    LocalBoundedExecutionManager,
    LocalInMemoryJobStore,
    LocalJobArtifactStore,
    build_quota_guard,
    build_rate_limiter,
)
from core.metrics import OperationalMetricsRegistry
from core.services.model_registry import ModelRegistry
from core.services.tts_service import TTSService


# START_CONTRACT: CoreRuntime
#   PURPOSE: Hold the fully assembled shared runtime components for transport adapters.
#   INPUTS: { settings: CoreSettings - Parsed runtime settings, backend_registry: BackendRegistry - Selected backend registry, registry: ModelRegistry - Model discovery and loading service, tts_service: TTSService - Core synthesis service, application: TTSApplicationService - Application-level synthesis facade, job_artifact_store: JobArtifactStore - Artifact persistence backend, job_store: JobMetadataStore - Job metadata persistence backend, job_executor: InMemoryJobExecutor - Job execution adapter, job_manager: JobExecutionBackend - Async execution backend, job_execution: JobExecutionGateway - Job orchestration gateway, rate_limiter: RateLimiter - Request throttling service, quota_guard: QuotaGuard - Quota enforcement service, inference_guard: InferenceGuard - Shared inference concurrency guard, metrics: OperationalMetricsRegistry - Operational metrics facade }
#   OUTPUTS: { instance - Immutable runtime composition root }
#   SIDE_EFFECTS: none
#   LINKS: M-BOOTSTRAP
# END_CONTRACT: CoreRuntime
@dataclass(frozen=True)
class CoreRuntime:
    settings: CoreSettings
    backend_registry: BackendRegistry
    registry: ModelRegistry
    tts_service: TTSService
    application: TTSApplicationService
    job_artifact_store: JobArtifactStore
    job_store: JobMetadataStore
    job_executor: InMemoryJobExecutor
    job_manager: JobExecutionBackend
    job_execution: JobExecutionGateway
    rate_limiter: RateLimiter
    quota_guard: QuotaGuard
    inference_guard: InferenceGuard
    metrics: OperationalMetricsRegistry


# START_CONTRACT: build_job_artifact_store
#   PURPOSE: Build the configured job artifact persistence backend from runtime settings.
#   INPUTS: { settings: CoreSettings - Runtime settings containing artifact backend selection }
#   OUTPUTS: { JobArtifactStore - Configured job artifact store implementation }
#   SIDE_EFFECTS: none
#   LINKS: M-BOOTSTRAP
# END_CONTRACT: build_job_artifact_store
def build_job_artifact_store(settings: CoreSettings) -> JobArtifactStore:
    if settings.job_artifact_backend == "local":
        return LocalJobArtifactStore()
    raise ValueError(
        f"Unsupported job artifact backend: {settings.job_artifact_backend}"
    )


# START_CONTRACT: build_job_metadata_store
#   PURPOSE: Build the configured job metadata store and bind it to the artifact store.
#   INPUTS: { settings: CoreSettings - Runtime settings containing metadata backend selection, artifact_store: JobArtifactStore - Artifact cleanup dependency for job records }
#   OUTPUTS: { JobMetadataStore - Configured job metadata storage implementation }
#   SIDE_EFFECTS: none
#   LINKS: M-BOOTSTRAP
# END_CONTRACT: build_job_metadata_store
def build_job_metadata_store(
    settings: CoreSettings, *, artifact_store: JobArtifactStore
) -> JobMetadataStore:
    if settings.job_metadata_backend == "local":
        return LocalInMemoryJobStore(artifact_store=artifact_store)
    raise ValueError(
        f"Unsupported job metadata backend: {settings.job_metadata_backend}"
    )


# START_CONTRACT: build_job_execution_backend
#   PURPOSE: Build the configured async job execution backend for the shared runtime.
#   INPUTS: { settings: CoreSettings - Runtime settings containing execution backend selection, store: JobMetadataStore - Job metadata store used by the backend, executor: InMemoryJobExecutor - Job executor used to run submissions, metrics: OperationalMetricsRegistry - Metrics registry for queue and execution observations }
#   OUTPUTS: { JobExecutionBackend - Configured job execution backend }
#   SIDE_EFFECTS: none
#   LINKS: M-BOOTSTRAP
# END_CONTRACT: build_job_execution_backend
def build_job_execution_backend(
    settings: CoreSettings,
    *,
    store: JobMetadataStore,
    executor: InMemoryJobExecutor,
    metrics: OperationalMetricsRegistry,
) -> JobExecutionBackend:
    if settings.job_execution_backend == "local":
        return LocalBoundedExecutionManager(
            store=store, executor=executor, metrics=metrics
        )
    raise ValueError(
        f"Unsupported job execution backend: {settings.job_execution_backend}"
    )


# START_CONTRACT: build_runtime
#   PURPOSE: Assemble the full CoreRuntime graph from normalized shared settings.
#   INPUTS: { settings: CoreSettings - Runtime settings controlling backend, storage, and policy wiring }
#   OUTPUTS: { CoreRuntime - Fully wired shared runtime for adapters }
#   SIDE_EFFECTS: Creates configured runtime directories on disk and instantiates process-local services
#   LINKS: M-BOOTSTRAP
# END_CONTRACT: build_runtime
def build_runtime(settings: CoreSettings) -> CoreRuntime:
    # START_BLOCK_INIT_INFRASTRUCTURE
    settings.ensure_directories()
    inference_guard = InferenceGuard()
    metrics = OperationalMetricsRegistry()
    # END_BLOCK_INIT_INFRASTRUCTURE
    # START_BLOCK_INIT_BACKENDS
    backend_registry = BackendRegistry(
        [
            MLXBackend(settings.mlx_models_dir, metrics=metrics),
            QwenFastBackend(
                settings.models_dir,
                enabled=settings.qwen_fast_enabled,
                metrics=metrics,
            ),
            TorchBackend(settings.models_dir, metrics=metrics),
            ONNXBackend(settings.models_dir, metrics=metrics),
        ],
        requested_backend=settings.backend,
        autoselect=settings.backend_autoselect,
        model_manifest_path=settings.model_manifest_path,
    )
    # END_BLOCK_INIT_BACKENDS
    # START_BLOCK_INIT_SERVICES
    registry = ModelRegistry(
        backend_registry=backend_registry,
        preload_policy=settings.model_preload_policy,
        preload_model_ids=settings.model_preload_ids,
        metrics=metrics,
    )
    tts_service = TTSService(
        registry=registry, settings=settings, inference_guard=inference_guard
    )
    application = TTSApplicationService(tts_service=tts_service)
    # END_BLOCK_INIT_SERVICES
    # START_BLOCK_INIT_JOB_SYSTEM
    job_artifact_store = build_job_artifact_store(settings)
    job_store = build_job_metadata_store(settings, artifact_store=job_artifact_store)
    job_executor = InMemoryJobExecutor(application_service=application)
    job_manager = build_job_execution_backend(
        settings, store=job_store, executor=job_executor, metrics=metrics
    )
    job_execution = JobExecutionGateway(store=job_store, manager=job_manager)
    rate_limiter = build_rate_limiter(settings)
    quota_guard = build_quota_guard(settings, store=job_store)
    # END_BLOCK_INIT_JOB_SYSTEM
    # START_BLOCK_ASSEMBLE_RUNTIME
    return CoreRuntime(
        settings=settings,
        backend_registry=backend_registry,
        registry=registry,
        tts_service=tts_service,
        application=application,
        job_artifact_store=job_artifact_store,
        job_store=job_store,
        job_executor=job_executor,
        job_manager=job_manager,
        job_execution=job_execution,
        rate_limiter=rate_limiter,
        quota_guard=quota_guard,
        inference_guard=inference_guard,
        metrics=metrics,
    )
    # END_BLOCK_ASSEMBLE_RUNTIME


__all__ = [
    "CoreRuntime",
    "build_job_artifact_store",
    "build_job_metadata_store",
    "build_job_execution_backend",
    "build_runtime",
]
