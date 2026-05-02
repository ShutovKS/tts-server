# FILE: core/bootstrap.py
# VERSION: 1.1.1
# START_MODULE_CONTRACT
#   PURPOSE: Assemble the full CoreRuntime from settings by wiring all components together via auto-discovery (backend classes via discover_backend_classes(), manifest via load_composite_manifest(), TTSBackend.from_settings() factory) so adding a new backend or model does not require editing this file.
#   SCOPE: CoreRuntime dataclass, build_runtime factory, sub-component factory helpers, build_backends helper that drives auto-discovery of TTSBackend subclasses
#   DEPENDS: M-CONFIG, M-BACKENDS, M-MODELS, M-DISCOVERY, M-MODEL-REGISTRY, M-TTS-SERVICE, M-APPLICATION, M-INFRASTRUCTURE, M-METRICS
#   LINKS: M-BOOTSTRAP
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   CoreRuntime - Frozen dataclass holding all wired runtime components
#   build_runtime - Factory function that assembles CoreRuntime from settings via auto-discovered backends + composite manifest
#   build_backends - Auto-discover concrete TTSBackend subclasses and build an instance per class via TTSBackend.from_settings()
#   build_job_artifact_store - Factory for job artifact store
#   build_job_metadata_store - Factory for job metadata store
#   build_job_execution_backend - Factory for job execution backend
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.1 - Allowed degraded runtime assembly when no backend is ready so readiness and validation flows can report host limitations without failing bootstrap]
# END_CHANGE_SUMMARY

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

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
from core.backends.base import TTSBackend
from core.backends.registry import BackendRegistry
from core.contracts import RuntimeExecutionRegistry
from core.config import CoreSettings
from core.discovery import discover_backend_classes
from core.infrastructure import (
    InferenceGuard,
    LocalBoundedExecutionManager,
    LocalInMemoryJobStore,
    LocalJobArtifactStore,
    build_quota_guard,
    build_rate_limiter,
)
from core.metrics import OperationalMetricsRegistry
from core.models.composite import load_composite_manifest
from core.services.model_lifecycle import ModelLifecycleService
from core.services.model_registry import ModelRegistry
from core.services.result_cache import (
    FileSystemResultCache,
    NullResultCache,
    ResultCache,
)
from core.services.telemetry import TelemetryState, configure_telemetry
from core.services.tts_service import TTSService

logger = logging.getLogger(__name__)


# START_CONTRACT: CoreRuntime
#   PURPOSE: Hold the fully assembled shared runtime components for transport adapters.
#   INPUTS: { settings: CoreSettings - Parsed runtime settings, backend_registry: BackendRegistry - Selected backend registry, registry: ModelRegistry - Model discovery and loading service, model_lifecycle: ModelLifecycleService - Lifecycle facade for delete/refresh/download submissions, result_cache: ResultCache - Process-wide synthesis result cache (NullResultCache when disabled), telemetry: TelemetryState - Active OpenTelemetry runtime state (disabled when otel_enabled is false), tts_service: TTSService - Core synthesis service, application: TTSApplicationService - Application-level synthesis facade, job_artifact_store: JobArtifactStore - Artifact persistence backend, job_store: JobMetadataStore - Job metadata persistence backend, job_executor: InMemoryJobExecutor - Job execution adapter, job_manager: JobExecutionBackend - Async execution backend, job_execution: JobExecutionGateway - Job orchestration gateway, rate_limiter: RateLimiter - Request throttling service, quota_guard: QuotaGuard - Quota enforcement service, inference_guard: InferenceGuard - Shared inference concurrency guard, metrics: OperationalMetricsRegistry - Operational metrics facade }
#   OUTPUTS: { instance - Immutable runtime composition root }
#   SIDE_EFFECTS: none
#   LINKS: M-BOOTSTRAP
# END_CONTRACT: CoreRuntime
@dataclass(frozen=True)
class CoreRuntime:
    settings: CoreSettings
    backend_registry: BackendRegistry
    registry: ModelRegistry
    model_lifecycle: ModelLifecycleService
    result_cache: ResultCache
    telemetry: TelemetryState
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
    raise ValueError(f"Unsupported job artifact backend: {settings.job_artifact_backend}")


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
    raise ValueError(f"Unsupported job metadata backend: {settings.job_metadata_backend}")


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
        return LocalBoundedExecutionManager(store=store, executor=executor, metrics=metrics)
    raise ValueError(f"Unsupported job execution backend: {settings.job_execution_backend}")


# START_CONTRACT: build_backends
#   PURPOSE: Auto-discover every concrete TTSBackend subclass currently registered (via __subclasses__() and entry_points) and build one instance per class through TTSBackend.from_settings(), so the bootstrap no longer hardcodes which backend classes exist.
#   INPUTS: { settings: CoreSettings - Runtime settings, metrics: OperationalMetricsRegistry - Shared metrics facade threaded into every backend }
#   OUTPUTS: { tuple[TTSBackend, ...] - Constructed backends in deterministic discovery order; backends whose construction raises are skipped with a warning log }
#   SIDE_EFFECTS: Logs a warning whenever a discovered backend class fails to construct
#   LINKS: M-BACKENDS, M-DISCOVERY, M-BOOTSTRAP
# END_CONTRACT: build_backends
def build_backends(
    settings: CoreSettings,
    *,
    metrics: OperationalMetricsRegistry,
) -> tuple[TTSBackend, ...]:
    # START_BLOCK_BUILD_BACKENDS
    instances: list[TTSBackend] = []
    for backend_cls in discover_backend_classes():
        try:
            instances.append(backend_cls.from_settings(settings, metrics=metrics))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "[Bootstrap][build_backends][SKIP] backend=%s reason=%s",
                f"{backend_cls.__module__}.{backend_cls.__qualname__}",
                exc,
            )
    return tuple(instances)
    # END_BLOCK_BUILD_BACKENDS


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
    telemetry = configure_telemetry(settings)
    inference_guard = InferenceGuard()
    metrics = OperationalMetricsRegistry()
    # END_BLOCK_INIT_INFRASTRUCTURE
    # START_BLOCK_INIT_BACKENDS
    backends = build_backends(settings, metrics=metrics)
    if not backends:
        raise RuntimeError(
            "No backend implementations were discovered. "
            "Ensure at least one TTSBackend subclass is importable or "
            "registered via the 'tts_server.backends' entry-point group."
        )
    model_manifest = load_composite_manifest(
        base_path=settings.model_manifest_path,
        models_dir=settings.models_dir,
    )
    backend_registry = BackendRegistry(
        backends,
        requested_backend=settings.backend,
        autoselect=settings.backend_autoselect,
        allow_unready_selection=True,
        model_manifest=model_manifest,
    )
    # END_BLOCK_INIT_BACKENDS
    # START_BLOCK_INIT_SERVICES
    registry = ModelRegistry(
        backend_registry=backend_registry,
        preload_policy=settings.model_preload_policy,
        preload_model_ids=settings.model_preload_ids,
        metrics=metrics,
    )
    if settings.result_cache_enabled:
        result_cache: ResultCache = FileSystemResultCache(
            settings.result_cache_dir,
            max_entries=settings.result_cache_max_entries,
        )
    else:
        result_cache = NullResultCache()
    tts_service = TTSService(
        registry=cast(RuntimeExecutionRegistry, registry),
        settings=settings,
        inference_guard=inference_guard,
        result_cache=result_cache,
    )
    application = TTSApplicationService(tts_service=tts_service)
    model_lifecycle = ModelLifecycleService(
        models_dir=settings.models_dir,
        registry=registry,
    )
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
        model_lifecycle=model_lifecycle,
        result_cache=result_cache,
        telemetry=telemetry,
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
    "build_backends",
    "build_job_artifact_store",
    "build_job_execution_backend",
    "build_job_metadata_store",
    "build_runtime",
]
