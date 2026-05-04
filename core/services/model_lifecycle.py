# FILE: core/services/model_lifecycle.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide a process-local lifecycle facade for model management — listing, deleting, refreshing, cache invalidation, and submitting/observing best-effort downloads — so the HTTP control plane can exercise these operations without reaching into the model registry directly.
#   SCOPE: ModelDownloadJob descriptor, ModelDownloadStatus literal-string set, default no-op downloader, and ModelLifecycleService with delete_model / submit_download / get_download / list_downloads / refresh helpers backed by the active ModelRegistry, optional EngineRegistry cache invalidation, and the configured models_dir on disk.
#   DEPENDS: M-MODEL-REGISTRY, M-MODELS, M-CONFIG, M-ENGINE-REGISTRY
#   LINKS: M-MODEL-LIFECYCLE
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LOGGER - Module logger for lifecycle events.
#   MODEL_DOWNLOAD_STATUSES - Tuple of legal ModelDownloadJob.status values.
#   ModelDownloadJob - Frozen descriptor for a single download submission with status, progress, and timestamps.
#   DownloaderCallable - Callable signature used by ModelLifecycleService to perform a download.
#   default_downloader - Default downloader that always reports "no_downloader_configured" so callers see a deterministic failure when no downloader is wired in.
#   ModelLifecycleService - Process-local lifecycle facade used by the HTTP control plane.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Added engine cache invalidation during refresh so model-lifecycle reload clears CachedEngine hot state together with registry/catalog refresh]
# END_CHANGE_SUMMARY

from __future__ import annotations

import shutil
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from core.engines.registry import EngineRegistry
from core.models.catalog import ModelSpec
from core.observability import get_logger, log_event

LOGGER = get_logger(__name__)

MODEL_DOWNLOAD_STATUSES: tuple[str, ...] = ("pending", "running", "succeeded", "failed")


# START_CONTRACT: ModelDownloadJob
#   PURPOSE: Describe a single model-download submission so the control plane can track its progress without coupling to a specific downloader implementation.
#   INPUTS: { id: str - Stable job identifier, model_id: str - Target model identifier, source: str | None - Optional source descriptor (e.g. a HuggingFace repo id), status: str - One of MODEL_DOWNLOAD_STATUSES, progress: float - Best-effort 0..1 progress estimate, error: str | None - Reason when status is "failed", created_at: float - Unix timestamp when the job was submitted, updated_at: float - Unix timestamp of the most recent status mutation, completed_at: float | None - Unix timestamp when the job reached a terminal state, details: Mapping[str, Any] - Free-form structured details supplied by the downloader }
#   OUTPUTS: { instance - Immutable job descriptor }
#   SIDE_EFFECTS: none
#   LINKS: M-MODEL-LIFECYCLE
# END_CONTRACT: ModelDownloadJob
@dataclass(frozen=True)
class ModelDownloadJob:
    id: str
    model_id: str
    source: str | None
    status: str
    progress: float
    error: str | None
    created_at: float
    updated_at: float
    completed_at: float | None
    details: dict[str, Any] = field(default_factory=dict)


DownloaderCallable = Callable[[ModelDownloadJob, Path], ModelDownloadJob]


# START_CONTRACT: default_downloader
#   PURPOSE: Provide a deterministic "no downloader configured" failure when ModelLifecycleService is instantiated without an injected downloader so submissions surface a stable error instead of hanging.
#   INPUTS: { job: ModelDownloadJob - Submission descriptor, target_dir: Path - Directory the downloader would have populated }
#   OUTPUTS: { ModelDownloadJob - Updated descriptor with status "failed" and a "no_downloader_configured" error }
#   SIDE_EFFECTS: none
#   LINKS: M-MODEL-LIFECYCLE
# END_CONTRACT: default_downloader
def default_downloader(job: ModelDownloadJob, target_dir: Path) -> ModelDownloadJob:
    now = time.time()
    return replace(
        job,
        status="failed",
        progress=0.0,
        error="no_downloader_configured",
        updated_at=now,
        completed_at=now,
        details={"target_dir": str(target_dir)},
    )


# START_CONTRACT: ModelLifecycleService
#   PURPOSE: Process-local lifecycle facade exposing model deletion, refresh, cache invalidation, and best-effort download orchestration to transports.
#   INPUTS: { models_dir: Path - Filesystem root where model folders live, registry: Any - Active ModelRegistry-like object exposing model_specs and reload_manifest hooks, engine_registry: EngineRegistry | None - Optional runtime engine registry whose cacheable engines should be invalidated on refresh, downloader: DownloaderCallable | None - Optional injected downloader; defaults to default_downloader }
#   OUTPUTS: { instance - Lifecycle service instance }
#   SIDE_EFFECTS: Mutates the local filesystem under models_dir when deleting models or running downloaders, and runs downloads on background threads.
#   LINKS: M-MODEL-LIFECYCLE
# END_CONTRACT: ModelLifecycleService
class ModelLifecycleService:
    def __init__(
        self,
        *,
        models_dir: Path,
        registry: Any,
        engine_registry: EngineRegistry | None = None,
        downloader: DownloaderCallable | None = None,
    ) -> None:
        self._models_dir = Path(models_dir)
        self._registry = registry
        self._engine_registry = engine_registry
        self._downloader: DownloaderCallable = downloader or default_downloader
        self._jobs: dict[str, ModelDownloadJob] = {}
        self._lock = threading.RLock()

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def _model_specs(self) -> tuple[ModelSpec, ...]:
        # START_BLOCK_RESOLVE_MODEL_SPECS
        try:
            specs = getattr(self._registry, "model_specs", None)
        except Exception:
            return ()
        if isinstance(specs, tuple):
            return specs
        if specs is None:
            return ()
        try:
            return tuple(specs)
        except TypeError:
            return ()
        # END_BLOCK_RESOLVE_MODEL_SPECS

    def find_spec(self, model_id: str) -> ModelSpec | None:
        # START_BLOCK_FIND_MODEL_SPEC
        for spec in self._model_specs():
            candidates = {spec.api_name, spec.folder, spec.key, spec.model_id}
            if model_id in candidates:
                return spec
        return None
        # END_BLOCK_FIND_MODEL_SPEC

    # START_CONTRACT: delete_model
    #   PURPOSE: Remove the on-disk artifacts of a registered model from the configured models_dir.
    #   INPUTS: { model_id: str - Identifier of the model to delete (matched against api_name/folder/key/model_id) }
    #   OUTPUTS: { bool - True when the folder existed and was removed, False when the spec is unknown or the folder was already absent }
    #   SIDE_EFFECTS: Recursively removes the model's folder under models_dir and emits a "[ModelLifecycle][delete_model][...]" log event.
    #   LINKS: M-MODEL-LIFECYCLE
    # END_CONTRACT: delete_model
    def delete_model(self, model_id: str) -> bool:
        spec = self.find_spec(model_id)
        if spec is None:
            log_event(
                LOGGER,
                level=30,
                event="[ModelLifecycle][delete_model][BLOCK_REJECT_UNKNOWN_MODEL]",
                message="Refusing to delete unknown model",
                model_id=model_id,
            )
            return False
        target = self._models_dir / spec.folder
        if not target.exists():
            log_event(
                LOGGER,
                level=20,
                event="[ModelLifecycle][delete_model][BLOCK_SKIP_MISSING_FOLDER]",
                message="Model folder is already absent",
                model_id=model_id,
                folder=spec.folder,
            )
            return False
        shutil.rmtree(target)
        log_event(
            LOGGER,
            level=20,
            event="[ModelLifecycle][delete_model][BLOCK_REMOVED_MODEL_FOLDER]",
            message="Removed model folder",
            model_id=model_id,
            folder=spec.folder,
            path=str(target),
        )
        return True

    # START_CONTRACT: refresh
    #   PURPOSE: Best-effort reload hook used after the on-disk model layout changes so transports can request a re-discovery without restarting the server.
    #   INPUTS: {}
    #   OUTPUTS: { dict[str, Any] - Refresh outcome with keys "supported" (bool), "model_count" (int), optionally "method" (str), and cache invalidation counts }
    #   SIDE_EFFECTS: May invoke the registry's reload_manifest/refresh hook, clears cacheable engine runtime state when an engine registry is supplied, and emits a "[ModelLifecycle][refresh][...]" log event either way.
    #   LINKS: M-MODEL-LIFECYCLE
    # END_CONTRACT: refresh
    def refresh(self) -> dict[str, Any]:
        # START_BLOCK_REFRESH_REGISTRY
        cleared_engine_caches = self._clear_engine_caches()
        for hook_name in ("reload_manifest", "refresh", "rebuild"):
            hook = getattr(self._registry, hook_name, None)
            if callable(hook):
                hook()
                count = len(self._model_specs())
                log_event(
                    LOGGER,
                    level=20,
                    event="[ModelLifecycle][refresh][BLOCK_REFRESHED_REGISTRY]",
                    message="Registry refresh hook invoked",
                    method=hook_name,
                    model_count=count,
                    cleared_engine_caches=cleared_engine_caches,
                )
                return {
                    "supported": True,
                    "method": hook_name,
                    "model_count": count,
                    "cleared_engine_caches": cleared_engine_caches,
                }
        count = len(self._model_specs())
        log_event(
            LOGGER,
            level=30,
            event="[ModelLifecycle][refresh][BLOCK_NO_REFRESH_HOOK]",
            message="Registry exposes no refresh hook; returning current snapshot",
            model_count=count,
            cleared_engine_caches=cleared_engine_caches,
        )
        return {
            "supported": False,
            "model_count": count,
            "cleared_engine_caches": cleared_engine_caches,
        }
        # END_BLOCK_REFRESH_REGISTRY

    def _clear_engine_caches(self) -> int:
        if self._engine_registry is None:
            return 0
        cleared = 0
        for engine in self._engine_registry.registered_engines:
            clear_cache = getattr(engine, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()
                cleared += 1
        return cleared

    # START_CONTRACT: submit_download
    #   PURPOSE: Submit a best-effort download for a registered model and run it on a background thread so the HTTP control plane returns immediately.
    #   INPUTS: { model_id: str - Target model identifier, source: str | None - Optional source descriptor passed through to the downloader, run_async: bool - True to run the download on a background thread (default), False to run synchronously (used by tests) }
    #   OUTPUTS: { ModelDownloadJob - Newly created job descriptor; when run_async is True its status is "pending" or "running", when False the returned descriptor reflects the final status }
    #   SIDE_EFFECTS: Stores the job descriptor in the in-memory job map, may spawn a background thread, and lets the configured downloader mutate the on-disk models_dir.
    #   LINKS: M-MODEL-LIFECYCLE
    # END_CONTRACT: submit_download
    def submit_download(
        self,
        model_id: str,
        *,
        source: str | None = None,
        run_async: bool = True,
    ) -> ModelDownloadJob:
        spec = self.find_spec(model_id)
        now = time.time()
        target_dir = (self._models_dir / spec.folder) if spec is not None else self._models_dir
        job = ModelDownloadJob(
            id=str(uuid.uuid4()),
            model_id=model_id,
            source=source,
            status="pending",
            progress=0.0,
            error=None,
            created_at=now,
            updated_at=now,
            completed_at=None,
            details={"target_dir": str(target_dir)},
        )
        with self._lock:
            self._jobs[job.id] = job
        log_event(
            LOGGER,
            level=20,
            event="[ModelLifecycle][submit_download][BLOCK_QUEUED_DOWNLOAD]",
            message="Model download submission queued",
            job_id=job.id,
            model_id=model_id,
            source=source,
            target_dir=str(target_dir),
        )
        if spec is None:
            failed = replace(
                job,
                status="failed",
                error="unknown_model_id",
                updated_at=time.time(),
                completed_at=time.time(),
            )
            with self._lock:
                self._jobs[job.id] = failed
            return failed
        if run_async:
            thread = threading.Thread(
                target=self._run_download,
                args=(job.id, target_dir),
                name=f"model-download-{job.id}",
                daemon=True,
            )
            thread.start()
            return job
        self._run_download(job.id, target_dir)
        with self._lock:
            return self._jobs[job.id]

    def _run_download(self, job_id: str, target_dir: Path) -> None:
        # START_BLOCK_RUN_DOWNLOAD_JOB
        with self._lock:
            current = self._jobs.get(job_id)
        if current is None:
            return
        running = replace(current, status="running", updated_at=time.time())
        with self._lock:
            self._jobs[job_id] = running
        try:
            outcome = self._downloader(running, target_dir)
        except Exception as exc:
            outcome = replace(
                running,
                status="failed",
                error=str(exc),
                updated_at=time.time(),
                completed_at=time.time(),
            )
            log_event(
                LOGGER,
                level=40,
                event="[ModelLifecycle][_run_download][BLOCK_DOWNLOAD_RAISED]",
                message="Downloader raised an exception",
                job_id=job_id,
                error=str(exc),
            )
        with self._lock:
            self._jobs[job_id] = outcome
        log_event(
            LOGGER,
            level=20,
            event="[ModelLifecycle][_run_download][BLOCK_DOWNLOAD_COMPLETED]",
            message="Model download finished",
            job_id=job_id,
            status=outcome.status,
            error=outcome.error,
        )
        # END_BLOCK_RUN_DOWNLOAD_JOB

    # START_CONTRACT: get_download
    #   PURPOSE: Look up a previously submitted download job by id.
    #   INPUTS: { job_id: str - Identifier returned by submit_download }
    #   OUTPUTS: { ModelDownloadJob | None - Stored descriptor or None when the job is unknown }
    #   SIDE_EFFECTS: none
    #   LINKS: M-MODEL-LIFECYCLE
    # END_CONTRACT: get_download
    def get_download(self, job_id: str) -> ModelDownloadJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    # START_CONTRACT: list_downloads
    #   PURPOSE: Return all submitted download jobs in submission order.
    #   INPUTS: {}
    #   OUTPUTS: { tuple[ModelDownloadJob, ...] - Snapshot of every job currently tracked }
    #   SIDE_EFFECTS: none
    #   LINKS: M-MODEL-LIFECYCLE
    # END_CONTRACT: list_downloads
    def list_downloads(self) -> tuple[ModelDownloadJob, ...]:
        with self._lock:
            return tuple(sorted(self._jobs.values(), key=lambda job: job.created_at))


__all__ = [
    "DownloaderCallable",
    "MODEL_DOWNLOAD_STATUSES",
    "ModelDownloadJob",
    "ModelLifecycleService",
    "default_downloader",
]
