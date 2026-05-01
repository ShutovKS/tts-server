# FILE: core/infrastructure/job_execution_local.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide local in-memory implementations for job execution and metadata storage.
#   SCOPE: LocalBoundedExecutionManager, LocalInMemoryJobStore, LocalJobArtifactStore
#   DEPENDS: M-CONTRACTS, M-ERRORS, M-METRICS
#   LINKS: M-INFRASTRUCTURE
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LocalBoundedExecutionManager - Local async job execution with bounded concurrency
#   LocalInMemoryJobStore - In-memory job metadata store
#   LocalJobArtifactStore - In-memory job artifact storage
#   LocalJobArtifactHandler - Alias for the local job artifact cleanup implementation
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from queue import Queue
from threading import BoundedSemaphore, Condition, Event, Lock, Thread

from core.application.job_execution import (
    JobArtifactStore,
    JobExecutionBackend,
    JobExecutor,
    JobMetadataStore,
)
from core.contracts.jobs import (
    JobCreateResolution,
    JobFailureSnapshot,
    JobResultResolution,
    JobSnapshot,
    JobStatus,
    JobStatusTransition,
    JobSubmission,
    JobSuccessSnapshot,
    StoredJob,
    apply_job_transition,
    create_queued_job,
)
from core.errors import (
    JobIdempotencyConflictError,
    JobNotCancellableError,
    JobQueueFullError,
)
from core.metrics import OperationalMetricsRegistry


# START_CONTRACT: LocalJobArtifactStore
#   PURPOSE: Clean up staged filesystem artifacts associated with locally executed jobs.
#   INPUTS: {}
#   OUTPUTS: { instance - Local artifact cleanup implementation }
#   SIDE_EFFECTS: Deletes staged files from the local filesystem
#   LINKS: M-INFRASTRUCTURE
# END_CONTRACT: LocalJobArtifactStore
@dataclass(frozen=True)
class LocalJobArtifactStore(JobArtifactStore):
    def cleanup_submission_artifacts(self, submission: JobSubmission) -> None:
        self.cleanup_paths(submission.staged_input_paths)

    def cleanup_paths(self, paths: tuple[Path, ...]) -> None:
        for path in paths:
            with suppress(FileNotFoundError):
                path.unlink()


LocalJobArtifactHandler = LocalJobArtifactStore


# START_CONTRACT: LocalInMemoryJobStore
#   PURPOSE: Persist async job metadata and idempotency state in process-local memory.
#   INPUTS: { artifact_store: JobArtifactStore - Artifact cleanup dependency for stored jobs, retention_ttl_seconds: float - Time-to-live for completed job records }
#   OUTPUTS: { instance - In-memory job metadata store }
#   SIDE_EFFECTS: Mutates in-memory job and idempotency indexes and triggers artifact cleanup on terminal jobs
#   LINKS: M-INFRASTRUCTURE
# END_CONTRACT: LocalInMemoryJobStore
@dataclass
class LocalInMemoryJobStore(JobMetadataStore):
    artifact_store: JobArtifactStore = field(default_factory=LocalJobArtifactStore)
    retention_ttl_seconds: float = 300.0

    def __post_init__(self) -> None:
        self._jobs: dict[str, StoredJob] = {}
        self._idempotency_index: dict[tuple[str | None, str], str] = {}
        self._lock = Lock()

    def create(self, submission: JobSubmission) -> JobSnapshot:
        resolution = self.create_or_get(submission)
        if not resolution.created and submission.idempotency_key is None:
            raise RuntimeError("Unexpected existing job resolution without idempotency key")
        return resolution.snapshot

    def create_or_get(self, submission: JobSubmission) -> JobCreateResolution:
        with self._lock:
            # START_BLOCK_PURGE_EXPIRED_JOBS
            self._purge_expired_locked()
            # END_BLOCK_PURGE_EXPIRED_JOBS
            # START_BLOCK_REUSE_IDEMPOTENT_JOB
            if submission.idempotency_key is not None:
                existing_job = self._get_by_idempotency_key_locked(
                    submission.idempotency_key,
                    scope=submission.idempotency_scope,
                )
                if existing_job is not None:
                    self._ensure_idempotency_match(existing_job, submission)
                    return JobCreateResolution(snapshot=existing_job.snapshot, created=False)
            # END_BLOCK_REUSE_IDEMPOTENT_JOB
            # START_BLOCK_CREATE_NEW_JOB
            job = create_queued_job(submission=submission)
            self._jobs[job.snapshot.job_id] = job
            if submission.idempotency_key is not None:
                self._idempotency_index[
                    (submission.idempotency_scope, submission.idempotency_key)
                ] = job.snapshot.job_id
            return JobCreateResolution(snapshot=job.snapshot, created=True)
            # END_BLOCK_CREATE_NEW_JOB

    def get(self, job_id: str) -> StoredJob | None:
        with self._lock:
            self._purge_expired_locked()
            return self._jobs.get(job_id)

    def get_snapshot(self, job_id: str) -> JobSnapshot | None:
        job = self.get(job_id)
        return None if job is None else job.snapshot

    def count_active_jobs_for_principal(self, principal_id: str) -> int:
        with self._lock:
            self._purge_expired_locked()
            return sum(
                1
                for job in self._jobs.values()
                if job.snapshot.owner_principal_id == principal_id
                and job.snapshot.status in {JobStatus.QUEUED, JobStatus.RUNNING}
            )

    def get_by_idempotency_key(
        self, idempotency_key: str, *, scope: str | None = None
    ) -> StoredJob | None:
        with self._lock:
            self._purge_expired_locked()
            return self._get_by_idempotency_key_locked(idempotency_key, scope=scope)

    def mark_running(self, job_id: str, *, started_at: datetime | None = None) -> JobSnapshot:
        changed_at = started_at or datetime.now(UTC)
        with self._lock:
            job = self._require_job_locked(job_id)
            updated = apply_job_transition(
                job=job,
                transition=JobStatusTransition(
                    from_status=JobStatus.QUEUED,
                    to_status=JobStatus.RUNNING,
                    changed_at=changed_at,
                ),
            )
            self._jobs[job_id] = updated
            return updated.snapshot

    def mark_succeeded(
        self,
        job_id: str,
        *,
        success: JobSuccessSnapshot,
        completed_at: datetime | None = None,
    ) -> JobSnapshot:
        changed_at = completed_at or datetime.now(UTC)
        with self._lock:
            job = self._require_job_locked(job_id)
            updated = apply_job_transition(
                job=job,
                transition=JobStatusTransition(
                    from_status=JobStatus.RUNNING,
                    to_status=JobStatus.SUCCEEDED,
                    changed_at=changed_at,
                ),
                backend=success.generation.backend,
                saved_path=success.generation.saved_path,
                success=success,
                retention_expires_at=self._build_retention_expiry(changed_at),
            )
            self._jobs[job_id] = updated
            self.artifact_store.cleanup_submission_artifacts(updated.submission)
            return updated.snapshot

    def mark_failed(
        self,
        job_id: str,
        *,
        status: JobStatus,
        failure: JobFailureSnapshot,
        completed_at: datetime | None = None,
    ) -> JobSnapshot:
        if status not in {JobStatus.FAILED, JobStatus.TIMEOUT, JobStatus.CANCELLED}:
            raise ValueError(f"Unsupported terminal failure status: {status.value}")
        changed_at = completed_at or datetime.now(UTC)
        with self._lock:
            job = self._require_job_locked(job_id)
            updated = apply_job_transition(
                job=job,
                transition=JobStatusTransition(
                    from_status=JobStatus.RUNNING,
                    to_status=status,
                    changed_at=changed_at,
                ),
                failure=failure,
                retention_expires_at=self._build_retention_expiry(changed_at),
            )
            self._jobs[job_id] = updated
            self.artifact_store.cleanup_submission_artifacts(updated.submission)
            return updated.snapshot

    def cancel(self, job_id: str, *, completed_at: datetime | None = None) -> JobSnapshot:
        changed_at = completed_at or datetime.now(UTC)
        with self._lock:
            job = self._require_job_locked(job_id)
            if job.snapshot.status is JobStatus.CANCELLED:
                return job.snapshot
            updated = apply_job_transition(
                job=job,
                transition=JobStatusTransition(
                    from_status=JobStatus.QUEUED,
                    to_status=JobStatus.CANCELLED,
                    changed_at=changed_at,
                ),
                failure=JobFailureSnapshot(
                    code="job_cancelled",
                    message="Job was cancelled before execution started",
                ),
                retention_expires_at=self._build_retention_expiry(changed_at),
            )
            self._jobs[job_id] = updated
            self.artifact_store.cleanup_submission_artifacts(updated.submission)
            return updated.snapshot

    def get_result(self, job_id: str) -> JobResultResolution | None:
        with self._lock:
            self._purge_expired_locked()
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return JobResultResolution(
                snapshot=job.snapshot,
                success=job.success,
                terminal_state=job.terminal_state,
            )

    def _build_retention_expiry(self, completed_at: datetime) -> datetime:
        return completed_at + timedelta(seconds=self.retention_ttl_seconds)

    def _purge_expired_locked(self) -> None:
        # START_BLOCK_COLLECT_EXPIRED_JOBS
        now = datetime.now(UTC)
        expired_ids = [
            job_id
            for job_id, job in self._jobs.items()
            if job.snapshot.retention_expires_at is not None
            and job.snapshot.retention_expires_at <= now
        ]
        # END_BLOCK_COLLECT_EXPIRED_JOBS
        # START_BLOCK_REMOVE_EXPIRED_JOBS
        for job_id in expired_ids:
            job = self._jobs.pop(job_id)
            if job.submission.idempotency_key is not None:
                index_key = (
                    job.submission.idempotency_scope,
                    job.submission.idempotency_key,
                )
                indexed_job_id = self._idempotency_index.get(index_key)
                if indexed_job_id == job_id:
                    self._idempotency_index.pop(index_key, None)
            self.artifact_store.cleanup_submission_artifacts(job.submission)
        # END_BLOCK_REMOVE_EXPIRED_JOBS

    def _require_job_locked(self, job_id: str) -> StoredJob:
        self._purge_expired_locked()
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def _get_by_idempotency_key_locked(
        self, idempotency_key: str, *, scope: str | None = None
    ) -> StoredJob | None:
        job_id = self._idempotency_index.get((scope, idempotency_key))
        if job_id is None:
            return None
        return self._jobs.get(job_id)

    @staticmethod
    def _ensure_idempotency_match(existing_job: StoredJob, submission: JobSubmission) -> None:
        existing_fingerprint = existing_job.submission.idempotency_fingerprint
        requested_fingerprint = submission.idempotency_fingerprint
        if existing_fingerprint != requested_fingerprint:
            raise JobIdempotencyConflictError(
                idempotency_key=submission.idempotency_key or "unknown",
                existing_job_id=existing_job.snapshot.job_id,
                reason="Idempotency key was already used with a different payload",
                details={
                    "job_id": existing_job.snapshot.job_id,
                    "operation": existing_job.snapshot.operation.value,
                    "mode": existing_job.snapshot.mode,
                },
            )


# START_CONTRACT: LocalBoundedExecutionManager
#   PURPOSE: Execute queued jobs locally with bounded queue capacity and worker concurrency.
#   INPUTS: { store: JobMetadataStore - Job metadata store used for lifecycle updates, executor: JobExecutor - Executor used to run submissions, worker_count: int - Number of worker threads, queue_capacity: int - Maximum number of queued jobs, metrics: OperationalMetricsRegistry | None - Optional metrics facade for execution tracking }
#   OUTPUTS: { instance - Local bounded execution backend }
#   SIDE_EFFECTS: Spawns worker threads, mutates in-memory queue state, updates job records, and records execution metrics
#   LINKS: M-INFRASTRUCTURE
# END_CONTRACT: LocalBoundedExecutionManager
@dataclass
class LocalBoundedExecutionManager(JobExecutionBackend):
    store: JobMetadataStore
    executor: JobExecutor
    worker_count: int = 1
    queue_capacity: int = 16
    metrics: OperationalMetricsRegistry | None = None
    _queue: deque[str] = field(init=False, repr=False)
    _condition: Condition = field(init=False, repr=False)
    _stop_event: Event = field(init=False, repr=False)
    _queue_slots: BoundedSemaphore = field(init=False, repr=False)
    _workers: list[Thread] = field(init=False, repr=False)
    _started: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if self.worker_count < 1:
            raise ValueError("worker_count must be at least 1")
        if self.queue_capacity < 1:
            raise ValueError("queue_capacity must be at least 1")
        self.metrics = self.metrics or OperationalMetricsRegistry()
        self._queue = deque()
        self._condition = Condition(Lock())
        self._stop_event = Event()
        self._queue_slots = BoundedSemaphore(self.queue_capacity)
        self._workers = []
        self._update_queue_depth_metrics()

    def start(self) -> None:
        with self._condition:
            # START_BLOCK_GUARD_MANAGER_STARTUP
            if self._started:
                return
            self._stop_event.clear()
            # END_BLOCK_GUARD_MANAGER_STARTUP
            # START_BLOCK_SPAWN_WORKERS
            self._workers = [
                Thread(target=self._worker_loop, name=f"job-worker-{index}", daemon=True)
                for index in range(self.worker_count)
            ]
            for worker in self._workers:
                worker.start()
            self._started = True
            # END_BLOCK_SPAWN_WORKERS

    def stop(self, *, drain: bool = False) -> None:
        workers: list[Thread]
        with self._condition:
            # START_BLOCK_GUARD_MANAGER_SHUTDOWN
            if not self._started:
                return
            self._stop_event.set()
            # END_BLOCK_GUARD_MANAGER_SHUTDOWN
            # START_BLOCK_PREPARE_WORKER_SHUTDOWN
            if not drain:
                self._queue.clear()
                self._update_queue_depth_metrics_locked()
            self._condition.notify_all()
            workers = list(self._workers)
            self._workers = []
            self._started = False
            # END_BLOCK_PREPARE_WORKER_SHUTDOWN
        # START_BLOCK_JOIN_WORKERS
        for worker in workers:
            worker.join(timeout=1.0)
        # END_BLOCK_JOIN_WORKERS

    def submit(self, submission: JobSubmission) -> JobSnapshot:
        return self.submit_idempotent(submission).snapshot

    def submit_idempotent(self, submission: JobSubmission) -> JobCreateResolution:
        # START_BLOCK_ENSURE_EXECUTION_STARTED
        self.start()
        # END_BLOCK_ENSURE_EXECUTION_STARTED
        # START_BLOCK_CHECK_EXISTING_SUBMISSION
        if submission.idempotency_key is not None:
            existing_job = self.store.get_by_idempotency_key(
                submission.idempotency_key,
                scope=submission.idempotency_scope,
            )
            if existing_job is not None:
                return self.store.create_or_get(submission)
        # END_BLOCK_CHECK_EXISTING_SUBMISSION

        # START_BLOCK_ACQUIRE_QUEUE_SLOT
        if not self._queue_slots.acquire(blocking=False):
            raise JobQueueFullError("Local job queue is full")
        # END_BLOCK_ACQUIRE_QUEUE_SLOT
        try:
            # START_BLOCK_PERSIST_AND_ENQUEUE_JOB
            resolution = self.store.create_or_get(submission)
            if not resolution.created:
                self._queue_slots.release()
                return resolution
            self.metrics.collector.increment("jobs.submitted")
            with self._condition:
                self._queue.append(resolution.snapshot.job_id)
                self._update_queue_depth_metrics_locked()
                self._condition.notify()
            return resolution
            # END_BLOCK_PERSIST_AND_ENQUEUE_JOB
        except Exception:
            # START_BLOCK_RELEASE_QUEUE_SLOT_ON_FAILURE
            self._queue_slots.release()
            raise
            # END_BLOCK_RELEASE_QUEUE_SLOT_ON_FAILURE

    def cancel(self, job_id: str) -> JobSnapshot | None:
        # START_BLOCK_RESOLVE_CANCELLATION_TARGET
        snapshot = self.store.get_snapshot(job_id)
        if snapshot is None:
            return None
        # END_BLOCK_RESOLVE_CANCELLATION_TARGET
        # START_BLOCK_CANCEL_QUEUED_JOB
        if snapshot.status is JobStatus.QUEUED:
            cancelled = self.store.cancel(job_id)
            self.metrics.collector.increment("jobs.cancelled")
            with self._condition:
                with suppress(ValueError):
                    self._queue.remove(job_id)
                self._update_queue_depth_metrics_locked()
            return cancelled
        # END_BLOCK_CANCEL_QUEUED_JOB
        # START_BLOCK_REJECT_RUNNING_CANCELLATION
        if snapshot.status is JobStatus.RUNNING:
            raise JobNotCancellableError(job_id, snapshot.status.value)
        # END_BLOCK_REJECT_RUNNING_CANCELLATION
        # START_BLOCK_RETURN_TERMINAL_SNAPSHOT
        return snapshot
        # END_BLOCK_RETURN_TERMINAL_SNAPSHOT

    def _worker_loop(self) -> None:
        while True:
            # START_BLOCK_DEQUEUE_WORK_ITEM
            job_id = self._dequeue_job_id()
            if job_id is None:
                return
            # END_BLOCK_DEQUEUE_WORK_ITEM
            try:
                # START_BLOCK_LOAD_JOB_STATE
                stored_job = self.store.get(job_id)
                if stored_job is None:
                    continue
                if stored_job.snapshot.status is JobStatus.CANCELLED:
                    continue
                # END_BLOCK_LOAD_JOB_STATE
                # START_BLOCK_MARK_JOB_RUNNING
                try:
                    self.store.mark_running(job_id)
                    self.metrics.collector.increment("jobs.started")
                except ValueError:
                    continue
                # END_BLOCK_MARK_JOB_RUNNING
                # START_BLOCK_EXECUTE_JOB_PAYLOAD
                running_job = self.store.get(job_id)
                if running_job is None:
                    continue
                self._execute_running_job(running_job)
                # END_BLOCK_EXECUTE_JOB_PAYLOAD
            finally:
                # START_BLOCK_RELEASE_WORKER_SLOT
                self._queue_slots.release()
                # END_BLOCK_RELEASE_WORKER_SLOT

    def _dequeue_job_id(self) -> str | None:
        with self._condition:
            while not self._queue and not self._stop_event.is_set():
                self._condition.wait(timeout=0.1)
            if not self._queue:
                return None
            job_id = self._queue.popleft()
            self._update_queue_depth_metrics_locked()
            return job_id

    def _execute_running_job(self, job: StoredJob) -> None:
        result_queue: Queue[tuple[str, object]] = Queue(maxsize=1)

        def run_job() -> None:
            try:
                result = self.executor.execute(job.submission)
            except Exception as exc:  # pragma: no cover - exercised via queue outcome assertion
                result_queue.put(("error", exc))
                return
            result_queue.put(("success", result))

        # START_BLOCK_START_EXECUTION_THREAD
        execution_thread = Thread(
            target=run_job, name=f"job-runner-{job.snapshot.job_id}", daemon=True
        )
        execution_thread.start()
        execution_thread.join(timeout=job.submission.execution_timeout_seconds)
        # END_BLOCK_START_EXECUTION_THREAD

        # START_BLOCK_HANDLE_TIMEOUT_RESULT
        if execution_thread.is_alive():
            self.store.mark_failed(
                job.snapshot.job_id,
                status=JobStatus.TIMEOUT,
                failure=JobFailureSnapshot(
                    code="job_execution_timeout",
                    message="Job execution exceeded configured timeout",
                    details={"timeout_seconds": job.submission.execution_timeout_seconds},
                ),
            )
            self.metrics.collector.increment("jobs.timeout")
            return
        # END_BLOCK_HANDLE_TIMEOUT_RESULT

        # START_BLOCK_HANDLE_MISSING_RESULT
        if result_queue.empty():
            self.store.mark_failed(
                job.snapshot.job_id,
                status=JobStatus.FAILED,
                failure=JobFailureSnapshot(
                    code="job_execution_failed",
                    message="Job execution finished without a result",
                ),
            )
            self.metrics.collector.increment("jobs.failed")
            return
        # END_BLOCK_HANDLE_MISSING_RESULT

        # START_BLOCK_HANDLE_EXECUTION_OUTCOME
        outcome, payload = result_queue.get_nowait()
        if outcome == "success":
            self.store.mark_succeeded(
                job.snapshot.job_id,
                success=JobSuccessSnapshot(generation=payload),
            )
            self.metrics.collector.increment("jobs.completed")
            return

        self.store.mark_failed(
            job.snapshot.job_id,
            status=JobStatus.FAILED,
            failure=JobFailureSnapshot(
                code="job_execution_failed",
                message="Job execution failed",
                details={"reason": str(payload), "error_type": type(payload).__name__},
            ),
        )
        self.metrics.collector.increment("jobs.failed")
        # END_BLOCK_HANDLE_EXECUTION_OUTCOME

    def _update_queue_depth_metrics(self) -> None:
        with self._condition:
            self._update_queue_depth_metrics_locked()

    def _update_queue_depth_metrics_locked(self) -> None:
        queue_depth = len(self._queue)
        self.metrics.collector.set_gauge("jobs.queue.depth.current", queue_depth)
        peak = self.metrics.execution_summary()["queue_depth"]["peak"]
        if queue_depth > peak:
            self.metrics.collector.set_gauge("jobs.queue.depth.peak", queue_depth)


__all__ = [
    "LocalBoundedExecutionManager",
    "LocalInMemoryJobStore",
    "LocalJobArtifactHandler",
    "LocalJobArtifactStore",
]
