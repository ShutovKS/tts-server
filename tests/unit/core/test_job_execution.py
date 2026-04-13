# FILE: tests/unit/core/test_job_execution.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for core job execution, storage, and lifecycle management.
#   SCOPE: Submission, execution, cancellation, artifacts, snapshots
#   DEPENDS: M-CORE
#   LINKS: V-M-CORE
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   StubApplicationService - Fake synthesis application used by job execution tests
#   BlockingStore - In-memory job store variant that blocks mark_running for contention tests
#   _make_submission - Helper that builds deterministic async job submissions
#   _make_generation_result - Helper that builds deterministic generation results
#   _make_success_snapshot - Helper that builds deterministic job success snapshots
#   _wait_for_status - Poll helper for async job state transitions in tests
#   test_in_memory_job_store_creates_queued_job_snapshot - Verifies queued snapshot creation
#   test_in_memory_job_store_marks_running_and_succeeded_with_result - Verifies running and success transitions persist result data
#   test_local_in_memory_job_store_uses_artifact_handler_for_terminal_cleanup - Verifies staged input cleanup on success
#   test_in_memory_job_store_cancels_only_queued_jobs - Verifies cancellation rules for queued jobs
#   test_in_memory_job_store_rejects_invalid_terminal_failure_status - Verifies invalid terminal states are rejected
#   test_in_memory_job_store_purges_expired_terminal_jobs - Verifies retention cleanup removes expired terminal jobs
#   test_in_memory_job_store_returns_existing_job_for_same_idempotency_key_and_payload - Verifies idempotent replay returns stored jobs
#   test_in_memory_job_store_rejects_idempotency_key_reuse_for_different_payload - Verifies conflicting idempotency reuse is rejected
#   test_in_memory_job_store_allows_same_idempotency_key_for_different_principals - Verifies idempotency scope remains principal-bound
#   test_job_execution_gateway_without_manager_delegates_to_store - Verifies gateway uses the store when no manager is configured
#   test_job_execution_gateway_submit_idempotent_without_manager_delegates_to_store - Verifies idempotent gateway path without a manager
#   test_in_memory_job_executor_dispatches_by_operation - Verifies executor dispatches custom/design/clone operations correctly
#   test_local_bounded_execution_manager_runs_job_to_success - Verifies local manager drives successful execution
#   test_local_bounded_execution_manager_returns_existing_job_for_idempotent_replay - Verifies manager idempotent replay behavior
#   test_local_bounded_execution_manager_marks_failures - Verifies worker failures become terminal job failures
#   test_local_bounded_execution_manager_marks_timeouts_from_running_state - Verifies execution timeouts become timeout terminal states
#   test_local_bounded_execution_manager_cancels_queued_jobs - Verifies queued jobs can be cancelled before running
#   test_local_bounded_execution_manager_rejects_running_job_cancellation - Verifies running jobs are not cancellable
#   test_local_bounded_execution_manager_rejects_submit_when_queue_is_full - Verifies bounded queues reject overflow
#   test_job_execution_gateway_with_manager_delegates_submit_and_cancel_to_manager - Verifies gateway delegates to configured manager
#   test_local_job_artifact_handler_cleans_up_staged_paths - Verifies artifact handler removes staged files
#   test_local_adapters_conform_to_explicit_job_ports - Verifies local adapters satisfy declared job protocols
#   test_build_job_wiring_uses_local_defaults - Verifies bootstrap wiring selects local job defaults
#   test_build_runtime_uses_local_job_ports_by_default - Verifies runtime bootstrap exposes local job ports by default
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT and MODULE_MAP]
# END_CHANGE_SUMMARY

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
import tempfile
from time import sleep

import pytest

from core.application.job_execution import (
    InMemoryJobExecutor,
    JobArtifactStore,
    JobExecutionBackend,
    JobExecutionGateway,
    JobMetadataStore,
    JobNotCancellableError,
    JobQueueFullError,
)
from core.contracts.commands import (
    CustomVoiceCommand,
    VoiceCloneCommand,
    VoiceDesignCommand,
)
from core.contracts.jobs import (
    JobOperation,
    JobStatus,
    JobSuccessSnapshot,
    create_job_submission,
)
from core.contracts.results import AudioResult, GenerationResult
from core.bootstrap import (
    build_job_artifact_store,
    build_job_execution_backend,
    build_job_metadata_store,
    build_runtime,
)
from core.config import (
    CoreSettings,
    DEFAULT_MODELS_DIR,
    DEFAULT_OUTPUTS_DIR,
    DEFAULT_UPLOAD_STAGING_DIR,
    DEFAULT_VOICES_DIR,
)
from core.errors import JobIdempotencyConflictError
from core.metrics import OperationalMetricsRegistry
from core.infrastructure.job_execution_local import (
    LocalBoundedExecutionManager,
    LocalInMemoryJobStore,
    LocalJobArtifactHandler,
    LocalJobArtifactStore,
)


pytestmark = pytest.mark.unit


TEST_TMP_ROOT = Path(tempfile.gettempdir()) / "qwen-tts-tests"


class StubApplicationService:
    def __init__(
        self,
        result: GenerationResult | None = None,
        error: Exception | None = None,
        *,
        started: Event | None = None,
        release: Event | None = None,
    ):
        self.result = result or _make_generation_result()
        self.error = error
        self.started = started
        self.release = release
        self.calls: list[tuple[str, object]] = []

    def synthesize_custom(self, command: CustomVoiceCommand) -> GenerationResult:
        self.calls.append(("custom", command))
        return self._resolve()

    def synthesize_design(self, command: VoiceDesignCommand) -> GenerationResult:
        self.calls.append(("design", command))
        return self._resolve()

    def synthesize_clone(self, command: VoiceCloneCommand) -> GenerationResult:
        self.calls.append(("clone", command))
        return self._resolve()

    def _resolve(self) -> GenerationResult:
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            self.release.wait(timeout=2.0)
        if self.error is not None:
            raise self.error
        return self.result


class BlockingStore(LocalInMemoryJobStore):
    def __init__(self) -> None:
        super().__init__()
        self.mark_running_started = Event()
        self.allow_mark_running = Event()

    def mark_running(self, job_id: str, *, started_at: datetime | None = None):
        self.mark_running_started.set()
        self.allow_mark_running.wait(timeout=2.0)
        return super().mark_running(job_id, started_at=started_at)


def _make_submission(
    *,
    request_id: str = "req-1",
    owner_principal_id: str = "local-default",
    timeout_seconds: float = 15.0,
    idempotency_key: str | None = None,
    idempotency_scope: str | None = None,
    idempotency_fingerprint: str | None = None,
):
    return create_job_submission(
        operation=JobOperation.SYNTHESIZE_CUSTOM,
        command=CustomVoiceCommand(text="hello", model="demo-model", save_output=False),
        submit_request_id=request_id,
        owner_principal_id=owner_principal_id,
        response_format="wav",
        save_output=False,
        execution_timeout_seconds=timeout_seconds,
        staged_input_paths=(TEST_TMP_ROOT / "staged-input.wav",),
        idempotency_key=idempotency_key,
        idempotency_scope=idempotency_scope,
        idempotency_fingerprint=idempotency_fingerprint,
    )


def _make_generation_result() -> GenerationResult:
    return GenerationResult(
        audio=AudioResult(path=TEST_TMP_ROOT / "audio.wav", bytes_data=b"audio-bytes"),
        saved_path=TEST_TMP_ROOT / "saved.wav",
        model="demo-model",
        mode="custom",
        backend="mlx",
    )


def _make_success_snapshot() -> JobSuccessSnapshot:
    return JobSuccessSnapshot(generation=_make_generation_result())


def _wait_for_status(
    store: LocalInMemoryJobStore,
    job_id: str,
    status: JobStatus,
    *,
    timeout: float = 2.0,
):
    deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout)
    last_snapshot = None
    while datetime.now(timezone.utc) < deadline:
        snapshot = store.get_snapshot(job_id)
        last_snapshot = snapshot
        if snapshot is not None and snapshot.status is status:
            return snapshot
        sleep(0.01)
    raise AssertionError(
        f"Timed out waiting for status {status.value}, last snapshot: {last_snapshot}"
    )


def test_in_memory_job_store_creates_queued_job_snapshot():
    store = LocalInMemoryJobStore()

    snapshot = store.create(_make_submission())

    assert snapshot.status is JobStatus.QUEUED
    assert snapshot.operation is JobOperation.SYNTHESIZE_CUSTOM
    assert snapshot.mode == "custom"
    assert snapshot.requested_model == "demo-model"
    assert snapshot.response_format == "wav"
    assert snapshot.submit_request_id == "req-1"
    assert snapshot.owner_principal_id == "local-default"
    assert snapshot.job_id


def test_in_memory_job_store_marks_running_and_succeeded_with_result():
    store = LocalInMemoryJobStore(retention_ttl_seconds=60.0)
    created = store.create(_make_submission())
    started_at = datetime.now(timezone.utc)
    completed_at = started_at + timedelta(seconds=3)

    running = store.mark_running(created.job_id, started_at=started_at)
    succeeded = store.mark_succeeded(
        created.job_id, success=_make_success_snapshot(), completed_at=completed_at
    )
    resolution = store.get_result(created.job_id)

    assert running.status is JobStatus.RUNNING
    assert running.started_at == started_at
    assert succeeded.status is JobStatus.SUCCEEDED
    assert succeeded.completed_at == completed_at
    assert succeeded.backend == "mlx"
    assert succeeded.saved_path == TEST_TMP_ROOT / "saved.wav"
    assert succeeded.retention_expires_at == completed_at + timedelta(seconds=60)
    assert resolution is not None
    assert resolution.snapshot.status is JobStatus.SUCCEEDED
    assert resolution.success is not None
    assert resolution.success.generation.audio.bytes_data == b"audio-bytes"
    assert resolution.terminal_state is not None
    assert resolution.terminal_state.status is JobStatus.SUCCEEDED
    assert resolution.terminal_state.success is not None


def test_local_in_memory_job_store_uses_artifact_handler_for_terminal_cleanup(
    tmp_path: Path,
):
    staged_path = tmp_path / "staged-input.wav"
    staged_path.write_bytes(b"audio")
    artifact_store = LocalJobArtifactStore()
    store = LocalInMemoryJobStore(
        artifact_store=artifact_store, retention_ttl_seconds=60.0
    )
    created = store.create(
        create_job_submission(
            operation=JobOperation.SYNTHESIZE_CUSTOM,
            command=CustomVoiceCommand(
                text="hello", model="demo-model", save_output=False
            ),
            submit_request_id="req-cleanup",
            owner_principal_id="local-default",
            response_format="wav",
            save_output=False,
            execution_timeout_seconds=15.0,
            staged_input_paths=(staged_path,),
        )
    )
    store.mark_running(created.job_id)

    store.mark_succeeded(created.job_id, success=_make_success_snapshot())

    assert staged_path.exists() is False


def test_in_memory_job_store_cancels_only_queued_jobs():
    store = LocalInMemoryJobStore()
    created = store.create(_make_submission())

    cancelled = store.cancel(created.job_id)

    assert cancelled.status is JobStatus.CANCELLED
    assert cancelled.terminal_error is not None
    assert cancelled.terminal_error.code == "job_cancelled"

    with pytest.raises(ValueError, match="Job state mismatch for transition"):
        store.mark_running(created.job_id)


def test_in_memory_job_store_rejects_invalid_terminal_failure_status():
    store = LocalInMemoryJobStore()
    created = store.create(_make_submission())
    store.mark_running(created.job_id)

    with pytest.raises(ValueError, match="Unsupported terminal failure status"):
        store.mark_failed(
            created.job_id,
            status=JobStatus.SUCCEEDED,
            failure=None,  # type: ignore[arg-type]
        )


def test_in_memory_job_store_purges_expired_terminal_jobs():
    store = LocalInMemoryJobStore(retention_ttl_seconds=1.0)
    created = store.create(
        _make_submission(
            idempotency_key="idem-1",
            idempotency_scope="principal-a",
            idempotency_fingerprint="fp-1",
        )
    )
    completed_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    store.mark_running(created.job_id, started_at=completed_at - timedelta(seconds=1))
    store.mark_succeeded(
        created.job_id, success=_make_success_snapshot(), completed_at=completed_at
    )

    assert store.get_snapshot(created.job_id) is None
    assert store.get_result(created.job_id) is None
    assert store.get_by_idempotency_key("idem-1", scope="principal-a") is None


def test_in_memory_job_store_returns_existing_job_for_same_idempotency_key_and_payload():
    store = LocalInMemoryJobStore()

    first = store.create_or_get(
        _make_submission(
            idempotency_key="idem-1",
            idempotency_scope="principal-a",
            idempotency_fingerprint="fp-1",
        )
    )
    second = store.create_or_get(
        _make_submission(
            request_id="req-2",
            owner_principal_id="principal-a",
            idempotency_key="idem-1",
            idempotency_scope="principal-a",
            idempotency_fingerprint="fp-1",
        )
    )

    assert first.created is True
    assert second.created is False
    assert second.snapshot.job_id == first.snapshot.job_id
    assert second.snapshot.idempotency_key == "idem-1"


def test_in_memory_job_store_rejects_idempotency_key_reuse_for_different_payload():
    store = LocalInMemoryJobStore()
    store.create_or_get(
        _make_submission(
            idempotency_key="idem-1",
            idempotency_scope="principal-a",
            idempotency_fingerprint="fp-1",
        )
    )

    with pytest.raises(JobIdempotencyConflictError, match="different payload"):
        store.create_or_get(
            _make_submission(
                idempotency_key="idem-1",
                idempotency_scope="principal-a",
                idempotency_fingerprint="fp-2",
            )
        )


def test_in_memory_job_store_allows_same_idempotency_key_for_different_principals():
    store = LocalInMemoryJobStore()

    first = store.create_or_get(
        _make_submission(
            owner_principal_id="principal-a",
            idempotency_key="idem-shared",
            idempotency_scope="principal-a",
            idempotency_fingerprint="fp-1",
        )
    )
    second = store.create_or_get(
        _make_submission(
            request_id="req-2",
            owner_principal_id="principal-b",
            idempotency_key="idem-shared",
            idempotency_scope="principal-b",
            idempotency_fingerprint="fp-1",
        )
    )

    assert first.created is True
    assert second.created is True
    assert first.snapshot.job_id != second.snapshot.job_id
    assert first.snapshot.owner_principal_id == "principal-a"
    assert second.snapshot.owner_principal_id == "principal-b"


def test_job_execution_gateway_without_manager_delegates_to_store():
    store = LocalInMemoryJobStore()
    gateway = JobExecutionGateway(store=store)

    created = gateway.submit(_make_submission(request_id="req-42"))
    fetched = gateway.get_job(created.job_id)
    cancelled = gateway.cancel(created.job_id)
    resolution = gateway.get_result(created.job_id)

    assert fetched is not None
    assert fetched.submit_request_id == "req-42"
    assert cancelled is not None
    assert cancelled.status is JobStatus.CANCELLED
    assert resolution is not None
    assert resolution.snapshot.status is JobStatus.CANCELLED
    assert resolution.success is None
    assert resolution.terminal_state is not None
    assert resolution.terminal_state.failure is not None


def test_job_execution_gateway_submit_idempotent_without_manager_delegates_to_store():
    store = LocalInMemoryJobStore()
    gateway = JobExecutionGateway(store=store)

    first = gateway.submit_idempotent(
        _make_submission(idempotency_key="idem-1", idempotency_fingerprint="fp-1")
    )
    second = gateway.submit_idempotent(
        _make_submission(idempotency_key="idem-1", idempotency_fingerprint="fp-1")
    )

    assert first.created is True
    assert second.created is False
    assert second.snapshot.job_id == first.snapshot.job_id


def test_in_memory_job_executor_dispatches_by_operation():
    application = StubApplicationService()
    executor = InMemoryJobExecutor(application_service=application)

    custom_result = executor.execute(_make_submission())
    design_result = executor.execute(
        create_job_submission(
            operation=JobOperation.SYNTHESIZE_DESIGN,
            command=VoiceDesignCommand(
                text="hello",
                voice_description="calm",
                model="demo-model",
                save_output=False,
            ),
            submit_request_id="req-2",
            owner_principal_id="local-default",
            response_format="wav",
            save_output=False,
            execution_timeout_seconds=10.0,
        )
    )
    clone_result = executor.execute(
        create_job_submission(
            operation=JobOperation.SYNTHESIZE_CLONE,
            command=VoiceCloneCommand(
                text="hello",
                ref_text="sample",
                ref_audio_path=TEST_TMP_ROOT / "source.wav",
                model="demo-model",
                save_output=False,
            ),
            submit_request_id="req-3",
            owner_principal_id="local-default",
            response_format="wav",
            save_output=False,
            execution_timeout_seconds=10.0,
        )
    )

    assert custom_result.backend == "mlx"
    assert design_result.backend == "mlx"
    assert clone_result.backend == "mlx"
    assert [mode for mode, _ in application.calls] == ["custom", "design", "clone"]


def test_local_bounded_execution_manager_runs_job_to_success():
    store = LocalInMemoryJobStore()
    manager = LocalBoundedExecutionManager(
        store=store,
        executor=InMemoryJobExecutor(application_service=StubApplicationService()),
        worker_count=1,
        queue_capacity=2,
    )

    try:
        created = manager.submit(_make_submission())
        succeeded = _wait_for_status(store, created.job_id, JobStatus.SUCCEEDED)
        resolution = store.get_result(created.job_id)

        assert succeeded.started_at is not None
        assert succeeded.completed_at is not None
        assert resolution is not None
        assert resolution.success is not None
        assert resolution.success.generation.audio.bytes_data == b"audio-bytes"
    finally:
        manager.stop()


def test_local_bounded_execution_manager_returns_existing_job_for_idempotent_replay():
    store = LocalInMemoryJobStore()
    manager = LocalBoundedExecutionManager(
        store=store,
        executor=InMemoryJobExecutor(application_service=StubApplicationService()),
        worker_count=1,
        queue_capacity=2,
    )

    try:
        first = manager.submit_idempotent(
            _make_submission(idempotency_key="idem-1", idempotency_fingerprint="fp-1")
        )
        second = manager.submit_idempotent(
            _make_submission(idempotency_key="idem-1", idempotency_fingerprint="fp-1")
        )

        assert first.created is True
        assert second.created is False
        assert second.snapshot.job_id == first.snapshot.job_id
    finally:
        manager.stop()


def test_local_bounded_execution_manager_marks_failures():
    store = LocalInMemoryJobStore()
    manager = LocalBoundedExecutionManager(
        store=store,
        executor=InMemoryJobExecutor(
            application_service=StubApplicationService(error=RuntimeError("boom"))
        ),
    )

    try:
        created = manager.submit(_make_submission())
        failed = _wait_for_status(store, created.job_id, JobStatus.FAILED)

        assert failed.terminal_error is not None
        assert failed.terminal_error.code == "job_execution_failed"
        assert failed.terminal_error.details == {
            "reason": "boom",
            "error_type": "RuntimeError",
        }
    finally:
        manager.stop()


def test_local_bounded_execution_manager_marks_timeouts_from_running_state():
    started = Event()
    release = Event()
    store = LocalInMemoryJobStore()
    manager = LocalBoundedExecutionManager(
        store=store,
        executor=InMemoryJobExecutor(
            application_service=StubApplicationService(started=started, release=release)
        ),
    )

    try:
        created = manager.submit(_make_submission(timeout_seconds=0.05))
        assert started.wait(timeout=1.0)
        timed_out = _wait_for_status(store, created.job_id, JobStatus.TIMEOUT)

        assert timed_out.started_at is not None
        assert timed_out.completed_at is not None
        assert timed_out.terminal_error is not None
        assert timed_out.terminal_error.code == "job_execution_timeout"
        assert timed_out.terminal_error.details == {"timeout_seconds": 0.05}
    finally:
        release.set()
        manager.stop()


def test_local_bounded_execution_manager_cancels_queued_jobs():
    store = BlockingStore()
    manager = LocalBoundedExecutionManager(
        store=store,
        executor=InMemoryJobExecutor(application_service=StubApplicationService()),
        worker_count=1,
        queue_capacity=2,
    )

    try:
        first = manager.submit(_make_submission(request_id="req-blocker"))
        assert store.mark_running_started.wait(timeout=1.0)
        second = manager.submit(_make_submission(request_id="req-cancel"))

        cancelled = manager.cancel(second.job_id)
        store.allow_mark_running.set()
        first_succeeded = _wait_for_status(store, first.job_id, JobStatus.SUCCEEDED)
        second_snapshot = store.get_snapshot(second.job_id)

        assert cancelled is not None
        assert cancelled.status is JobStatus.CANCELLED
        assert first_succeeded.status is JobStatus.SUCCEEDED
        assert second_snapshot is not None
        assert second_snapshot.status is JobStatus.CANCELLED
    finally:
        store.allow_mark_running.set()
        manager.stop()


def test_local_bounded_execution_manager_rejects_running_job_cancellation():
    started = Event()
    release = Event()
    store = LocalInMemoryJobStore()
    manager = LocalBoundedExecutionManager(
        store=store,
        executor=InMemoryJobExecutor(
            application_service=StubApplicationService(started=started, release=release)
        ),
    )

    try:
        created = manager.submit(_make_submission())
        assert started.wait(timeout=1.0)
        running = _wait_for_status(store, created.job_id, JobStatus.RUNNING)

        with pytest.raises(JobNotCancellableError, match="not cancellable"):
            manager.cancel(running.job_id)
    finally:
        release.set()
        manager.stop()


def test_local_bounded_execution_manager_rejects_submit_when_queue_is_full():
    store = BlockingStore()
    manager = LocalBoundedExecutionManager(
        store=store,
        executor=InMemoryJobExecutor(application_service=StubApplicationService()),
        worker_count=1,
        queue_capacity=1,
    )

    try:
        manager.submit(_make_submission(request_id="req-1"))
        assert store.mark_running_started.wait(timeout=1.0)

        with pytest.raises(JobQueueFullError, match="queue is full"):
            manager.submit(_make_submission(request_id="req-2"))
    finally:
        store.allow_mark_running.set()
        manager.stop()


def test_job_execution_gateway_with_manager_delegates_submit_and_cancel_to_manager():
    store = LocalInMemoryJobStore()
    manager = LocalBoundedExecutionManager(
        store=store,
        executor=InMemoryJobExecutor(application_service=StubApplicationService()),
    )
    gateway = JobExecutionGateway(store=store, manager=manager)

    try:
        created = gateway.submit(_make_submission(request_id="req-99"))
        succeeded = _wait_for_status(store, created.job_id, JobStatus.SUCCEEDED)
        cancelled = gateway.cancel(created.job_id)

        assert created.submit_request_id == "req-99"
        assert succeeded.status is JobStatus.SUCCEEDED
        assert cancelled is not None
        assert cancelled.status is JobStatus.SUCCEEDED
    finally:
        manager.stop()


def test_local_job_artifact_handler_cleans_up_staged_paths(tmp_path: Path):
    staged_path = tmp_path / "staged-input.wav"
    staged_path.write_bytes(b"audio")

    LocalJobArtifactHandler().cleanup_paths((staged_path,))

    assert staged_path.exists() is False


@pytest.mark.parametrize(
    ("port_type", "instance"),
    [
        (JobArtifactStore, LocalJobArtifactStore()),
        (JobMetadataStore, LocalInMemoryJobStore()),
        (
            JobExecutionBackend,
            LocalBoundedExecutionManager(
                store=LocalInMemoryJobStore(),
                executor=InMemoryJobExecutor(
                    application_service=StubApplicationService()
                ),
            ),
        ),
    ],
)
def test_local_adapters_conform_to_explicit_job_ports(
    port_type: type[object], instance: object
):
    assert isinstance(instance, port_type)
    if isinstance(instance, LocalBoundedExecutionManager):
        instance.stop()


def test_build_job_wiring_uses_local_defaults():
    settings = CoreSettings(
        models_dir=DEFAULT_MODELS_DIR,
        mlx_models_dir=DEFAULT_MODELS_DIR / "mlx",
        outputs_dir=DEFAULT_OUTPUTS_DIR,
        voices_dir=DEFAULT_VOICES_DIR,
        upload_staging_dir=DEFAULT_UPLOAD_STAGING_DIR,
    )
    artifact_store = build_job_artifact_store(settings)
    metadata_store = build_job_metadata_store(settings, artifact_store=artifact_store)
    executor = InMemoryJobExecutor(application_service=StubApplicationService())
    execution_backend = build_job_execution_backend(
        settings,
        store=metadata_store,
        executor=executor,
        metrics=OperationalMetricsRegistry(),
    )

    assert isinstance(artifact_store, LocalJobArtifactStore)
    assert isinstance(metadata_store, LocalInMemoryJobStore)
    assert metadata_store.artifact_store is artifact_store
    assert isinstance(execution_backend, LocalBoundedExecutionManager)
    assert execution_backend.store is metadata_store

    execution_backend.stop()


def test_build_runtime_uses_local_job_ports_by_default(tmp_path: Path):
    settings = CoreSettings(
        models_dir=tmp_path / "models",
        mlx_models_dir=tmp_path / "mlx-models",
        outputs_dir=tmp_path / "outputs",
        voices_dir=tmp_path / "voices",
        upload_staging_dir=tmp_path / "uploads",
    )

    runtime = build_runtime(settings)

    assert isinstance(runtime.job_artifact_store, LocalJobArtifactStore)
    assert isinstance(runtime.job_store, LocalInMemoryJobStore)
    assert isinstance(runtime.job_manager, LocalBoundedExecutionManager)
    assert runtime.job_store.artifact_store is runtime.job_artifact_store
    assert runtime.job_manager.store is runtime.job_store

    runtime.job_manager.stop()
