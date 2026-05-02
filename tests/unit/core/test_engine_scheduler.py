# FILE: tests/unit/core/test_engine_scheduler.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for the standalone engine scheduler and worker-pool behavior.
#   SCOPE: Default single-slot contention, queue-full and timeout handling, shutdown semantics, and exception-safe slot release
#   DEPENDS: M-ENGINE-SCHEDULER, M-ERRORS
#   LINKS: V-M-ENGINE-SCHEDULER
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_engine_scheduler_preserves_default_single_slot_busy_behavior - Verifies the default policy matches the current single-slot inference guard semantics.
#   test_engine_scheduler_does_not_over_release_after_post_enqueue_submit_failure - Verifies a post-enqueue submit failure does not return capacity before the worker-owned task completes.
#   test_engine_scheduler_raises_queue_full_when_policy_allows_queue_but_capacity_is_exhausted - Verifies queued pools fail with a queue-full error once active and queued capacity are both occupied.
#   test_engine_scheduler_releases_slot_after_task_exception - Verifies raised task exceptions do not leak worker-pool capacity.
#   test_engine_scheduler_raises_request_timeout_for_inference_deadline - Verifies inference deadlines produce controlled timeout errors.
#   test_engine_scheduler_rejects_submit_after_shutdown - Verifies post-shutdown submit fails deterministically.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.1 - Task 11 review-fix: added deterministic coverage for post-enqueue submit failures so queue-slot permits cannot be over-released]
# END_CHANGE_SUMMARY

from __future__ import annotations

from threading import Event, Thread

import pytest

from core.engines import EngineScheduler, EngineSchedulerStoppedError, EngineWorkerPoolPolicy
from core.errors import InferenceBusyError, JobQueueFullError, RequestTimeoutError
from core.engines import scheduler as scheduler_module

pytestmark = pytest.mark.unit


def test_engine_scheduler_preserves_default_single_slot_busy_behavior() -> None:
    scheduler = EngineScheduler()
    started = Event()
    release = Event()
    first_result: list[str] = []
    first_error: list[BaseException] = []

    def blocking_task() -> str:
        started.set()
        assert release.wait(timeout=1.0)
        return "first-done"

    def run_first() -> None:
        try:
            first_result.append(
                scheduler.submit_engine_task(engine_key="piper", device_key="cpu", call=blocking_task)
            )
        except BaseException as exc:  # pragma: no cover - defensive test capture
            first_error.append(exc)

    thread = Thread(target=run_first, daemon=True)
    thread.start()
    assert started.wait(timeout=1.0)

    with pytest.raises(InferenceBusyError, match="already in progress"):
        scheduler.submit_engine_task(
            engine_key="piper",
            device_key="cpu",
            call=lambda: "second",
        )

    release.set()
    thread.join(timeout=1.0)
    scheduler.shutdown()

    assert first_error == []
    assert first_result == ["first-done"]


def test_engine_scheduler_does_not_over_release_after_post_enqueue_submit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = EngineScheduler()
    started = Event()
    release = Event()
    first_error: list[BaseException] = []
    second_error: list[BaseException] = []
    second_result: list[str] = []

    original_log_event = scheduler_module.log_event

    def fail_after_enqueue(*args, **kwargs):
        if kwargs.get("event") == "[EngineScheduler][submit][BLOCK_SCHEDULER_DISPATCH]":
            raise RuntimeError("synthetic log failure")
        return original_log_event(*args, **kwargs)

    monkeypatch.setattr(scheduler_module, "log_event", fail_after_enqueue)

    def blocking_task() -> str:
        started.set()
        assert release.wait(timeout=1.0)
        return "first-done"

    def run_first() -> None:
        try:
            scheduler.submit_engine_task(engine_key="piper", device_key="cpu", call=blocking_task)
        except BaseException as exc:  # pragma: no cover - defensive test capture
            first_error.append(exc)

    def run_second() -> None:
        try:
            second_result.append(
                scheduler.submit_engine_task(
                    engine_key="piper",
                    device_key="cpu",
                    call=lambda: "second-done",
                )
            )
        except BaseException as exc:  # pragma: no cover - defensive test capture
            second_error.append(exc)

    first_thread = Thread(target=run_first, daemon=True)
    first_thread.start()
    assert started.wait(timeout=1.0)
    first_thread.join(timeout=1.0)

    second_thread = Thread(target=run_second, daemon=True)
    second_thread.start()
    second_thread.join(timeout=0.2)

    release.set()
    second_thread.join(timeout=1.0)
    scheduler.shutdown()

    assert len(first_error) == 1
    assert isinstance(first_error[0], RuntimeError)
    assert str(first_error[0]) == "synthetic log failure"
    assert second_result == []
    assert len(second_error) == 1
    assert isinstance(second_error[0], InferenceBusyError)


def test_engine_scheduler_raises_queue_full_when_policy_allows_queue_but_capacity_is_exhausted() -> None:
    scheduler = EngineScheduler()
    policy = EngineWorkerPoolPolicy(max_active=1, max_queued=1, submit_timeout_seconds=0.01)
    started = Event()
    release = Event()
    waiting_to_start = Event()

    def blocking_task() -> str:
        started.set()
        assert release.wait(timeout=1.0)
        return "running"

    def queued_task() -> str:
        waiting_to_start.set()
        return "queued"

    running_thread = Thread(
        target=lambda: scheduler.submit_engine_task(
            engine_key="piper",
            device_key="cpu",
            call=blocking_task,
            policy=policy,
        ),
        daemon=True,
    )
    queued_thread = Thread(
        target=lambda: scheduler.submit_engine_task(
            engine_key="piper",
            device_key="cpu",
            call=queued_task,
        ),
        daemon=True,
    )

    running_thread.start()
    assert started.wait(timeout=1.0)
    queued_thread.start()

    with pytest.raises(JobQueueFullError, match="queue is full"):
        scheduler.submit_engine_task(
            engine_key="piper",
            device_key="cpu",
            call=lambda: "overflow",
        )

    release.set()
    running_thread.join(timeout=1.0)
    queued_thread.join(timeout=1.0)
    scheduler.shutdown()
    assert waiting_to_start.wait(timeout=1.0)


def test_engine_scheduler_releases_slot_after_task_exception() -> None:
    scheduler = EngineScheduler()

    with pytest.raises(RuntimeError, match="boom"):
        scheduler.submit_engine_task(
            engine_key="piper",
            device_key="cpu",
            call=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    result = scheduler.submit_engine_task(
        engine_key="piper",
        device_key="cpu",
        call=lambda: "recovered",
    )
    scheduler.shutdown()

    assert result == "recovered"


def test_engine_scheduler_raises_request_timeout_for_inference_deadline() -> None:
    scheduler = EngineScheduler(
        default_policy=EngineWorkerPoolPolicy(inference_timeout_seconds=0.05)
    )
    started = Event()
    release = Event()

    def blocking_task() -> str:
        started.set()
        assert release.wait(timeout=1.0)
        return "too-late"

    with pytest.raises(RequestTimeoutError, match="timed out") as exc_info:
        scheduler.submit_engine_task(
            engine_key="piper",
            device_key="cpu",
            call=blocking_task,
        )

    release.set()
    scheduler.shutdown()

    assert started.wait(timeout=1.0)
    assert getattr(exc_info.value, "context").details == {
        "engine": "piper",
        "device": "cpu",
        "timeout_seconds": 0.05,
        "phase": "inference",
    }


def test_engine_scheduler_rejects_submit_after_shutdown() -> None:
    scheduler = EngineScheduler()
    scheduler.shutdown()

    with pytest.raises(EngineSchedulerStoppedError, match="shut down"):
        scheduler.submit_engine_task(
            engine_key="piper",
            device_key="cpu",
            call=lambda: "never",
        )
