# FILE: core/engines/scheduler.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide a per-engine and per-device worker scheduler with bounded concurrency, bounded queueing, and deterministic timeout behavior.
#   SCOPE: EngineScheduler, WorkerPool, scheduler policy/key DTOs, bounded submit/execute lifecycle, and shutdown handling
#   DEPENDS: M-ERRORS, M-OBSERVABILITY
#   LINKS: M-ENGINE-SCHEDULER, M-ENGINE-CONTRACTS
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   EngineWorkerPoolKey - Deterministic engine/device pool key used to isolate worker limits.
#   EngineWorkerPoolPolicy - Explicit worker-pool policy describing active concurrency, queue depth, and timeout behavior.
#   EngineSchedulerStoppedError - Typed scheduler-local error raised after shutdown.
#   WorkerPool - Per-key bounded worker pool that executes submitted callables.
#   EngineScheduler - Process-local scheduler facade that manages per-engine/per-device pools.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.1 - Task 11 review-fix: made queue-slot ownership explicit so only pre-enqueue submit failures release permits and worker-owned tasks release exactly once]
# END_CHANGE_SUMMARY

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Queue
from threading import BoundedSemaphore, Condition, Event, Lock, Thread
from typing import TypeVar

from core.errors import CoreError, InferenceBusyError, JobQueueFullError, RequestTimeoutError
from core.observability import log_event

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


# START_CONTRACT: EngineWorkerPoolKey
#   PURPOSE: Identify the bounded worker pool that owns execution for one engine and device combination.
#   INPUTS: { engine_key: str - Stable engine identifier, device_key: str | None - Optional device identifier such as cpu, cuda:0, or mps }
#   OUTPUTS: { instance - Immutable per-engine/per-device scheduler key }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-SCHEDULER
# END_CONTRACT: EngineWorkerPoolKey
@dataclass(frozen=True)
class EngineWorkerPoolKey:
    engine_key: str
    device_key: str | None = None


# START_CONTRACT: EngineWorkerPoolPolicy
#   PURPOSE: Describe explicit concurrency and timeout policy for one worker pool while defaulting to the current single-slot runtime behavior.
#   INPUTS: { max_active: int - Maximum concurrently running tasks, max_queued: int - Maximum queued tasks waiting behind active work, submit_timeout_seconds: float - Maximum time allowed to wait for a slot before submit fails, inference_timeout_seconds: float | None - Maximum time allowed for task execution before inference timeout is raised }
#   OUTPUTS: { instance - Immutable worker-pool policy }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-SCHEDULER
# END_CONTRACT: EngineWorkerPoolPolicy
@dataclass(frozen=True)
class EngineWorkerPoolPolicy:
    max_active: int = 1
    max_queued: int = 0
    submit_timeout_seconds: float = 0.0
    inference_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_active < 1:
            raise ValueError("max_active must be at least 1")
        if self.max_queued < 0:
            raise ValueError("max_queued must be at least 0")
        if self.submit_timeout_seconds < 0:
            raise ValueError("submit_timeout_seconds must be at least 0")
        if self.inference_timeout_seconds is not None and self.inference_timeout_seconds <= 0:
            raise ValueError("inference_timeout_seconds must be greater than 0 when provided")

    @property
    def capacity(self) -> int:
        return self.max_active + self.max_queued

    @classmethod
    def single_slot(cls) -> EngineWorkerPoolPolicy:
        return cls()


# START_CONTRACT: EngineSchedulerStoppedError
#   PURPOSE: Report deterministic submit failures after the scheduler or a worker pool has been shut down.
#   INPUTS: { reason: str - Human-readable shutdown reason }
#   OUTPUTS: { instance - Typed scheduler shutdown error }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-SCHEDULER
# END_CONTRACT: EngineSchedulerStoppedError
class EngineSchedulerStoppedError(CoreError):
    pass


@dataclass
class _TaskEnvelope:
    key: EngineWorkerPoolKey
    call: Callable[[], object]
    result_queue: Queue[tuple[str, object]]
    timeout_seconds: float | None


# START_CONTRACT: WorkerPool
#   PURPOSE: Execute submitted callables for one engine/device key with bounded running and queued capacity.
#   INPUTS: { key: EngineWorkerPoolKey - Pool identity, policy: EngineWorkerPoolPolicy - Active/queued/timeout policy }
#   OUTPUTS: { instance - Started-on-demand worker pool }
#   SIDE_EFFECTS: Spawns worker threads, tracks in-memory queue state, and emits structured scheduler logs
#   LINKS: M-ENGINE-SCHEDULER
# END_CONTRACT: WorkerPool
@dataclass
class WorkerPool:
    key: EngineWorkerPoolKey
    policy: EngineWorkerPoolPolicy = field(default_factory=EngineWorkerPoolPolicy.single_slot)
    _queue: deque[_TaskEnvelope] = field(init=False, repr=False)
    _condition: Condition = field(init=False, repr=False)
    _stop_event: Event = field(init=False, repr=False)
    _queue_slots: BoundedSemaphore = field(init=False, repr=False)
    _workers: list[Thread] = field(init=False, repr=False)
    _started: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._queue = deque()
        self._condition = Condition(Lock())
        self._stop_event = Event()
        self._queue_slots = BoundedSemaphore(self.policy.capacity)
        self._workers = []

    # START_CONTRACT: start
    #   PURPOSE: Spawn the worker threads for the pool exactly once.
    #   INPUTS: {}
    #   OUTPUTS: { None }
    #   SIDE_EFFECTS: Starts daemon worker threads
    #   LINKS: M-ENGINE-SCHEDULER
    # END_CONTRACT: start
    def start(self) -> None:
        with self._condition:
            # START_BLOCK_GUARD_POOL_STARTUP
            if self._started:
                return
            if self._stop_event.is_set():
                raise EngineSchedulerStoppedError("Engine scheduler worker pool is shut down")
            # END_BLOCK_GUARD_POOL_STARTUP
            # START_BLOCK_SPAWN_POOL_WORKERS
            self._workers = [
                Thread(
                    target=self._worker_loop,
                    name=f"engine-worker-{self.key.engine_key}-{self.key.device_key or 'default'}-{index}",
                    daemon=True,
                )
                for index in range(self.policy.max_active)
            ]
            for worker in self._workers:
                worker.start()
            self._started = True
            # END_BLOCK_SPAWN_POOL_WORKERS

    # START_CONTRACT: shutdown
    #   PURPOSE: Stop the pool and reject future submissions deterministically.
    #   INPUTS: { wait: bool - Whether to join worker threads before returning }
    #   OUTPUTS: { None }
    #   SIDE_EFFECTS: Stops new work admission, wakes waiting workers, and optionally joins worker threads
    #   LINKS: M-ENGINE-SCHEDULER
    # END_CONTRACT: shutdown
    def shutdown(self, *, wait: bool = True) -> None:
        workers: list[Thread]
        with self._condition:
            # START_BLOCK_MARK_POOL_STOPPED
            self._stop_event.set()
            self._condition.notify_all()
            workers = list(self._workers)
            self._workers = []
            self._started = False
            # END_BLOCK_MARK_POOL_STOPPED
        if not wait:
            return
        # START_BLOCK_JOIN_POOL_WORKERS
        for worker in workers:
            worker.join(timeout=1.0)
        # END_BLOCK_JOIN_POOL_WORKERS

    # START_CONTRACT: submit
    #   PURPOSE: Execute one callable under bounded worker-pool policy and return its result or typed failure.
    #   INPUTS: { call: Callable[[], T] - Zero-argument callable to run, submit_timeout_seconds: float | None - Optional per-submit override for waiting on capacity, inference_timeout_seconds: float | None - Optional per-submit override for execution deadline }
    #   OUTPUTS: { T - Callable result }
    #   SIDE_EFFECTS: May block waiting for capacity, enqueue work, wait for worker completion, and emit structured scheduler logs
    #   LINKS: M-ENGINE-SCHEDULER
    # END_CONTRACT: submit
    def submit(
        self,
        call: Callable[[], T],
        *,
        submit_timeout_seconds: float | None = None,
        inference_timeout_seconds: float | None = None,
    ) -> T:
        self.start()
        if self._stop_event.is_set():
            raise EngineSchedulerStoppedError("Engine scheduler worker pool is shut down")

        effective_submit_timeout = (
            self.policy.submit_timeout_seconds
            if submit_timeout_seconds is None
            else submit_timeout_seconds
        )
        effective_inference_timeout = (
            self.policy.inference_timeout_seconds
            if inference_timeout_seconds is None
            else inference_timeout_seconds
        )
        if effective_submit_timeout < 0:
            raise ValueError("submit_timeout_seconds must be at least 0")
        if effective_inference_timeout is not None and effective_inference_timeout <= 0:
            raise ValueError("inference_timeout_seconds must be greater than 0 when provided")

        # START_BLOCK_ACQUIRE_SCHEDULER_SLOT
        acquired = self._queue_slots.acquire(timeout=effective_submit_timeout)
        if not acquired:
            if self.policy.max_queued == 0:
                raise InferenceBusyError(
                    "Inference is already in progress",
                    details={
                        "engine": self.key.engine_key,
                        "device": self.key.device_key,
                        "submit_timeout_seconds": effective_submit_timeout,
                    },
                )
            raise JobQueueFullError(
                "Engine worker queue is full",
                details={
                    "engine": self.key.engine_key,
                    "device": self.key.device_key,
                    "max_active": self.policy.max_active,
                    "max_queued": self.policy.max_queued,
                    "submit_timeout_seconds": effective_submit_timeout,
                },
            )
        # END_BLOCK_ACQUIRE_SCHEDULER_SLOT

        result_queue: Queue[tuple[str, object]] = Queue(maxsize=1)
        task = _TaskEnvelope(
            key=self.key,
            call=call,
            result_queue=result_queue,
            timeout_seconds=effective_inference_timeout,
        )

        slot_owned_by_submit = True
        try:
            # START_BLOCK_SCHEDULER_DISPATCH
            with self._condition:
                if self._stop_event.is_set():
                    raise EngineSchedulerStoppedError("Engine scheduler worker pool is shut down")
                self._queue.append(task)
                slot_owned_by_submit = False
                self._condition.notify()
            log_event(
                LOGGER,
                level=20,
                event="[EngineScheduler][submit][BLOCK_SCHEDULER_DISPATCH]",
                message="Queued engine work item",
                engine=self.key.engine_key,
                device=self.key.device_key,
                max_active=self.policy.max_active,
                max_queued=self.policy.max_queued,
                submit_timeout_seconds=effective_submit_timeout,
                inference_timeout_seconds=effective_inference_timeout,
            )
            # END_BLOCK_SCHEDULER_DISPATCH
            outcome, payload = result_queue.get()
        except Exception:
            # START_BLOCK_RELEASE_SLOT_ON_SUBMIT_FAILURE
            if slot_owned_by_submit:
                self._queue_slots.release()
            raise
            # END_BLOCK_RELEASE_SLOT_ON_SUBMIT_FAILURE

        if outcome == "success":
            return payload  # type: ignore[return-value]
        raise payload  # type: ignore[misc]

    def _worker_loop(self) -> None:
        while True:
            # START_BLOCK_DEQUEUE_ENGINE_TASK
            task = self._dequeue_task()
            if task is None:
                return
            # END_BLOCK_DEQUEUE_ENGINE_TASK
            try:
                self._execute_task(task)
            finally:
                # START_BLOCK_RELEASE_SCHEDULER_SLOT
                self._queue_slots.release()
                # END_BLOCK_RELEASE_SCHEDULER_SLOT

    def _dequeue_task(self) -> _TaskEnvelope | None:
        with self._condition:
            while not self._queue and not self._stop_event.is_set():
                self._condition.wait(timeout=0.1)
            if not self._queue:
                return None
            return self._queue.popleft()

    def _execute_task(self, task: _TaskEnvelope) -> None:
        result_queue: Queue[tuple[str, object]] = Queue(maxsize=1)

        def run_task() -> None:
            try:
                result = task.call()
            except Exception as exc:
                result_queue.put(("error", exc))
                return
            result_queue.put(("success", result))

        # START_BLOCK_START_EXECUTION_THREAD
        execution_thread = Thread(
            target=run_task,
            name=f"engine-task-{task.key.engine_key}-{task.key.device_key or 'default'}",
            daemon=True,
        )
        execution_thread.start()
        execution_thread.join(timeout=task.timeout_seconds)
        # END_BLOCK_START_EXECUTION_THREAD

        # START_BLOCK_HANDLE_EXECUTION_TIMEOUT
        if execution_thread.is_alive():
            task.result_queue.put(
                (
                    "error",
                    RequestTimeoutError(
                        details={
                            "engine": task.key.engine_key,
                            "device": task.key.device_key,
                            "timeout_seconds": task.timeout_seconds,
                            "phase": "inference",
                        }
                    ),
                )
            )
            return
        # END_BLOCK_HANDLE_EXECUTION_TIMEOUT

        # START_BLOCK_HANDLE_EXECUTION_OUTCOME
        if result_queue.empty():
            task.result_queue.put(
                (
                    "error",
                    RequestTimeoutError(
                        "Engine work finished without a result",
                        details={
                            "engine": task.key.engine_key,
                            "device": task.key.device_key,
                            "phase": "result",
                        },
                    ),
                )
            )
            return

        outcome, payload = result_queue.get_nowait()
        task.result_queue.put((outcome, payload))
        # END_BLOCK_HANDLE_EXECUTION_OUTCOME


# START_CONTRACT: EngineScheduler
#   PURPOSE: Manage per-engine/per-device worker pools under explicit policy objects while keeping runtime integration separate from the current TTSService path.
#   INPUTS: { default_policy: EngineWorkerPoolPolicy - Default policy applied when a submitter does not supply a per-call override }
#   OUTPUTS: { instance - Process-local engine scheduler facade }
#   SIDE_EFFECTS: Creates worker pools lazily and manages their lifetime until shutdown
#   LINKS: M-ENGINE-SCHEDULER
# END_CONTRACT: EngineScheduler
class EngineScheduler:
    def __init__(self, default_policy: EngineWorkerPoolPolicy | None = None) -> None:
        self._default_policy = default_policy or EngineWorkerPoolPolicy.single_slot()
        self._pools: dict[EngineWorkerPoolKey, WorkerPool] = {}
        self._lock = Lock()
        self._shutdown = False

    # START_CONTRACT: submit_engine_task
    #   PURPOSE: Submit work to the pool for the requested engine/device key using either the scheduler default policy or an explicit override.
    #   INPUTS: { engine_key: str - Stable engine identifier, device_key: str | None - Optional device identifier, call: Callable[[], T] - Zero-argument callable that performs the work, policy: EngineWorkerPoolPolicy | None - Optional per-key policy override when the pool is first created, submit_timeout_seconds: float | None - Optional per-call submit wait override, inference_timeout_seconds: float | None - Optional per-call execution deadline override }
    #   OUTPUTS: { T - Callable result }
    #   SIDE_EFFECTS: Creates a keyed worker pool on first use and executes the task under its bounded policy
    #   LINKS: M-ENGINE-SCHEDULER
    # END_CONTRACT: submit_engine_task
    def submit_engine_task(
        self,
        *,
        engine_key: str,
        call: Callable[[], T],
        device_key: str | None = None,
        policy: EngineWorkerPoolPolicy | None = None,
        submit_timeout_seconds: float | None = None,
        inference_timeout_seconds: float | None = None,
    ) -> T:
        pool = self._get_or_create_pool(
            EngineWorkerPoolKey(engine_key=engine_key, device_key=device_key),
            policy=policy,
        )
        return pool.submit(
            call,
            submit_timeout_seconds=submit_timeout_seconds,
            inference_timeout_seconds=inference_timeout_seconds,
        )

    # START_CONTRACT: shutdown
    #   PURPOSE: Stop all worker pools and reject future submissions deterministically.
    #   INPUTS: { wait: bool - Whether to join worker threads before returning }
    #   OUTPUTS: { None }
    #   SIDE_EFFECTS: Shuts down all existing pools
    #   LINKS: M-ENGINE-SCHEDULER
    # END_CONTRACT: shutdown
    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            self._shutdown = True
            pools = list(self._pools.values())
        for pool in pools:
            pool.shutdown(wait=wait)

    def _get_or_create_pool(
        self,
        key: EngineWorkerPoolKey,
        *,
        policy: EngineWorkerPoolPolicy | None,
    ) -> WorkerPool:
        with self._lock:
            if self._shutdown:
                raise EngineSchedulerStoppedError("Engine scheduler is shut down")
            existing = self._pools.get(key)
            if existing is not None:
                return existing
            created = WorkerPool(key=key, policy=policy or self._default_policy)
            self._pools[key] = created
            return created


__all__ = [
    "EngineScheduler",
    "EngineSchedulerStoppedError",
    "EngineWorkerPoolKey",
    "EngineWorkerPoolPolicy",
    "WorkerPool",
]
