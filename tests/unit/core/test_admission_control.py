# FILE: tests/unit/core/test_admission_control.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for core admission control and quota enforcement.
#   SCOPE: Local rate limiting, quota guards, principal-scoped admission decisions
#   DEPENDS: M-CORE
#   LINKS: V-M-CORE
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_build_rate_limiter_keeps_disabled_local_default - Verifies disabled rate limiting keeps requests allowed
#   test_local_rate_limiter_is_principal_scoped - Verifies fixed-window rate limits are principal-scoped
#   test_local_quota_guard_uses_active_job_cap_per_principal - Verifies active-job quota checks are principal-scoped
#   test_local_quota_guard_consumes_compute_window_budget - Verifies compute budget is consumed within the configured window
#   test_local_quota_guard_is_deterministic_under_parallel_compute_contention - Verifies compute quota remains deterministic under parallel contention
#   test_local_quota_guard_counts_active_jobs_per_principal_under_parallel_checks - Verifies active-job quota checks remain deterministic under parallel reads
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT and MODULE_MAP]
# END_CHANGE_SUMMARY

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile

import pytest

from core.application.admission_control import RATE_LIMIT_POLICY_SYNC_TTS
from core.config import CoreSettings
from core.infrastructure.admission_control_local import (
    build_quota_guard,
    build_rate_limiter,
)
from core.infrastructure.job_execution_local import LocalInMemoryJobStore
from tests.unit.core.test_job_execution import _make_submission


pytestmark = pytest.mark.unit


DEFAULT_SETTINGS = dict(
    models_dir=Path(tempfile.gettempdir()) / "qwen-tts-tests" / "models",
    mlx_models_dir=Path(tempfile.gettempdir()) / "qwen-tts-tests" / "models" / "mlx",
    outputs_dir=Path(tempfile.gettempdir()) / "qwen-tts-tests" / "outputs",
    voices_dir=Path(tempfile.gettempdir()) / "qwen-tts-tests" / "voices",
)


def test_build_rate_limiter_keeps_disabled_local_default() -> None:
    settings = CoreSettings(**DEFAULT_SETTINGS)

    limiter = build_rate_limiter(settings)
    first = limiter.check_and_consume(
        principal_id="local-default", policy=RATE_LIMIT_POLICY_SYNC_TTS
    )
    second = limiter.check_and_consume(
        principal_id="local-default", policy=RATE_LIMIT_POLICY_SYNC_TTS
    )

    assert first.allowed is True
    assert second.allowed is True
    assert first.limit == 0


def test_local_rate_limiter_is_principal_scoped() -> None:
    settings = CoreSettings(
        **DEFAULT_SETTINGS, rate_limit_enabled=True, rate_limit_sync_tts_per_minute=1
    )

    limiter = build_rate_limiter(settings)
    allowed_a = limiter.check_and_consume(
        principal_id="principal-a", policy=RATE_LIMIT_POLICY_SYNC_TTS
    )
    blocked_a = limiter.check_and_consume(
        principal_id="principal-a", policy=RATE_LIMIT_POLICY_SYNC_TTS
    )
    allowed_b = limiter.check_and_consume(
        principal_id="principal-b", policy=RATE_LIMIT_POLICY_SYNC_TTS
    )

    assert allowed_a.allowed is True
    assert blocked_a.allowed is False
    assert blocked_a.retry_after_seconds is not None
    assert allowed_b.allowed is True


def test_local_quota_guard_uses_active_job_cap_per_principal() -> None:
    store = LocalInMemoryJobStore()
    store.create(_make_submission(owner_principal_id="principal-a"))
    store.create(_make_submission(request_id="req-2", owner_principal_id="principal-a"))
    store.create(_make_submission(request_id="req-3", owner_principal_id="principal-b"))
    settings = CoreSettings(
        **DEFAULT_SETTINGS, quota_enabled=True, quota_max_active_jobs_per_principal=2
    )

    quota_guard = build_quota_guard(settings, store=store)
    decision_a = quota_guard.check_active_async_jobs(principal_id="principal-a")
    decision_b = quota_guard.check_active_async_jobs(principal_id="principal-b")

    assert decision_a.allowed is False
    assert decision_a.current_usage == 2
    assert decision_b.allowed is True
    assert decision_b.current_usage == 1


def test_local_quota_guard_consumes_compute_window_budget() -> None:
    store = LocalInMemoryJobStore()
    settings = CoreSettings(
        **DEFAULT_SETTINGS,
        quota_enabled=True,
        quota_compute_requests_per_window=1,
        quota_compute_window_seconds=60,
    )

    quota_guard = build_quota_guard(settings, store=store)
    first = quota_guard.check_and_consume_compute(principal_id="principal-a")
    second = quota_guard.check_and_consume_compute(principal_id="principal-a")
    other_principal = quota_guard.check_and_consume_compute(principal_id="principal-b")

    assert first.allowed is True
    assert second.allowed is False
    assert second.retry_after_seconds is not None
    assert other_principal.allowed is True


def test_local_quota_guard_is_deterministic_under_parallel_compute_contention() -> None:
    store = LocalInMemoryJobStore()
    settings = CoreSettings(
        **DEFAULT_SETTINGS,
        quota_enabled=True,
        quota_compute_requests_per_window=2,
        quota_compute_window_seconds=60,
    )

    quota_guard = build_quota_guard(settings, store=store)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                quota_guard.check_and_consume_compute, principal_id="principal-a"
            )
            for _ in range(4)
        ]
        results = [future.result(timeout=1.0) for future in futures]

    allowed = [result for result in results if result.allowed]
    blocked = [result for result in results if not result.allowed]
    assert len(allowed) == 2, (
        f"Expected exactly two compute admissions before quota enforcement, got: {results}"
    )
    assert all(result.current_usage in {1, 2} for result in allowed)
    assert len(blocked) == 2, (
        f"Expected remaining parallel requests to be rejected, got: {results}"
    )
    assert all(result.current_usage == 2 for result in blocked)


def test_local_quota_guard_counts_active_jobs_per_principal_under_parallel_checks() -> (
    None
):
    store = LocalInMemoryJobStore()
    store.create(_make_submission(owner_principal_id="principal-a"))
    store.create(_make_submission(request_id="req-2", owner_principal_id="principal-a"))
    store.create(_make_submission(request_id="req-3", owner_principal_id="principal-b"))
    settings = CoreSettings(
        **DEFAULT_SETTINGS, quota_enabled=True, quota_max_active_jobs_per_principal=2
    )

    quota_guard = build_quota_guard(settings, store=store)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                quota_guard.check_active_async_jobs, principal_id="principal-a"
            )
            for _ in range(2)
        ]
        futures.extend(
            executor.submit(
                quota_guard.check_active_async_jobs, principal_id="principal-b"
            )
            for _ in range(2)
        )
        results = [future.result(timeout=1.0) for future in futures]

    principal_a_results = results[:2]
    principal_b_results = results[2:]
    assert all(
        result.allowed is False and result.current_usage == 2
        for result in principal_a_results
    )
    assert all(
        result.allowed is True and result.current_usage == 1
        for result in principal_b_results
    )
