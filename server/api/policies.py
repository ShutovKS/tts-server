# FILE: server/api/policies.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define rate limiting and quota policy enforcement for API endpoints.
#   SCOPE: Rate limit and quota dependency injection
#   DEPENDS: M-APPLICATION, M-ERRORS
#   LINKS: M-SERVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   enforce_sync_tts_admission - Enforce admission checks for synchronous TTS routes
#   enforce_async_submit_admission - Enforce admission checks for async submission routes
#   enforce_job_read_admission - Enforce admission checks for async job read routes
#   enforce_job_cancel_admission - Enforce admission checks for async job cancel routes
#   enforce_control_plane_admission - Enforce admission checks for control-plane routes
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

from fastapi import Request

from core.application.admission_control import (
    QUOTA_POLICY_ACTIVE_ASYNC_JOBS,
    RATE_LIMIT_POLICY_ASYNC_SUBMIT,
    RATE_LIMIT_POLICY_CONTROL_PLANE,
    RATE_LIMIT_POLICY_JOB_CANCEL,
    RATE_LIMIT_POLICY_JOB_READ,
    RATE_LIMIT_POLICY_SYNC_TTS,
)
from core.errors import QuotaExceededError, RateLimitExceededError


# START_CONTRACT: enforce_sync_tts_admission
#   PURPOSE: Apply rate-limit and compute quota checks for synchronous TTS endpoints.
#   INPUTS: { request: Request - incoming request with principal and admission services in app state }
#   OUTPUTS: { None - returns when admission is allowed }
#   SIDE_EFFECTS: Consumes rate-limit/quota capacity and may raise admission errors
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: enforce_sync_tts_admission
async def enforce_sync_tts_admission(request: Request) -> None:
    principal_id = request.state.principal.principal_id
    _enforce_rate_limit(request, principal_id=principal_id, policy=RATE_LIMIT_POLICY_SYNC_TTS)
    _enforce_compute_quota(request, principal_id=principal_id)


# START_CONTRACT: enforce_async_submit_admission
#   PURPOSE: Apply admission checks for asynchronous job submission endpoints.
#   INPUTS: { request: Request - incoming request with principal and admission services in app state }
#   OUTPUTS: { None - returns when submission is allowed }
#   SIDE_EFFECTS: Consumes rate-limit/quota capacity and may raise admission errors
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: enforce_async_submit_admission
async def enforce_async_submit_admission(request: Request) -> None:
    principal_id = request.state.principal.principal_id
    _enforce_rate_limit(request, principal_id=principal_id, policy=RATE_LIMIT_POLICY_ASYNC_SUBMIT)
    _enforce_compute_quota(request, principal_id=principal_id)
    _enforce_active_async_jobs_quota(request, principal_id=principal_id)


# START_CONTRACT: enforce_job_read_admission
#   PURPOSE: Apply admission checks for async job status and result reads.
#   INPUTS: { request: Request - incoming request with principal and admission services in app state }
#   OUTPUTS: { None - returns when read access is admitted }
#   SIDE_EFFECTS: Consumes job-read rate-limit capacity and may raise admission errors
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: enforce_job_read_admission
async def enforce_job_read_admission(request: Request) -> None:
    principal_id = request.state.principal.principal_id
    _enforce_rate_limit(request, principal_id=principal_id, policy=RATE_LIMIT_POLICY_JOB_READ)


# START_CONTRACT: enforce_job_cancel_admission
#   PURPOSE: Apply admission checks for async job cancellation requests.
#   INPUTS: { request: Request - incoming request with principal and admission services in app state }
#   OUTPUTS: { None - returns when cancellation is admitted }
#   SIDE_EFFECTS: Consumes job-cancel rate-limit capacity and may raise admission errors
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: enforce_job_cancel_admission
async def enforce_job_cancel_admission(request: Request) -> None:
    principal_id = request.state.principal.principal_id
    _enforce_rate_limit(request, principal_id=principal_id, policy=RATE_LIMIT_POLICY_JOB_CANCEL)


# START_CONTRACT: enforce_control_plane_admission
#   PURPOSE: Apply admission checks for control-plane routes like health and model discovery.
#   INPUTS: { request: Request - incoming request with principal and admission services in app state }
#   OUTPUTS: { None - returns when control-plane access is admitted }
#   SIDE_EFFECTS: Consumes control-plane rate-limit capacity and may raise admission errors
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: enforce_control_plane_admission
async def enforce_control_plane_admission(request: Request) -> None:
    principal_id = request.state.principal.principal_id
    _enforce_rate_limit(request, principal_id=principal_id, policy=RATE_LIMIT_POLICY_CONTROL_PLANE)


def _enforce_rate_limit(request: Request, *, principal_id: str, policy: str) -> None:
    decision = request.app.state.rate_limiter.check_and_consume(
        principal_id=principal_id, policy=policy
    )
    if decision.allowed:
        return
    raise RateLimitExceededError(
        policy=decision.policy,
        limit=decision.limit,
        window_seconds=decision.window_seconds,
        retry_after_seconds=decision.retry_after_seconds,
    )


def _enforce_compute_quota(request: Request, *, principal_id: str) -> None:
    decision = request.app.state.quota_guard.check_and_consume_compute(principal_id=principal_id)
    if decision.allowed:
        return
    raise QuotaExceededError(
        policy=decision.policy,
        limit=decision.limit,
        window_seconds=decision.window_seconds,
        retry_after_seconds=decision.retry_after_seconds,
        details={"current_usage": decision.current_usage},
    )


def _enforce_active_async_jobs_quota(request: Request, *, principal_id: str) -> None:
    decision = request.app.state.quota_guard.check_active_async_jobs(principal_id=principal_id)
    if decision.allowed:
        return
    raise QuotaExceededError(
        policy=QUOTA_POLICY_ACTIVE_ASYNC_JOBS,
        limit=decision.limit,
        window_seconds=decision.window_seconds,
        retry_after_seconds=decision.retry_after_seconds,
        details={"current_usage": decision.current_usage},
    )


__all__ = [
    "enforce_sync_tts_admission",
    "enforce_async_submit_admission",
    "enforce_job_read_admission",
    "enforce_job_cancel_admission",
    "enforce_control_plane_admission",
]
