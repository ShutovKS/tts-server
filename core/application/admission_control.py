# FILE: core/application/admission_control.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define admission control abstractions for rate limiting and quota enforcement.
#   SCOPE: QuotaGuard and RateLimiter abstract interfaces
#   DEPENDS: M-ERRORS
#   LINKS: M-APPLICATION
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   RATE_LIMIT_POLICY_SYNC_TTS - Policy key for synchronous synthesis requests
#   RATE_LIMIT_POLICY_ASYNC_SUBMIT - Policy key for async job submissions
#   RATE_LIMIT_POLICY_JOB_READ - Policy key for async job status/result reads
#   RATE_LIMIT_POLICY_JOB_CANCEL - Policy key for async job cancellation requests
#   RATE_LIMIT_POLICY_CONTROL_PLANE - Policy key for health and control-plane requests
#   QUOTA_POLICY_COMPUTE - Policy key for compute-budget quota enforcement
#   QUOTA_POLICY_ACTIVE_ASYNC_JOBS - Policy key for active async job quota enforcement
#   RateLimitDecision - Rate-limit admission decision DTO
#   QuotaDecision - Quota admission decision DTO
#   QuotaGuard - Quota enforcement abstraction
#   RateLimiter - Rate limit enforcement abstraction
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

RATE_LIMIT_POLICY_SYNC_TTS = "sync_tts"
RATE_LIMIT_POLICY_ASYNC_SUBMIT = "async_submit"
RATE_LIMIT_POLICY_JOB_READ = "job_read"
RATE_LIMIT_POLICY_JOB_CANCEL = "job_cancel"
RATE_LIMIT_POLICY_CONTROL_PLANE = "control_plane"
QUOTA_POLICY_COMPUTE = "compute_requests"
QUOTA_POLICY_ACTIVE_ASYNC_JOBS = "active_async_jobs"


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    policy: str
    limit: int
    window_seconds: int
    retry_after_seconds: int | None = None


@dataclass(frozen=True)
class QuotaDecision:
    allowed: bool
    policy: str
    limit: int
    window_seconds: int | None = None
    retry_after_seconds: int | None = None
    current_usage: int | None = None


# START_CONTRACT: RateLimiter
#   PURPOSE: Define the contract for checking and consuming request rate limit capacity.
#   INPUTS: {}
#   OUTPUTS: { instance - Protocol describing rate limit enforcement }
#   SIDE_EFFECTS: none
#   LINKS: M-APPLICATION
# END_CONTRACT: RateLimiter
@runtime_checkable
class RateLimiter(Protocol):
    def check_and_consume(self, *, principal_id: str, policy: str) -> RateLimitDecision: ...


# START_CONTRACT: QuotaGuard
#   PURPOSE: Define the contract for compute and async job quota enforcement.
#   INPUTS: {}
#   OUTPUTS: { instance - Protocol describing quota enforcement }
#   SIDE_EFFECTS: none
#   LINKS: M-APPLICATION
# END_CONTRACT: QuotaGuard
@runtime_checkable
class QuotaGuard(Protocol):
    def check_and_consume_compute(self, *, principal_id: str) -> QuotaDecision: ...

    def check_active_async_jobs(self, *, principal_id: str) -> QuotaDecision: ...


__all__ = [
    "QUOTA_POLICY_ACTIVE_ASYNC_JOBS",
    "QUOTA_POLICY_COMPUTE",
    "RATE_LIMIT_POLICY_ASYNC_SUBMIT",
    "RATE_LIMIT_POLICY_CONTROL_PLANE",
    "RATE_LIMIT_POLICY_JOB_CANCEL",
    "RATE_LIMIT_POLICY_JOB_READ",
    "RATE_LIMIT_POLICY_SYNC_TTS",
    "QuotaDecision",
    "QuotaGuard",
    "RateLimitDecision",
    "RateLimiter",
]
