# FILE: core/infrastructure/admission_control_local.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide local in-memory implementations for rate limiting and quota enforcement.
#   SCOPE: Local rate limiter and quota guard factories, in-memory token bucket implementations
#   DEPENDS: M-CONFIG, M-ERRORS, M-APPLICATION
#   LINKS: M-INFRASTRUCTURE
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   DisabledRateLimiter - No-op rate limiter used when rate limiting is disabled
#   DisabledQuotaGuard - No-op quota guard used when quota enforcement is disabled
#   LocalFixedWindowRateLimiter - In-memory principal-scoped fixed-window rate limiter
#   LocalQuotaGuard - In-memory principal-scoped quota guard
#   build_rate_limiter - Factory for rate limiter from settings
#   build_quota_guard - Factory for quota guard from settings
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import monotonic

from core.application.admission_control import (
    QuotaDecision,
    QuotaGuard,
    RateLimitDecision,
    RateLimiter,
)
from core.application.job_execution import JobMetadataStore
from core.config import CoreSettings


@dataclass(frozen=True)
class RateLimitPolicyConfig:
    limit: int
    window_seconds: int = 60


@dataclass(frozen=True)
class QuotaPolicyConfig:
    compute_requests_per_window: int
    compute_window_seconds: int
    max_active_jobs_per_principal: int


class DisabledRateLimiter(RateLimiter):
    def check_and_consume(self, *, principal_id: str, policy: str) -> RateLimitDecision:
        return RateLimitDecision(
            allowed=True,
            policy=policy,
            limit=0,
            window_seconds=60,
            retry_after_seconds=None,
        )


class DisabledQuotaGuard(QuotaGuard):
    def check_and_consume_compute(self, *, principal_id: str) -> QuotaDecision:
        return QuotaDecision(
            allowed=True,
            policy="compute_requests",
            limit=0,
            window_seconds=None,
            retry_after_seconds=None,
        )

    def check_active_async_jobs(self, *, principal_id: str) -> QuotaDecision:
        return QuotaDecision(
            allowed=True,
            policy="active_async_jobs",
            limit=0,
            window_seconds=None,
            retry_after_seconds=None,
        )


@dataclass
class LocalFixedWindowRateLimiter(RateLimiter):
    policies: dict[str, RateLimitPolicyConfig]

    def __post_init__(self) -> None:
        self._lock = Lock()
        self._windows: dict[tuple[str, str], tuple[float, int]] = {}

    def check_and_consume(self, *, principal_id: str, policy: str) -> RateLimitDecision:
        config = self.policies.get(policy)
        if config is None or config.limit <= 0:
            return RateLimitDecision(
                allowed=True,
                policy=policy,
                limit=0,
                window_seconds=60,
                retry_after_seconds=None,
            )

        now = monotonic()
        key = (principal_id, policy)
        with self._lock:
            window_started_at, count = self._windows.get(key, (now, 0))
            elapsed = now - window_started_at
            if elapsed >= config.window_seconds:
                window_started_at = now
                count = 0
            if count >= config.limit:
                retry_after_seconds = max(1, int(config.window_seconds - elapsed))
                return RateLimitDecision(
                    allowed=False,
                    policy=policy,
                    limit=config.limit,
                    window_seconds=config.window_seconds,
                    retry_after_seconds=retry_after_seconds,
                )
            self._windows[key] = (window_started_at, count + 1)
        return RateLimitDecision(
            allowed=True,
            policy=policy,
            limit=config.limit,
            window_seconds=config.window_seconds,
            retry_after_seconds=None,
        )


@dataclass
class LocalQuotaGuard(QuotaGuard):
    store: JobMetadataStore
    compute_requests_per_window: int
    compute_window_seconds: int
    max_active_jobs_per_principal: int

    def __post_init__(self) -> None:
        self._lock = Lock()
        self._compute_windows: dict[str, tuple[float, int]] = {}

    def check_and_consume_compute(self, *, principal_id: str) -> QuotaDecision:
        if self.compute_requests_per_window <= 0:
            return QuotaDecision(
                allowed=True,
                policy="compute_requests",
                limit=0,
                window_seconds=self.compute_window_seconds,
                retry_after_seconds=None,
                current_usage=0,
            )

        now = monotonic()
        with self._lock:
            window_started_at, count = self._compute_windows.get(principal_id, (now, 0))
            elapsed = now - window_started_at
            if elapsed >= self.compute_window_seconds:
                window_started_at = now
                count = 0
            if count >= self.compute_requests_per_window:
                retry_after_seconds = max(1, int(self.compute_window_seconds - elapsed))
                return QuotaDecision(
                    allowed=False,
                    policy="compute_requests",
                    limit=self.compute_requests_per_window,
                    window_seconds=self.compute_window_seconds,
                    retry_after_seconds=retry_after_seconds,
                    current_usage=count,
                )
            self._compute_windows[principal_id] = (window_started_at, count + 1)
            current_usage = count + 1
        return QuotaDecision(
            allowed=True,
            policy="compute_requests",
            limit=self.compute_requests_per_window,
            window_seconds=self.compute_window_seconds,
            retry_after_seconds=None,
            current_usage=current_usage,
        )

    def check_active_async_jobs(self, *, principal_id: str) -> QuotaDecision:
        if self.max_active_jobs_per_principal <= 0:
            return QuotaDecision(
                allowed=True,
                policy="active_async_jobs",
                limit=0,
                window_seconds=None,
                retry_after_seconds=None,
                current_usage=0,
            )

        active_jobs = self.store.count_active_jobs_for_principal(principal_id)
        allowed = active_jobs < self.max_active_jobs_per_principal
        return QuotaDecision(
            allowed=allowed,
            policy="active_async_jobs",
            limit=self.max_active_jobs_per_principal,
            window_seconds=None,
            retry_after_seconds=None,
            current_usage=active_jobs,
        )


# START_CONTRACT: build_rate_limiter
#   PURPOSE: Build the configured local or disabled rate limiter from runtime settings.
#   INPUTS: { settings: CoreSettings - Runtime settings containing rate limit policy configuration }
#   OUTPUTS: { RateLimiter - Configured rate limiter implementation }
#   SIDE_EFFECTS: none
#   LINKS: M-INFRASTRUCTURE
# END_CONTRACT: build_rate_limiter
def build_rate_limiter(settings: CoreSettings) -> RateLimiter:
    if not settings.rate_limit_enabled:
        return DisabledRateLimiter()
    if settings.rate_limit_backend != "local":
        raise ValueError(f"Unsupported rate limit backend: {settings.rate_limit_backend}")
    return LocalFixedWindowRateLimiter(
        policies={
            "sync_tts": RateLimitPolicyConfig(limit=settings.rate_limit_sync_tts_per_minute),
            "async_submit": RateLimitPolicyConfig(
                limit=settings.rate_limit_async_submit_per_minute
            ),
            "job_read": RateLimitPolicyConfig(limit=settings.rate_limit_job_read_per_minute),
            "job_cancel": RateLimitPolicyConfig(limit=settings.rate_limit_job_cancel_per_minute),
            "control_plane": RateLimitPolicyConfig(
                limit=settings.rate_limit_control_plane_per_minute
            ),
        }
    )


# START_CONTRACT: build_quota_guard
#   PURPOSE: Build the configured local or disabled quota guard from runtime settings.
#   INPUTS: { settings: CoreSettings - Runtime settings containing quota configuration, store: JobMetadataStore - Job metadata store used for active job counting }
#   OUTPUTS: { QuotaGuard - Configured quota enforcement implementation }
#   SIDE_EFFECTS: none
#   LINKS: M-INFRASTRUCTURE
# END_CONTRACT: build_quota_guard
def build_quota_guard(settings: CoreSettings, *, store: JobMetadataStore) -> QuotaGuard:
    if not settings.quota_enabled:
        return DisabledQuotaGuard()
    if settings.quota_backend != "local":
        raise ValueError(f"Unsupported quota backend: {settings.quota_backend}")
    return LocalQuotaGuard(
        store=store,
        compute_requests_per_window=settings.quota_compute_requests_per_window,
        compute_window_seconds=settings.quota_compute_window_seconds,
        max_active_jobs_per_principal=settings.quota_max_active_jobs_per_principal,
    )


__all__ = [
    "DisabledQuotaGuard",
    "DisabledRateLimiter",
    "LocalFixedWindowRateLimiter",
    "LocalQuotaGuard",
    "build_quota_guard",
    "build_rate_limiter",
]
