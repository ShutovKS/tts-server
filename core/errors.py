# FILE: core/errors.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define typed domain error hierarchy for all core operations.
#   SCOPE: CoreError base, ErrorContext payload, specific error subclasses
#   DEPENDS: none
#   LINKS: M-ERRORS
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   CoreError - Base exception for all domain errors
#   ErrorContext - Structured error payload container
#   ModelNotAvailableError - Requested model not found
#   BackendError - Base for backend failures
#   BackendNotAvailableError - No suitable backend found
#   BackendCapabilityError - Backend lacks required capability
#   ModelCapabilityError - Requested model family lacks the requested synthesis capability
#   ModelLoadError - Model loading failure
#   TTSGenerationError - Audio generation failure
#   InferenceBusyError - Inference slot occupied
#   AudioConversionError - Audio format conversion failure
#   AudioArtifactNotFoundError - Generated audio file missing
#   RequestTimeoutError - Inference timeout
#   JobQueueFullError - Local job queue capacity exceeded
#   JobNotFoundError - Job ID not in store
#   JobNotReadyError - Job result not yet available
#   JobNotSucceededError - Job completed with non-success status
#   JobNotCancellableError - Job in non-cancellable state
#   JobIdempotencyConflictError - Idempotency key reuse with different payload
#   UnauthorizedError - Missing authentication
#   ForbiddenError - Insufficient authorization
#   RateLimitExceededError - Rate limit policy violated
#   QuotaExceededError - Quota policy violated
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.1 - Added explicit model capability errors for unsupported family and operation combinations]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# START_CONTRACT: CoreError
#   PURPOSE: Provide the shared base exception type for all reusable core runtime failures.
#   INPUTS: { args: object - Standard exception constructor arguments }
#   OUTPUTS: { instance - Core exception carrying a human-readable failure reason }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: CoreError
class CoreError(Exception):
    """Base class for reusable core errors."""


# START_CONTRACT: ErrorContext
#   PURPOSE: Capture structured error metadata that can be serialized for diagnostics and APIs.
#   INPUTS: { reason: str - Primary failure reason, details: dict[str, Any] | None - Optional structured context fields }
#   OUTPUTS: { instance - Immutable error payload container }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: ErrorContext
@dataclass(frozen=True)
class ErrorContext:
    reason: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {"reason": self.reason}
        if self.details:
            payload.update(self.details)
        return payload


# START_CONTRACT: ModelNotAvailableError
#   PURPOSE: Report that the requested model or mode could not be resolved from configured assets.
#   INPUTS: { model_name: str | None - Requested model identifier when available, reason: str | None - Optional override for the failure message, details: dict[str, Any] | None - Optional structured diagnostics }
#   OUTPUTS: { instance - Domain error with structured model lookup context }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: ModelNotAvailableError
class ModelNotAvailableError(CoreError):
    def __init__(
        self,
        model_name: str | None = None,
        *,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        resolved_reason = reason or (
            f"Requested model is not available: {model_name}"
            if model_name
            else "Requested model is not available"
        )
        super().__init__(resolved_reason)
        self.model_name = model_name
        self.context = ErrorContext(reason=resolved_reason, details=details)


# START_CONTRACT: BackendError
#   PURPOSE: Represent backend-specific failures with structured diagnostic context.
#   INPUTS: { reason: str - Human-readable backend failure reason, details: dict[str, Any] | None - Optional structured diagnostics }
#   OUTPUTS: { instance - Backend-related domain error }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: BackendError
class BackendError(CoreError):
    def __init__(self, reason: str, *, details: dict[str, Any] | None = None):
        super().__init__(reason)
        self.context = ErrorContext(reason=reason, details=details)


# START_CONTRACT: BackendNotAvailableError
#   PURPOSE: Report that no configured or requested backend can be selected for runtime use.
#   INPUTS: { args: object - Backend selection failure details }
#   OUTPUTS: { instance - Specialized backend availability error }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: BackendNotAvailableError
class BackendNotAvailableError(BackendError):
    pass


# START_CONTRACT: BackendCapabilityError
#   PURPOSE: Report that a backend does not support the requested model or synthesis mode.
#   INPUTS: { args: object - Backend capability failure details }
#   OUTPUTS: { instance - Specialized backend capability error }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: BackendCapabilityError
class BackendCapabilityError(BackendError):
    pass


# START_CONTRACT: ModelCapabilityError
#   PURPOSE: Report that a resolved model or family does not support the requested normalized synthesis capability.
#   INPUTS: { model_id: str - Requested model identifier, capability: str - Requested normalized synthesis capability, supported_capabilities: tuple[str, ...] - Supported capability identifiers for the model, family: str | None - Optional resolved family label, reason: str | None - Optional override for the failure message, details: dict[str, Any] | None - Optional structured diagnostics }
#   OUTPUTS: { instance - Domain error with model capability context }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: ModelCapabilityError
class ModelCapabilityError(CoreError):
    def __init__(
        self,
        *,
        model_id: str,
        capability: str,
        supported_capabilities: tuple[str, ...],
        family: str | None = None,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        resolved_reason = (
            reason or f"Model '{model_id}' does not support capability '{capability}'"
        )
        super().__init__(resolved_reason)
        payload: dict[str, Any] = {
            "model": model_id,
            "capability": capability,
            "supported_capabilities": list(supported_capabilities),
        }
        if family:
            payload["family"] = family
        if details:
            payload.update(details)
        self.context = ErrorContext(reason=resolved_reason, details=payload)


# START_CONTRACT: ModelLoadError
#   PURPOSE: Report failures that occur while preparing a model runtime for inference.
#   INPUTS: { reason: str - Human-readable load failure reason, details: dict[str, Any] | None - Optional structured diagnostics }
#   OUTPUTS: { instance - Domain error with model loading context }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: ModelLoadError
class ModelLoadError(CoreError):
    def __init__(self, reason: str, *, details: dict[str, Any] | None = None):
        super().__init__(reason)
        self.context = ErrorContext(reason=reason, details=details)


# START_CONTRACT: TTSGenerationError
#   PURPOSE: Report failures that occur during text-to-speech generation or output preparation.
#   INPUTS: { reason: str - Human-readable generation failure reason, details: dict[str, Any] | None - Optional structured diagnostics }
#   OUTPUTS: { instance - Domain error with synthesis context }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: TTSGenerationError
class TTSGenerationError(CoreError):
    def __init__(self, reason: str, *, details: dict[str, Any] | None = None):
        super().__init__(reason)
        self.context = ErrorContext(reason=reason, details=details)


# START_CONTRACT: InferenceBusyError
#   PURPOSE: Report that the shared inference slot is already occupied by another request.
#   INPUTS: { reason: str - Human-readable contention message, details: dict[str, Any] | None - Optional structured diagnostics }
#   OUTPUTS: { instance - Domain error describing inference contention }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: InferenceBusyError
class InferenceBusyError(CoreError):
    def __init__(
        self,
        reason: str = "Inference is already in progress",
        *,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(reason)
        self.context = ErrorContext(reason=reason, details=details)


# START_CONTRACT: AudioConversionError
#   PURPOSE: Report failures while validating or converting audio artifacts for clone workflows.
#   INPUTS: { reason: str - Human-readable conversion failure reason, details: dict[str, Any] | None - Optional structured diagnostics }
#   OUTPUTS: { instance - Domain error for audio normalization failures }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: AudioConversionError
class AudioConversionError(CoreError):
    def __init__(self, reason: str, *, details: dict[str, Any] | None = None):
        super().__init__(reason)
        self.context = ErrorContext(reason=reason, details=details)


# START_CONTRACT: AudioArtifactNotFoundError
#   PURPOSE: Report that a generation workflow completed without producing an expected audio artifact.
#   INPUTS: { reason: str - Human-readable missing artifact reason, details: dict[str, Any] | None - Optional structured diagnostics }
#   OUTPUTS: { instance - Domain error for missing generated audio }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: AudioArtifactNotFoundError
class AudioArtifactNotFoundError(CoreError):
    def __init__(self, reason: str, *, details: dict[str, Any] | None = None):
        super().__init__(reason)
        self.context = ErrorContext(reason=reason, details=details)


# START_CONTRACT: RequestTimeoutError
#   PURPOSE: Report that an inference request exceeded its configured execution deadline.
#   INPUTS: { reason: str - Human-readable timeout reason, details: dict[str, Any] | None - Optional structured diagnostics }
#   OUTPUTS: { instance - Domain error describing request timeout }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: RequestTimeoutError
class RequestTimeoutError(CoreError):
    def __init__(
        self,
        reason: str = "Inference request timed out",
        *,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(reason)
        self.context = ErrorContext(reason=reason, details=details)


# START_CONTRACT: JobQueueFullError
#   PURPOSE: Report that the bounded local job queue cannot accept another submission.
#   INPUTS: { reason: str - Human-readable capacity failure reason, details: dict[str, Any] | None - Optional structured diagnostics }
#   OUTPUTS: { instance - Domain error for queue capacity exhaustion }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: JobQueueFullError
class JobQueueFullError(CoreError):
    def __init__(
        self,
        reason: str = "Local job queue is full",
        *,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(reason)
        self.context = ErrorContext(reason=reason, details=details)


# START_CONTRACT: JobNotFoundError
#   PURPOSE: Report that a requested async job identifier does not exist in the current store.
#   INPUTS: { job_id: str - Missing job identifier, reason: str | None - Optional override for the failure message, details: dict[str, Any] | None - Optional structured diagnostics }
#   OUTPUTS: { instance - Domain error with job lookup context }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: JobNotFoundError
class JobNotFoundError(CoreError):
    def __init__(
        self,
        job_id: str,
        *,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        resolved_reason = reason or f"Job was not found: {job_id}"
        super().__init__(resolved_reason)
        self.job_id = job_id
        payload = {"job_id": job_id}
        if details:
            payload.update(details)
        self.context = ErrorContext(reason=resolved_reason, details=payload)


# START_CONTRACT: JobNotReadyError
#   PURPOSE: Report that an async job result was requested before the job reached a ready state.
#   INPUTS: { job_id: str - Job identifier being queried, status: str - Current job status preventing result retrieval, reason: str | None - Optional override for the failure message, details: dict[str, Any] | None - Optional structured diagnostics }
#   OUTPUTS: { instance - Domain error with job readiness context }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: JobNotReadyError
class JobNotReadyError(CoreError):
    def __init__(
        self,
        job_id: str,
        status: str,
        *,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        resolved_reason = reason or f"Job result is not ready while status is {status}"
        super().__init__(resolved_reason)
        payload = {"job_id": job_id, "status": status}
        if details:
            payload.update(details)
        self.context = ErrorContext(reason=resolved_reason, details=payload)


# START_CONTRACT: JobNotSucceededError
#   PURPOSE: Report that a completed job cannot yield output because it did not finish successfully.
#   INPUTS: { job_id: str - Job identifier being queried, status: str - Terminal job status, reason: str | None - Optional override for the failure message, details: dict[str, Any] | None - Optional structured diagnostics }
#   OUTPUTS: { instance - Domain error with terminal job context }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: JobNotSucceededError
class JobNotSucceededError(CoreError):
    def __init__(
        self,
        job_id: str,
        status: str,
        *,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        resolved_reason = (
            reason
            or f"Job did not succeed and cannot produce audio while status is {status}"
        )
        super().__init__(resolved_reason)
        payload = {"job_id": job_id, "status": status}
        if details:
            payload.update(details)
        self.context = ErrorContext(reason=resolved_reason, details=payload)


# START_CONTRACT: JobNotCancellableError
#   PURPOSE: Report that a job cannot be cancelled from its current lifecycle state.
#   INPUTS: { job_id: str - Job identifier being cancelled, status: str - Current job status blocking cancellation, reason: str | None - Optional override for the failure message, details: dict[str, Any] | None - Optional structured diagnostics }
#   OUTPUTS: { instance - Domain error with cancellation context }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: JobNotCancellableError
class JobNotCancellableError(CoreError):
    def __init__(
        self,
        job_id: str,
        status: str,
        *,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        resolved_reason = reason or f"Job is not cancellable while status is {status}"
        super().__init__(resolved_reason)
        payload = {"job_id": job_id, "status": status}
        if details:
            payload.update(details)
        self.context = ErrorContext(reason=resolved_reason, details=payload)


# START_CONTRACT: JobIdempotencyConflictError
#   PURPOSE: Report that an idempotency key was reused for a different async job payload.
#   INPUTS: { idempotency_key: str - Conflicting idempotency key, existing_job_id: str - Existing job bound to the key, reason: str | None - Optional override for the failure message, details: dict[str, Any] | None - Optional structured diagnostics }
#   OUTPUTS: { instance - Domain error with idempotency conflict context }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: JobIdempotencyConflictError
class JobIdempotencyConflictError(CoreError):
    def __init__(
        self,
        *,
        idempotency_key: str,
        existing_job_id: str,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        resolved_reason = (
            reason or "Idempotency key was already used with a different payload"
        )
        super().__init__(resolved_reason)
        payload = {
            "idempotency_key": idempotency_key,
            "job_id": existing_job_id,
        }
        if details:
            payload.update(details)
        self.context = ErrorContext(reason=resolved_reason, details=payload)


# START_CONTRACT: UnauthorizedError
#   PURPOSE: Report that the caller has not provided valid authentication credentials.
#   INPUTS: { reason: str - Human-readable authentication failure reason, details: dict[str, Any] | None - Optional structured diagnostics }
#   OUTPUTS: { instance - Domain error for authentication failures }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: UnauthorizedError
class UnauthorizedError(CoreError):
    def __init__(
        self,
        reason: str = "Authentication is required",
        *,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(reason)
        self.context = ErrorContext(reason=reason, details=details)


# START_CONTRACT: ForbiddenError
#   PURPOSE: Report that the caller is authenticated but not authorized for the requested action.
#   INPUTS: { reason: str - Human-readable authorization failure reason, details: dict[str, Any] | None - Optional structured diagnostics }
#   OUTPUTS: { instance - Domain error for authorization failures }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: ForbiddenError
class ForbiddenError(CoreError):
    def __init__(
        self,
        reason: str = "Access to the requested resource is forbidden",
        *,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(reason)
        self.context = ErrorContext(reason=reason, details=details)


# START_CONTRACT: RateLimitExceededError
#   PURPOSE: Report that a request exceeded the configured rate limiting policy.
#   INPUTS: { policy: str - Rate limit policy identifier, limit: int - Allowed request count for the window, window_seconds: int - Window duration in seconds, retry_after_seconds: int | None - Suggested retry delay, reason: str | None - Optional override for the failure message, details: dict[str, Any] | None - Optional structured diagnostics }
#   OUTPUTS: { instance - Domain error with rate limit context }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: RateLimitExceededError
class RateLimitExceededError(CoreError):
    def __init__(
        self,
        *,
        policy: str,
        limit: int,
        window_seconds: int,
        retry_after_seconds: int | None = None,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        resolved_reason = reason or "Request rate limit was exceeded"
        super().__init__(resolved_reason)
        payload = {
            "policy": policy,
            "limit": limit,
            "window_seconds": window_seconds,
        }
        if retry_after_seconds is not None:
            payload["retry_after_seconds"] = retry_after_seconds
        if details:
            payload.update(details)
        self.retry_after_seconds = retry_after_seconds
        self.context = ErrorContext(reason=resolved_reason, details=payload)


# START_CONTRACT: QuotaExceededError
#   PURPOSE: Report that a caller exceeded the configured quota policy for compute or job usage.
#   INPUTS: { policy: str - Quota policy identifier, limit: int - Allowed usage limit, window_seconds: int | None - Optional quota window duration, retry_after_seconds: int | None - Suggested retry delay, reason: str | None - Optional override for the failure message, details: dict[str, Any] | None - Optional structured diagnostics }
#   OUTPUTS: { instance - Domain error with quota enforcement context }
#   SIDE_EFFECTS: none
#   LINKS: M-ERRORS
# END_CONTRACT: QuotaExceededError
class QuotaExceededError(CoreError):
    def __init__(
        self,
        *,
        policy: str,
        limit: int,
        window_seconds: int | None = None,
        retry_after_seconds: int | None = None,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        resolved_reason = reason or "Quota policy was exceeded"
        super().__init__(resolved_reason)
        payload = {
            "policy": policy,
            "limit": limit,
        }
        if window_seconds is not None:
            payload["window_seconds"] = window_seconds
        if retry_after_seconds is not None:
            payload["retry_after_seconds"] = retry_after_seconds
        if details:
            payload.update(details)
        self.retry_after_seconds = retry_after_seconds
        self.context = ErrorContext(reason=resolved_reason, details=payload)


__all__ = [
    "CoreError",
    "ErrorContext",
    "ModelNotAvailableError",
    "BackendError",
    "BackendNotAvailableError",
    "BackendCapabilityError",
    "ModelCapabilityError",
    "ModelLoadError",
    "TTSGenerationError",
    "InferenceBusyError",
    "AudioConversionError",
    "AudioArtifactNotFoundError",
    "RequestTimeoutError",
    "JobQueueFullError",
    "JobNotFoundError",
    "JobNotReadyError",
    "JobNotSucceededError",
    "JobNotCancellableError",
    "JobIdempotencyConflictError",
    "UnauthorizedError",
    "ForbiddenError",
    "RateLimitExceededError",
    "QuotaExceededError",
]
