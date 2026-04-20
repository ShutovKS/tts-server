# FILE: server/api/errors.py
# VERSION: 1.0.1
# START_MODULE_CONTRACT
#   PURPOSE: Map domain errors to HTTP error responses with structured JSON bodies.
#   SCOPE: Exception handlers for all CoreError subclasses
#   DEPENDS: M-ERRORS
#   LINKS: M-SERVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   register_exception_handlers - Register mapped exception handlers on FastAPI app
#   build_exception_mappings - Build exception-to-error-descriptor mappings
#   map_exception_to_descriptor - Convert exceptions into public API descriptors
#   build_generation_error_descriptor - Build generation-specific API error descriptors
#   build_error_details - Build sanitized public error details from exceptions
#   build_retry_after_headers - Build Retry-After headers from retry hints
#   build_model_not_available_details - Build model-not-available error details
#   sanitize_validation_errors - Sanitize FastAPI validation errors for public responses
#   sanitize_public_error_details - Sanitize structured error detail payloads for public responses
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.1 - Restored build_error_details compatibility with default_reason call sites so mapped API errors keep returning structured payloads]
# END_CHANGE_SUMMARY

from __future__ import annotations

import logging
import re
from pathlib import PurePath
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from core.errors import (
    AudioConversionError,
    BackendCapabilityError,
    BackendNotAvailableError,
    ForbiddenError,
    InferenceBusyError,
    JobIdempotencyConflictError,
    JobNotCancellableError,
    JobNotFoundError,
    JobNotReadyError,
    JobNotSucceededError,
    JobQueueFullError,
    ModelCapabilityError,
    ModelLoadError,
    ModelNotAvailableError,
    QuotaExceededError,
    RateLimitExceededError,
    RequestTimeoutError,
    RuntimeCapabilityNotConfiguredError,
    TTSGenerationError,
    UnauthorizedError,
)
from core.observability import log_event
from server.api.contracts import ErrorDescriptor, ExceptionMapping
from server.api.responses import build_error_response
from server.bootstrap import ServerSettings


_PATH_KEY_RE = re.compile(
    r"(^|_)(path|paths|dir|dirs|directory|directories|file|filename|filenames)$",
    re.IGNORECASE,
)
_PATH_VALUE_RE = re.compile(
    r"([A-Za-z]:[\\/]|/Users/|/tmp/|/var/|/private/|\.uploads/|\.outputs/|\.models/|\.voices/)"
)


# START_CONTRACT: register_exception_handlers
#   PURPOSE: Register request validation, mapped domain error, and fallback exception handlers on the FastAPI app.
#   INPUTS: { app: FastAPI - application to attach handlers to, logger: Any - structured logger used by mapped handlers }
#   OUTPUTS: { None - handlers are attached in place }
#   SIDE_EFFECTS: Mutates FastAPI exception handler registry
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: register_exception_handlers
def register_exception_handlers(app: FastAPI, logger) -> None:
    # START_BLOCK_RESOLVE_EXCEPTION_MAPPINGS
    mappings = app.state.exception_mappings
    # END_BLOCK_RESOLVE_EXCEPTION_MAPPINGS

    # START_BLOCK_REGISTER_VALIDATION_HANDLER
    @app.exception_handler(RequestValidationError)
    # START_CONTRACT: handle_validation_error
    #   PURPOSE: Convert FastAPI request validation errors into the public API error response shape.
    #   INPUTS: { request: Request - failed request context, exc: RequestValidationError - validation error raised by FastAPI }
    #   OUTPUTS: { JSONResponse - standardized validation error response }
    #   SIDE_EFFECTS: none
    #   LINKS: M-SERVER, M-ERRORS
    # END_CONTRACT: handle_validation_error
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        descriptor = ErrorDescriptor(
            status_code=422,
            code="validation_error",
            message="Request validation failed",
            details={"errors": sanitize_validation_errors(exc.errors())},
        )
        return build_error_response(request=request, descriptor=descriptor)

    # END_BLOCK_REGISTER_VALIDATION_HANDLER

    # START_BLOCK_REGISTER_HANDLERS
    for exception_type in mappings:

        @app.exception_handler(exception_type)
        # START_CONTRACT: handle_mapped_error
        #   PURPOSE: Convert known mapped domain errors into standardized API error responses.
        #   INPUTS: { request: Request - failed request context, exc: Exception - mapped exception instance, _exception_type: type[Exception] - captured exception class for handler binding }
        #   OUTPUTS: { JSONResponse - standardized mapped error response }
        #   SIDE_EFFECTS: Emits structured error mapping logs through the shared logger
        #   LINKS: M-SERVER, M-ERRORS
        # END_CONTRACT: handle_mapped_error
        async def handle_mapped_error(
            request: Request, exc: Exception, _exception_type=exception_type
        ) -> JSONResponse:
            descriptor = map_exception_to_descriptor(
                request, exc, mappings[_exception_type], logger
            )
            return build_error_response(request=request, descriptor=descriptor)
    # END_BLOCK_REGISTER_HANDLERS

    # START_BLOCK_REGISTER_FALLBACK_HANDLER
    @app.exception_handler(Exception)
    # START_CONTRACT: handle_unexpected_error
    #   PURPOSE: Convert unexpected uncaught exceptions into the generic internal error response.
    #   INPUTS: { request: Request - failed request context, exc: Exception - uncaught exception instance }
    #   OUTPUTS: { JSONResponse - generic internal server error response }
    #   SIDE_EFFECTS: none
    #   LINKS: M-SERVER, M-ERRORS
    # END_CONTRACT: handle_unexpected_error
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        descriptor = ErrorDescriptor(
            status_code=500,
            code="internal_error",
            message="Unexpected internal server error",
            details={"reason": "Unexpected internal server error"},
        )
        return build_error_response(request=request, descriptor=descriptor)

    # END_BLOCK_REGISTER_FALLBACK_HANDLER


# START_CONTRACT: build_exception_mappings
#   PURPOSE: Build the exception-to-error-descriptor mapping table used by API handlers.
#   INPUTS: { settings: ServerSettings - server settings supplying status-code policy values }
#   OUTPUTS: { dict[type[Exception], ExceptionMapping] - mapping of exception classes to descriptor builders }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: build_exception_mappings
def build_exception_mappings(
    settings: ServerSettings,
) -> dict[type[Exception], ExceptionMapping]:
    return {
        ModelNotAvailableError: ExceptionMapping(
            error_type=ModelNotAvailableError,
            builder=lambda exc: ErrorDescriptor(
                status_code=404,
                code="model_not_available",
                message="Requested model is not available",
                details=build_model_not_available_details(exc),
            ),
        ),
        BackendNotAvailableError: ExceptionMapping(
            error_type=BackendNotAvailableError,
            builder=lambda exc: ErrorDescriptor(
                status_code=503,
                code="backend_not_available",
                message="Configured backend is not available",
                details=build_error_details(exc, default_reason=str(exc)),
                retryable=True,
            ),
        ),
        BackendCapabilityError: ExceptionMapping(
            error_type=BackendCapabilityError,
            builder=lambda exc: ErrorDescriptor(
                status_code=422,
                code="backend_capability_missing",
                message="Selected backend does not support the requested operation",
                details=build_error_details(exc, default_reason=str(exc)),
            ),
        ),
        ModelCapabilityError: ExceptionMapping(
            error_type=ModelCapabilityError,
            builder=lambda exc: ErrorDescriptor(
                status_code=422,
                code="model_capability_not_supported",
                message="Requested model does not support the requested operation",
                details=build_error_details(exc, default_reason=str(exc)),
            ),
        ),
        RuntimeCapabilityNotConfiguredError: ExceptionMapping(
            error_type=RuntimeCapabilityNotConfiguredError,
            builder=lambda exc: ErrorDescriptor(
                status_code=422,
                code="runtime_capability_not_configured",
                message="Requested mode is not configured for the current runtime",
                details=build_error_details(exc, default_reason=str(exc)),
            ),
        ),
        ModelLoadError: ExceptionMapping(
            error_type=ModelLoadError,
            builder=lambda exc: ErrorDescriptor(
                status_code=500,
                code="model_load_failed",
                message="Failed to load model",
                details=build_error_details(exc, default_reason=str(exc)),
            ),
        ),
        AudioConversionError: ExceptionMapping(
            error_type=AudioConversionError,
            builder=lambda exc: ErrorDescriptor(
                status_code=400,
                code="audio_conversion_failed",
                message="Could not process reference audio",
                details=build_error_details(exc, default_reason=str(exc)),
            ),
        ),
        InferenceBusyError: ExceptionMapping(
            error_type=InferenceBusyError,
            builder=lambda exc: ErrorDescriptor(
                status_code=settings.inference_busy_status_code,
                code="inference_busy",
                message="Server is busy processing another inference request",
                details=build_error_details(exc, default_reason=str(exc)),
                retryable=True,
            ),
        ),
        TTSGenerationError: ExceptionMapping(
            error_type=TTSGenerationError,
            builder=lambda exc: build_generation_error_descriptor(exc),
        ),
        RequestTimeoutError: ExceptionMapping(
            error_type=RequestTimeoutError,
            builder=lambda exc: ErrorDescriptor(
                status_code=504,
                code="request_timeout",
                message="Inference request timed out",
                details=build_error_details(exc, default_reason=str(exc)),
                retryable=True,
            ),
        ),
        JobQueueFullError: ExceptionMapping(
            error_type=JobQueueFullError,
            builder=lambda exc: ErrorDescriptor(
                status_code=settings.inference_busy_status_code,
                code="job_queue_full",
                message="Job queue is full",
                details=build_error_details(exc, default_reason=str(exc)),
                retryable=True,
            ),
        ),
        JobNotFoundError: ExceptionMapping(
            error_type=JobNotFoundError,
            builder=lambda exc: ErrorDescriptor(
                status_code=404,
                code="job_not_found",
                message="Job was not found",
                details=build_error_details(exc, default_reason=str(exc)),
            ),
        ),
        JobNotReadyError: ExceptionMapping(
            error_type=JobNotReadyError,
            builder=lambda exc: ErrorDescriptor(
                status_code=409,
                code="job_not_ready",
                message="Job result is not ready",
                details=build_error_details(exc, default_reason=str(exc)),
            ),
        ),
        JobNotSucceededError: ExceptionMapping(
            error_type=JobNotSucceededError,
            builder=lambda exc: ErrorDescriptor(
                status_code=409,
                code="job_not_succeeded",
                message="Job did not finish successfully",
                details=build_error_details(exc, default_reason=str(exc)),
            ),
        ),
        JobNotCancellableError: ExceptionMapping(
            error_type=JobNotCancellableError,
            builder=lambda exc: ErrorDescriptor(
                status_code=409,
                code="job_not_cancellable",
                message="Job cannot be cancelled in its current state",
                details=build_error_details(exc, default_reason=str(exc)),
            ),
        ),
        JobIdempotencyConflictError: ExceptionMapping(
            error_type=JobIdempotencyConflictError,
            builder=lambda exc: ErrorDescriptor(
                status_code=409,
                code="job_idempotency_conflict",
                message="Idempotency key conflicts with an existing job payload",
                details=build_error_details(exc, default_reason=str(exc)),
            ),
        ),
        UnauthorizedError: ExceptionMapping(
            error_type=UnauthorizedError,
            builder=lambda exc: ErrorDescriptor(
                status_code=401,
                code="unauthorized",
                message="Authentication is required or invalid",
                details=build_error_details(exc, default_reason=str(exc)),
            ),
        ),
        ForbiddenError: ExceptionMapping(
            error_type=ForbiddenError,
            builder=lambda exc: ErrorDescriptor(
                status_code=403,
                code="forbidden",
                message="Access to the requested resource is forbidden",
                details=build_error_details(exc, default_reason=str(exc)),
            ),
        ),
        RateLimitExceededError: ExceptionMapping(
            error_type=RateLimitExceededError,
            builder=lambda exc: ErrorDescriptor(
                status_code=429,
                code="rate_limit_exceeded",
                message="Request rate limit exceeded",
                details=build_error_details(exc, default_reason=str(exc)),
                retryable=True,
                headers=build_retry_after_headers(exc.retry_after_seconds),
            ),
        ),
        QuotaExceededError: ExceptionMapping(
            error_type=QuotaExceededError,
            builder=lambda exc: ErrorDescriptor(
                status_code=429,
                code="quota_exceeded",
                message="Quota exceeded for requested operation",
                details=build_error_details(exc, default_reason=str(exc)),
                retryable=True,
                headers=build_retry_after_headers(exc.retry_after_seconds),
            ),
        ),
    }


# START_CONTRACT: map_exception_to_descriptor
#   PURPOSE: Translate a raised exception into a public error descriptor and emit a structured log record.
#   INPUTS: { request: Request - failed request context, exc: Exception - raised exception instance, mapping: ExceptionMapping - descriptor builder mapping, logger: Any - structured logger for error events }
#   OUTPUTS: { ErrorDescriptor - transport-ready public error descriptor }
#   SIDE_EFFECTS: Emits structured mapped-error logs
#   LINKS: M-SERVER, M-ERRORS, M-OBSERVABILITY
# END_CONTRACT: map_exception_to_descriptor
def map_exception_to_descriptor(
    request: Request, exc: Exception, mapping: ExceptionMapping, logger
) -> ErrorDescriptor:
    descriptor = mapping.builder(exc)
    log_event(
        logger,
        level=logging.ERROR,
        event="[ErrorHandlers][map_exception_to_descriptor][MAP_EXCEPTION_TO_DESCRIPTOR]",
        message="Mapped exception to API error response",
        path=request.url.path,
        error_type=type(exc).__name__,
        code=descriptor.code,
        status_code=descriptor.status_code,
        retryable=descriptor.retryable,
        details=descriptor.details,
    )
    return descriptor


# START_CONTRACT: build_generation_error_descriptor
#   PURPOSE: Build the public error descriptor for synthesis generation failures.
#   INPUTS: { exc: TTSGenerationError - generation failure exception }
#   OUTPUTS: { ErrorDescriptor - API error descriptor for failed generation }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: build_generation_error_descriptor
def build_generation_error_descriptor(exc: TTSGenerationError) -> ErrorDescriptor:
    details = build_error_details(exc, default_message=str(exc))
    return ErrorDescriptor(
        status_code=500,
        code="generation_failed",
        message="Audio generation failed",
        details=details,
    )


# START_CONTRACT: build_error_details
#   PURPOSE: Convert exception context into sanitized public error details.
#   INPUTS: { exc: Exception - exception carrying optional public context, default_message: str | None - default message when context is absent, default_reason: str | None - compatibility alias for the fallback reason }
#   OUTPUTS: { dict[str, object] - sanitized public error detail payload }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: build_error_details
def build_error_details(
    exc: Exception,
    *,
    default_message: str | None = None,
    default_reason: str | None = None,
) -> dict[str, object]:
    context = getattr(exc, "context", None)
    if context is not None and hasattr(context, "to_dict"):
        return sanitize_public_error_details(context.to_dict())
    fallback_reason = default_reason or default_message or str(exc)
    return sanitize_public_error_details({"reason": fallback_reason})


# START_CONTRACT: build_retry_after_headers
#   PURPOSE: Build retry-related HTTP headers for rate-limit and quota errors.
#   INPUTS: { retry_after_seconds: int | None - retry delay in seconds if available }
#   OUTPUTS: { dict[str, str] | None - response headers containing Retry-After when applicable }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: build_retry_after_headers
def build_retry_after_headers(retry_after_seconds: int | None) -> dict[str, str] | None:
    if retry_after_seconds is None:
        return None
    return {"Retry-After": str(retry_after_seconds)}


# START_CONTRACT: build_model_not_available_details
#   PURPOSE: Build sanitized public error details for missing-model failures.
#   INPUTS: { exc: ModelNotAvailableError - missing model exception }
#   OUTPUTS: { dict[str, object] - public error details including model id when available }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: build_model_not_available_details
def build_model_not_available_details(exc: ModelNotAvailableError) -> dict[str, object]:
    details = build_error_details(exc, default_reason=str(exc))
    if exc.model_name is not None and "model" not in details:
        details["model"] = exc.model_name
    return details


# START_CONTRACT: sanitize_validation_errors
#   PURPOSE: Normalize request validation error payloads into JSON-serializable public structures.
#   INPUTS: { errors: list[dict] - raw validation error items from FastAPI/Pydantic }
#   OUTPUTS: { list[dict] - sanitized validation error items }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: sanitize_validation_errors
def sanitize_validation_errors(errors: list[dict]) -> list[dict]:
    sanitized: list[dict] = []
    for item in errors:
        normalized = dict(item)
        if "ctx" in normalized and isinstance(normalized["ctx"], dict):
            normalized["ctx"] = {
                key: str(value) for key, value in normalized["ctx"].items()
            }
        sanitized.append(normalized)
    return sanitized


# START_CONTRACT: sanitize_public_error_details
#   PURPOSE: Recursively sanitize public error detail values to avoid leaking local filesystem paths.
#   INPUTS: { value: Any - nested error detail payload }
#   OUTPUTS: { Any - sanitized payload safe for public API responses }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: sanitize_public_error_details
def sanitize_public_error_details(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if _is_path_key(key):
                if isinstance(item, (list, tuple)):
                    sanitized[key] = [_sanitize_path_value(part) for part in item]
                else:
                    sanitized[key] = _sanitize_path_value(item)
                continue
            sanitized[key] = sanitize_public_error_details(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_public_error_details(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_public_error_details(item) for item in value]
    if isinstance(value, str) and _looks_like_local_path(value):
        return _sanitize_path_string(value)
    return value


def _is_path_key(key: str) -> bool:
    return bool(_PATH_KEY_RE.search(key))


def _looks_like_local_path(value: str) -> bool:
    if _PATH_VALUE_RE.search(value):
        return True
    try:
        path = PurePath(value)
    except Exception:
        return False
    return path.is_absolute() and len(path.parts) > 1


def _sanitize_path_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return PurePath(value).name or value


def _sanitize_path_string(value: str) -> str:
    sanitized = value
    for token in value.split():
        normalized = token.strip("\"'(),[]{}")
        if not _looks_like_local_path(normalized):
            continue
        sanitized = sanitized.replace(normalized, _sanitize_path_value(normalized))
    return sanitized


__all__ = [
    "register_exception_handlers",
    "build_exception_mappings",
    "map_exception_to_descriptor",
    "build_generation_error_descriptor",
    "build_error_details",
    "build_retry_after_headers",
    "build_model_not_available_details",
    "sanitize_validation_errors",
    "sanitize_public_error_details",
]
