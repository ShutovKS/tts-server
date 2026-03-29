from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from core.errors import (
    AudioConversionError,
    BackendCapabilityError,
    BackendNotAvailableError,
    InferenceBusyError,
    ModelLoadError,
    ModelNotAvailableError,
    TTSGenerationError,
)
from core.observability import log_event
from server.api.contracts import ErrorDescriptor, ExceptionMapping
from server.api.responses import build_error_response
from server.bootstrap import ServerSettings



def register_exception_handlers(app: FastAPI, logger) -> None:
    mappings = app.state.exception_mappings

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        descriptor = ErrorDescriptor(
            status_code=422,
            code="validation_error",
            message="Request validation failed",
            details={"errors": sanitize_validation_errors(exc.errors())},
        )
        return build_error_response(request=request, descriptor=descriptor)

    for exception_type in mappings:

        @app.exception_handler(exception_type)
        async def handle_mapped_error(request: Request, exc: Exception, _exception_type=exception_type) -> JSONResponse:
            descriptor = map_exception_to_descriptor(request, exc, mappings[_exception_type], logger)
            return build_error_response(request=request, descriptor=descriptor)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        descriptor = ErrorDescriptor(
            status_code=500,
            code="internal_error",
            message="Unexpected internal server error",
            details={"reason": str(exc)},
        )
        return build_error_response(request=request, descriptor=descriptor)



def build_exception_mappings(settings: ServerSettings) -> dict[type[Exception], ExceptionMapping]:
    return {
        ModelNotAvailableError: ExceptionMapping(
            error_type=ModelNotAvailableError,
            builder=lambda exc: ErrorDescriptor(
                status_code=404,
                code="model_not_available",
                message="Requested model is not available",
                details={"model": exc.model_name},
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
    }



def map_exception_to_descriptor(request: Request, exc: Exception, mapping: ExceptionMapping, logger) -> ErrorDescriptor:
    descriptor = mapping.builder(exc)
    log_event(
        logger,
        level=logging.ERROR,
        event="http.error.mapped",
        message="Mapped exception to API error response",
        path=request.url.path,
        error_type=type(exc).__name__,
        code=descriptor.code,
        status_code=descriptor.status_code,
        retryable=descriptor.retryable,
        details=descriptor.details,
    )
    return descriptor



def build_generation_error_descriptor(exc: TTSGenerationError) -> ErrorDescriptor:
    details = build_error_details(exc, default_reason=str(exc))
    return ErrorDescriptor(
        status_code=500,
        code="generation_failed",
        message="Audio generation failed",
        details=details,
    )



def build_error_details(exc: Exception, *, default_reason: str) -> dict[str, object]:
    context = getattr(exc, "context", None)
    if context is not None and hasattr(context, "to_dict"):
        return context.to_dict()
    return {"reason": default_reason}



def sanitize_validation_errors(errors: list[dict]) -> list[dict]:
    sanitized: list[dict] = []
    for item in errors:
        normalized = dict(item)
        if "ctx" in normalized and isinstance(normalized["ctx"], dict):
            normalized["ctx"] = {key: str(value) for key, value in normalized["ctx"].items()}
        sanitized.append(normalized)
    return sanitized
