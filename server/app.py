# FILE: server/app.py
# VERSION: 1.0.3
# START_MODULE_CONTRACT
#   PURPOSE: Compose the FastAPI application with all routes, middleware, and error handlers.
#   SCOPE: FastAPI app factory, route registration, middleware setup, lifespan management
#   DEPENDS: M-BOOTSTRAP, M-CONFIG, M-ERRORS, M-OBSERVABILITY
#   LINKS: M-SERVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LOGGER - Module logger for server application events
#   create_app - Build and configure the FastAPI application
#   app - FastAPI application instance
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.3 - Exposed async correlation headers through CORS so remote multi-client consumers can read stable job and submit identifiers]
# END_CHANGE_SUMMARY

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from core.observability import (
    Timer,
    bind_request_context,
    get_logger,
    log_event,
    operation_scope,
    reset_request_context,
)
from server.api.auth import resolve_request_principal
from server.api.errors import (
    build_exception_mappings,
    map_exception_to_descriptor,
    register_exception_handlers,
)
from server.api.responses import build_error_response
from server.api.routes_health import register_health_routes
from server.api.routes_models import register_models_routes
from server.api.routes_tts import register_tts_routes
from server.bootstrap import ServerSettings, build_server_runtime


LOGGER = get_logger(__name__)
DEFAULT_DEMO_CORS_ORIGINS = (
    "http://127.0.0.1:8030",
    "http://localhost:8030",
    "http://0.0.0.0:8030",
    "https://split-tts.drive-vr.ru",
)


# START_CONTRACT: create_app
#   PURPOSE: Build and configure the FastAPI server application from resolved runtime settings.
#   INPUTS: { settings: Optional[ServerSettings] - optional server settings override for runtime assembly }
#   OUTPUTS: { FastAPI - configured API application with routes, middleware, and handlers }
#   SIDE_EFFECTS: Initializes app state, registers middleware and routes, and starts/stops job manager during lifespan
#   LINKS: M-SERVER, M-BOOTSTRAP
# END_CONTRACT: create_app
def create_app(settings: Optional[ServerSettings] = None) -> FastAPI:
    runtime = build_server_runtime(settings)
    app_settings = runtime.settings
    cors_allowed_origins = list(app_settings.cors_allowed_origins or DEFAULT_DEMO_CORS_ORIGINS)

    @asynccontextmanager
    # START_CONTRACT: lifespan
    #   PURPOSE: Manage server startup and shutdown around the FastAPI lifespan.
    #   INPUTS: { _: FastAPI - application instance supplied by FastAPI lifespan handling }
    #   OUTPUTS: { AsyncIterator[None] - async context manager yielding control during app lifetime }
    #   SIDE_EFFECTS: Ensures directories exist and starts then stops the shared job manager
    #   LINKS: M-SERVER, M-BOOTSTRAP
    # END_CONTRACT: lifespan
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # START_BLOCK_STARTUP
        app_settings.ensure_directories()
        runtime.core.job_manager.start()
        # END_BLOCK_STARTUP
        try:
            yield
        finally:
            # START_BLOCK_SHUTDOWN
            runtime.core.job_manager.stop()
            # END_BLOCK_SHUTDOWN

    app = FastAPI(
        title="Qwen3-TTS API Server",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "x-request-id",
            "x-model-id",
            "x-backend-id",
            "x-job-id",
            "x-submit-request-id",
            "x-tts-mode",
            "x-saved-output-file",
        ],
    )

    # START_BLOCK_CONFIGURE_APP_STATE
    app.state.settings = app_settings
    app.state.runtime = runtime.core
    app.state.registry = runtime.core.registry
    app.state.tts_service = runtime.core.tts_service
    app.state.application = runtime.core.application
    app.state.job_store = runtime.core.job_store
    app.state.job_execution = runtime.core.job_execution
    app.state.rate_limiter = runtime.core.rate_limiter
    app.state.quota_guard = runtime.core.quota_guard
    app.state.metrics = runtime.core.metrics
    app.state.logger = LOGGER
    app.state.exception_mappings = build_exception_mappings(app_settings)
    # END_BLOCK_CONFIGURE_APP_STATE

    # START_BLOCK_REGISTER_MIDDLEWARE
    @app.middleware("http")
    # START_CONTRACT: request_context_middleware
    #   PURPOSE: Bind request-scoped context, principal resolution, and uniform completion/error logging.
    #   INPUTS: { request: Request - incoming HTTP request, call_next: Any - downstream FastAPI middleware handler }
    #   OUTPUTS: { Response - downstream response or mapped error response }
    #   SIDE_EFFECTS: Writes request state, binds observability context, logs request lifecycle events, and sets response headers
    #   LINKS: M-SERVER, M-ERRORS, M-OBSERVABILITY
    # END_CONTRACT: request_context_middleware
    async def request_context_middleware(request: Request, call_next):
        # START_BLOCK_INITIALIZE_REQUEST_CONTEXT
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        request_timer = Timer()
        request_token = bind_request_context(request_id)
        # END_BLOCK_INITIALIZE_REQUEST_CONTEXT
        with operation_scope(f"http.{request.method.lower()} {request.url.path}"):
            # START_BLOCK_LOG_REQUEST_START
            log_event(
                LOGGER,
                level=logging.INFO,
                event="[ServerApp][request_context_middleware][BLOCK_LOG_REQUEST_START]",
                message="HTTP request started",
                method=request.method,
                path=request.url.path,
                query=str(request.url.query),
                client=getattr(request.client, "host", None),
            )
            # END_BLOCK_LOG_REQUEST_START
            try:
                # START_BLOCK_RESOLVE_PRINCIPAL_AND_CALL_NEXT
                principal = resolve_request_principal(request)
                request.state.principal = principal
                request.state.principal_id = principal.principal_id
                response = await call_next(request)
                # END_BLOCK_RESOLVE_PRINCIPAL_AND_CALL_NEXT
            except Exception as exc:
                mapping = app.state.exception_mappings.get(type(exc))
                if mapping is not None:
                    # START_BLOCK_BUILD_MAPPED_ERROR_RESPONSE
                    descriptor = map_exception_to_descriptor(
                        request, exc, mapping, LOGGER
                    )
                    response = build_error_response(
                        request=request, descriptor=descriptor
                    )
                    response.headers["x-request-id"] = request_id
                    log_event(
                        LOGGER,
                        level=logging.INFO,
                        event="[ServerApp][request_context_middleware][BLOCK_BUILD_MAPPED_ERROR_RESPONSE]",
                        message="HTTP request completed",
                        method=request.method,
                        path=request.url.path,
                        status_code=response.status_code,
                        duration_ms=request_timer.elapsed_ms,
                    )
                    return response
                    # END_BLOCK_BUILD_MAPPED_ERROR_RESPONSE
                # START_BLOCK_LOG_UNMAPPED_REQUEST_FAILURE
                log_event(
                    LOGGER,
                    level=logging.ERROR,
                    event="[ServerApp][request_context_middleware][BLOCK_LOG_UNMAPPED_REQUEST_FAILURE]",
                    message="HTTP request failed before response serialization",
                    method=request.method,
                    path=request.url.path,
                    duration_ms=request_timer.elapsed_ms,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise
                # END_BLOCK_LOG_UNMAPPED_REQUEST_FAILURE
            else:
                # START_BLOCK_LOG_SUCCESS_RESPONSE
                response.headers["x-request-id"] = request_id
                log_event(
                    LOGGER,
                    level=logging.INFO,
                    event="[ServerApp][request_context_middleware][BLOCK_LOG_SUCCESS_RESPONSE]",
                    message="HTTP request completed",
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    duration_ms=request_timer.elapsed_ms,
                )
                return response
                # END_BLOCK_LOG_SUCCESS_RESPONSE
            finally:
                # START_BLOCK_RESET_REQUEST_CONTEXT
                reset_request_context(request_token)
                # END_BLOCK_RESET_REQUEST_CONTEXT

    # END_BLOCK_REGISTER_MIDDLEWARE

    # START_BLOCK_REGISTER_ROUTES
    register_exception_handlers(app, LOGGER)
    register_health_routes(app, LOGGER)
    register_models_routes(app, LOGGER)
    register_tts_routes(app, LOGGER)
    # END_BLOCK_REGISTER_ROUTES
    return app


app = create_app()

__all__ = [
    "LOGGER",
    "create_app",
    "app",
]
