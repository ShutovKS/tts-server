from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request

from core.observability import Timer, bind_request_context, get_logger, log_event, operation_scope, reset_request_context
from server.api.errors import build_exception_mappings, register_exception_handlers
from server.api.routes_health import register_health_routes
from server.api.routes_models import register_models_routes
from server.api.routes_tts import register_tts_routes
from server.bootstrap import ServerSettings, build_server_runtime


LOGGER = get_logger(__name__)



def create_app(settings: Optional[ServerSettings] = None) -> FastAPI:
    runtime = build_server_runtime(settings)
    app_settings = runtime.settings

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        app_settings.ensure_directories()
        yield

    app = FastAPI(
        title="Qwen3-TTS API Server",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.settings = app_settings
    app.state.runtime = runtime.core
    app.state.registry = runtime.core.registry
    app.state.tts_service = runtime.core.tts_service
    app.state.application = runtime.core.application
    app.state.exception_mappings = build_exception_mappings(app_settings)

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        request_timer = Timer()
        request_token = bind_request_context(request_id)
        with operation_scope(f"http.{request.method.lower()} {request.url.path}"):
            log_event(
                LOGGER,
                level=logging.INFO,
                event="http.request.started",
                message="HTTP request started",
                method=request.method,
                path=request.url.path,
                query=str(request.url.query),
                client=getattr(request.client, "host", None),
            )
            try:
                response = await call_next(request)
            except Exception as exc:
                log_event(
                    LOGGER,
                    level=logging.ERROR,
                    event="http.request.failed",
                    message="HTTP request failed before response serialization",
                    method=request.method,
                    path=request.url.path,
                    duration_ms=request_timer.elapsed_ms,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise
            else:
                response.headers["x-request-id"] = request_id
                log_event(
                    LOGGER,
                    level=logging.INFO,
                    event="http.request.completed",
                    message="HTTP request completed",
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    duration_ms=request_timer.elapsed_ms,
                )
                return response
            finally:
                reset_request_context(request_token)

    register_exception_handlers(app, LOGGER)
    register_health_routes(app, LOGGER)
    register_models_routes(app, LOGGER)
    register_tts_routes(app, LOGGER)
    return app


app = create_app()
