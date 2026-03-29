from __future__ import annotations

from fastapi import FastAPI, Request

from core.infrastructure.audio_io import check_ffmpeg_available
from core.observability import log_event, operation_scope
from server.schemas.audio import HealthResponse



def register_health_routes(app: FastAPI, logger) -> None:
    @app.get("/health/live", response_model=HealthResponse, tags=["health"])
    async def health_live() -> HealthResponse:
        return HealthResponse(status="ok", checks={"process": "alive"})

    @app.get("/health/ready", response_model=HealthResponse, tags=["health"])
    async def health_ready(request: Request) -> HealthResponse:
        with operation_scope("server.health_ready"):
            readiness = build_readiness_report(request)
            log_event(
                logger,
                level=20,
                event="health.ready.checked",
                message="Readiness probe evaluated",
                status=readiness.status,
                available_models=readiness.checks["models"]["available_models"],
                runtime_ready_models=readiness.checks["models"]["runtime_ready_models"],
                ffmpeg_available=readiness.checks["ffmpeg"]["available"],
            )
            return readiness



def build_readiness_report(request: Request) -> HealthResponse:
    registry_report = request.app.state.registry.readiness_report()
    ffmpeg_ready = check_ffmpeg_available()
    settings = request.app.state.settings
    config = {
        "models_dir": str(settings.models_dir),
        "models_dir_exists": settings.models_dir.exists(),
        "outputs_dir": str(settings.outputs_dir),
        "outputs_dir_exists": settings.outputs_dir.exists(),
        "voices_dir": str(settings.voices_dir),
        "voices_dir_exists": settings.voices_dir.exists(),
        "sample_rate": settings.sample_rate,
        "max_upload_size_bytes": settings.max_upload_size_bytes,
    }
    runtime = {
        "inference_busy": request.app.state.runtime.inference_guard.is_busy(),
        "streaming_enabled": settings.enable_streaming,
        "default_save_output": settings.default_save_output,
        "request_timeout_seconds": settings.request_timeout_seconds,
        "configured_backend": settings.backend,
        "backend_autoselect": settings.backend_autoselect,
    }
    status = "ok" if registry_report["registry_ready"] and ffmpeg_ready and config["models_dir_exists"] else "degraded"
    return HealthResponse(
        status=status,
        checks={
            "models": registry_report,
            "ffmpeg": {"available": ffmpeg_ready},
            "config": config,
            "runtime": runtime,
        },
    )
