# FILE: server/api/routes_models.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Define model discovery and lifecycle HTTP endpoints.
#   SCOPE: GET /api/v1/models, DELETE /api/v1/models/{model_id}, POST /api/v1/models/{model_id}/download, GET /api/v1/models/{model_id}/downloads/{job_id}, POST /api/v1/models/refresh
#   DEPENDS: M-MODEL-REGISTRY, M-MODEL-LIFECYCLE
#   LINKS: M-SERVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _serialize_download_job - Convert a ModelDownloadJob into a JSON-friendly dict.
#   register_models_routes - Register model discovery and lifecycle routes on the FastAPI app.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Phase 4.13: registered DELETE /api/v1/models/{model_id}, POST /api/v1/models/{model_id}/download, GET /api/v1/models/{model_id}/downloads/{job_id}, and POST /api/v1/models/refresh that delegate to ModelLifecycleService via app.state.model_lifecycle]
# END_CHANGE_SUMMARY

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.observability import log_event, operation_scope
from core.services.model_lifecycle import ModelDownloadJob
from server.api.policies import enforce_control_plane_admission
from server.schemas.audio import ModelsResponse


class ModelDownloadRequest(BaseModel):
    source: str | None = None


def _serialize_download_job(job: ModelDownloadJob) -> dict[str, Any]:
    # START_BLOCK_SERIALIZE_DOWNLOAD_JOB
    return {
        "id": job.id,
        "model_id": job.model_id,
        "source": job.source,
        "status": job.status,
        "progress": job.progress,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "completed_at": job.completed_at,
        "details": dict(job.details),
    }
    # END_BLOCK_SERIALIZE_DOWNLOAD_JOB


# START_CONTRACT: register_models_routes
#   PURPOSE: Register model discovery and lifecycle HTTP routes on the FastAPI application.
#   INPUTS: { app: FastAPI - application to attach routes to, logger: Any - structured logger used by endpoint handlers }
#   OUTPUTS: { None - routes are attached in place }
#   SIDE_EFFECTS: Mutates FastAPI routing table by registering model discovery and lifecycle endpoints
#   LINKS: M-SERVER, M-MODEL-REGISTRY, M-MODEL-LIFECYCLE
# END_CONTRACT: register_models_routes
def register_models_routes(app: FastAPI, logger) -> None:
    @app.get("/api/v1/models", response_model=ModelsResponse, tags=["models"])
    # START_CONTRACT: list_models
    #   PURPOSE: Return the list of models currently registered for the API.
    #   INPUTS: { request: Request - incoming request used to access registry state }
    #   OUTPUTS: { ModelsResponse - model discovery payload }
    #   SIDE_EFFECTS: Consumes control-plane admission checks and emits model listing logs
    #   LINKS: M-SERVER, M-MODEL-REGISTRY
    # END_CONTRACT: list_models
    async def list_models(request: Request) -> ModelsResponse:
        with operation_scope("server.list_models"):
            await enforce_control_plane_admission(request)
            models = request.app.state.registry.list_models()
            log_event(
                logger,
                level=20,
                event="[RoutesModels][list_models][LIST_MODELS]",
                message="Model listing completed",
                model_count=len(models),
            )
            return ModelsResponse(data=models)

    @app.delete("/api/v1/models/{model_id}", tags=["models"])
    # START_CONTRACT: delete_model
    #   PURPOSE: Remove the on-disk artifacts of a registered model from the configured models_dir.
    #   INPUTS: { request: Request - incoming request used to access lifecycle state, model_id: str - identifier of the model to delete }
    #   OUTPUTS: { JSONResponse - 200 with summary on success, 404 when the spec is unknown or the folder is already absent }
    #   SIDE_EFFECTS: Consumes control-plane admission checks, deletes the model folder, and emits model deletion logs
    #   LINKS: M-SERVER, M-MODEL-LIFECYCLE
    # END_CONTRACT: delete_model
    async def delete_model(request: Request, model_id: str) -> JSONResponse:
        with operation_scope("server.delete_model"):
            await enforce_control_plane_admission(request)
            lifecycle = request.app.state.model_lifecycle
            removed = lifecycle.delete_model(model_id)
            log_event(
                logger,
                level=20,
                event="[RoutesModels][delete_model][BLOCK_REPORT_DELETE_OUTCOME]",
                message="Delete model request handled",
                model_id=model_id,
                removed=removed,
            )
            if not removed:
                raise HTTPException(
                    status_code=404,
                    detail={"code": "model_not_found", "model_id": model_id},
                )
            return JSONResponse({"model_id": model_id, "removed": True})

    @app.post("/api/v1/models/{model_id}/download", tags=["models"])
    # START_CONTRACT: submit_model_download
    #   PURPOSE: Submit a best-effort download for a registered model and return the freshly created job descriptor.
    #   INPUTS: { request: Request - incoming request used to access lifecycle state, model_id: str - target model identifier, payload: ModelDownloadRequest - optional source descriptor }
    #   OUTPUTS: { JSONResponse - 202 with the job descriptor and a Location header pointing at the status endpoint }
    #   SIDE_EFFECTS: Consumes control-plane admission checks, queues the download via ModelLifecycleService, and emits a download submission log
    #   LINKS: M-SERVER, M-MODEL-LIFECYCLE
    # END_CONTRACT: submit_model_download
    async def submit_model_download(
        request: Request, model_id: str, payload: ModelDownloadRequest | None = None
    ) -> JSONResponse:
        with operation_scope("server.submit_model_download"):
            await enforce_control_plane_admission(request)
            source = payload.source if payload is not None else None
            lifecycle = request.app.state.model_lifecycle
            job = lifecycle.submit_download(model_id, source=source)
            log_event(
                logger,
                level=20,
                event="[RoutesModels][submit_model_download][BLOCK_QUEUE_JOB]",
                message="Model download submission accepted",
                model_id=model_id,
                job_id=job.id,
                source=source,
                status=job.status,
            )
            location = f"/api/v1/models/{model_id}/downloads/{job.id}"
            return JSONResponse(
                _serialize_download_job(job),
                status_code=202,
                headers={"Location": location},
            )

    @app.get("/api/v1/models/{model_id}/downloads/{job_id}", tags=["models"])
    # START_CONTRACT: get_model_download
    #   PURPOSE: Look up the status of a previously submitted model download job.
    #   INPUTS: { request: Request - incoming request used to access lifecycle state, model_id: str - target model identifier (used for cross-checking), job_id: str - identifier returned by submit_model_download }
    #   OUTPUTS: { JSONResponse - 200 with the job descriptor when found, 404 when the job is unknown or addressed under a different model_id }
    #   SIDE_EFFECTS: Consumes control-plane admission checks and emits a status lookup log
    #   LINKS: M-SERVER, M-MODEL-LIFECYCLE
    # END_CONTRACT: get_model_download
    async def get_model_download(request: Request, model_id: str, job_id: str) -> JSONResponse:
        with operation_scope("server.get_model_download"):
            await enforce_control_plane_admission(request)
            lifecycle = request.app.state.model_lifecycle
            job = lifecycle.get_download(job_id)
            if job is None or job.model_id != model_id:
                raise HTTPException(
                    status_code=404,
                    detail={"code": "download_job_not_found", "job_id": job_id},
                )
            log_event(
                logger,
                level=20,
                event="[RoutesModels][get_model_download][BLOCK_REPORT_STATUS]",
                message="Returning model download job status",
                model_id=model_id,
                job_id=job_id,
                status=job.status,
            )
            return JSONResponse(_serialize_download_job(job))

    @app.post("/api/v1/models/refresh", tags=["models"])
    # START_CONTRACT: refresh_models
    #   PURPOSE: Trigger a best-effort registry refresh so newly added or removed model folders are picked up without restarting the server.
    #   INPUTS: { request: Request - incoming request used to access lifecycle state }
    #   OUTPUTS: { JSONResponse - 200 with refresh outcome (supported flag, model_count, optionally method) }
    #   SIDE_EFFECTS: Consumes control-plane admission checks, may invoke the registry's reload hook, and emits a refresh log
    #   LINKS: M-SERVER, M-MODEL-LIFECYCLE
    # END_CONTRACT: refresh_models
    async def refresh_models(request: Request) -> JSONResponse:
        with operation_scope("server.refresh_models"):
            await enforce_control_plane_admission(request)
            lifecycle = request.app.state.model_lifecycle
            outcome = lifecycle.refresh()
            log_event(
                logger,
                level=20,
                event="[RoutesModels][refresh_models][BLOCK_REPORT_REFRESH]",
                message="Model registry refresh completed",
                supported=outcome.get("supported"),
                model_count=outcome.get("model_count"),
            )
            return JSONResponse(outcome)


__all__ = [
    "ModelDownloadRequest",
    "register_models_routes",
]
