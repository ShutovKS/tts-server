# FILE: server/api/tts/jobs.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Register shared async TTS job resource HTTP routes.
#   SCOPE: GET /api/v1/tts/jobs/{job_id}, GET /api/v1/tts/jobs/{job_id}/result, POST /api/v1/tts/jobs/{job_id}/cancel route registration only
#   DEPENDS: M-APPLICATION, M-CONTRACTS, M-ERRORS, M-OBSERVABILITY, M-SERVER
#   LINKS: M-SERVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   register_tts_job_routes - Register shared async TTS job status, result, and cancel routes on the FastAPI app
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Split shared async TTS job-resource route registration out of server/api/routes_tts.py without changing public HTTP behavior]
# END_CHANGE_SUMMARY

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, Response  # pyright: ignore[reportMissingImports]
from fastapi.responses import JSONResponse  # pyright: ignore[reportMissingImports]

from core.contracts.jobs import JobStatus
from core.errors import JobNotCancellableError, JobNotFoundError, JobNotReadyError, JobNotSucceededError
from core.observability import log_event
from server.api.auth import ensure_job_owner_access
from server.api.policies import enforce_job_cancel_admission, enforce_job_read_admission
from server.api.responses import apply_async_job_headers, build_audio_response
from server.api.tts._helpers import build_job_snapshot_payload, get_job_snapshot_or_raise, public_job_status
from server.schemas.audio import JobSnapshotPayload
from server.schemas.errors import ErrorResponse


# START_CONTRACT: register_tts_job_routes
#   PURPOSE: Register shared async TTS job-resource routes on the FastAPI application.
#   INPUTS: { app: FastAPI - application to attach routes to, logger: Any - structured logger used by endpoint handlers }
#   OUTPUTS: { None - routes are attached in place }
#   SIDE_EFFECTS: Mutates FastAPI routing table by registering async TTS job-resource endpoints
#   LINKS: M-SERVER, M-APPLICATION
# END_CONTRACT: register_tts_job_routes
def register_tts_job_routes(app: FastAPI, logger) -> None:
    @app.get(
        "/api/v1/tts/jobs/{job_id}",
        name="tts_job_status",
        tags=["tts"],
        response_model=JobSnapshotPayload,
        responses={
            401: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
        },
    )
    # START_CONTRACT: tts_job_status
    #   PURPOSE: Return the current async job snapshot for an owned TTS job.
    #   INPUTS: { request: Request - incoming HTTP request, job_id: str - async job identifier }
    #   OUTPUTS: { JobSnapshotPayload - current async job snapshot }
    #   SIDE_EFFECTS: Consumes admission quota and enforces owner access checks
    #   LINKS: M-SERVER, M-APPLICATION
    # END_CONTRACT: tts_job_status
    async def tts_job_status(request: Request, job_id: str) -> JobSnapshotPayload:
        await enforce_job_read_admission(request)
        snapshot = get_job_snapshot_or_raise(request, job_id)
        log_event(
            logger,
            level=logging.INFO,
            event="[RoutesTTS][tts_job_status][BLOCK_READ_ASYNC_JOB_STATUS]",
            message="Async job status retrieved",
            endpoint="/api/v1/tts/jobs/{job_id}",
            job_id=snapshot.job_id,
            submit_request_id=snapshot.submit_request_id,
            current_request_id=request.state.request_id,
            public_status=public_job_status(snapshot.status),
        )
        response = JSONResponse(
            status_code=200,
            content=build_job_snapshot_payload(request, snapshot).model_dump(mode="json"),
        )
        return apply_async_job_headers(response, snapshot)

    @app.get(
        "/api/v1/tts/jobs/{job_id}/result",
        name="tts_job_result",
        tags=["tts"],
        responses={
            401: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
        },
    )
    # START_CONTRACT: tts_job_result
    #   PURPOSE: Return completed audio for an owned async TTS job when it has succeeded.
    #   INPUTS: { request: Request - incoming HTTP request, job_id: str - async job identifier }
    #   OUTPUTS: { Response - generated audio response for the completed job }
    #   SIDE_EFFECTS: Consumes admission quota and may raise job state or ownership errors
    #   LINKS: M-SERVER, M-APPLICATION
    # END_CONTRACT: tts_job_result
    async def tts_job_result(request: Request, job_id: str) -> Response:
        # START_BLOCK_LOAD_JOB_RESULT
        await enforce_job_read_admission(request)
        resolution = request.app.state.job_execution.get_result(job_id)
        if resolution is None:
            raise JobNotFoundError(job_id)
        # END_BLOCK_LOAD_JOB_RESULT

        # START_BLOCK_VALIDATE_JOB_RESULT
        snapshot = resolution.snapshot
        ensure_job_owner_access(request, owner_principal_id=snapshot.owner_principal_id)
        if snapshot.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
            raise JobNotReadyError(job_id, snapshot.status.value)
        if snapshot.status is not JobStatus.SUCCEEDED or resolution.success is None:
            raise JobNotSucceededError(
                job_id,
                public_job_status(snapshot.status),
                details={
                    "terminal_error": (
                        {
                            "code": snapshot.terminal_error.code,
                            "message": snapshot.terminal_error.message,
                            "details": snapshot.terminal_error.details,
                        }
                        if snapshot.terminal_error is not None
                        else None
                    )
                },
            )
        # END_BLOCK_VALIDATE_JOB_RESULT
        log_event(
            logger,
            level=logging.INFO,
            event="[RoutesTTS][tts_job_result][BLOCK_DELIVER_ASYNC_JOB_RESULT]",
            message="Async job result delivered",
            endpoint="/api/v1/tts/jobs/{job_id}/result",
            job_id=snapshot.job_id,
            submit_request_id=snapshot.submit_request_id,
            current_request_id=request.state.request_id,
            public_status=public_job_status(snapshot.status),
        )

        # START_BLOCK_BUILD_JOB_RESULT_RESPONSE
        response = build_audio_response(
            request,
            resolution.success.generation,
            snapshot.response_format or "wav",
            logger,
        )
        response = apply_async_job_headers(response, snapshot)
        return response
        # END_BLOCK_BUILD_JOB_RESULT_RESPONSE

    @app.post(
        "/api/v1/tts/jobs/{job_id}/cancel",
        name="tts_job_cancel",
        tags=["tts"],
        response_model=JobSnapshotPayload,
        responses={
            401: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
        },
    )
    # START_CONTRACT: tts_job_cancel
    #   PURPOSE: Cancel an owned async TTS job when it is still cancellable.
    #   INPUTS: { request: Request - incoming HTTP request, job_id: str - async job identifier }
    #   OUTPUTS: { Response - job snapshot response reflecting cancellation state }
    #   SIDE_EFFECTS: Consumes admission quota and mutates async job execution state when cancellation succeeds
    #   LINKS: M-SERVER, M-APPLICATION
    # END_CONTRACT: tts_job_cancel
    async def tts_job_cancel(request: Request, job_id: str) -> Response:
        # START_BLOCK_VALIDATE_CANCELLATION_REQUEST
        await enforce_job_cancel_admission(request)
        snapshot = get_job_snapshot_or_raise(request, job_id)
        if snapshot.status not in {JobStatus.QUEUED, JobStatus.CANCELLED}:
            raise JobNotCancellableError(job_id, snapshot.status.value)
        # END_BLOCK_VALIDATE_CANCELLATION_REQUEST

        # START_BLOCK_SUBMIT_CANCELLATION
        cancelled = request.app.state.job_execution.cancel(job_id)
        if cancelled is None:
            raise JobNotFoundError(job_id)
        # END_BLOCK_SUBMIT_CANCELLATION
        log_event(
            logger,
            level=logging.INFO,
            event="[RoutesTTS][tts_job_cancel][BLOCK_CANCEL_ASYNC_JOB]",
            message="Async job cancellation resolved",
            endpoint="/api/v1/tts/jobs/{job_id}/cancel",
            job_id=cancelled.job_id,
            submit_request_id=cancelled.submit_request_id,
            current_request_id=request.state.request_id,
            public_status=public_job_status(cancelled.status),
        )

        # START_BLOCK_BUILD_CANCEL_RESPONSE
        status_code = 200 if snapshot.status is JobStatus.CANCELLED else 202
        response = JSONResponse(
            status_code=status_code,
            content=build_job_snapshot_payload(request, cancelled).model_dump(mode="json"),
        )
        return apply_async_job_headers(response, cancelled)
        # END_BLOCK_BUILD_CANCEL_RESPONSE


__all__ = ["register_tts_job_routes"]
