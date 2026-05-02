# FILE: server/api/tts/design.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Register voice design synchronous and async TTS HTTP routes.
#   SCOPE: POST /api/v1/tts/design and POST /api/v1/tts/design/jobs route registration only
#   DEPENDS: M-APPLICATION, M-CONTRACTS, M-ERRORS, M-OBSERVABILITY, M-SERVER
#   LINKS: M-SERVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   register_design_tts_routes - Register voice design sync and async routes on the FastAPI app
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Split voice design TTS route registration out of server/api/routes_tts.py without changing public HTTP behavior]
# END_CHANGE_SUMMARY

from __future__ import annotations

import logging

from fastapi import FastAPI, Header, Request, Response  # pyright: ignore[reportMissingImports]
from fastapi.responses import JSONResponse  # pyright: ignore[reportMissingImports]

from core.contracts.commands import VoiceDesignCommand
from core.observability import log_event, operation_scope
from server.api.policies import enforce_async_submit_admission, enforce_sync_tts_admission
from server.api.responses import apply_async_job_headers, build_audio_response, resolve_save_output
from server.api.tts._helpers import (
    build_job_snapshot_payload,
    build_text_length_error,
    create_design_job_submission,
    ensure_requested_model_capability,
    enforce_text_length,
    public_job_status,
)
from server.api.tts._timeout import run_inference_with_timeout
from server.schemas.audio import DesignTTSRequest, JobSnapshotPayload
from server.schemas.errors import ErrorResponse


# START_CONTRACT: register_design_tts_routes
#   PURPOSE: Register voice design synchronous and asynchronous TTS routes on the FastAPI application.
#   INPUTS: { app: FastAPI - application to attach routes to, logger: Any - structured logger used by endpoint handlers }
#   OUTPUTS: { None - routes are attached in place }
#   SIDE_EFFECTS: Mutates FastAPI routing table by registering voice design TTS endpoints
#   LINKS: M-SERVER, M-APPLICATION
# END_CONTRACT: register_design_tts_routes
def register_design_tts_routes(app: FastAPI, logger) -> None:
    @app.post(
        "/api/v1/tts/design",
        tags=["tts"],
        responses={
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    # START_CONTRACT: tts_design
    #   PURPOSE: Handle synchronous voice design synthesis requests.
    #   INPUTS: { request: Request - incoming HTTP request, payload: DesignTTSRequest - validated voice design payload }
    #   OUTPUTS: { Response - generated audio response or validation error response }
    #   SIDE_EFFECTS: Consumes admission quota, emits endpoint logs, and may trigger synthesis execution
    #   LINKS: M-SERVER, M-APPLICATION
    # END_CONTRACT: tts_design
    async def tts_design(request: Request, payload: DesignTTSRequest) -> Response:
        with operation_scope("server.tts_design"):  # type: ignore[reportGeneralTypeIssues]
            # START_BLOCK_VALIDATE_DESIGN_REQUEST
            await enforce_sync_tts_admission(request)
            resolved_save_output = resolve_save_output(
                payload.save_output, request.app.state.settings.default_save_output
            )
            log_event(
                logger,
                level=logging.INFO,
                event="[RoutesTTS][tts_design][BLOCK_VALIDATE_DESIGN_REQUEST]",
                message="Voice design request received",
                endpoint="/api/v1/tts/design",
                model=payload.model,
                mode="design",
                language=payload.language,
                save_output=resolved_save_output,
            )
            try:
                text = enforce_text_length(
                    value=payload.text,
                    field_name="text",
                    max_chars=request.app.state.settings.max_input_text_chars,
                )
            except ValueError as exc:
                return build_text_length_error(request=request, field_name="text", message=str(exc))
            ensure_requested_model_capability(request, payload.model, execution_mode="design")
            # END_BLOCK_VALIDATE_DESIGN_REQUEST
            # START_BLOCK_EXECUTE_DESIGN_SYNTHESIS
            result = await run_inference_with_timeout(
                request=request,
                operation_name="synthesize_design",
                call=lambda: request.app.state.application.synthesize_design(
                    VoiceDesignCommand(
                        text=text,
                        model=payload.model,
                        save_output=resolved_save_output,
                        language=payload.language,
                        voice_description=payload.voice_description,
                    )
                ),
            )
            # END_BLOCK_EXECUTE_DESIGN_SYNTHESIS
            # START_BLOCK_BUILD_DESIGN_RESPONSE
            return build_audio_response(request, result, "wav", logger)
            # END_BLOCK_BUILD_DESIGN_RESPONSE

    @app.post(
        "/api/v1/tts/design/jobs",
        tags=["tts"],
        response_model=JobSnapshotPayload,
        status_code=202,
        responses={
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    # START_CONTRACT: tts_design_job_submit
    #   PURPOSE: Submit a voice design request for asynchronous execution.
    #   INPUTS: { request: Request - incoming HTTP request, payload: DesignTTSRequest - validated voice design payload, idempotency_key: Optional[str] - optional idempotency key header }
    #   OUTPUTS: { JobSnapshotPayload - submitted or reused async job snapshot }
    #   SIDE_EFFECTS: Consumes admission quota and enqueues or reuses async job execution state
    #   LINKS: M-SERVER, M-APPLICATION
    # END_CONTRACT: tts_design_job_submit
    async def tts_design_job_submit(
        request: Request,
        payload: DesignTTSRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JobSnapshotPayload:
        # START_BLOCK_CHECK_IDEMPOTENCY_DESIGN_JOB
        await enforce_async_submit_admission(request)
        # END_BLOCK_CHECK_IDEMPOTENCY_DESIGN_JOB
        # START_BLOCK_SUBMIT_DESIGN_JOB
        resolution = request.app.state.job_execution.submit_idempotent(
            create_design_job_submission(request, payload, idempotency_key=idempotency_key)
        )
        # END_BLOCK_SUBMIT_DESIGN_JOB
        log_event(
            logger,
            level=logging.INFO,
            event="[RoutesTTS][tts_design_job_submit][BLOCK_SUBMIT_ASYNC_JOB]",
            message="Async design job submission resolved",
            endpoint="/api/v1/tts/design/jobs",
            job_id=resolution.snapshot.job_id,
            submit_request_id=resolution.snapshot.submit_request_id,
            current_request_id=request.state.request_id,
            reused_existing_job=not resolution.created,
            public_status=public_job_status(resolution.snapshot.status),
        )
        # START_BLOCK_BUILD_DESIGN_JOB_RESPONSE
        response = JSONResponse(
            status_code=202,
            content=build_job_snapshot_payload(request, resolution.snapshot).model_dump(
                mode="json"
            ),
        )
        return apply_async_job_headers(response, resolution.snapshot)
        # END_BLOCK_BUILD_DESIGN_JOB_RESPONSE


__all__ = ["register_design_tts_routes"]
