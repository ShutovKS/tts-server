# FILE: server/api/tts/custom.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Register custom synchronous and async TTS HTTP routes.
#   SCOPE: POST /api/v1/tts/custom and POST /api/v1/tts/custom/jobs route registration only
#   DEPENDS: M-APPLICATION, M-CONTRACTS, M-ERRORS, M-OBSERVABILITY, M-SERVER
#   LINKS: M-SERVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   register_custom_tts_routes - Register custom sync and async speech routes on the FastAPI app
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Split custom TTS route registration out of server/api/routes_tts.py without changing public HTTP behavior]
# END_CHANGE_SUMMARY

from __future__ import annotations

import logging

from fastapi import FastAPI, Header, Request, Response  # pyright: ignore[reportMissingImports]
from fastapi.responses import JSONResponse  # pyright: ignore[reportMissingImports]

from core.contracts.commands import CustomVoiceCommand
from core.observability import log_event, operation_scope
from server.api.policies import enforce_async_submit_admission, enforce_sync_tts_admission
from server.api.responses import apply_async_job_headers, build_audio_response, resolve_save_output
from server.api.tts._helpers import (
    build_job_snapshot_payload,
    build_text_length_error,
    create_custom_job_submission_from_custom,
    ensure_requested_model_capability,
    enforce_text_length,
    public_job_status,
)
from server.api.tts._timeout import run_inference_with_timeout
from server.schemas.audio import CustomTTSRequest, JobSnapshotPayload
from server.schemas.errors import ErrorResponse


# START_CONTRACT: register_custom_tts_routes
#   PURPOSE: Register custom synchronous and asynchronous TTS routes on the FastAPI application.
#   INPUTS: { app: FastAPI - application to attach routes to, logger: Any - structured logger used by endpoint handlers }
#   OUTPUTS: { None - routes are attached in place }
#   SIDE_EFFECTS: Mutates FastAPI routing table by registering custom TTS endpoints
#   LINKS: M-SERVER, M-APPLICATION
# END_CONTRACT: register_custom_tts_routes
def register_custom_tts_routes(app: FastAPI, logger) -> None:
    @app.post(
        "/api/v1/tts/custom",
        tags=["tts"],
        responses={
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    # START_CONTRACT: tts_custom
    #   PURPOSE: Handle synchronous custom voice synthesis requests.
    #   INPUTS: { request: Request - incoming HTTP request, payload: CustomTTSRequest - validated custom synthesis payload }
    #   OUTPUTS: { Response - generated audio response or validation error response }
    #   SIDE_EFFECTS: Consumes admission quota, emits endpoint logs, and may trigger synthesis execution
    #   LINKS: M-SERVER, M-APPLICATION
    # END_CONTRACT: tts_custom
    async def tts_custom(request: Request, payload: CustomTTSRequest) -> Response:
        with operation_scope("server.tts_custom"):  # type: ignore[reportGeneralTypeIssues]
            # START_BLOCK_PREPARE_CUSTOM_REQUEST
            instruct = payload.instruct or payload.emotion or "Normal tone"
            resolved_save_output = resolve_save_output(
                payload.save_output, request.app.state.settings.default_save_output
            )
            log_event(
                logger,
                level=logging.INFO,
                event="[RoutesTTS][tts_custom][BLOCK_PREPARE_CUSTOM_REQUEST]",
                message="Custom TTS request received",
                endpoint="/api/v1/tts/custom",
                model=payload.model,
                mode="custom",
                language=payload.language,
                save_output=resolved_save_output,
            )
            # END_BLOCK_PREPARE_CUSTOM_REQUEST
            # START_BLOCK_VALIDATE_CUSTOM_REQUEST
            await enforce_sync_tts_admission(request)
            try:
                text = enforce_text_length(
                    value=payload.text,
                    field_name="text",
                    max_chars=request.app.state.settings.max_input_text_chars,
                )
            except ValueError as exc:
                return build_text_length_error(request=request, field_name="text", message=str(exc))
            ensure_requested_model_capability(request, payload.model, execution_mode="custom")
            # END_BLOCK_VALIDATE_CUSTOM_REQUEST
            # START_BLOCK_EXECUTE_CUSTOM_SYNTHESIS
            result = await run_inference_with_timeout(
                request=request,
                operation_name="synthesize_custom",
                call=lambda: request.app.state.application.synthesize_custom(
                    CustomVoiceCommand(
                        text=text,
                        model=payload.model,
                        save_output=resolved_save_output,
                        language=payload.language,
                        speaker=payload.speaker,
                        instruct=instruct,
                        speed=payload.speed,
                    )
                ),
            )
            # END_BLOCK_EXECUTE_CUSTOM_SYNTHESIS
            # START_BLOCK_BUILD_CUSTOM_RESPONSE
            return build_audio_response(request, result, "wav", logger)
            # END_BLOCK_BUILD_CUSTOM_RESPONSE

    @app.post(
        "/api/v1/tts/custom/jobs",
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
    # START_CONTRACT: tts_custom_job_submit
    #   PURPOSE: Submit a custom TTS request for asynchronous execution.
    #   INPUTS: { request: Request - incoming HTTP request, payload: CustomTTSRequest - validated custom synthesis payload, idempotency_key: Optional[str] - optional idempotency key header }
    #   OUTPUTS: { JobSnapshotPayload - submitted or reused async job snapshot }
    #   SIDE_EFFECTS: Consumes admission quota and enqueues or reuses async job execution state
    #   LINKS: M-SERVER, M-APPLICATION
    # END_CONTRACT: tts_custom_job_submit
    async def tts_custom_job_submit(
        request: Request,
        payload: CustomTTSRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JobSnapshotPayload:
        # START_BLOCK_CHECK_IDEMPOTENCY_CUSTOM_JOB
        await enforce_async_submit_admission(request)
        # END_BLOCK_CHECK_IDEMPOTENCY_CUSTOM_JOB
        # START_BLOCK_SUBMIT_CUSTOM_JOB
        resolution = request.app.state.job_execution.submit_idempotent(
            create_custom_job_submission_from_custom(
                request, payload, idempotency_key=idempotency_key
            )
        )
        # END_BLOCK_SUBMIT_CUSTOM_JOB
        log_event(
            logger,
            level=logging.INFO,
            event="[RoutesTTS][tts_custom_job_submit][BLOCK_SUBMIT_ASYNC_JOB]",
            message="Async custom job submission resolved",
            endpoint="/api/v1/tts/custom/jobs",
            job_id=resolution.snapshot.job_id,
            submit_request_id=resolution.snapshot.submit_request_id,
            current_request_id=request.state.request_id,
            reused_existing_job=not resolution.created,
            public_status=public_job_status(resolution.snapshot.status),
        )
        # START_BLOCK_BUILD_CUSTOM_JOB_RESPONSE
        response = JSONResponse(
            status_code=202,
            content=build_job_snapshot_payload(request, resolution.snapshot).model_dump(
                mode="json"
            ),
        )
        return apply_async_job_headers(response, resolution.snapshot)
        # END_BLOCK_BUILD_CUSTOM_JOB_RESPONSE


__all__ = ["register_custom_tts_routes"]
