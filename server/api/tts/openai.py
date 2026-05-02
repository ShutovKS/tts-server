# FILE: server/api/tts/openai.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Register OpenAI-compatible synchronous and async TTS HTTP routes.
#   SCOPE: POST /v1/audio/speech and POST /v1/audio/speech/jobs route registration only
#   DEPENDS: M-APPLICATION, M-CONTRACTS, M-ERRORS, M-OBSERVABILITY, M-SERVER
#   LINKS: M-SERVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   register_openai_tts_routes - Register OpenAI-compatible sync and async speech routes on the FastAPI app
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Split OpenAI-compatible TTS route registration out of server/api/routes_tts.py without changing public HTTP behavior]
# END_CHANGE_SUMMARY

from __future__ import annotations

import logging

from fastapi import FastAPI, Header, Request, Response  # pyright: ignore[reportMissingImports]
from fastapi.responses import JSONResponse  # pyright: ignore[reportMissingImports]

from core.contracts.commands import CustomVoiceCommand
from core.observability import log_event, operation_scope
from server.api.policies import enforce_async_submit_admission, enforce_sync_tts_admission
from server.api.responses import apply_async_job_headers, build_audio_response
from server.api.tts._helpers import (
    build_job_snapshot_payload,
    build_text_length_error,
    create_custom_job_submission_from_openai,
    ensure_requested_model_capability,
    enforce_text_length,
    public_job_status,
)
from server.api.tts._timeout import run_inference_with_timeout
from server.schemas.audio import JobSnapshotPayload, OpenAISpeechRequest
from server.schemas.errors import ErrorResponse


# START_CONTRACT: register_openai_tts_routes
#   PURPOSE: Register OpenAI-compatible synchronous and asynchronous speech routes on the FastAPI application.
#   INPUTS: { app: FastAPI - application to attach routes to, logger: Any - structured logger used by endpoint handlers }
#   OUTPUTS: { None - routes are attached in place }
#   SIDE_EFFECTS: Mutates FastAPI routing table by registering OpenAI-compatible TTS endpoints
#   LINKS: M-SERVER, M-APPLICATION
# END_CONTRACT: register_openai_tts_routes
def register_openai_tts_routes(app: FastAPI, logger) -> None:
    @app.post(
        "/v1/audio/speech",
        tags=["tts"],
        responses={
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    # START_CONTRACT: openai_speech
    #   PURPOSE: Handle synchronous OpenAI-compatible speech synthesis requests.
    #   INPUTS: { request: Request - incoming HTTP request, payload: OpenAISpeechRequest - validated speech payload }
    #   OUTPUTS: { Response - generated audio response or validation error response }
    #   SIDE_EFFECTS: Consumes admission quota, emits endpoint logs, and may trigger synthesis execution
    #   LINKS: M-SERVER, M-APPLICATION
    # END_CONTRACT: openai_speech
    async def openai_speech(request: Request, payload: OpenAISpeechRequest) -> Response:
        with operation_scope("server.openai_speech"):  # type: ignore[reportGeneralTypeIssues]
            # START_BLOCK_LOG_OPENAI_REQUEST
            log_event(
                logger,
                level=logging.INFO,
                event="[RoutesTTS][openai_speech][BLOCK_LOG_OPENAI_REQUEST]",
                message="OpenAI-compatible speech request received",
                endpoint="/v1/audio/speech",
                model=payload.model,
                mode="custom",
                language=payload.language,
                response_format=payload.response_format,
            )
            # END_BLOCK_LOG_OPENAI_REQUEST
            # START_BLOCK_VALIDATE_OPENAI_REQUEST
            await enforce_sync_tts_admission(request)
            try:
                input_text = enforce_text_length(
                    value=payload.input,
                    field_name="input",
                    max_chars=request.app.state.settings.max_input_text_chars,
                )
            except ValueError as exc:
                return build_text_length_error(
                    request=request, field_name="input", message=str(exc)
                )
            ensure_requested_model_capability(request, payload.model, execution_mode="custom")
            # END_BLOCK_VALIDATE_OPENAI_REQUEST
            # START_BLOCK_EXECUTE_OPENAI_SYNTHESIS
            result = await run_inference_with_timeout(
                request=request,
                operation_name="synthesize_custom",
                call=lambda: request.app.state.application.synthesize_custom(
                    CustomVoiceCommand(
                        text=input_text,
                        model=payload.model,
                        save_output=request.app.state.settings.default_save_output,
                        language=payload.language,
                        speaker=payload.voice,
                        instruct="Normal tone",
                        speed=payload.speed,
                    )
                ),
            )
            # END_BLOCK_EXECUTE_OPENAI_SYNTHESIS
            # START_BLOCK_BUILD_OPENAI_RESPONSE
            return build_audio_response(request, result, payload.response_format, logger)
            # END_BLOCK_BUILD_OPENAI_RESPONSE

    @app.post(
        "/v1/audio/speech/jobs",
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
    # START_CONTRACT: openai_speech_job_submit
    #   PURPOSE: Submit an OpenAI-compatible speech request for asynchronous execution.
    #   INPUTS: { request: Request - incoming HTTP request, payload: OpenAISpeechRequest - validated speech payload, idempotency_key: Optional[str] - optional idempotency key header }
    #   OUTPUTS: { JobSnapshotPayload - submitted or reused async job snapshot }
    #   SIDE_EFFECTS: Consumes admission quota and enqueues or reuses async job execution state
    #   LINKS: M-SERVER, M-APPLICATION
    # END_CONTRACT: openai_speech_job_submit
    async def openai_speech_job_submit(
        request: Request,
        payload: OpenAISpeechRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JobSnapshotPayload:
        # START_BLOCK_CHECK_IDEMPOTENCY
        await enforce_async_submit_admission(request)
        # END_BLOCK_CHECK_IDEMPOTENCY
        # START_BLOCK_SUBMIT_OPENAI_JOB
        resolution = request.app.state.job_execution.submit_idempotent(
            create_custom_job_submission_from_openai(
                request, payload, idempotency_key=idempotency_key
            )
        )
        # END_BLOCK_SUBMIT_OPENAI_JOB
        log_event(
            logger,
            level=logging.INFO,
            event="[RoutesTTS][openai_speech_job_submit][BLOCK_SUBMIT_ASYNC_JOB]",
            message="Async speech job submission resolved",
            endpoint="/v1/audio/speech/jobs",
            job_id=resolution.snapshot.job_id,
            submit_request_id=resolution.snapshot.submit_request_id,
            current_request_id=request.state.request_id,
            reused_existing_job=not resolution.created,
            public_status=public_job_status(resolution.snapshot.status),
        )
        # START_BLOCK_BUILD_JOB_RESPONSE
        response = JSONResponse(
            status_code=202,
            content=build_job_snapshot_payload(request, resolution.snapshot).model_dump(
                mode="json"
            ),
        )
        return apply_async_job_headers(response, resolution.snapshot)
        # END_BLOCK_BUILD_JOB_RESPONSE


__all__ = ["register_openai_tts_routes"]
