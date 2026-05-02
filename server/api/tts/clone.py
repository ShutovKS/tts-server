# FILE: server/api/tts/clone.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Register voice clone synchronous and async TTS HTTP routes.
#   SCOPE: POST /api/v1/tts/clone and POST /api/v1/tts/clone/jobs route registration only
#   DEPENDS: M-APPLICATION, M-CONTRACTS, M-ERRORS, M-OBSERVABILITY, M-SERVER
#   LINKS: M-SERVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   register_clone_tts_routes - Register voice clone sync and async routes on the FastAPI app
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Split voice clone TTS route registration out of server/api/routes_tts.py without changing public HTTP behavior]
# END_CHANGE_SUMMARY

from __future__ import annotations

import logging

from fastapi import FastAPI, File, Form, Header, Request, Response, UploadFile  # pyright: ignore[reportMissingImports]
from fastapi.responses import JSONResponse  # pyright: ignore[reportMissingImports]

from core.contracts.commands import VoiceCloneCommand
from core.observability import log_event, operation_scope
from core.infrastructure.audio_io import convert_audio_to_wav_if_needed
from server.api.policies import enforce_async_submit_admission, enforce_sync_tts_admission
from server.api.responses import apply_async_job_headers, build_audio_response, resolve_save_output
from server.api.tts._helpers import (
    build_clone_staged_path,
    build_job_snapshot_payload,
    build_text_length_error,
    ensure_requested_model_capability,
    enforce_text_length,
    public_job_status,
    stage_clone_job_submission,
    validate_clone_upload,
)
from server.api.tts._timeout import run_inference_with_timeout
from server.schemas.audio import JobSnapshotPayload, normalize_language_value
from server.schemas.errors import ErrorResponse


# START_CONTRACT: register_clone_tts_routes
#   PURPOSE: Register voice clone synchronous and asynchronous TTS routes on the FastAPI application.
#   INPUTS: { app: FastAPI - application to attach routes to, logger: Any - structured logger used by endpoint handlers }
#   OUTPUTS: { None - routes are attached in place }
#   SIDE_EFFECTS: Mutates FastAPI routing table by registering voice clone TTS endpoints
#   LINKS: M-SERVER, M-APPLICATION
# END_CONTRACT: register_clone_tts_routes
def register_clone_tts_routes(app: FastAPI, logger) -> None:
    @app.post(
        "/api/v1/tts/clone",
        tags=["tts"],
        responses={
            400: {"model": ErrorResponse},
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    # START_CONTRACT: tts_clone
    #   PURPOSE: Handle synchronous voice clone synthesis requests with uploaded reference audio.
    #   INPUTS: { request: Request - incoming HTTP request, text: str - synthesis text, ref_audio: UploadFile - uploaded reference audio, ref_text: Optional[str] - optional reference transcript, language: Optional[str] - requested language value, model: Optional[str] - optional model override, save_output: Optional[bool] - output persistence override }
    #   OUTPUTS: { Response - generated audio response or validation error response }
    #   SIDE_EFFECTS: Consumes admission quota, reads uploaded bytes, stages temporary files, emits endpoint logs, and may trigger synthesis execution
    #   LINKS: M-SERVER, M-APPLICATION
    # END_CONTRACT: tts_clone
    async def tts_clone(
        request: Request,
        text: str = Form(...),
        ref_audio: UploadFile = File(...),
        ref_text: str | None = Form(default=None),
        language: str | None = Form(default="auto"),
        model: str | None = Form(default=None),
        save_output: bool | None = Form(default=None),
    ) -> Response:
        with operation_scope("server.tts_clone"):  # type: ignore[reportGeneralTypeIssues]
            # START_BLOCK_VALIDATE_CLONE_REQUEST
            await enforce_sync_tts_admission(request)
            text = text.strip()
            normalized_language = normalize_language_value(language or "auto")
            resolved_save_output = resolve_save_output(
                save_output, request.app.state.settings.default_save_output
            )
            if not text:
                return build_text_length_error(
                    request=request, field_name="text", message="Text must not be empty"
                )
            try:
                text = enforce_text_length(
                    value=text,
                    field_name="text",
                    max_chars=request.app.state.settings.max_input_text_chars,
                )
            except ValueError as exc:
                return build_text_length_error(request=request, field_name="text", message=str(exc))
            ensure_requested_model_capability(request, model, execution_mode="clone")
            # END_BLOCK_VALIDATE_CLONE_REQUEST
            # START_BLOCK_LOG_CLONE_REQUEST
            log_event(
                logger,
                level=logging.INFO,
                event="[RoutesTTS][tts_clone][BLOCK_LOG_CLONE_REQUEST]",
                message="Voice clone request received",
                endpoint="/api/v1/tts/clone",
                model=model,
                mode="clone",
                language=normalized_language,
                save_output=resolved_save_output,
                ref_audio_filename=ref_audio.filename,
                ref_text_provided=bool(ref_text),
            )
            # END_BLOCK_LOG_CLONE_REQUEST
            # START_BLOCK_VALIDATE_SYNC_CLONE_UPLOAD
            upload_bytes = await ref_audio.read()
            upload_error = validate_clone_upload(request, ref_audio, upload_bytes)
            if upload_error is not None:
                return upload_error
            # END_BLOCK_VALIDATE_SYNC_CLONE_UPLOAD

            # START_BLOCK_EXECUTE_CLONE_SYNTHESIS
            temp_path = build_clone_staged_path(request, ref_audio, prefix="upload")
            temp_path.write_bytes(upload_bytes)
            normalized_ref_audio = temp_path
            normalized_was_converted = False
            try:
                normalized_ref_audio, normalized_was_converted = convert_audio_to_wav_if_needed(
                    temp_path, request.app.state.settings
                )
                result = await run_inference_with_timeout(
                    request=request,
                    operation_name="synthesize_clone",
                    call=lambda: request.app.state.application.synthesize_clone(
                        VoiceCloneCommand(
                            text=text,
                            model=model,
                            save_output=resolved_save_output,
                            language=normalized_language,
                            ref_audio_path=normalized_ref_audio,
                            ref_text=ref_text,
                        )
                    ),
                )
            finally:
                if normalized_was_converted and normalized_ref_audio.exists():
                    normalized_ref_audio.unlink(missing_ok=True)
                temp_path.unlink(missing_ok=True)
            # END_BLOCK_EXECUTE_CLONE_SYNTHESIS
            # START_BLOCK_BUILD_CLONE_RESPONSE
            return build_audio_response(request, result, "wav", logger)
            # END_BLOCK_BUILD_CLONE_RESPONSE

    @app.post(
        "/api/v1/tts/clone/jobs",
        tags=["tts"],
        response_model=JobSnapshotPayload,
        status_code=202,
        responses={
            400: {"model": ErrorResponse},
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    # START_CONTRACT: tts_clone_job_submit
    #   PURPOSE: Submit a voice clone request with uploaded reference audio for asynchronous execution.
    #   INPUTS: { request: Request - incoming HTTP request, text: str - synthesis text, ref_audio: UploadFile - uploaded reference audio, ref_text: Optional[str] - optional reference transcript, language: Optional[str] - requested language value, model: Optional[str] - optional model override, save_output: Optional[bool] - output persistence override, idempotency_key: Optional[str] - optional idempotency key header }
    #   OUTPUTS: { Response - accepted async job snapshot or validation error response }
    #   SIDE_EFFECTS: Consumes admission quota, reads uploaded bytes, stages files, and enqueues or reuses async job execution state
    #   LINKS: M-SERVER, M-APPLICATION
    # END_CONTRACT: tts_clone_job_submit
    async def tts_clone_job_submit(
        request: Request,
        text: str = Form(...),
        ref_audio: UploadFile = File(...),
        ref_text: str | None = Form(default=None),
        language: str | None = Form(default="auto"),
        model: str | None = Form(default=None),
        save_output: bool | None = Form(default=None),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> Response:
        # START_BLOCK_CHECK_IDEMPOTENCY_CLONE_JOB
        await enforce_async_submit_admission(request)
        # END_BLOCK_CHECK_IDEMPOTENCY_CLONE_JOB
        # START_BLOCK_SUBMIT_CLONE_JOB
        submission, error_response = await stage_clone_job_submission(
            request,
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language or "auto",
            model=model,
            save_output=save_output,
            idempotency_key=idempotency_key,
        )
        if error_response is not None:
            return error_response
        assert submission is not None
        # END_BLOCK_SUBMIT_CLONE_JOB
        # START_BLOCK_PERSIST_CLONE_JOB_INPUTS
        staged_paths = submission.staged_input_paths
        try:
            resolution = request.app.state.job_execution.submit_idempotent(submission)
        except Exception:
            for staged_path in staged_paths:
                staged_path.unlink(missing_ok=True)
            raise
        if not resolution.created:
            for staged_path in staged_paths:
                staged_path.unlink(missing_ok=True)
        # END_BLOCK_PERSIST_CLONE_JOB_INPUTS
        log_event(
            logger,
            level=logging.INFO,
            event="[RoutesTTS][tts_clone_job_submit][BLOCK_SUBMIT_ASYNC_JOB]",
            message="Async clone job submission resolved",
            endpoint="/api/v1/tts/clone/jobs",
            job_id=resolution.snapshot.job_id,
            submit_request_id=resolution.snapshot.submit_request_id,
            current_request_id=request.state.request_id,
            reused_existing_job=not resolution.created,
            public_status=public_job_status(resolution.snapshot.status),
        )
        # START_BLOCK_BUILD_CLONE_JOB_RESPONSE
        response = JSONResponse(
            status_code=202,
            content=build_job_snapshot_payload(request, resolution.snapshot).model_dump(
                mode="json"
            ),
        )
        return apply_async_job_headers(response, resolution.snapshot)
        # END_BLOCK_BUILD_CLONE_JOB_RESPONSE


__all__ = ["register_clone_tts_routes"]
