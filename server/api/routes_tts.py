# FILE: server/api/routes_tts.py
# VERSION: 1.2.0
# START_MODULE_CONTRACT
#   PURPOSE: Define synchronous and async TTS HTTP endpoints.
#   SCOPE: POST /v1/audio/speech, POST /api/v1/tts/custom|design|clone, POST /api/v1/tts/custom/stream, async job endpoints
#   DEPENDS: M-APPLICATION, M-CONTRACTS, M-ERRORS, M-OBSERVABILITY, M-STREAMING
#   LINKS: M-SERVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   T - Generic type variable used by async timeout helper utilities
#   build_text_length_error - Build validation errors for invalid text lengths
#   build_upload_validation_error - Build validation errors for invalid clone uploads
#   enforce_text_length - Validate text against configured length limits
#   current_principal_id - Read current principal id from request state
#   resolve_idempotency_scope - Resolve idempotency scope for async submissions
#   ensure_requested_model_capability - Validate an explicitly requested model against the requested normalized synthesis capability
#   build_job_urls - Build status, result, and cancel URLs for async jobs
#   public_job_status - Convert internal job statuses into the frozen public async lifecycle
#   build_job_snapshot_payload - Convert internal job snapshots to public payloads
#   get_job_snapshot_or_raise - Load a job snapshot and enforce owner access
#   build_idempotency_fingerprint - Build deterministic async job idempotency fingerprints
#   create_custom_job_submission_from_openai - Build async custom submissions from OpenAI payloads
#   create_custom_job_submission_from_custom - Build async custom submissions from custom payloads
#   create_design_job_submission - Build async voice design submissions
#   validate_clone_upload - Validate clone upload metadata and content type
#   build_clone_staged_path - Build a staging path for uploaded clone audio
#   stage_clone_job_submission - Persist uploaded clone media and build async clone submissions
#   run_inference_with_timeout - Run synthesis with timeout handling for sync routes
#   register_tts_routes - Register synchronous and async TTS routes on the FastAPI app
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.2.0 - Phase 4.12: registered POST /api/v1/tts/custom/stream that returns the WAV bytes of a completed custom synthesis result as a chunked StreamingResponse using core.services.streaming.stream_generation_result, with the chunk count surfaced via the x-tts-stream-chunks header]
# END_CHANGE_SUMMARY

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, File, Form, Header, Request, Response, UploadFile  # pyright: ignore[reportMissingImports]
from fastapi.responses import JSONResponse, StreamingResponse  # pyright: ignore[reportMissingImports]

from core.contracts.commands import CustomVoiceCommand, VoiceCloneCommand, VoiceDesignCommand
from core.contracts.jobs import JobOperation, JobStatus
from core.errors import JobNotCancellableError, JobNotFoundError, JobNotReadyError, JobNotSucceededError
from core.infrastructure.audio_io import convert_audio_to_wav_if_needed
from core.observability import log_event, operation_scope
from core.services.streaming import DEFAULT_AUDIO_STREAM_CHUNK_SIZE, stream_generation_result
from server.api.auth import ensure_job_owner_access
from server.api.policies import (
    enforce_async_submit_admission,
    enforce_job_cancel_admission,
    enforce_job_read_admission,
    enforce_sync_tts_admission,
)
from server.api.responses import apply_async_job_headers, build_audio_response, resolve_save_output
from server.api.tts._helpers import (  # pyright: ignore[reportMissingImports]
    build_idempotency_fingerprint,
    build_clone_staged_path,
    build_job_snapshot_payload,
    build_text_length_error,
    create_custom_job_submission_from_custom,
    create_custom_job_submission_from_openai,
    create_design_job_submission,
    ensure_requested_model_capability,
    get_job_snapshot_or_raise,
    public_job_status,
    resolve_idempotency_scope,
    stage_clone_job_submission,
    validate_clone_upload,
    enforce_text_length,
)
from server.api.tts._timeout import run_inference_with_timeout  # pyright: ignore[reportMissingImports]
from server.schemas.audio import (
    CustomTTSRequest,
    DesignTTSRequest,
    JobSnapshotPayload,
    OpenAISpeechRequest,
    normalize_language_value,
)
from server.schemas.errors import ErrorResponse

# START_CONTRACT: register_tts_routes
#   PURPOSE: Register synchronous and asynchronous TTS HTTP endpoints on the FastAPI application.
#   INPUTS: { app: FastAPI - application to attach routes to, logger: Any - structured logger used by endpoint handlers }
#   OUTPUTS: { None - routes are attached in place }
#   SIDE_EFFECTS: Mutates FastAPI routing table by registering TTS endpoints
#   LINKS: M-SERVER, M-APPLICATION
# END_CONTRACT: register_tts_routes
def register_tts_routes(app: FastAPI, logger) -> None:
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
        "/api/v1/tts/custom/stream",
        tags=["tts"],
        responses={
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    # START_CONTRACT: tts_custom_stream
    #   PURPOSE: Stream the WAV bytes of a custom voice synthesis result to the client over a chunked HTTP response.
    #   INPUTS: { request: Request - incoming HTTP request, payload: CustomTTSRequest - validated custom synthesis payload }
    #   OUTPUTS: { StreamingResponse - chunked WAV audio stream with x-request-id, x-model-id, x-tts-mode, x-backend-id, and x-tts-stream-chunks headers }
    #   SIDE_EFFECTS: Consumes admission quota, emits endpoint logs, triggers synthesis execution, and writes chunked HTTP frames
    #   LINKS: M-SERVER, M-APPLICATION, M-STREAMING
    # END_CONTRACT: tts_custom_stream
    async def tts_custom_stream(request: Request, payload: CustomTTSRequest) -> Response:
        with operation_scope("server.tts_custom_stream"):  # type: ignore[reportGeneralTypeIssues]
            # START_BLOCK_PREPARE_CUSTOM_STREAM_REQUEST
            instruct = payload.instruct or payload.emotion or "Normal tone"
            resolved_save_output = resolve_save_output(
                payload.save_output, request.app.state.settings.default_save_output
            )
            log_event(
                logger,
                level=logging.INFO,
                event="[RoutesTTS][tts_custom_stream][BLOCK_PREPARE_CUSTOM_STREAM_REQUEST]",
                message="Custom TTS streaming request received",
                endpoint="/api/v1/tts/custom/stream",
                model=payload.model,
                mode="custom",
                language=payload.language,
                save_output=resolved_save_output,
            )
            # END_BLOCK_PREPARE_CUSTOM_STREAM_REQUEST
            # START_BLOCK_VALIDATE_CUSTOM_STREAM_REQUEST
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
            # END_BLOCK_VALIDATE_CUSTOM_STREAM_REQUEST
            # START_BLOCK_EXECUTE_CUSTOM_STREAM_SYNTHESIS
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
            # END_BLOCK_EXECUTE_CUSTOM_STREAM_SYNTHESIS
            # START_BLOCK_BUILD_CUSTOM_STREAM_RESPONSE
            chunks = list(
                stream_generation_result(result, chunk_size=DEFAULT_AUDIO_STREAM_CHUNK_SIZE)
            )
            request_id = getattr(request.state, "request_id", "unknown")
            headers = {
                "x-request-id": request_id,
                "x-model-id": result.model,
                "x-tts-mode": result.mode,
                "x-backend-id": result.backend,
                "x-tts-stream-chunks": str(len(chunks)),
            }

            def _iter_chunks():
                for chunk in chunks:
                    yield chunk.data

            log_event(
                logger,
                level=logging.INFO,
                event="[RoutesTTS][tts_custom_stream][BLOCK_BUILD_CUSTOM_STREAM_RESPONSE]",
                message="Custom TTS streaming response is ready",
                endpoint="/api/v1/tts/custom/stream",
                model=result.model,
                mode=result.mode,
                backend=result.backend,
                stream_chunks=len(chunks),
                media_type=result.audio.media_type,
            )
            return StreamingResponse(
                _iter_chunks(),
                media_type=result.audio.media_type,
                headers=headers,
            )
            # END_BLOCK_BUILD_CUSTOM_STREAM_RESPONSE

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
