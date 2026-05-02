# FILE: server/api/tts/stream.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Register streaming custom TTS HTTP routes.
#   SCOPE: POST /api/v1/tts/custom/stream route registration only
#   DEPENDS: M-APPLICATION, M-CONTRACTS, M-ERRORS, M-OBSERVABILITY, M-STREAMING, M-SERVER
#   LINKS: M-SERVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   register_stream_tts_routes - Register custom streaming speech routes on the FastAPI app
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Split custom streaming TTS route registration out of server/api/routes_tts.py without changing public HTTP behavior]
# END_CHANGE_SUMMARY

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, Response  # pyright: ignore[reportMissingImports]
from fastapi.responses import StreamingResponse  # pyright: ignore[reportMissingImports]

from core.contracts.commands import CustomVoiceCommand
from core.observability import log_event, operation_scope
from core.services.streaming import DEFAULT_AUDIO_STREAM_CHUNK_SIZE, stream_generation_result
from server.api.policies import enforce_sync_tts_admission
from server.api.responses import resolve_save_output
from server.api.tts._helpers import (
    build_text_length_error,
    ensure_requested_model_capability,
    enforce_text_length,
)
from server.api.tts._timeout import run_inference_with_timeout
from server.schemas.audio import CustomTTSRequest
from server.schemas.errors import ErrorResponse


# START_CONTRACT: register_stream_tts_routes
#   PURPOSE: Register streaming custom TTS routes on the FastAPI application.
#   INPUTS: { app: FastAPI - application to attach routes to, logger: Any - structured logger used by endpoint handlers }
#   OUTPUTS: { None - routes are attached in place }
#   SIDE_EFFECTS: Mutates FastAPI routing table by registering streaming TTS endpoints
#   LINKS: M-SERVER, M-APPLICATION, M-STREAMING
# END_CONTRACT: register_stream_tts_routes
def register_stream_tts_routes(app: FastAPI, logger) -> None:
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


__all__ = ["register_stream_tts_routes"]
