# FILE: server/api/responses.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Define HTTP response formatting utilities.
#   SCOPE: Audio response builders, error response builders, async job correlation header helpers
#   DEPENDS: M-CONTRACTS
#   LINKS: M-SERVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   resolve_save_output - Resolve request-level output persistence override against server defaults
#   build_error_response - Serialize API error descriptors into JSON HTTP responses
#   build_audio_response - Build HTTP audio responses and emit response-ready observability events
#   apply_async_job_headers - Attach stable async job correlation headers to a response
#   public_artifact_name - Reduce internal artifact paths to public-safe filenames
#   wav_to_pcm_bytes - Extract raw PCM bytes from WAV payloads
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Added async job correlation header helper so submit, status, cancel, and result responses expose stable request and job identifiers]
# END_CHANGE_SUMMARY

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from core.contracts.jobs import JobSnapshot
from core.contracts.results import GenerationResult
from core.observability import log_event
from server.api.contracts import ErrorDescriptor
from server.schemas.errors import ErrorResponse


# START_CONTRACT: resolve_save_output
#   PURPOSE: Resolve whether output persistence is enabled for a request using explicit override or default.
#   INPUTS: { save_output: Optional[bool] - request-level override, default_save_output: bool - server default value }
#   OUTPUTS: { bool - resolved output persistence flag }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: resolve_save_output
def resolve_save_output(save_output: Optional[bool], default_save_output: bool) -> bool:
    return default_save_output if save_output is None else save_output


# START_CONTRACT: build_error_response
#   PURPOSE: Build a structured JSON HTTP response from a public error descriptor.
#   INPUTS: { request: Request - request carrying correlation state, descriptor: ErrorDescriptor - public error response descriptor }
#   OUTPUTS: { JSONResponse - serialized API error response }
#   SIDE_EFFECTS: Reads request state and sets HTTP response headers from descriptor metadata
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: build_error_response
def build_error_response(
    *, request: Request, descriptor: ErrorDescriptor
) -> JSONResponse:
    payload = ErrorResponse(
        code=descriptor.code,
        message=descriptor.message,
        details=descriptor.details,
        request_id=getattr(request.state, "request_id", "unknown"),
    )
    response = JSONResponse(
        status_code=descriptor.status_code, content=payload.model_dump()
    )
    for header_name, header_value in (descriptor.headers or {}).items():
        response.headers[header_name] = header_value
    return response


# START_CONTRACT: build_audio_response
#   PURPOSE: Build an HTTP audio response from a generation result and requested response format.
#   INPUTS: { request: Request - request carrying correlation state, result: GenerationResult - completed generation output, response_format: str - requested wire format, logger: Any - structured logger for response events }
#   OUTPUTS: { Response - HTTP response containing generated audio bytes }
#   SIDE_EFFECTS: May transcode WAV to PCM, emits response logs, and sets response headers
#   LINKS: M-SERVER, M-OBSERVABILITY
# END_CONTRACT: build_audio_response
def build_audio_response(
    request: Request, result: GenerationResult, response_format: str, logger
) -> Response:
    request_id = getattr(request.state, "request_id", "unknown")
    audio_bytes = result.audio.bytes_data
    media_type = result.audio.media_type

    if response_format == "pcm":
        audio_bytes = wav_to_pcm_bytes(audio_bytes)
        media_type = "audio/pcm"

    headers = {
        "x-request-id": request_id,
        "x-model-id": result.model,
        "x-tts-mode": result.mode,
        "x-backend-id": result.backend,
    }
    if result.saved_path:
        headers["x-saved-output-file"] = public_artifact_name(result.saved_path)

    log_event(
        logger,
        level=20,
        event="[Responses][build_audio_response][BUILD_AUDIO_RESPONSE]",
        message="Audio response is ready",
        model=result.model,
        mode=result.mode,
        backend=result.backend,
        response_format=response_format,
        media_type=media_type,
        saved_path=str(result.saved_path) if result.saved_path else None,
    )

    return Response(content=audio_bytes, media_type=media_type, headers=headers)


# START_CONTRACT: apply_async_job_headers
#   PURPOSE: Attach stable async job correlation headers to a response.
#   INPUTS: { response: Response - HTTP response to mutate, snapshot: JobSnapshot - internal async job snapshot carrying correlation identifiers }
#   OUTPUTS: { Response - the same response with async correlation headers attached }
#   SIDE_EFFECTS: Mutates response headers in place
#   LINKS: M-SERVER
# END_CONTRACT: apply_async_job_headers
def apply_async_job_headers(response: Response, snapshot: JobSnapshot) -> Response:
    response.headers["x-job-id"] = snapshot.job_id
    response.headers["x-submit-request-id"] = snapshot.submit_request_id
    return response


# START_CONTRACT: public_artifact_name
#   PURPOSE: Convert an internal artifact path into a public-safe filename.
#   INPUTS: { path: str | Path - artifact path or filename }
#   OUTPUTS: { str - basename safe to expose in API responses }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: public_artifact_name
def public_artifact_name(path: str | Path) -> str:
    return Path(path).name


# START_CONTRACT: wav_to_pcm_bytes
#   PURPOSE: Extract raw PCM frame bytes from a WAV audio payload.
#   INPUTS: { wav_bytes: bytes - WAV container bytes }
#   OUTPUTS: { bytes - PCM frame data without WAV headers }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: wav_to_pcm_bytes
def wav_to_pcm_bytes(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        return wav_file.readframes(wav_file.getnframes())

__all__ = [
    "resolve_save_output",
    "build_error_response",
    "build_audio_response",
    "apply_async_job_headers",
    "public_artifact_name",
    "wav_to_pcm_bytes",
]
