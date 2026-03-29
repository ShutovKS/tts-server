from __future__ import annotations

import io
import wave
from typing import Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from core.contracts.results import GenerationResult
from core.observability import log_event
from server.api.contracts import ErrorDescriptor
from server.schemas.errors import ErrorResponse



def resolve_save_output(save_output: Optional[bool], default_save_output: bool) -> bool:
    return default_save_output if save_output is None else save_output



def build_error_response(*, request: Request, descriptor: ErrorDescriptor) -> JSONResponse:
    payload = ErrorResponse(
        code=descriptor.code,
        message=descriptor.message,
        details=descriptor.details,
        request_id=getattr(request.state, "request_id", "unknown"),
    )
    return JSONResponse(status_code=descriptor.status_code, content=payload.model_dump())



def build_audio_response(request: Request, result: GenerationResult, response_format: str, logger) -> Response:
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
        headers["x-saved-output-path"] = str(result.saved_path)

    log_event(
        logger,
        level=20,
        event="http.audio_response.ready",
        message="Audio response is ready",
        model=result.model,
        mode=result.mode,
        backend=result.backend,
        response_format=response_format,
        media_type=media_type,
        saved_path=str(result.saved_path) if result.saved_path else None,
    )

    if request.app.state.settings.enable_streaming:
        return StreamingResponse(io.BytesIO(audio_bytes), media_type=media_type, headers=headers)
    return Response(content=audio_bytes, media_type=media_type, headers=headers)



def wav_to_pcm_bytes(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        return wav_file.readframes(wav_file.getnframes())
