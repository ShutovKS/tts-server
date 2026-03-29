from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Request, Response, UploadFile

from core.contracts.commands import CustomVoiceCommand, VoiceCloneCommand, VoiceDesignCommand
from core.observability import log_event, operation_scope
from server.api.contracts import ErrorDescriptor
from server.api.responses import build_audio_response, build_error_response, resolve_save_output
from server.schemas.audio import CustomTTSRequest, DesignTTSRequest, OpenAISpeechRequest
from server.schemas.errors import ErrorResponse



def register_tts_routes(app: FastAPI, logger) -> None:
    @app.post(
        "/v1/audio/speech",
        tags=["tts"],
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    async def openai_speech(request: Request, payload: OpenAISpeechRequest) -> Response:
        with operation_scope("server.openai_speech"):
            log_event(
                logger,
                level=logging.INFO,
                event="tts.endpoint.started",
                message="OpenAI-compatible speech request received",
                endpoint="/v1/audio/speech",
                model=payload.model,
                mode="custom",
                response_format=payload.response_format,
            )
            result = request.app.state.application.synthesize_custom(
                CustomVoiceCommand(
                    text=payload.input,
                    model=payload.model,
                    save_output=request.app.state.settings.default_save_output,
                    speaker=payload.voice,
                    instruct="Normal tone",
                    speed=payload.speed,
                )
            )
            return build_audio_response(request, result, payload.response_format, logger)

    @app.post(
        "/api/v1/tts/custom",
        tags=["tts"],
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    async def tts_custom(request: Request, payload: CustomTTSRequest) -> Response:
        with operation_scope("server.tts_custom"):
            instruct = payload.instruct or payload.emotion or "Normal tone"
            resolved_save_output = resolve_save_output(payload.save_output, request.app.state.settings.default_save_output)
            log_event(
                logger,
                level=logging.INFO,
                event="tts.endpoint.started",
                message="Custom TTS request received",
                endpoint="/api/v1/tts/custom",
                model=payload.model,
                mode="custom",
                save_output=resolved_save_output,
            )
            result = request.app.state.application.synthesize_custom(
                CustomVoiceCommand(
                    text=payload.text,
                    model=payload.model,
                    save_output=resolved_save_output,
                    speaker=payload.speaker,
                    instruct=instruct,
                    speed=payload.speed,
                )
            )
            return build_audio_response(request, result, "wav", logger)

    @app.post(
        "/api/v1/tts/design",
        tags=["tts"],
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    async def tts_design(request: Request, payload: DesignTTSRequest) -> Response:
        with operation_scope("server.tts_design"):
            resolved_save_output = resolve_save_output(payload.save_output, request.app.state.settings.default_save_output)
            log_event(
                logger,
                level=logging.INFO,
                event="tts.endpoint.started",
                message="Voice design request received",
                endpoint="/api/v1/tts/design",
                model=payload.model,
                mode="design",
                save_output=resolved_save_output,
            )
            result = request.app.state.application.synthesize_design(
                VoiceDesignCommand(
                    text=payload.text,
                    model=payload.model,
                    save_output=resolved_save_output,
                    voice_description=payload.voice_description,
                )
            )
            return build_audio_response(request, result, "wav", logger)

    @app.post(
        "/api/v1/tts/clone",
        tags=["tts"],
        responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    async def tts_clone(
        request: Request,
        text: str = Form(...),
        ref_audio: UploadFile = File(...),
        ref_text: Optional[str] = Form(default=None),
        model: Optional[str] = Form(default=None),
        save_output: Optional[bool] = Form(default=None),
    ) -> Response:
        with operation_scope("server.tts_clone"):
            text = text.strip()
            resolved_save_output = resolve_save_output(save_output, request.app.state.settings.default_save_output)
            log_event(
                logger,
                level=logging.INFO,
                event="tts.endpoint.started",
                message="Voice clone request received",
                endpoint="/api/v1/tts/clone",
                model=model,
                mode="clone",
                save_output=resolved_save_output,
                ref_audio_filename=ref_audio.filename,
                ref_text_provided=bool(ref_text),
            )
            if not text:
                return build_error_response(
                    request=request,
                    descriptor=ErrorDescriptor(
                        status_code=422,
                        code="validation_error",
                        message="Request validation failed",
                        details={"errors": [{"loc": ["body", "text"], "msg": "Text must not be empty", "type": "value_error"}]},
                    ),
                )

            upload_bytes = await ref_audio.read()
            if len(upload_bytes) > request.app.state.settings.max_upload_size_bytes:
                return build_error_response(
                    request=request,
                    descriptor=ErrorDescriptor(
                        status_code=400,
                        code="upload_too_large",
                        message="Uploaded file exceeds configured size limit",
                        details={"max_upload_size_bytes": request.app.state.settings.max_upload_size_bytes},
                    ),
                )

            suffix = Path(ref_audio.filename or "reference.wav").suffix or ".wav"
            temp_path = request.app.state.settings.outputs_dir / f"upload_{uuid.uuid4().hex}{suffix}"
            temp_path.write_bytes(upload_bytes)
            try:
                result = request.app.state.application.synthesize_clone(
                    VoiceCloneCommand(
                        text=text,
                        model=model,
                        save_output=resolved_save_output,
                        ref_audio_path=temp_path,
                        ref_text=ref_text,
                    )
                )
            finally:
                temp_path.unlink(missing_ok=True)
            return build_audio_response(request, result, "wav", logger)
