# FILE: server/api/routes_tts.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define synchronous and async TTS HTTP endpoints.
#   SCOPE: POST /v1/audio/speech, POST /api/v1/tts/custom|design|clone, async job endpoints
#   DEPENDS: M-APPLICATION, M-CONTRACTS, M-ERRORS, M-OBSERVABILITY
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
#   LAST_CHANGE: [v1.0.1 - Added controlled model-capability validation before sync and async family-specific execution paths]
# END_CHANGE_SUMMARY

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Callable, Optional, TypeVar

from starlette.datastructures import UploadFile as StarletteUploadFile

from fastapi import FastAPI, File, Form, Header, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from core.contracts.commands import (
    CustomVoiceCommand,
    VoiceCloneCommand,
    VoiceDesignCommand,
)
from core.contracts.synthesis import legacy_mode_to_capability
from core.contracts.jobs import (
    JobOperation,
    JobSnapshot,
    JobStatus,
    create_job_submission,
)
from core.errors import (
    JobNotCancellableError,
    JobNotFoundError,
    JobNotReadyError,
    JobNotSucceededError,
    ModelCapabilityError,
    RequestTimeoutError,
)
from core.observability import Timer, log_event, operation_scope
from server.api.auth import ensure_job_owner_access
from server.api.contracts import ErrorDescriptor
from server.api.policies import (
    enforce_async_submit_admission,
    enforce_job_cancel_admission,
    enforce_job_read_admission,
    enforce_sync_tts_admission,
)
from server.api.responses import (
    build_audio_response,
    build_error_response,
    public_artifact_name,
    resolve_save_output,
)
from server.schemas.audio import (
    CustomTTSRequest,
    DesignTTSRequest,
    JobFailurePayload,
    JobSnapshotPayload,
    OpenAISpeechRequest,
    normalize_language_value,
    validate_text_length,
)
from server.schemas.errors import ErrorResponse


T = TypeVar("T")


_ALLOWED_CLONE_UPLOAD_CONTENT_TYPES = frozenset(
    {
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/vnd.wave",
        "audio/mpeg",
        "audio/mp3",
        "audio/flac",
        "audio/x-flac",
        "audio/ogg",
        "audio/webm",
        "audio/mp4",
        "audio/x-m4a",
        "video/webm",
        "application/octet-stream",
    }
)
_ALLOWED_CLONE_UPLOAD_SUFFIXES = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a", ".mp4"}
)


# START_CONTRACT: build_text_length_error
#   PURPOSE: Build a standardized validation error response for oversized or empty text fields.
#   INPUTS: { request: Request - request carrying correlation state, field_name: str - field that failed validation, message: str - validation message }
#   OUTPUTS: { JSONResponse - standardized validation error response }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: build_text_length_error
def build_text_length_error(
    *, request: Request, field_name: str, message: str
) -> JSONResponse:
    return build_error_response(
        request=request,
        descriptor=ErrorDescriptor(
            status_code=422,
            code="validation_error",
            message="Request validation failed",
            details={
                "errors": [
                    {"loc": ["body", field_name], "msg": message, "type": "value_error"}
                ]
            },
        ),
    )


# START_CONTRACT: build_upload_validation_error
#   PURPOSE: Build a standardized error response for invalid clone upload inputs.
#   INPUTS: { request: Request - request carrying correlation state, code: str - machine-readable error code, message: str - human-readable summary, details: dict[str, object] - structured validation details }
#   OUTPUTS: { JSONResponse - standardized upload validation error response }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: build_upload_validation_error
def build_upload_validation_error(
    *, request: Request, code: str, message: str, details: dict[str, object]
) -> JSONResponse:
    return build_error_response(
        request=request,
        descriptor=ErrorDescriptor(
            status_code=400,
            code=code,
            message=message,
            details=details,
        ),
    )


# START_CONTRACT: enforce_text_length
#   PURPOSE: Validate a text field against the configured character limit.
#   INPUTS: { value: str - text to validate, field_name: str - field label used in errors, max_chars: int - maximum allowed characters }
#   OUTPUTS: { str - validated text value }
#   SIDE_EFFECTS: Raises ValueError when text exceeds the configured limit
#   LINKS: M-SERVER
# END_CONTRACT: enforce_text_length
def enforce_text_length(*, value: str, field_name: str, max_chars: int) -> str:
    return validate_text_length(value, field_name=field_name, max_chars=max_chars)


# START_CONTRACT: current_principal_id
#   PURPOSE: Return the current request principal identifier from request state.
#   INPUTS: { request: Request - request containing resolved principal state }
#   OUTPUTS: { str - current principal identifier }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: current_principal_id
def current_principal_id(request: Request) -> str:
    return request.state.principal.principal_id


# START_CONTRACT: resolve_idempotency_scope
#   PURPOSE: Resolve the idempotency scope key used for async job submissions.
#   INPUTS: { request: Request - request containing resolved principal state }
#   OUTPUTS: { str - idempotency scope identifier }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: resolve_idempotency_scope
def resolve_idempotency_scope(request: Request) -> str:
    return current_principal_id(request)


# START_CONTRACT: ensure_requested_model_capability
#   PURPOSE: Validate that an explicitly requested model supports the requested synthesis capability.
#   INPUTS: { request: Request - request carrying the active model registry, model_name: Optional[str] - explicit model identifier to validate, legacy_mode: str - compatibility mode that maps to a normalized capability }
#   OUTPUTS: { None - completes when the model supports the capability }
#   SIDE_EFFECTS: May resolve model metadata through the registry and raise model capability errors
#   LINKS: M-SERVER, M-MODEL-REGISTRY, M-ERRORS
# END_CONTRACT: ensure_requested_model_capability
def ensure_requested_model_capability(
    request: Request, model_name: Optional[str], *, legacy_mode: str
) -> None:
    if not model_name:
        return
    registry = request.app.state.registry
    if not hasattr(registry, "get_model_spec"):
        return
    spec = registry.get_model_spec(model_name=model_name)
    capability = legacy_mode_to_capability(legacy_mode)
    if capability in spec.supported_capabilities:
        return
    raise ModelCapabilityError(
        model_id=spec.model_id,
        capability=capability,
        supported_capabilities=spec.supported_capabilities,
        family=spec.family,
    )


# START_CONTRACT: build_job_urls
#   PURPOSE: Build status, result, and cancel URLs for an async TTS job.
#   INPUTS: { request: Request - request used for route URL generation, job_id: str - async job identifier }
#   OUTPUTS: { tuple[str, str, str] - status, result, and cancel endpoint URLs }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: build_job_urls
def build_job_urls(request: Request, job_id: str) -> tuple[str, str, str]:
    return (
        str(request.url_for("tts_job_status", job_id=job_id)),
        str(request.url_for("tts_job_result", job_id=job_id)),
        str(request.url_for("tts_job_cancel", job_id=job_id)),
    )


# START_CONTRACT: build_job_snapshot_payload
#   PURPOSE: Convert an internal job snapshot into the public async job response payload.
#   INPUTS: { request: Request - request used for URL and request id resolution, snapshot: JobSnapshot - internal job snapshot }
#   OUTPUTS: { JobSnapshotPayload - public async job snapshot payload }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER, M-CONTRACTS
# END_CONTRACT: build_job_snapshot_payload
def build_job_snapshot_payload(
    request: Request, snapshot: JobSnapshot
) -> JobSnapshotPayload:
    status_url, result_url, cancel_url = build_job_urls(request, snapshot.job_id)
    terminal_error = snapshot.terminal_error
    return JobSnapshotPayload(
        request_id=getattr(request.state, "request_id", "unknown"),
        job_id=snapshot.job_id,
        submit_request_id=snapshot.submit_request_id,
        status=snapshot.status.value,
        operation=snapshot.operation.value,
        mode=snapshot.mode,
        model=snapshot.requested_model,
        backend=snapshot.backend,
        response_format=snapshot.response_format,
        save_output=snapshot.save_output,
        created_at=snapshot.created_at,
        started_at=snapshot.started_at,
        completed_at=snapshot.completed_at,
        saved_path=public_artifact_name(snapshot.saved_path)
        if snapshot.saved_path is not None
        else None,
        terminal_error=(
            JobFailurePayload(
                code=terminal_error.code,
                message=terminal_error.message,
                details=terminal_error.details,
            )
            if terminal_error is not None
            else None
        ),
        status_url=status_url,
        result_url=result_url,
        cancel_url=cancel_url,
        idempotency_key=snapshot.idempotency_key,
    )


# START_CONTRACT: get_job_snapshot_or_raise
#   PURPOSE: Load a job snapshot from execution state and enforce owner access.
#   INPUTS: { request: Request - request carrying job execution state and principal, job_id: str - async job identifier }
#   OUTPUTS: { JobSnapshot - loaded job snapshot owned by the current principal }
#   SIDE_EFFECTS: Raises job-not-found or forbidden errors when access fails
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: get_job_snapshot_or_raise
def get_job_snapshot_or_raise(request: Request, job_id: str) -> JobSnapshot:
    snapshot = request.app.state.job_execution.get_job(job_id)
    if snapshot is None:
        raise JobNotFoundError(job_id)
    ensure_job_owner_access(request, owner_principal_id=snapshot.owner_principal_id)
    return snapshot


# START_CONTRACT: build_idempotency_fingerprint
#   PURPOSE: Build a deterministic payload fingerprint for async job idempotency handling.
#   INPUTS: { operation: JobOperation - async job operation type, payload: dict[str, object] - normalized submission payload }
#   OUTPUTS: { str - SHA-256 fingerprint for idempotency comparison }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: build_idempotency_fingerprint
def build_idempotency_fingerprint(
    *, operation: JobOperation, payload: dict[str, object]
) -> str:
    normalized_payload = json.dumps(
        {
            "operation": operation.value,
            "payload": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(normalized_payload.encode("utf-8")).hexdigest()


# START_CONTRACT: create_custom_job_submission_from_openai
#   PURPOSE: Convert an OpenAI-compatible speech payload into an async custom synthesis job submission.
#   INPUTS: { request: Request - request carrying app state and principal, payload: OpenAISpeechRequest - validated OpenAI speech payload, idempotency_key: Optional[str] - optional client idempotency key }
#   OUTPUTS: { Any - job submission object for async execution }
#   SIDE_EFFECTS: Reads app settings and may raise validation errors for oversized text
#   LINKS: M-SERVER, M-CONTRACTS
# END_CONTRACT: create_custom_job_submission_from_openai
def create_custom_job_submission_from_openai(
    request: Request,
    payload: OpenAISpeechRequest,
    *,
    idempotency_key: Optional[str] = None,
):
    ensure_requested_model_capability(request, payload.model, legacy_mode="custom")
    input_text = enforce_text_length(
        value=payload.input,
        field_name="input",
        max_chars=request.app.state.settings.max_input_text_chars,
    )
    save_output = request.app.state.settings.default_save_output
    return create_job_submission(
        operation=JobOperation.SYNTHESIZE_CUSTOM,
        command=CustomVoiceCommand(
            text=input_text,
            model=payload.model,
            save_output=save_output,
            language=payload.language,
            speaker=payload.voice,
            instruct="Normal tone",
            speed=payload.speed,
        ),
        submit_request_id=request.state.request_id,
        owner_principal_id=current_principal_id(request),
        response_format=payload.response_format,
        save_output=save_output,
        execution_timeout_seconds=request.app.state.settings.request_timeout_seconds,
        idempotency_key=idempotency_key,
        idempotency_scope=resolve_idempotency_scope(request)
        if idempotency_key is not None
        else None,
        idempotency_fingerprint=(
            build_idempotency_fingerprint(
                operation=JobOperation.SYNTHESIZE_CUSTOM,
                payload={
                    "model": payload.model,
                    "input": input_text,
                    "voice": payload.voice,
                    "language": payload.language,
                    "response_format": payload.response_format,
                    "speed": payload.speed,
                    "save_output": save_output,
                },
            )
            if idempotency_key is not None
            else None
        ),
    )


# START_CONTRACT: create_custom_job_submission_from_custom
#   PURPOSE: Convert a custom TTS payload into an async custom synthesis job submission.
#   INPUTS: { request: Request - request carrying app state and principal, payload: CustomTTSRequest - validated custom TTS payload, idempotency_key: Optional[str] - optional client idempotency key }
#   OUTPUTS: { Any - job submission object for async execution }
#   SIDE_EFFECTS: Reads app settings and may raise validation errors for oversized text
#   LINKS: M-SERVER, M-CONTRACTS
# END_CONTRACT: create_custom_job_submission_from_custom
def create_custom_job_submission_from_custom(
    request: Request,
    payload: CustomTTSRequest,
    *,
    idempotency_key: Optional[str] = None,
):
    ensure_requested_model_capability(request, payload.model, legacy_mode="custom")
    text = enforce_text_length(
        value=payload.text,
        field_name="text",
        max_chars=request.app.state.settings.max_input_text_chars,
    )
    save_output = resolve_save_output(
        payload.save_output, request.app.state.settings.default_save_output
    )
    instruct = payload.instruct or payload.emotion or "Normal tone"
    return create_job_submission(
        operation=JobOperation.SYNTHESIZE_CUSTOM,
        command=CustomVoiceCommand(
            text=text,
            model=payload.model,
            save_output=save_output,
            language=payload.language,
            speaker=payload.speaker,
            instruct=instruct,
            speed=payload.speed,
        ),
        submit_request_id=request.state.request_id,
        owner_principal_id=current_principal_id(request),
        response_format="wav",
        save_output=save_output,
        execution_timeout_seconds=request.app.state.settings.request_timeout_seconds,
        idempotency_key=idempotency_key,
        idempotency_scope=resolve_idempotency_scope(request)
        if idempotency_key is not None
        else None,
        idempotency_fingerprint=(
            build_idempotency_fingerprint(
                operation=JobOperation.SYNTHESIZE_CUSTOM,
                payload={
                    "model": payload.model,
                    "text": text,
                    "speaker": payload.speaker,
                    "emotion": payload.emotion,
                    "instruct": instruct,
                    "language": payload.language,
                    "speed": payload.speed,
                    "save_output": save_output,
                    "response_format": "wav",
                },
            )
            if idempotency_key is not None
            else None
        ),
    )


# START_CONTRACT: create_design_job_submission
#   PURPOSE: Convert a voice design payload into an async voice design job submission.
#   INPUTS: { request: Request - request carrying app state and principal, payload: DesignTTSRequest - validated design payload, idempotency_key: Optional[str] - optional client idempotency key }
#   OUTPUTS: { Any - job submission object for async execution }
#   SIDE_EFFECTS: Reads app settings and may raise validation errors for oversized text
#   LINKS: M-SERVER, M-CONTRACTS
# END_CONTRACT: create_design_job_submission
def create_design_job_submission(
    request: Request,
    payload: DesignTTSRequest,
    *,
    idempotency_key: Optional[str] = None,
):
    ensure_requested_model_capability(request, payload.model, legacy_mode="design")
    text = enforce_text_length(
        value=payload.text,
        field_name="text",
        max_chars=request.app.state.settings.max_input_text_chars,
    )
    save_output = resolve_save_output(
        payload.save_output, request.app.state.settings.default_save_output
    )
    voice_description = payload.voice_description
    return create_job_submission(
        operation=JobOperation.SYNTHESIZE_DESIGN,
        command=VoiceDesignCommand(
            text=text,
            model=payload.model,
            save_output=save_output,
            language=payload.language,
            voice_description=voice_description,
        ),
        submit_request_id=request.state.request_id,
        owner_principal_id=current_principal_id(request),
        response_format="wav",
        save_output=save_output,
        execution_timeout_seconds=request.app.state.settings.request_timeout_seconds,
        idempotency_key=idempotency_key,
        idempotency_scope=resolve_idempotency_scope(request)
        if idempotency_key is not None
        else None,
        idempotency_fingerprint=(
            build_idempotency_fingerprint(
                operation=JobOperation.SYNTHESIZE_DESIGN,
                payload={
                    "model": payload.model,
                    "text": text,
                    "voice_description": voice_description,
                    "language": payload.language,
                    "save_output": save_output,
                    "response_format": "wav",
                },
            )
            if idempotency_key is not None
            else None
        ),
    )


# START_CONTRACT: validate_clone_upload
#   PURPOSE: Validate clone reference upload bytes, extension, and media type against server policy.
#   INPUTS: { request: Request - request carrying server settings, ref_audio: UploadFile - uploaded reference audio, upload_bytes: bytes - uploaded file bytes }
#   OUTPUTS: { JSONResponse | None - validation error response when invalid, otherwise none }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: validate_clone_upload
def validate_clone_upload(
    request: Request, ref_audio: UploadFile, upload_bytes: bytes
) -> JSONResponse | None:
    if not upload_bytes:
        return build_upload_validation_error(
            request=request,
            code="invalid_upload_audio",
            message="Uploaded reference audio is empty",
            details={"field": "ref_audio"},
        )

    if len(upload_bytes) > request.app.state.settings.max_upload_size_bytes:
        return build_upload_validation_error(
            request=request,
            code="upload_too_large",
            message="Uploaded file exceeds configured size limit",
            details={
                "max_upload_size_bytes": request.app.state.settings.max_upload_size_bytes
            },
        )

    filename = ref_audio.filename or "reference.wav"
    suffix = Path(filename).suffix.lower()
    content_type = (ref_audio.content_type or "application/octet-stream").lower()

    if suffix not in _ALLOWED_CLONE_UPLOAD_SUFFIXES:
        return build_upload_validation_error(
            request=request,
            code="unsupported_upload_media_type",
            message="Unsupported reference audio file type",
            details={
                "field": "ref_audio",
                "content_type": content_type,
                "allowed_extensions": sorted(_ALLOWED_CLONE_UPLOAD_SUFFIXES),
            },
        )

    if content_type not in _ALLOWED_CLONE_UPLOAD_CONTENT_TYPES:
        return build_upload_validation_error(
            request=request,
            code="unsupported_upload_media_type",
            message="Unsupported reference audio media type",
            details={
                "field": "ref_audio",
                "content_type": content_type,
                "allowed_content_types": sorted(_ALLOWED_CLONE_UPLOAD_CONTENT_TYPES),
            },
        )

    return None


# START_CONTRACT: build_clone_staged_path
#   PURPOSE: Build a unique staging path for a clone upload inside the configured upload directory.
#   INPUTS: { request: Request - request carrying server settings, ref_audio: StarletteUploadFile - uploaded reference audio metadata, prefix: str - filename prefix for the staged artifact }
#   OUTPUTS: { Path - unique filesystem path for the staged upload }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: build_clone_staged_path
def build_clone_staged_path(
    request: Request, ref_audio: StarletteUploadFile, *, prefix: str
) -> Path:
    suffix = Path(ref_audio.filename or "reference.wav").suffix.lower() or ".wav"
    return (
        request.app.state.settings.upload_staging_dir
        / f"{prefix}_{uuid.uuid4().hex}{suffix}"
    )


# START_CONTRACT: stage_clone_job_submission
#   PURPOSE: Validate clone inputs, stage uploaded audio, and build an async clone job submission.
#   INPUTS: { request: Request - request carrying app state and principal, text: str - synthesis text, ref_audio: UploadFile - uploaded reference audio, ref_text: Optional[str] - optional reference transcript, language: str - requested language value, model: Optional[str] - optional model override, save_output: Optional[bool] - output persistence override, idempotency_key: Optional[str] - optional client idempotency key }
#   OUTPUTS: { tuple[Any | None, JSONResponse | None] - staged job submission or validation error response }
#   SIDE_EFFECTS: Reads uploaded file bytes and writes a staged upload file to disk when validation succeeds
#   LINKS: M-SERVER, M-CONTRACTS
# END_CONTRACT: stage_clone_job_submission
async def stage_clone_job_submission(
    request: Request,
    *,
    text: str,
    ref_audio: UploadFile,
    ref_text: Optional[str],
    language: str,
    model: Optional[str],
    save_output: Optional[bool],
    idempotency_key: Optional[str] = None,
):
    # START_BLOCK_VALIDATE_CLONE_JOB_TEXT
    stripped_text = text.strip()
    if not stripped_text:
        return None, build_text_length_error(
            request=request, field_name="text", message="Text must not be empty"
        )
    try:
        stripped_text = enforce_text_length(
            value=stripped_text,
            field_name="text",
            max_chars=request.app.state.settings.max_input_text_chars,
        )
    except ValueError as exc:
        return None, build_text_length_error(
            request=request, field_name="text", message=str(exc)
        )
    # END_BLOCK_VALIDATE_CLONE_JOB_TEXT

    # START_BLOCK_VALIDATE_CLONE_UPLOAD
    upload_bytes = await ref_audio.read()
    upload_error = validate_clone_upload(request, ref_audio, upload_bytes)
    if upload_error is not None:
        return None, upload_error
    # END_BLOCK_VALIDATE_CLONE_UPLOAD

    # START_BLOCK_BUILD_CLONE_JOB_SUBMISSION
    resolved_save_output = resolve_save_output(
        save_output, request.app.state.settings.default_save_output
    )
    normalized_language = normalize_language_value(language)
    ensure_requested_model_capability(request, model, legacy_mode="clone")
    staged_path = build_clone_staged_path(request, ref_audio, prefix="job_upload")
    staged_path.write_bytes(upload_bytes)
    submission = create_job_submission(
        operation=JobOperation.SYNTHESIZE_CLONE,
        command=VoiceCloneCommand(
            text=stripped_text,
            model=model,
            save_output=resolved_save_output,
            language=normalized_language,
            ref_audio_path=staged_path,
            ref_text=ref_text,
        ),
        submit_request_id=request.state.request_id,
        owner_principal_id=current_principal_id(request),
        response_format="wav",
        save_output=resolved_save_output,
        execution_timeout_seconds=request.app.state.settings.request_timeout_seconds,
        staged_input_paths=(staged_path,),
        idempotency_key=idempotency_key,
        idempotency_scope=resolve_idempotency_scope(request)
        if idempotency_key is not None
        else None,
        idempotency_fingerprint=(
            build_idempotency_fingerprint(
                operation=JobOperation.SYNTHESIZE_CLONE,
                payload={
                    "model": model,
                    "text": stripped_text,
                    "ref_text": ref_text,
                    "language": normalized_language,
                    "save_output": resolved_save_output,
                    "response_format": "wav",
                    "ref_audio_filename": ref_audio.filename,
                    "ref_audio_size": len(upload_bytes),
                    "ref_audio_sha256": hashlib.sha256(upload_bytes).hexdigest(),
                },
            )
            if idempotency_key is not None
            else None
        ),
    )
    return submission, None
    # END_BLOCK_BUILD_CLONE_JOB_SUBMISSION


# START_CONTRACT: run_inference_with_timeout
#   PURPOSE: Execute a blocking inference call in a worker thread with request timeout enforcement.
#   INPUTS: { request: Request - request carrying timeout settings and logger, operation_name: str - operation label for logs, call: Callable[[], T] - blocking inference callable }
#   OUTPUTS: { T - result returned by the inference callable }
#   SIDE_EFFECTS: Offloads work to a thread, emits execution logs, and raises timeout errors when execution exceeds limits
#   LINKS: M-SERVER, M-ERRORS, M-OBSERVABILITY
# END_CONTRACT: run_inference_with_timeout
async def run_inference_with_timeout(
    *, request: Request, operation_name: str, call: Callable[[], T]
) -> T:
    # START_BLOCK_PREPARE_INFERENCE_TIMEOUT
    timeout_seconds = request.app.state.settings.request_timeout_seconds
    logger = request.app.state.logger
    # END_BLOCK_PREPARE_INFERENCE_TIMEOUT

    with operation_scope(f"tts.{operation_name}.execution"):
        # START_BLOCK_EXECUTE_SYNTHESIS
        wrapper_timer = Timer()
        log_event(
            logger,
            level=logging.INFO,
            event="[RoutesTTS][run_inference_with_timeout][BLOCK_EXECUTE_SYNTHESIS]",
            message="Inference execution wrapper started",
            inference_operation=operation_name,
            execution_mode="thread_offload",
            offloaded_from_event_loop=True,
            timeout_seconds=timeout_seconds,
        )

        def worker_call() -> T:
            log_event(
                logger,
                level=logging.INFO,
                event="[RoutesTTS][run_inference_with_timeout][BLOCK_EXECUTE_SYNTHESIS]",
                message="Adapter-level inference execution started",
                inference_operation=operation_name,
                execution_mode="thread_offload",
                offloaded_from_event_loop=True,
                timeout_seconds=timeout_seconds,
            )
            return call()

        # END_BLOCK_EXECUTE_SYNTHESIS

        try:
            # START_BLOCK_AWAIT_WORKER_RESULT
            result = await asyncio.wait_for(
                asyncio.to_thread(worker_call), timeout=timeout_seconds
            )
            # END_BLOCK_AWAIT_WORKER_RESULT
        except asyncio.TimeoutError as exc:
            # START_BLOCK_HANDLE_INFERENCE_TIMEOUT
            log_event(
                logger,
                level=logging.WARNING,
                event="[RoutesTTS][run_inference_with_timeout][BLOCK_HANDLE_INFERENCE_TIMEOUT]",
                message="Inference execution timed out",
                inference_operation=operation_name,
                execution_mode="thread_offload",
                offloaded_from_event_loop=True,
                timeout_seconds=timeout_seconds,
                duration_ms=wrapper_timer.elapsed_ms,
            )
            raise RequestTimeoutError(
                details={
                    "operation": operation_name,
                    "timeout_seconds": timeout_seconds,
                }
            ) from exc
            # END_BLOCK_HANDLE_INFERENCE_TIMEOUT
        except Exception as exc:
            # START_BLOCK_HANDLE_INFERENCE_FAILURE
            log_event(
                logger,
                level=logging.ERROR,
                event="[RoutesTTS][run_inference_with_timeout][BLOCK_HANDLE_INFERENCE_FAILURE]",
                message="Inference execution failed",
                inference_operation=operation_name,
                execution_mode="thread_offload",
                offloaded_from_event_loop=True,
                timeout_seconds=timeout_seconds,
                duration_ms=wrapper_timer.elapsed_ms,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
            # END_BLOCK_HANDLE_INFERENCE_FAILURE
        # START_BLOCK_LOG_INFERENCE_COMPLETION
        log_event(
            logger,
            level=logging.INFO,
            event="[RoutesTTS][run_inference_with_timeout][BLOCK_LOG_INFERENCE_COMPLETION]",
            message="Inference execution wrapper completed",
            inference_operation=operation_name,
            execution_mode="thread_offload",
            offloaded_from_event_loop=True,
            timeout_seconds=timeout_seconds,
            duration_ms=wrapper_timer.elapsed_ms,
        )
        return result
        # END_BLOCK_LOG_INFERENCE_COMPLETION


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
        with operation_scope("server.openai_speech"):
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
            ensure_requested_model_capability(
                request, payload.model, legacy_mode="custom"
            )
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
            return build_audio_response(
                request, result, payload.response_format, logger
            )
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
        with operation_scope("server.tts_custom"):
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
                return build_text_length_error(
                    request=request, field_name="text", message=str(exc)
                )
            ensure_requested_model_capability(
                request, payload.model, legacy_mode="custom"
            )
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
        with operation_scope("server.tts_design"):
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
                return build_text_length_error(
                    request=request, field_name="text", message=str(exc)
                )
            ensure_requested_model_capability(
                request, payload.model, legacy_mode="design"
            )
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
        ref_text: Optional[str] = Form(default=None),
        language: Optional[str] = Form(default="auto"),
        model: Optional[str] = Form(default=None),
        save_output: Optional[bool] = Form(default=None),
    ) -> Response:
        with operation_scope("server.tts_clone"):
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
                return build_text_length_error(
                    request=request, field_name="text", message=str(exc)
                )
            ensure_requested_model_capability(request, model, legacy_mode="clone")
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
            try:
                result = await run_inference_with_timeout(
                    request=request,
                    operation_name="synthesize_clone",
                    call=lambda: request.app.state.application.synthesize_clone(
                        VoiceCloneCommand(
                            text=text,
                            model=model,
                            save_output=resolved_save_output,
                            language=normalized_language,
                            ref_audio_path=temp_path,
                            ref_text=ref_text,
                        )
                    ),
                )
            finally:
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
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
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
        # START_BLOCK_BUILD_JOB_RESPONSE
        return build_job_snapshot_payload(request, resolution.snapshot)
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
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
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
        # START_BLOCK_BUILD_CUSTOM_JOB_RESPONSE
        return build_job_snapshot_payload(request, resolution.snapshot)
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
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    ) -> JobSnapshotPayload:
        # START_BLOCK_CHECK_IDEMPOTENCY_DESIGN_JOB
        await enforce_async_submit_admission(request)
        # END_BLOCK_CHECK_IDEMPOTENCY_DESIGN_JOB
        # START_BLOCK_SUBMIT_DESIGN_JOB
        resolution = request.app.state.job_execution.submit_idempotent(
            create_design_job_submission(
                request, payload, idempotency_key=idempotency_key
            )
        )
        # END_BLOCK_SUBMIT_DESIGN_JOB
        # START_BLOCK_BUILD_DESIGN_JOB_RESPONSE
        return build_job_snapshot_payload(request, resolution.snapshot)
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
        ref_text: Optional[str] = Form(default=None),
        language: Optional[str] = Form(default="auto"),
        model: Optional[str] = Form(default=None),
        save_output: Optional[bool] = Form(default=None),
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
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
        # START_BLOCK_BUILD_CLONE_JOB_RESPONSE
        return JSONResponse(
            status_code=202,
            content=build_job_snapshot_payload(request, resolution.snapshot).model_dump(
                mode="json"
            ),
        )
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
        return build_job_snapshot_payload(
            request, get_job_snapshot_or_raise(request, job_id)
        )

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
                snapshot.status.value,
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

        # START_BLOCK_BUILD_JOB_RESULT_RESPONSE
        response = build_audio_response(
            request,
            resolution.success.generation,
            snapshot.response_format or "wav",
            logger,
        )
        response.headers["x-job-id"] = snapshot.job_id
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

        # START_BLOCK_BUILD_CANCEL_RESPONSE
        status_code = 200 if snapshot.status is JobStatus.CANCELLED else 202
        return JSONResponse(
            status_code=status_code,
            content=build_job_snapshot_payload(request, cancelled).model_dump(
                mode="json"
            ),
        )
        # END_BLOCK_BUILD_CANCEL_RESPONSE


__all__ = [
    "T",
    "build_text_length_error",
    "build_upload_validation_error",
    "enforce_text_length",
    "current_principal_id",
    "resolve_idempotency_scope",
    "ensure_requested_model_capability",
    "build_job_urls",
    "build_job_snapshot_payload",
    "get_job_snapshot_or_raise",
    "build_idempotency_fingerprint",
    "create_custom_job_submission_from_openai",
    "create_custom_job_submission_from_custom",
    "create_design_job_submission",
    "validate_clone_upload",
    "build_clone_staged_path",
    "stage_clone_job_submission",
    "run_inference_with_timeout",
    "register_tts_routes",
]
