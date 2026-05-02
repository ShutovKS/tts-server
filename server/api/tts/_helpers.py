# FILE: server/api/tts/_helpers.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Host private TTS route helper logic for validation, job submission shaping, and clone staging.
#   SCOPE: Validation helpers, request-principal helpers, job snapshot helpers, async job submission shaping, clone upload validation, and staged upload path construction
#   DEPENDS: M-CONTRACTS, M-ERRORS, M-OBSERVABILITY, M-STREAMING, M-SERVER
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
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Extracted route helper logic out of server/api/routes_tts.py without changing route registration or observable behavior]
# END_CHANGE_SUMMARY

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from core.contracts.commands import CustomVoiceCommand, VoiceCloneCommand, VoiceDesignCommand
from core.contracts.jobs import JobOperation, JobSnapshot, JobStatus, create_job_submission
from core.contracts.synthesis import execution_mode_to_capability
from core.errors import JobNotFoundError, ModelCapabilityError
from server.api.auth import ensure_job_owner_access
from server.api.contracts import ErrorDescriptor
from server.api.responses import build_error_response, public_artifact_name, resolve_save_output
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
_ALLOWED_CLONE_UPLOAD_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a", ".mp4"})


# START_CONTRACT: build_text_length_error
#   PURPOSE: Build a standardized validation error response for oversized or empty text fields.
#   INPUTS: { request: Request - request carrying correlation state, field_name: str - field that failed validation, message: str - validation message }
#   OUTPUTS: { JSONResponse - standardized validation error response }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: build_text_length_error
def build_text_length_error(*, request: Any, field_name: str, message: str):
    return build_error_response(
        request=request,
        descriptor=ErrorDescriptor(
            status_code=422,
            code="validation_error",
            message="Request validation failed",
            details={"errors": [{"loc": ["body", field_name], "msg": message, "type": "value_error"}]},
        ),
    )


# START_CONTRACT: build_upload_validation_error
#   PURPOSE: Build a standardized error response for invalid clone upload inputs.
#   INPUTS: { request: Request - request carrying correlation state, code: str - machine-readable error code, message: str - human-readable summary, details: dict[str, object] - structured validation details }
#   OUTPUTS: { JSONResponse - standardized upload validation error response }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: build_upload_validation_error
def build_upload_validation_error(*, request: Any, code: str, message: str, details: dict[str, object]):
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
def current_principal_id(request: Any) -> str:
    return request.state.principal.principal_id


# START_CONTRACT: resolve_idempotency_scope
#   PURPOSE: Resolve the idempotency scope key used for async job submissions.
#   INPUTS: { request: Request - request containing resolved principal state }
#   OUTPUTS: { str - idempotency scope identifier }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: resolve_idempotency_scope
def resolve_idempotency_scope(request: Any) -> str:
    return current_principal_id(request)


# START_CONTRACT: ensure_requested_model_capability
#   PURPOSE: Validate that an explicitly requested model supports the requested synthesis capability.
#   INPUTS: { request: Request - request carrying the active model registry, model_name: Optional[str] - explicit model identifier to validate, execution_mode: str - execution mode that maps to a normalized capability }
#   OUTPUTS: { None - completes when the model supports the capability }
#   SIDE_EFFECTS: May resolve model metadata through the registry and raise model capability errors
#   LINKS: M-SERVER, M-MODEL-REGISTRY, M-ERRORS
# END_CONTRACT: ensure_requested_model_capability
def ensure_requested_model_capability(request: Any, model_name: str | None, *, execution_mode: str) -> None:
    if not model_name:
        return
    registry = request.app.state.registry
    if not hasattr(registry, "get_model_spec"):
        return
    spec = registry.get_model_spec(model_name=model_name)
    capability = execution_mode_to_capability(execution_mode)
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
def build_job_urls(request: Any, job_id: str) -> tuple[str, str, str]:
    return (
        str(request.url_for("tts_job_status", job_id=job_id)),
        str(request.url_for("tts_job_result", job_id=job_id)),
        str(request.url_for("tts_job_cancel", job_id=job_id)),
    )


# START_CONTRACT: public_job_status
#   PURPOSE: Convert an internal async job status into the frozen public Phase 1 lifecycle state set.
#   INPUTS: { status: JobStatus - internal async job status }
#   OUTPUTS: { str - public job state limited to queued, running, succeeded, failed, or cancelled }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: public_job_status
def public_job_status(status: JobStatus) -> str:
    if status is JobStatus.TIMEOUT:
        return JobStatus.FAILED.value
    return status.value


# START_CONTRACT: build_job_snapshot_payload
#   PURPOSE: Convert an internal job snapshot into the public async job response payload.
#   INPUTS: { request: Request - request used for URL and request id resolution, snapshot: JobSnapshot - internal job snapshot }
#   OUTPUTS: { JobSnapshotPayload - public async job snapshot payload }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER, M-CONTRACTS
# END_CONTRACT: build_job_snapshot_payload
def build_job_snapshot_payload(request: Any, snapshot: JobSnapshot) -> JobSnapshotPayload:
    status_url, result_url, cancel_url = build_job_urls(request, snapshot.job_id)
    terminal_error = snapshot.terminal_error
    return JobSnapshotPayload(
        request_id=getattr(request.state, "request_id", "unknown"),
        job_id=snapshot.job_id,
        submit_request_id=snapshot.submit_request_id,
        status=public_job_status(snapshot.status),
        operation=snapshot.operation.value,
        mode=snapshot.mode,
        model=snapshot.requested_model,
        backend=snapshot.backend,
        response_format=snapshot.response_format,
        save_output=snapshot.save_output,
        created_at=snapshot.created_at,
        started_at=snapshot.started_at,
        completed_at=snapshot.completed_at,
        saved_path=public_artifact_name(snapshot.saved_path) if snapshot.saved_path is not None else None,
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
def get_job_snapshot_or_raise(request: Any, job_id: str) -> JobSnapshot:
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
def build_idempotency_fingerprint(*, operation: JobOperation, payload: dict[str, object]) -> str:
    normalized_payload = json.dumps(
        {"operation": operation.value, "payload": payload},
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
def create_custom_job_submission_from_openai(request: Any, payload: OpenAISpeechRequest, *, idempotency_key: str | None = None):
    ensure_requested_model_capability(request, payload.model, execution_mode="custom")
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
        idempotency_scope=resolve_idempotency_scope(request) if idempotency_key is not None else None,
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
def create_custom_job_submission_from_custom(request: Any, payload: CustomTTSRequest, *, idempotency_key: str | None = None):
    ensure_requested_model_capability(request, payload.model, execution_mode="custom")
    text = enforce_text_length(
        value=payload.text,
        field_name="text",
        max_chars=request.app.state.settings.max_input_text_chars,
    )
    save_output = resolve_save_output(payload.save_output, request.app.state.settings.default_save_output)
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
        idempotency_scope=resolve_idempotency_scope(request) if idempotency_key is not None else None,
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
def create_design_job_submission(request: Any, payload: DesignTTSRequest, *, idempotency_key: str | None = None):
    ensure_requested_model_capability(request, payload.model, execution_mode="design")
    text = enforce_text_length(
        value=payload.text,
        field_name="text",
        max_chars=request.app.state.settings.max_input_text_chars,
    )
    save_output = resolve_save_output(payload.save_output, request.app.state.settings.default_save_output)
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
        idempotency_scope=resolve_idempotency_scope(request) if idempotency_key is not None else None,
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
def validate_clone_upload(request: Any, ref_audio: Any, upload_bytes: bytes):
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
            details={"max_upload_size_bytes": request.app.state.settings.max_upload_size_bytes},
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
def build_clone_staged_path(request: Any, ref_audio: Any, *, prefix: str) -> Path:
    suffix = Path(ref_audio.filename or "reference.wav").suffix.lower() or ".wav"
    return request.app.state.settings.upload_staging_dir / f"{prefix}_{uuid.uuid4().hex}{suffix}"


# START_CONTRACT: stage_clone_job_submission
#   PURPOSE: Validate clone inputs, stage uploaded audio, and build an async clone job submission.
#   INPUTS: { request: Request - request carrying app state and principal, text: str - synthesis text, ref_audio: UploadFile - uploaded reference audio, ref_text: Optional[str] - optional reference transcript, language: str - requested language value, model: Optional[str] - optional model override, save_output: Optional[bool] - output persistence override, idempotency_key: Optional[str] - optional client idempotency key }
#   OUTPUTS: { tuple[Any | None, JSONResponse | None] - staged job submission or validation error response }
#   SIDE_EFFECTS: Reads uploaded file bytes and writes a staged upload file to disk when validation succeeds
#   LINKS: M-SERVER, M-CONTRACTS
# END_CONTRACT: stage_clone_job_submission
async def stage_clone_job_submission(
    request: Any,
    *,
    text: str,
    ref_audio: Any,
    ref_text: str | None,
    language: str,
    model: str | None,
    save_output: bool | None,
    idempotency_key: str | None = None,
):
    stripped_text = text.strip()
    if not stripped_text:
        return None, build_text_length_error(request=request, field_name="text", message="Text must not be empty")
    try:
        stripped_text = enforce_text_length(
            value=stripped_text,
            field_name="text",
            max_chars=request.app.state.settings.max_input_text_chars,
        )
    except ValueError as exc:
        return None, build_text_length_error(request=request, field_name="text", message=str(exc))

    upload_bytes = await ref_audio.read()
    upload_error = validate_clone_upload(request, ref_audio, upload_bytes)
    if upload_error is not None:
        return None, upload_error

    resolved_save_output = resolve_save_output(save_output, request.app.state.settings.default_save_output)
    normalized_language = normalize_language_value(language)
    ensure_requested_model_capability(request, model, execution_mode="clone")
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
        idempotency_scope=resolve_idempotency_scope(request) if idempotency_key is not None else None,
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


__all__ = [
    "build_text_length_error",
    "build_upload_validation_error",
    "enforce_text_length",
    "current_principal_id",
    "resolve_idempotency_scope",
    "ensure_requested_model_capability",
    "build_job_urls",
    "public_job_status",
    "build_job_snapshot_payload",
    "get_job_snapshot_or_raise",
    "build_idempotency_fingerprint",
    "create_custom_job_submission_from_openai",
    "create_custom_job_submission_from_custom",
    "create_design_job_submission",
    "validate_clone_upload",
    "build_clone_staged_path",
    "stage_clone_job_submission",
]
