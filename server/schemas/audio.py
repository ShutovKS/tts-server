# FILE: server/schemas/audio.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define Pydantic schemas for audio-related API request/response payloads.
#   SCOPE: Request validation schemas for TTS endpoints
#   DEPENDS: none
#   LINKS: M-SERVER
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   normalize_text_value - Normalize and reject empty text inputs
#   normalize_language_value - Normalize and validate language fields
#   validate_text_length - Enforce configured max text length
#   OpenAISpeechRequest - OpenAI-compatible synchronous speech request schema
#   CustomTTSRequest - Custom voice synthesis request schema
#   DesignTTSRequest - Voice design synthesis request schema
#   JobFailurePayload - Async job terminal failure payload schema
#   JobSnapshotPayload - Async job snapshot payload schema
#   TTSSuccessMetadata - Success metadata schema for generated responses
#   ModelInfo - Model discovery payload item schema
#   ModelsResponse - Model discovery response schema
#   HealthResponse - Health/readiness response schema
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.1 - Expanded model discovery payloads with family, routing, and readiness metadata]
# END_CHANGE_SUMMARY

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# START_CONTRACT: normalize_text_value
#   PURPOSE: Strip and validate a text field to ensure it is not empty.
#   INPUTS: { value: str - raw text value, empty_message: str - validation error message for empty input }
#   OUTPUTS: { str - normalized non-empty text value }
#   SIDE_EFFECTS: Raises ValueError when the normalized text is empty
#   LINKS: M-SERVER
# END_CONTRACT: normalize_text_value
def normalize_text_value(value: str, *, empty_message: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(empty_message)
    return value


# START_CONTRACT: normalize_language_value
#   PURPOSE: Normalize and validate a language field for API request payloads.
#   INPUTS: { value: str - raw language value }
#   OUTPUTS: { str - normalized lowercase language value }
#   SIDE_EFFECTS: Raises ValueError when the normalized language is empty
#   LINKS: M-SERVER
# END_CONTRACT: normalize_language_value
def normalize_language_value(value: str) -> str:
    value = value.strip().lower()
    if not value:
        raise ValueError("Language must not be empty")
    return value


# START_CONTRACT: validate_text_length
#   PURPOSE: Enforce the configured maximum character length for a text field.
#   INPUTS: { value: str - text value to validate, field_name: str - field label used in errors, max_chars: int - maximum allowed characters }
#   OUTPUTS: { str - original text value when within limit }
#   SIDE_EFFECTS: Raises ValueError when the value exceeds the configured length
#   LINKS: M-SERVER
# END_CONTRACT: validate_text_length
def validate_text_length(value: str, *, field_name: str, max_chars: int) -> str:
    if len(value) > max_chars:
        raise ValueError(f"{field_name} must be at most {max_chars} characters")
    return value


# START_CONTRACT: OpenAISpeechRequest
#   PURPOSE: Define the OpenAI-compatible speech request payload schema.
#   INPUTS: { model: str - requested model id, input: str - source text, voice: str - speaker name, language: str - requested language code, response_format: Literal['wav', 'pcm'] - desired audio format, speed: float - playback speed multiplier }
#   OUTPUTS: { OpenAISpeechRequest - validated OpenAI-style speech request model }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: OpenAISpeechRequest
class OpenAISpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description="Model identifier")
    input: str = Field(..., min_length=1, description="Input text")
    voice: str = Field(default="Vivian", description="Speaker/voice name")
    language: str = Field(default="auto", description="Language code or auto")
    response_format: Literal["wav", "pcm"] = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

    @field_validator("input")
    @classmethod
    # START_CONTRACT: validate_input
    #   PURPOSE: Validate that the OpenAI input text field is non-empty after trimming.
    #   INPUTS: { cls: type[OpenAISpeechRequest] - model class, value: str - raw input text }
    #   OUTPUTS: { str - normalized non-empty input text }
    #   SIDE_EFFECTS: Raises ValueError for empty normalized input
    #   LINKS: M-SERVER
    # END_CONTRACT: validate_input
    def validate_input(cls, value: str) -> str:
        return normalize_text_value(value, empty_message="Input text must not be empty")

    @field_validator("language")
    @classmethod
    # START_CONTRACT: validate_language
    #   PURPOSE: Validate and normalize the OpenAI request language field.
    #   INPUTS: { cls: type[OpenAISpeechRequest] - model class, value: str - raw language field }
    #   OUTPUTS: { str - normalized language value }
    #   SIDE_EFFECTS: Raises ValueError for empty normalized language values
    #   LINKS: M-SERVER
    # END_CONTRACT: validate_language
    def validate_language(cls, value: str) -> str:
        return normalize_language_value(value)


# START_CONTRACT: CustomTTSRequest
#   PURPOSE: Define the custom voice synthesis request payload schema.
#   INPUTS: { model: Optional[str] - optional model override, text: str - synthesis text, speaker: str - speaker name, emotion: Optional[str] - optional emotion hint, instruct: Optional[str] - optional instruction text, language: str - requested language code, speed: float - playback speed multiplier, save_output: Optional[bool] - output persistence override }
#   OUTPUTS: { CustomTTSRequest - validated custom TTS request model }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: CustomTTSRequest
class CustomTTSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = Field(
        default=None, description="Optional custom voice model override"
    )
    text: str = Field(..., min_length=1)
    speaker: str = Field(default="Vivian")
    emotion: Optional[str] = Field(default=None)
    instruct: Optional[str] = Field(default=None)
    language: str = Field(default="auto")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    save_output: Optional[bool] = Field(default=None)

    @field_validator("text")
    @classmethod
    # START_CONTRACT: validate_text
    #   PURPOSE: Validate that the custom TTS text field is non-empty after trimming.
    #   INPUTS: { cls: type[CustomTTSRequest] - model class, value: str - raw text field }
    #   OUTPUTS: { str - normalized non-empty text value }
    #   SIDE_EFFECTS: Raises ValueError for empty normalized text
    #   LINKS: M-SERVER
    # END_CONTRACT: validate_text
    def validate_text(cls, value: str) -> str:
        return normalize_text_value(value, empty_message="Text must not be empty")

    @field_validator("language")
    @classmethod
    # START_CONTRACT: validate_language
    #   PURPOSE: Validate and normalize the custom TTS language field.
    #   INPUTS: { cls: type[CustomTTSRequest] - model class, value: str - raw language field }
    #   OUTPUTS: { str - normalized language value }
    #   SIDE_EFFECTS: Raises ValueError for empty normalized language values
    #   LINKS: M-SERVER
    # END_CONTRACT: validate_language
    def validate_language(cls, value: str) -> str:
        return normalize_language_value(value)


# START_CONTRACT: DesignTTSRequest
#   PURPOSE: Define the voice design synthesis request payload schema.
#   INPUTS: { model: Optional[str] - optional model override, text: str - synthesis text, voice_description: str - natural-language voice description, language: str - requested language code, save_output: Optional[bool] - output persistence override }
#   OUTPUTS: { DesignTTSRequest - validated voice design request model }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: DesignTTSRequest
class DesignTTSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = Field(
        default=None, description="Optional voice design model override"
    )
    text: str = Field(..., min_length=1)
    voice_description: str = Field(..., min_length=1)
    language: str = Field(default="auto")
    save_output: Optional[bool] = Field(default=None)

    @field_validator("text", "voice_description")
    @classmethod
    # START_CONTRACT: validate_non_empty
    #   PURPOSE: Validate that design request text fields are non-empty after trimming.
    #   INPUTS: { cls: type[DesignTTSRequest] - model class, value: str - raw field value }
    #   OUTPUTS: { str - normalized non-empty field value }
    #   SIDE_EFFECTS: Raises ValueError for empty normalized values
    #   LINKS: M-SERVER
    # END_CONTRACT: validate_non_empty
    def validate_non_empty(cls, value: str) -> str:
        return normalize_text_value(value, empty_message="Value must not be empty")

    @field_validator("language")
    @classmethod
    # START_CONTRACT: validate_language
    #   PURPOSE: Validate and normalize the design request language field.
    #   INPUTS: { cls: type[DesignTTSRequest] - model class, value: str - raw language field }
    #   OUTPUTS: { str - normalized language value }
    #   SIDE_EFFECTS: Raises ValueError for empty normalized language values
    #   LINKS: M-SERVER
    # END_CONTRACT: validate_language
    def validate_language(cls, value: str) -> str:
        return normalize_language_value(value)


# START_CONTRACT: JobFailurePayload
#   PURPOSE: Define the schema for terminal job failure details exposed by async job APIs.
#   INPUTS: { code: str - machine-readable failure code, message: str - failure summary, details: Optional[dict[str, object]] - optional structured failure details }
#   OUTPUTS: { JobFailurePayload - validated job failure payload model }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: JobFailurePayload
class JobFailurePayload(BaseModel):
    code: str
    message: str
    details: Optional[dict[str, object]] = None


# START_CONTRACT: JobSnapshotPayload
#   PURPOSE: Define the schema returned for async job snapshots and state transitions.
#   INPUTS: { request_id: str - submit or query correlation id, job_id: str - async job identifier, submit_request_id: str - original submit request id, status: str - job lifecycle state, operation: str - queued operation type, mode: str - TTS mode, model: Optional[str] - requested model id, backend: Optional[str] - execution backend id, response_format: Optional[str] - stored response format, save_output: bool - persistence flag, created_at: datetime - creation time, started_at: Optional[datetime] - execution start time, completed_at: Optional[datetime] - completion time, saved_path: Optional[str] - public artifact name, terminal_error: Optional[JobFailurePayload] - terminal failure payload, status_url: str - status endpoint URL, result_url: str - result endpoint URL, cancel_url: str - cancel endpoint URL, idempotency_key: Optional[str] - client idempotency key }
#   OUTPUTS: { JobSnapshotPayload - validated async job snapshot model }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: JobSnapshotPayload
class JobSnapshotPayload(BaseModel):
    request_id: str
    job_id: str
    submit_request_id: str
    status: str
    operation: str
    mode: str
    model: Optional[str] = None
    backend: Optional[str] = None
    response_format: Optional[str] = None
    save_output: bool
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    saved_path: Optional[str] = None
    terminal_error: Optional[JobFailurePayload] = None
    status_url: str
    result_url: str
    cancel_url: str
    idempotency_key: Optional[str] = None


# START_CONTRACT: TTSSuccessMetadata
#   PURPOSE: Define metadata fields associated with a successful TTS response.
#   INPUTS: { request_id: str - request correlation id, model: str - model id used for synthesis, mode: str - synthesis mode, backend: str - backend id, saved_path: Optional[str] - public saved artifact name }
#   OUTPUTS: { TTSSuccessMetadata - validated success metadata model }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: TTSSuccessMetadata
class TTSSuccessMetadata(BaseModel):
    request_id: str
    model: str
    mode: str
    backend: str
    saved_path: Optional[str] = None


# START_CONTRACT: ModelInfo
#   PURPOSE: Define the schema for a model discovery record returned by the API.
#   INPUTS: { key: str - registry key, id: str - public model id, name: str - display name, mode: str - supported synthesis mode, family: str | None - resolved model family label, family_key: str | None - normalized family identifier, capabilities_supported: list[str] - normalized supported synthesis capabilities, backend_support: list[str] - backends declared for the model in the manifest, folder: str - model folder name, available: bool - local availability flag, loadable: bool | None - artifact validation flag, runtime_ready: bool | None - runtime readiness flag, backend: str | None - execution backend id, selected_backend: str | None - globally selected backend id, execution_backend: str | None - effective per-model backend route, capabilities: dict[str, object] - backend capability metadata, route: dict[str, object] - backend route explanation, missing_artifacts: list[str] - missing artifact names, required_artifacts: list[str] - required artifact names }
#   OUTPUTS: { ModelInfo - validated model discovery payload item }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: ModelInfo
class ModelInfo(BaseModel):
    key: str
    id: str
    name: str
    mode: str
    family: Optional[str] = None
    family_key: Optional[str] = None
    capabilities_supported: list[str] = Field(default_factory=list)
    backend_support: list[str] = Field(default_factory=list)
    folder: str
    available: bool
    loadable: Optional[bool] = None
    runtime_ready: Optional[bool] = None
    backend: Optional[str] = None
    selected_backend: Optional[str] = None
    selected_backend_label: Optional[str] = None
    execution_backend: Optional[str] = None
    execution_backend_label: Optional[str] = None
    capabilities: dict[str, object]
    route: dict[str, object] = Field(default_factory=dict)
    missing_artifacts: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)


# START_CONTRACT: ModelsResponse
#   PURPOSE: Define the response schema for model discovery endpoints.
#   INPUTS: { data: list[ModelInfo] - available model records }
#   OUTPUTS: { ModelsResponse - validated model list response model }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: ModelsResponse
class ModelsResponse(BaseModel):
    data: list[ModelInfo]


# START_CONTRACT: HealthResponse
#   PURPOSE: Define the response schema for server health and readiness probes.
#   INPUTS: { status: Literal['ok', 'degraded'] - overall health status, checks: dict[str, object] - structured probe diagnostics }
#   OUTPUTS: { HealthResponse - validated health response model }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: HealthResponse
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, object]


__all__ = [
    "normalize_text_value",
    "normalize_language_value",
    "validate_text_length",
    "OpenAISpeechRequest",
    "CustomTTSRequest",
    "DesignTTSRequest",
    "JobFailurePayload",
    "JobSnapshotPayload",
    "TTSSuccessMetadata",
    "ModelInfo",
    "ModelsResponse",
    "HealthResponse",
]
