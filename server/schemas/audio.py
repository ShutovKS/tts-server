from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OpenAISpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description="Model identifier")
    input: str = Field(..., min_length=1, description="Input text")
    voice: str = Field(default="Vivian", description="Speaker/voice name")
    response_format: Literal["wav", "pcm"] = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Input text must not be empty")
        return value


class CustomTTSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = Field(default=None, description="Optional custom voice model override")
    text: str = Field(..., min_length=1)
    speaker: str = Field(default="Vivian")
    emotion: Optional[str] = Field(default=None)
    instruct: Optional[str] = Field(default=None)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    save_output: Optional[bool] = Field(default=None)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Text must not be empty")
        return value


class DesignTTSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = Field(default=None, description="Optional voice design model override")
    text: str = Field(..., min_length=1)
    voice_description: str = Field(..., min_length=1)
    save_output: Optional[bool] = Field(default=None)

    @field_validator("text", "voice_description")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value must not be empty")
        return value


class TTSSuccessMetadata(BaseModel):
    request_id: str
    model: str
    mode: str
    backend: str
    saved_path: Optional[str] = None


class ModelInfo(BaseModel):
    key: str
    id: str
    name: str
    mode: str
    folder: str
    available: bool
    backend: str
    capabilities: dict[str, object]


class ModelsResponse(BaseModel):
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, object]
