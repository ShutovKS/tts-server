# FILE: server/schemas/errors.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define Pydantic schemas for structured API error payloads.
#   SCOPE: Error response schema types
#   DEPENDS: none
#   LINKS: M-SERVER
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   ErrorResponse - Structured API error payload schema
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# START_CONTRACT: ErrorResponse
#   PURPOSE: Define the structured JSON schema returned for API error responses.
#   INPUTS: { code: str - machine-readable error code, message: str - human-readable summary, details: dict[str, Any] - structured public error context, request_id: str - request correlation identifier }
#   OUTPUTS: { ErrorResponse - validated API error payload model }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: ErrorResponse
class ErrorResponse(BaseModel):
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional error details")
    request_id: str = Field(..., description="Per-request correlation identifier")


__all__ = [
    "ErrorResponse",
]
