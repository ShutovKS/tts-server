# FILE: server/api/contracts.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define API-level request/response contract types.
#   SCOPE: API-specific contract types and helpers
#   DEPENDS: M-CONTRACTS
#   LINKS: M-SERVER
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   ErrorDescriptor - Immutable transport-ready API error descriptor
#   ExceptionMapping - Immutable exception-to-error-descriptor mapping record
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
# START_CONTRACT: ErrorDescriptor
#   PURPOSE: Describe a public API error response in transport-ready form.
#   INPUTS: { status_code: int - HTTP status code, code: str - machine-readable error code, message: str - human-readable summary, details: dict[str, Any] - safe public error details, retryable: bool - retry hint flag, headers: dict[str, str] | None - optional response headers }
#   OUTPUTS: { ErrorDescriptor - immutable API error descriptor }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: ErrorDescriptor
class ErrorDescriptor:
    status_code: int
    code: str
    message: str
    details: dict[str, Any]
    retryable: bool = False
    headers: dict[str, str] | None = None


@dataclass(frozen=True)
# START_CONTRACT: ExceptionMapping
#   PURPOSE: Bind an exception type to a descriptor builder for API error translation.
#   INPUTS: { error_type: type[Exception] - exception class to match, builder: Any - callable that builds an ErrorDescriptor }
#   OUTPUTS: { ExceptionMapping - immutable exception-to-descriptor mapping }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: ExceptionMapping
class ExceptionMapping:
    error_type: type[Exception]
    builder: Any


__all__ = [
    "ErrorDescriptor",
    "ExceptionMapping",
]
