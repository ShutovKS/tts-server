# FILE: core/observability.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide request context propagation, structured logging, and timing utilities.
#   SCOPE: Context variables for request ID and operation, structured JSON log emitter, Timer
#   DEPENDS: none
#   LINKS: M-OBSERVABILITY
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   log_event - Emit structured JSON log with request context
#   bind_request_context - Set request ID in context var
#   reset_request_context - Reset request ID context var
#   get_request_id - Read current request ID from context
#   get_operation - Read current operation from context
#   operation_scope - Context manager for operation tracking
#   OperationScope - Context manager class for operation name propagation
#   Timer - High-resolution elapsed time tracker
#   get_logger - Get named logger instance
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Phase 4.15: OperationScope bridges into OpenTelemetry via core.services.telemetry.start_span when telemetry is enabled, while remaining a pure contextvars helper otherwise]
# END_CHANGE_SUMMARY

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextvars import ContextVar
from time import perf_counter
from typing import Any

_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="system")
_OPERATION: ContextVar[str] = ContextVar("operation", default="system")


# START_CONTRACT: OperationScope
#   PURPOSE: Manage scoped propagation of the current logical operation name through context variables.
#   INPUTS: { operation: str - Operation name to bind while the scope is active }
#   OUTPUTS: { instance - Context manager that restores the previous operation on exit }
#   SIDE_EFFECTS: Modifies the operation context variable for the current execution context
#   LINKS: M-OBSERVABILITY
# END_CONTRACT: OperationScope
class OperationScope:
    def __init__(self, operation: str):
        self.operation = operation
        self._token = None
        self._span_cm: Any | None = None
        self._span: Any | None = None

    def __enter__(self) -> OperationScope:
        self._token = _OPERATION.set(self.operation)
        # START_BLOCK_BRIDGE_OTEL_SPAN
        try:
            from core.services.telemetry import get_active_state, start_span

            state = get_active_state()
            if state is not None and state.enabled:
                self._span_cm = start_span(
                    self.operation,
                    attributes={"request_id": _REQUEST_ID.get()},
                )
                self._span = self._span_cm.__enter__()
        except Exception:
            self._span_cm = None
            self._span = None
        # END_BLOCK_BRIDGE_OTEL_SPAN
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # START_BLOCK_CLOSE_OTEL_SPAN
        if self._span_cm is not None:
            try:
                self._span_cm.__exit__(exc_type, exc, tb)
            except Exception:
                pass
            self._span_cm = None
            self._span = None
        # END_BLOCK_CLOSE_OTEL_SPAN
        if self._token is not None:
            _OPERATION.reset(self._token)


# START_CONTRACT: Timer
#   PURPOSE: Track elapsed wall-clock time in milliseconds for runtime operations.
#   INPUTS: {}
#   OUTPUTS: { instance - High-resolution timer with elapsed_ms property }
#   SIDE_EFFECTS: none
#   LINKS: M-OBSERVABILITY
# END_CONTRACT: Timer
class Timer:
    def __init__(self) -> None:
        self._started_at = perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return round((perf_counter() - self._started_at) * 1000, 3)


# START_CONTRACT: bind_request_context
#   PURPOSE: Bind a request identifier into the current context for downstream logging.
#   INPUTS: { request_id: str - Correlation identifier for the active request }
#   OUTPUTS: { object - Context token required to reset the request binding }
#   SIDE_EFFECTS: Modifies the request ID context variable for the current execution context
#   LINKS: M-OBSERVABILITY
# END_CONTRACT: bind_request_context
def bind_request_context(request_id: str) -> object:
    return _REQUEST_ID.set(request_id)


# START_CONTRACT: reset_request_context
#   PURPOSE: Restore the previous request identifier after a scoped request binding completes.
#   INPUTS: { token: object - Context token previously returned by bind_request_context }
#   OUTPUTS: { None - Completes the request context reset }
#   SIDE_EFFECTS: Modifies the request ID context variable for the current execution context
#   LINKS: M-OBSERVABILITY
# END_CONTRACT: reset_request_context
def reset_request_context(token: object) -> None:
    _REQUEST_ID.reset(token)


# START_CONTRACT: get_request_id
#   PURPOSE: Read the current request correlation identifier from context.
#   INPUTS: {}
#   OUTPUTS: { str - Active request identifier, or the module default }
#   SIDE_EFFECTS: none
#   LINKS: M-OBSERVABILITY
# END_CONTRACT: get_request_id
def get_request_id() -> str:
    return _REQUEST_ID.get()


# START_CONTRACT: get_operation
#   PURPOSE: Read the current logical operation name from context.
#   INPUTS: {}
#   OUTPUTS: { str - Active operation name, or the module default }
#   SIDE_EFFECTS: none
#   LINKS: M-OBSERVABILITY
# END_CONTRACT: get_operation
def get_operation() -> str:
    return _OPERATION.get()


# START_CONTRACT: operation_scope
#   PURPOSE: Create an operation scope helper for structured context propagation.
#   INPUTS: { operation: str - Operation name to bind for the scope }
#   OUTPUTS: { Iterator[OperationScope] - Context manager for operation propagation }
#   SIDE_EFFECTS: none
#   LINKS: M-OBSERVABILITY
# END_CONTRACT: operation_scope
def operation_scope(operation: str) -> Iterator[OperationScope]:
    return OperationScope(operation)


# START_CONTRACT: get_logger
#   PURPOSE: Return a named Python logger used by core modules for structured events.
#   INPUTS: { name: str - Logger namespace name }
#   OUTPUTS: { logging.Logger - Logger instance for the requested namespace }
#   SIDE_EFFECTS: none
#   LINKS: M-OBSERVABILITY
# END_CONTRACT: get_logger
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# START_CONTRACT: log_event
#   PURPOSE: Emit a structured JSON log entry enriched with request and operation context.
#   INPUTS: { logger: logging.Logger - Logger receiving the event, level: int - Python logging level, event: str - Stable event identifier, message: str - Human-readable log summary, fields: Any - Additional structured event fields }
#   OUTPUTS: { None - Emits the structured log record }
#   SIDE_EFFECTS: Writes a log record through the provided logger
#   LINKS: M-OBSERVABILITY
# END_CONTRACT: log_event
def log_event(
    logger: logging.Logger,
    *,
    level: int,
    event: str,
    message: str,
    **fields: Any,
) -> None:
    payload = {
        "event": event,
        "message": message,
        "request_id": get_request_id(),
        "operation": get_operation(),
        **fields,
    }
    logger.log(level, json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


__all__ = [
    "OperationScope",
    "Timer",
    "bind_request_context",
    "reset_request_context",
    "get_request_id",
    "get_operation",
    "operation_scope",
    "get_logger",
    "log_event",
]
