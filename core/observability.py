from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from time import perf_counter
from typing import Any, Iterator


_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="system")
_OPERATION: ContextVar[str] = ContextVar("operation", default="system")


class OperationScope:
    def __init__(self, operation: str):
        self.operation = operation
        self._token = None

    def __enter__(self) -> "OperationScope":
        self._token = _OPERATION.set(self.operation)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _OPERATION.reset(self._token)


class Timer:
    def __init__(self) -> None:
        self._started_at = perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return round((perf_counter() - self._started_at) * 1000, 3)


def bind_request_context(request_id: str) -> object:
    return _REQUEST_ID.set(request_id)


def reset_request_context(token: object) -> None:
    _REQUEST_ID.reset(token)


def get_request_id() -> str:
    return _REQUEST_ID.get()


def get_operation() -> str:
    return _OPERATION.get()


def operation_scope(operation: str) -> Iterator[OperationScope]:
    return OperationScope(operation)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


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
