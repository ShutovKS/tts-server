from __future__ import annotations

from threading import Lock

from core.errors import InferenceBusyError


class InferenceGuard:
    def __init__(self) -> None:
        self._lock = Lock()
        self._busy = False

    def acquire(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise InferenceBusyError("Inference is already in progress")
        self._busy = True

    def release(self) -> None:
        self._busy = False
        self._lock.release()

    def is_busy(self) -> bool:
        return self._busy
