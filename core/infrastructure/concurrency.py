# FILE: core/infrastructure/concurrency.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide single-slot inference concurrency guard.
#   SCOPE: InferenceGuard class with acquire/release semantics
#   DEPENDS: M-ERRORS
#   LINKS: M-INFRASTRUCTURE
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   InferenceGuard - Single-slot inference concurrency guard
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

from threading import Lock

from core.errors import InferenceBusyError


# START_CONTRACT: InferenceGuard
#   PURPOSE: Enforce single-slot access to shared inference resources across concurrent requests.
#   INPUTS: {}
#   OUTPUTS: { instance - Inference concurrency guard }
#   SIDE_EFFECTS: Manages an in-memory process-local lock protecting inference execution
#   LINKS: M-INFRASTRUCTURE
# END_CONTRACT: InferenceGuard
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


__all__ = [
    "InferenceGuard",
]
