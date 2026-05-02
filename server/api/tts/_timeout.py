# FILE: server/api/tts/_timeout.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Host the bounded synchronous inference timeout wrapper used by TTS HTTP routes.
#   SCOPE: Offload blocking synthesis calls to a worker thread, enforce request timeout, and preserve route-level timeout logging markers
#   DEPENDS: M-ERRORS, M-OBSERVABILITY, M-SERVER
#   LINKS: M-SERVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   T - Generic type variable used by async timeout helper utilities
#   run_inference_with_timeout - Run synthesis with timeout handling for sync routes
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Extracted bounded synchronous inference timeout handling out of server/api/routes_tts.py]
# END_CHANGE_SUMMARY

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TypeVar

from typing import Any

from core.errors import RequestTimeoutError
from core.observability import Timer, log_event, operation_scope

T = TypeVar("T")


# START_CONTRACT: run_inference_with_timeout
#   PURPOSE: Execute a blocking inference call in a worker thread with request timeout enforcement.
#   INPUTS: { request: Request - request carrying timeout settings and logger, operation_name: str - operation label for logs, call: Callable[[], T] - blocking inference callable }
#   OUTPUTS: { T - result returned by the inference callable }
#   SIDE_EFFECTS: Offloads work to a thread, emits execution logs, and raises timeout errors when execution exceeds limits
#   LINKS: M-SERVER, M-ERRORS, M-OBSERVABILITY
# END_CONTRACT: run_inference_with_timeout
async def run_inference_with_timeout(*, request: Any, operation_name: str, call: Callable[[], T]) -> T:
    timeout_seconds = request.app.state.settings.request_timeout_seconds
    logger = request.app.state.logger

    with operation_scope(f"tts.{operation_name}.execution"):  # type: ignore[reportGeneralTypeIssues]
        wrapper_timer = Timer()
        log_event(
            logger,
            level=logging.INFO,
            event="[RoutesTTS][run_inference_with_timeout][BLOCK_EXECUTE_SYNTHESIS]",
            message="Inference execution wrapper started with bounded synchronous semantics",
            inference_operation=operation_name,
            execution_mode="thread_offload",
            offloaded_from_event_loop=True,
            timeout_seconds=timeout_seconds,
            sync_semantics="bounded_sync_no_server_fallback",
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
                sync_semantics="bounded_sync_no_server_fallback",
            )
            return call()

        try:
            result = await asyncio.wait_for(asyncio.to_thread(worker_call), timeout=timeout_seconds)
        except TimeoutError as exc:
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
                sync_semantics="bounded_sync_no_server_fallback",
            )
            raise RequestTimeoutError(
                details={
                    "operation": operation_name,
                    "timeout_seconds": timeout_seconds,
                    "sync_semantics": "bounded_sync_no_server_fallback",
                }
            ) from exc
        except Exception as exc:
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
                sync_semantics="bounded_sync_no_server_fallback",
            )
            raise

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
            sync_semantics="bounded_sync_no_server_fallback",
        )
        return result


__all__ = ["T", "run_inference_with_timeout"]
