# FILE: core/services/telemetry.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide an opt-in OpenTelemetry tracing facade that bridges OperationScope and HTTP request handling into OTLP-compatible spans without making OpenTelemetry a hard runtime dependency.
#   SCOPE: TelemetryState container, configure_telemetry factory, instrument_fastapi bridge, start_span helper, get_active_state lookup, OperationScope -> OTel span bridge
#   DEPENDS: M-CONFIG, M-OBSERVABILITY
#   LINKS: M-TELEMETRY
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   TelemetryState - Frozen dataclass describing the active telemetry runtime (enabled flag, tracer, service name)
#   configure_telemetry - Build a TelemetryState from CoreSettings; produces a disabled state when OTel is unavailable
#   instrument_fastapi - Best-effort installer that wires a FastAPI app into OpenTelemetry HTTP instrumentation
#   start_span - Context manager that emits an OTel span when telemetry is active or yields silently otherwise
#   get_active_state - Read the process-local active TelemetryState
#   reset_active_state - Test-only helper that drops the cached TelemetryState
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Phase 4.15: introduced OpenTelemetry bridge module supporting opt-in OTLP/HTTP exporter, FastAPI instrumentation hook, OperationScope span bridging, and graceful no-op fallback when opentelemetry packages are missing or disabled]
# END_CHANGE_SUMMARY

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.config import CoreSettings

LOGGER = logging.getLogger(__name__)

_STATE_LOCK = threading.RLock()
_ACTIVE_STATE: TelemetryState | None = None


# START_CONTRACT: TelemetryState
#   PURPOSE: Hold the resolved OpenTelemetry runtime state for the current process so that context bridges can emit spans without re-resolving the tracer on every call.
#   INPUTS: { enabled: bool - Whether OTel instrumentation is active, service_name: str - Logical service name attached to spans, exporter_endpoint: str | None - Optional OTLP/HTTP collector endpoint, tracer: Any | None - Cached OpenTelemetry tracer instance, attributes: dict - Resource attributes attached to every span }
#   OUTPUTS: { instance - Immutable telemetry runtime descriptor }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEMETRY
# END_CONTRACT: TelemetryState
@dataclass(frozen=True)
class TelemetryState:
    enabled: bool
    service_name: str
    exporter_endpoint: str | None
    tracer: Any | None = None
    attributes: dict[str, str] = field(default_factory=dict)


# START_CONTRACT: configure_telemetry
#   PURPOSE: Build a TelemetryState from the provided CoreSettings, configuring an OTLP/HTTP-backed TracerProvider when telemetry is enabled and the opentelemetry packages are importable.
#   INPUTS: { settings: CoreSettings - Parsed runtime settings carrying otel_* fields }
#   OUTPUTS: { TelemetryState - Active telemetry runtime; enabled=False when telemetry is off or OTel is missing }
#   SIDE_EFFECTS: Caches the resolved state in module-level storage and may install a global TracerProvider when OTel is available
#   LINKS: M-TELEMETRY, M-BOOTSTRAP
# END_CONTRACT: configure_telemetry
def configure_telemetry(settings: CoreSettings) -> TelemetryState:
    global _ACTIVE_STATE

    # START_BLOCK_BUILD_DISABLED_STATE
    disabled = TelemetryState(
        enabled=False,
        service_name=settings.otel_service_name,
        exporter_endpoint=settings.otel_exporter_endpoint,
    )
    if not settings.otel_enabled:
        with _STATE_LOCK:
            _ACTIVE_STATE = disabled
        return disabled
    # END_BLOCK_BUILD_DISABLED_STATE

    # START_BLOCK_IMPORT_OTEL
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
    except ImportError:
        LOGGER.warning(
            '{"event":"[Telemetry][configure_telemetry][OTEL_UNAVAILABLE]",'
            '"message":"OpenTelemetry packages not installed; telemetry disabled"}'
        )
        with _STATE_LOCK:
            _ACTIVE_STATE = disabled
        return disabled
    # END_BLOCK_IMPORT_OTEL

    # START_BLOCK_BUILD_PROVIDER
    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)

    exporter: Any | None = None
    if settings.otel_exporter_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_endpoint)
        except ImportError:
            LOGGER.warning(
                '{"event":"[Telemetry][configure_telemetry][OTLP_UNAVAILABLE]",'
                '"message":"opentelemetry-exporter-otlp-proto-http missing; falling back to console exporter"}'
            )
            exporter = ConsoleSpanExporter()
    else:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(settings.otel_service_name)
    # END_BLOCK_BUILD_PROVIDER

    state = TelemetryState(
        enabled=True,
        service_name=settings.otel_service_name,
        exporter_endpoint=settings.otel_exporter_endpoint,
        tracer=tracer,
        attributes={"service.name": settings.otel_service_name},
    )
    with _STATE_LOCK:
        _ACTIVE_STATE = state
    LOGGER.info(
        '{"event":"[Telemetry][configure_telemetry][TELEMETRY_READY]",'
        '"message":"OpenTelemetry tracing enabled",'
        f'"service":"{settings.otel_service_name}",'
        f'"endpoint":"{settings.otel_exporter_endpoint or "console"}"}}'
    )
    return state


# START_CONTRACT: instrument_fastapi
#   PURPOSE: Best-effort wiring that installs OpenTelemetry's FastAPI instrumentation on the supplied app when telemetry is active.
#   INPUTS: { app: Any - FastAPI application instance, state: TelemetryState - Active telemetry runtime }
#   OUTPUTS: { bool - True when instrumentation was installed, False otherwise }
#   SIDE_EFFECTS: Mutates the FastAPI app middleware stack via FastAPIInstrumentor
#   LINKS: M-TELEMETRY, M-SERVER
# END_CONTRACT: instrument_fastapi
def instrument_fastapi(app: Any, state: TelemetryState) -> bool:
    if not state.enabled:
        return False
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        LOGGER.warning(
            '{"event":"[Telemetry][instrument_fastapi][FASTAPI_INSTRUMENTOR_MISSING]",'
            '"message":"opentelemetry-instrumentation-fastapi missing; HTTP spans skipped"}'
        )
        return False
    FastAPIInstrumentor.instrument_app(app)
    return True


# START_CONTRACT: start_span
#   PURPOSE: Provide a context manager that emits an OpenTelemetry span when telemetry is active and yields silently otherwise so callers stay decoupled from the underlying SDK.
#   INPUTS: { name: str - Span name (typically [Module][operation][BLOCK_NAME]), attributes: dict[str, Any] | None - Optional span attributes }
#   OUTPUTS: { Iterator[Any] - Active span object or None when telemetry is disabled }
#   SIDE_EFFECTS: Records a span in the active TracerProvider when telemetry is enabled
#   LINKS: M-TELEMETRY, M-OBSERVABILITY
# END_CONTRACT: start_span
@contextmanager
def start_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    state = get_active_state()
    if state is None or not state.enabled or state.tracer is None:
        yield None
        return
    with state.tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                try:
                    span.set_attribute(key, value)
                except Exception:  # pragma: no cover - defensive against exotic types
                    span.set_attribute(key, str(value))
        yield span


# START_CONTRACT: get_active_state
#   PURPOSE: Read the process-local TelemetryState that was last installed by configure_telemetry.
#   INPUTS: {}
#   OUTPUTS: { TelemetryState | None - Active telemetry state or None if configure_telemetry was never called }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEMETRY
# END_CONTRACT: get_active_state
def get_active_state() -> TelemetryState | None:
    with _STATE_LOCK:
        return _ACTIVE_STATE


# START_CONTRACT: reset_active_state
#   PURPOSE: Test-only helper that drops the cached telemetry state so suites can re-run configure_telemetry from a clean slate.
#   INPUTS: {}
#   OUTPUTS: { None }
#   SIDE_EFFECTS: Clears the module-level active TelemetryState
#   LINKS: M-TELEMETRY
# END_CONTRACT: reset_active_state
def reset_active_state() -> None:
    global _ACTIVE_STATE
    with _STATE_LOCK:
        _ACTIVE_STATE = None


__all__ = [
    "TelemetryState",
    "configure_telemetry",
    "instrument_fastapi",
    "start_span",
    "get_active_state",
    "reset_active_state",
]
