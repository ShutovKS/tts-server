# FILE: tests/unit/core/test_telemetry.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Verify the telemetry facade builds disabled state when OTel is off or missing, exposes start_span as a no-op fallback, and lets OperationScope bridge to OTel only when enabled.
#   SCOPE: configure_telemetry disabled state, configure_telemetry missing-package fallback, start_span no-op, OperationScope bridge inactive when telemetry disabled.
#   DEPENDS: M-TELEMETRY, M-OBSERVABILITY
#   LINKS: V-M-TELEMETRY
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _make_settings - Build a CoreSettings instance with optional otel overrides.
#   test_configure_telemetry_disabled_when_flag_off - Verifies disabled flag yields TelemetryState(enabled=False).
#   test_configure_telemetry_falls_back_when_otel_missing - Verifies disabled fallback when opentelemetry packages are missing.
#   test_start_span_yields_none_when_disabled - Verifies start_span yields silently when telemetry is disabled.
#   test_operation_scope_does_not_emit_span_when_telemetry_disabled - Verifies OperationScope stays a pure context-var helper when telemetry is off.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Phase 4.15: introduced unit coverage for the telemetry facade covering disabled state, missing-package fallback, no-op start_span, and OperationScope bridge inactivity]
# END_CHANGE_SUMMARY

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from core.config import CoreSettings
from core.observability import operation_scope
from core.services.telemetry import (
    configure_telemetry,
    get_active_state,
    reset_active_state,
    start_span,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_telemetry() -> None:
    reset_active_state()
    yield
    reset_active_state()


def _make_settings(tmp_path: Path, **overrides: object) -> CoreSettings:
    settings = CoreSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        **overrides,
    )
    settings.ensure_directories()
    return settings


def test_configure_telemetry_disabled_when_flag_off(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, otel_enabled=False)

    state = configure_telemetry(settings)

    assert state.enabled is False
    assert state.tracer is None
    assert get_active_state() is state


def test_configure_telemetry_falls_back_when_otel_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _make_settings(tmp_path, otel_enabled=True)

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ImportError("simulated missing opentelemetry package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    state = configure_telemetry(settings)

    assert state.enabled is False
    assert state.tracer is None


def test_start_span_yields_none_when_disabled(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, otel_enabled=False)
    configure_telemetry(settings)

    with start_span("test.operation", attributes={"k": "v"}) as span:
        assert span is None


def test_operation_scope_does_not_emit_span_when_telemetry_disabled(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, otel_enabled=False)
    configure_telemetry(settings)

    with operation_scope("test.scope") as scope:
        assert scope.operation == "test.scope"
        assert scope._span is None
