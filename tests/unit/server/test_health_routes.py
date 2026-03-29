from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from server.api.routes_health import build_readiness_report
from server.bootstrap import ServerSettings
from tests.support.api_fakes import DegradedRegistry, DummyRegistry


pytestmark = pytest.mark.unit


def _make_request(settings: ServerSettings, registry) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                registry=registry,
                settings=settings,
                runtime=SimpleNamespace(
                    inference_guard=SimpleNamespace(is_busy=lambda: False),
                ),
            )
        )
    )


def test_build_readiness_report_returns_deep_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        enable_streaming=True,
        default_save_output=False,
        sample_rate=24000,
        max_upload_size_bytes=25 * 1024 * 1024,
        request_timeout_seconds=300,
    )
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    report = build_readiness_report(_make_request(settings, DummyRegistry(settings)))

    assert report.status == "ok"
    assert report.checks["models"]["runtime_ready_models"] == 2
    assert report.checks["models"]["items"][0]["runtime_ready"] is True
    assert report.checks["ffmpeg"]["available"] is True
    assert report.checks["config"]["models_dir_exists"] is True
    assert report.checks["runtime"]["streaming_enabled"] is True
    assert report.checks["runtime"]["configured_backend"] is None
    assert report.checks["runtime"]["backend_autoselect"] is True


def test_build_readiness_report_returns_degraded_status_when_runtime_not_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
    )
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: False)

    report = build_readiness_report(_make_request(settings, DegradedRegistry(settings)))

    assert report.status == "degraded"
    assert report.checks["models"]["registry_ready"] is False
    assert report.checks["ffmpeg"]["available"] is False
    assert report.checks["models"]["items"][0]["missing_artifacts"] == ["config.json"]
