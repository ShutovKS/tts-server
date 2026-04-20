# FILE: tests/unit/server/test_health_routes.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for server readiness and health route reporting.
#   SCOPE: Readiness reports, model diagnostics, runtime status aggregation
#   DEPENDS: M-SERVER
#   LINKS: V-M-SERVER
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _make_request - Helper that builds a readiness-report request stub with metrics and runtime state
#   test_build_readiness_report_returns_deep_diagnostics - Verifies readiness reports include model, runtime, and metrics diagnostics
#   test_build_readiness_report_returns_degraded_status_when_runtime_not_ready - Verifies degraded readiness is reported when ffmpeg or registry checks fail
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT and MODULE_MAP]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
                metrics=SimpleNamespace(
                    readiness_summary=lambda: {
                        "execution": {
                            "submitted": 1,
                            "started": 1,
                            "completed": 1,
                            "failed": 0,
                            "timeout": 0,
                            "cancelled": 0,
                            "queue_depth": {"current": 0, "peak": 1},
                        },
                        "models": {
                            "cache": {"hit": {"mlx": 1}, "miss": {"mlx": 1}},
                            "load": {
                                "failures": {},
                                "duration_ms": {
                                    "mlx": {
                                        "count": 1,
                                        "avg_ms": 1.0,
                                        "max_ms": 1.0,
                                        "last_ms": 1.0,
                                    }
                                },
                            },
                        },
                    }
                ),
                runtime=SimpleNamespace(
                    inference_guard=SimpleNamespace(is_busy=lambda: False),
                ),
            )
        )
    )


def test_build_readiness_report_returns_deep_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        enable_streaming=True,
        default_save_output=False,
        sample_rate=24000,
        max_upload_size_bytes=25 * 1024 * 1024,
        request_timeout_seconds=300,
        model_preload_policy="listed",
        model_preload_ids=("Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",),
    )
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    report = build_readiness_report(_make_request(settings, DummyRegistry(settings)))

    assert report.status == "ok"
    assert report.checks["models"]["runtime_ready_models"] == 3
    assert report.checks["models"]["loaded_models"] == 1
    assert report.checks["models"]["preload"]["status"] == "completed"
    assert report.checks["models"]["cache_diagnostics"]["cached_model_count"] == 1
    assert report.checks["models"]["items"][0]["runtime_ready"] is True
    assert report.checks["models"]["items"][0]["preload"]["status"] == "loaded"
    assert report.checks["models"]["items"][0]["cache"]["loaded"] is True
    assert report.checks["models"]["routing"]["mixed_backend_routing"] is True
    assert report.checks["models"]["host"]["platform_system"] == "darwin"
    assert any(
        item["key"] == "qwen_fast"
        and item["diagnostics"]["reason"] == "platform_unsupported"
        for item in report.checks["models"]["available_backends"]
    )
    assert report.checks["ffmpeg"]["available"] is True
    assert report.checks["config"]["models_dir_exists"] is True
    assert report.checks["config"]["model_preload_policy"] == "listed"
    assert report.checks["config"]["model_preload_ids"] == [
        "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    ]
    assert report.checks["runtime"]["streaming_enabled"] is True
    assert report.checks["runtime"]["configured_backend"] is None
    assert report.checks["runtime"]["backend_autoselect"] is True
    assert report.checks["runtime"]["runtime_capability_map"] == {
        "family": None,
        "custom_model": None,
        "design_model": None,
        "clone_model": None,
    }
    assert report.checks["capabilities"]["capability_status"]["clone"]["bound"] is False
    assert report.checks["capabilities"]["capability_status"]["clone"]["reason"] == "runtime_binding_missing"
    assert any(
        candidate["key"] == "qwen_fast"
        for item in report.checks["models"]["items"]
        for candidate in item.get("route", {}).get("candidates", [])
    )
    assert report.checks["runtime"]["metrics"]["execution"]["submitted"] == 1
    assert (
        report.checks["models"]["metrics"]["operational"]["execution"]["completed"] == 1
    )


def test_build_readiness_report_returns_degraded_status_when_runtime_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
    )
    settings.ensure_directories()
    monkeypatch.setattr(
        "server.api.routes_health.check_ffmpeg_available", lambda: False
    )

    report = build_readiness_report(_make_request(settings, DegradedRegistry(settings)))

    assert report.status == "degraded"
    assert report.checks["models"]["registry_ready"] is False
    assert report.checks["models"]["routing"]["degraded_routes"] == 1
    assert report.checks["models"]["preload"]["status"] == "failed"
    assert report.checks["ffmpeg"]["available"] is False
    assert report.checks["models"]["items"][0]["missing_artifacts"] == ["config.json"]
    assert report.checks["capabilities"]["capability_status"]["custom"]["bound"] is False
