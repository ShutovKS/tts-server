# FILE: tests/integration/test_backend_api_metadata.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Integration tests for backend metadata exposure in the HTTP API.
#   SCOPE: Backend headers, readiness reporting, model metadata responses
#   DEPENDS: M-SERVER
#   LINKS: V-M-SERVER
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   client - Fixture that builds a test client with explicit backend settings
#   test_audio_response_includes_backend_header - Verifies audio responses expose selected backend metadata
#   test_readiness_report_exposes_backend_configuration - Verifies readiness payload reports configured and selected backends
#   test_models_endpoint_exposes_backend_and_capabilities - Verifies model listing exposes backend and capability metadata
#   test_models_endpoint_exposes_route_candidates_for_design_mode - Verifies design-capable models expose mixed-backend routing diagnostics
#   test_models_endpoint_exposes_route_candidates_for_clone_mode - Verifies clone-capable models expose mixed-backend routing diagnostics
#   test_clone_readiness_prefers_selected_backend_route - Verifies clone readiness keeps selected-backend routing semantics for the active host/backend combination
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Updated backend metadata expectations so qwen_fast clone routing reflects full-mode support]
# END_CHANGE_SUMMARY

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.bootstrap import ServerSettings
from tests.support.api_fakes import DummyRegistry, DummyTTSService


pytestmark = pytest.mark.integration


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        upload_staging_dir=tmp_path / ".uploads",
        active_family="qwen",
        default_custom_model="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        default_design_model="Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        default_clone_model="Qwen3-TTS-12Hz-1.7B-Base-8bit",
        backend=None,
        backend_autoselect=True,
        enable_streaming=True,
        default_save_output=False,
    )
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    app = create_app(settings)
    object.__setattr__(app.state.settings, "backend", "torch")
    app.state.registry = DummyRegistry(settings)
    app.state.tts_service = DummyTTSService(settings)
    app.state.application = DummyTTSService(settings)
    object.__setattr__(
        app.state.job_execution.manager.executor,
        "application_service",
        app.state.application,
    )
    app.state.metrics = SimpleNamespace(
        readiness_summary=lambda: {
            "execution": {
                "submitted": 0,
                "started": 0,
                "completed": 0,
                "failed": 0,
                "timeout": 0,
                "cancelled": 0,
                "queue_depth": {"current": 0, "peak": 0},
            },
            "models": {
                "cache": {"hit": {"mlx": 1}, "miss": {"mlx": 1}},
                "load": {"failures": {}, "duration_ms": {}},
            },
        }
    )

    with TestClient(app) as test_client:
        yield test_client


def test_audio_response_includes_backend_header(client: TestClient):
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "input": "Hello world",
            "voice": "Vivian",
            "response_format": "wav",
            "speed": 1.0,
        },
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"]
    assert response.headers["x-model-id"] == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    assert response.headers["x-tts-mode"] == "custom"
    assert response.headers["x-backend-id"] == "mlx"


def test_readiness_report_exposes_backend_configuration(client: TestClient):
    response = client.get("/health/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["checks"]["runtime"]["configured_backend"] == "torch"
    assert payload["checks"]["runtime"]["backend_autoselect"] is True
    assert payload["checks"]["runtime"]["runtime_capability_map"] == {
        "family": "qwen",
        "custom_model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "design_model": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        "clone_model": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
    }
    assert payload["checks"]["models"]["selected_backend"] == "mlx"
    assert (
        payload["checks"]["models"]["backend_selection"]["selection_reason"]
        == "platform_and_runtime_match"
    )
    assert payload["checks"]["models"]["backend_diagnostics"]["backend"] == "mlx"
    assert payload["checks"]["models"]["backend_diagnostics"]["ready"] is True
    assert payload["checks"]["models"]["backend_capabilities"]["supports_clone"] is True
    assert payload["checks"]["models"]["routing"]["mixed_backend_routing"] is True
    assert payload["checks"]["models"]["routing"]["per_model_backend_overrides"] == 1
    assert payload["checks"]["models"]["family_summary"]["qwen3_tts"] == {
        "family": "Qwen3-TTS",
        "configured_models": 3,
        "available_models": 3,
        "runtime_ready_models": 3,
    }
    assert payload["checks"]["models"]["cache_diagnostics"]["cached_model_count"] == 1
    assert (
        payload["checks"]["models"]["preload"]["loaded_model_ids"]
        == ["Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"]
    )
    assert payload["checks"]["models"]["host"]["platform_system"] == "darwin"
    assert payload["checks"]["models"]["available_backends"][0]["key"] == "mlx"
    assert payload["checks"]["models"]["available_backends"][0]["selected"] is True
    assert payload["checks"]["capabilities"]["capability_status"]["custom"]["bound"] is True
    assert payload["checks"]["capabilities"]["capability_status"]["custom"]["bound_model"] == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    assert payload["checks"]["capabilities"]["capability_status"]["clone"]["runtime_ready"] is True
    assert any(
        item["key"] == "qwen_fast"
        and item["diagnostics"]["reason"] == "platform_unsupported"
        for item in payload["checks"]["models"]["available_backends"]
    )
    assert any(
        item["id"] == "Piper-en_US-lessac-medium"
        and item["backend"] == "onnx"
        and item["available"] is False
        and item["runtime_ready"] is False
        for item in payload["checks"]["models"]["items"]
    )


def test_models_endpoint_exposes_backend_and_capabilities(client: TestClient):
    response = client.get("/api/v1/models")

    assert response.status_code == 200
    payload = response.json()["data"]
    qwen_model = next(
        item for item in payload if item["id"] == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    )
    piper_model = next(
        item for item in payload if item["id"] == "Piper-en_US-lessac-medium"
    )

    assert qwen_model["backend"] == "mlx"
    assert qwen_model["backend_support"] == ["mlx", "qwen_fast", "torch"]
    assert qwen_model["selected_backend"] == "mlx"
    assert qwen_model["execution_backend"] == "mlx"
    assert qwen_model["selected_backend_label"] == "MLX Apple Silicon"
    assert qwen_model["execution_backend_label"] == "MLX Apple Silicon"
    assert qwen_model["capabilities"]["supports_clone"] is True
    assert qwen_model["runtime_ready"] is True
    assert qwen_model["missing_artifacts"] == []
    assert qwen_model["route"]["candidates"][0]["key"] == "qwen_fast"
    assert (
        qwen_model["route"]["candidates"][0]["route_reason"] == "platform_unsupported"
    )
    assert qwen_model["route"]["candidates"][0]["diagnostics"]["backend"] == "qwen_fast"
    assert qwen_model["route"]["candidates"][0]["diagnostics"]["details"]["enabled"] is True
    assert piper_model["backend"] == "onnx"
    assert piper_model["selected_backend"] == "mlx"
    assert piper_model["execution_backend"] == "onnx"
    assert piper_model["selected_backend_label"] == "MLX Apple Silicon"
    assert piper_model["execution_backend_label"] == "ONNX Runtime"
    assert piper_model["family_key"] == "piper"
    assert piper_model["capabilities_supported"] == ["preset_speaker_tts"]
    assert piper_model["capabilities"]["supports_clone"] is False
    assert piper_model["capabilities"]["supports_voice_description_tts"] is False
    assert piper_model["runtime_ready"] is False
    assert piper_model["missing_artifacts"] == ["model.onnx", "model.onnx.json"]
    assert (
        piper_model["route"]["route_reason"]
        == "selected_backend_incompatible_with_model"
    )
    assert piper_model["route"]["routing_mode"] == "per_model_backend_override"
    assert piper_model["route"]["selected_backend_ready_for_model"] is False


def test_models_endpoint_exposes_route_candidates_for_clone_mode(client: TestClient):
    response = client.get("/api/v1/models")

    assert response.status_code == 200
    payload = response.json()["data"]
    clone_model = next(
        item for item in payload if item["id"] == "Qwen3-TTS-12Hz-1.7B-Base-8bit"
    )

    assert clone_model["mode"] == "clone"
    assert clone_model["selected_backend"] == "mlx"
    assert clone_model["execution_backend"] == "mlx"
    assert clone_model["route"]["route_reason"] == "selected_backend_supports_model"
    assert clone_model["route"]["routing_mode"] == "selected_backend"
    assert clone_model["route"]["candidates"][0]["key"] == "qwen_fast"
    assert clone_model["route"]["candidates"][0]["supports_mode"] is True
    assert clone_model["route"]["candidates"][0]["route_reason"] == "unsupported_platform"
    assert clone_model["capabilities"]["supports_clone"] is True
    assert clone_model["runtime_ready"] is True


def test_models_endpoint_exposes_route_candidates_for_design_mode(client: TestClient):
    response = client.get("/api/v1/models")

    assert response.status_code == 200
    payload = response.json()["data"]
    design_model = next(
        item for item in payload if item["id"] == "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"
    )

    assert design_model["mode"] == "design"
    assert design_model["selected_backend"] == "mlx"
    assert design_model["execution_backend"] == "mlx"
    assert design_model["route"]["route_reason"] == "selected_backend_supports_model"
    assert design_model["route"]["routing_mode"] == "selected_backend"
    assert design_model["route"]["candidates"][0]["key"] == "qwen_fast"
    assert design_model["route"]["candidates"][0]["supports_mode"] is True
    assert design_model["route"]["candidates"][0]["route_reason"] == "unsupported_platform"
    assert design_model["capabilities"]["supports_voice_description_tts"] is True
    assert design_model["runtime_ready"] is True


def test_clone_readiness_prefers_selected_backend_route(client: TestClient):
    response = client.get("/health/ready")

    assert response.status_code == 200
    payload = response.json()
    clone_model = next(
        item
        for item in payload["checks"]["models"]["items"]
        if item["id"] == "Qwen3-TTS-12Hz-1.7B-Base-8bit"
    )

    assert clone_model["mode"] == "clone"
    assert clone_model["execution_backend"] == "mlx"
    assert clone_model["route"]["routing_mode"] == "selected_backend"
    assert clone_model["route"]["route_reason"] == "selected_backend_supports_model"
    assert clone_model["route"]["candidates"][0]["key"] == "qwen_fast"
    assert clone_model["route"]["candidates"][0]["supports_mode"] is True
    assert clone_model["route"]["candidates"][0]["route_reason"] == "unsupported_platform"
    assert clone_model["route"]["candidates"][0]["diagnostics"]["backend"] == "qwen_fast"
    assert any(
        backend["key"] == "qwen_fast"
        and backend["diagnostics"]["reason"] == "platform_unsupported"
        for backend in payload["checks"]["models"]["available_backends"]
    )
