# FILE: tests/integration/test_backend_api_metadata.py
# VERSION: 1.0.0
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
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT and MODULE_MAP]
# END_CHANGE_SUMMARY

from __future__ import annotations

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
        backend="torch",
        backend_autoselect=True,
        enable_streaming=True,
        default_save_output=False,
    )
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    app = create_app(settings)
    app.state.registry = DummyRegistry(settings)
    app.state.tts_service = DummyTTSService(settings)
    app.state.application = DummyTTSService(settings)

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
    assert response.headers["x-backend-id"] == "mlx"


def test_readiness_report_exposes_backend_configuration(client: TestClient):
    response = client.get("/health/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["checks"]["runtime"]["configured_backend"] == "torch"
    assert payload["checks"]["runtime"]["backend_autoselect"] is True
    assert payload["checks"]["models"]["selected_backend"] == "mlx"
    assert payload["checks"]["models"]["routing"]["mixed_backend_routing"] is True
    assert payload["checks"]["models"]["host"]["platform_system"] == "darwin"
    assert payload["checks"]["models"]["available_backends"][0]["key"] == "mlx"
    assert any(
        item["key"] == "qwen_fast"
        and item["diagnostics"]["reason"] == "platform_unsupported"
        for item in payload["checks"]["models"]["available_backends"]
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
    assert qwen_model["capabilities"]["supports_clone"] is True
    assert qwen_model["route"]["candidates"][0]["key"] == "qwen_fast"
    assert (
        qwen_model["route"]["candidates"][0]["route_reason"] == "platform_unsupported"
    )
    assert piper_model["backend"] == "onnx"
    assert piper_model["selected_backend"] == "mlx"
    assert piper_model["execution_backend"] == "onnx"
    assert piper_model["family_key"] == "piper"
    assert piper_model["capabilities_supported"] == ["preset_speaker_tts"]
    assert (
        piper_model["route"]["route_reason"]
        == "selected_backend_incompatible_with_model"
    )
