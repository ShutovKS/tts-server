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
    assert payload["checks"]["models"]["available_backends"][0]["key"] == "mlx"


def test_models_endpoint_exposes_backend_and_capabilities(client: TestClient):
    response = client.get("/api/v1/models")

    assert response.status_code == 200
    model = response.json()["data"][0]
    assert model["backend"] == "mlx"
    assert model["capabilities"]["supports_clone"] is True
