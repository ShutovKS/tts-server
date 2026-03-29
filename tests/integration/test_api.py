from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.application import TTSApplicationService
from core.services.tts_service import TTSService
from server.app import create_app
from server.bootstrap import ServerSettings
from tests.support.api_fakes import (
    BusyTTSService,
    DummyRegistry,
    DummyTTSService,
    FailingTTSService,
    extract_json_logs,
    make_wav_bytes,
)


pytestmark = pytest.mark.integration


class StubRegistry:
    def get_model(self, model_name=None, mode=None):
        from core.models.catalog import MODEL_SPECS

        spec = next(spec for spec in MODEL_SPECS.values() if spec.mode == (mode or "clone"))
        return spec, object()


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        enable_streaming=True,
        default_save_output=False,
    )
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    settings.voices_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    app = create_app(settings)
    app.state.registry = DummyRegistry(settings)
    app.state.tts_service = DummyTTSService(settings)
    app.state.application = DummyTTSService(settings)

    with TestClient(app) as test_client:
        yield test_client


def test_liveness_endpoint(client: TestClient):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models_endpoint(client: TestClient):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["data"]) == 2
    assert payload["data"][0]["available"] is True


def test_openai_speech_returns_audio(client: TestClient):
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
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["x-model-id"] == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    assert response.headers["x-request-id"]
    assert response.content.startswith(b"RIFF")


def test_openai_speech_pcm_returns_binary_pcm(client: TestClient):
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "input": "Hello world",
            "voice": "Vivian",
            "response_format": "pcm",
            "speed": 1.0,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/pcm")
    assert not response.content.startswith(b"RIFF")


def test_custom_tts_happy_path(client: TestClient):
    response = client.post(
        "/api/v1/tts/custom",
        json={
            "text": "Hello custom",
            "speaker": "Vivian",
            "emotion": "Happy",
            "speed": 1.1,
            "save_output": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["x-saved-output-path"].endswith("saved_custom.wav")


def test_design_tts_happy_path(client: TestClient):
    response = client.post(
        "/api/v1/tts/design",
        json={
            "text": "Hello design",
            "voice_description": "calm narrator",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")


def test_clone_tts_happy_path(client: TestClient):
    files = {"ref_audio": ("reference.wav", make_wav_bytes(), "audio/wav")}
    data = {"text": "Clone this", "ref_text": "Clone this"}
    response = client.post("/api/v1/tts/clone", data=data, files=files)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")


def test_clone_upload_too_large_returns_json_error(client: TestClient):
    object.__setattr__(client.app.state.settings, "max_upload_size_bytes", 8)
    files = {"ref_audio": ("reference.wav", make_wav_bytes(), "audio/wav")}
    data = {"text": "Clone this"}
    response = client.post("/api/v1/tts/clone", data=data, files=files)
    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == "upload_too_large"
    assert payload["request_id"]


def test_validation_error_uses_unified_error_format(client: TestClient):
    response = client.post(
        "/api/v1/tts/custom",
        json={"text": "   ", "speaker": "Vivian"},
    )
    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "validation_error"
    assert payload["request_id"]


def test_model_load_error_uses_centralized_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
    )
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    app = create_app(settings)
    app.state.registry = DummyRegistry(settings)
    app.state.tts_service = FailingTTSService(settings)
    app.state.application = FailingTTSService(settings)

    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/audio/speech",
            json={
                "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                "input": "Hello world",
                "voice": "Vivian",
                "response_format": "wav",
                "speed": 1.0,
            },
        )

    assert response.status_code == 500
    payload = response.json()
    assert payload["code"] == "model_load_failed"
    assert payload["message"] == "Failed to load model"
    assert payload["details"]["reason"] == "mlx runtime failed"
    assert payload["details"]["runtime_dependency"] == "mlx_audio"


def test_inference_busy_error_preserves_status_and_details(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        inference_busy_status_code=429,
    )
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    app = create_app(settings)
    app.state.registry = DummyRegistry(settings)
    app.state.tts_service = BusyTTSService(settings)
    app.state.application = BusyTTSService(settings)

    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/audio/speech",
            json={
                "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                "input": "Hello world",
                "voice": "Vivian",
                "response_format": "wav",
                "speed": 1.0,
            },
        )

    assert response.status_code == 429
    payload = response.json()
    assert payload["code"] == "inference_busy"
    assert payload["details"]["reason"] == "Inference is already in progress"
    assert payload["details"]["queue_depth"] == 1


def test_request_logging_includes_request_id_and_endpoint_context(client: TestClient, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)

    response = client.post(
        "/v1/audio/speech",
        headers={"x-request-id": "req-123"},
        json={
            "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "input": "Hello world",
            "voice": "Vivian",
            "response_format": "wav",
            "speed": 1.0,
        },
    )

    assert response.status_code == 200
    started_logs = extract_json_logs(caplog, "http.request.started")
    completed_logs = extract_json_logs(caplog, "http.request.completed")
    endpoint_logs = extract_json_logs(caplog, "tts.endpoint.started")
    audio_logs = extract_json_logs(caplog, "http.audio_response.ready")

    assert any(item["request_id"] == "req-123" and item["path"] == "/v1/audio/speech" for item in started_logs)
    assert any(item["request_id"] == "req-123" and item["status_code"] == 200 for item in completed_logs)
    assert any(item["request_id"] == "req-123" and item["endpoint"] == "/v1/audio/speech" for item in endpoint_logs)
    assert any(item["request_id"] == "req-123" and item["model"] == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit" for item in audio_logs)


def test_clone_endpoint_returns_controlled_error_when_generation_artifact_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        enable_streaming=True,
        default_save_output=False,
    )
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    def fake_generate_audio(**kwargs):
        return None

    monkeypatch.setattr("core.services.tts_service.generate_audio", fake_generate_audio)

    app = create_app(settings)
    app.state.registry = StubRegistry()
    app.state.tts_service = TTSService(registry=StubRegistry(), settings=settings)
    app.state.application = TTSApplicationService(tts_service=app.state.tts_service)
    with TestClient(app) as test_client:
        files = {"ref_audio": ("reference.wav", make_wav_bytes(), "audio/wav")}
        data = {"text": "Clone this", "ref_text": "Clone this"}
        response = test_client.post("/api/v1/tts/clone", data=data, files=files)

    assert response.status_code == 500
    payload = response.json()
    assert payload["code"] == "generation_failed"
    assert payload["message"] == "Audio generation failed"
    assert payload["details"]["reason"].startswith("Generated audio file not found")
    assert payload["details"]["failure_kind"] == "missing_artifact"
    assert payload["request_id"]
