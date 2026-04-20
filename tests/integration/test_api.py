# FILE: tests/integration/test_api.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Integration tests for HTTP API behavior and endpoint responses.
#   SCOPE: Health checks, model listing, speech endpoints, request handling
#   DEPENDS: M-SERVER
#   LINKS: V-M-SERVER
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _state - Helper that exposes mutable app state from a TestClient
#   ContentionTTSService - Test double that coordinates concurrent request contention scenarios
#   StubRegistry - Minimal registry stub for integration tests
#   client - Fixture that builds a deterministic FastAPI test client with fake runtime services
#   Synchronous endpoint tests - Verify liveness, models, OpenAI/custom/design/clone happy paths, validation, and upload handling
#   Error mapping tests - Verify model load, missing model, busy inference, timeout, and queue-full errors use unified responses
#   Async job flow tests - Verify async submit, status, result, cancel, idempotency, and clone staging behavior
#   Auth and ownership tests - Verify static bearer protection and job owner isolation across async endpoints
#   Contention and quota tests - Verify concurrent reads, cancellations, owner isolation, and quota enforcement remain deterministic
#   Observability tests - Verify request, inference, timeout, and failure logs include expected structured context
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT and MODULE_MAP]
# END_CHANGE_SUMMARY

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from time import sleep, time
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from core.application import TTSApplicationService
from core.services.tts_service import TTSService
from core.infrastructure.admission_control_local import (
    build_quota_guard,
    build_rate_limiter,
)
from server.app import create_app
from server.bootstrap import ServerSettings
from tests.support.api_fakes import (
    BusyTTSService,
    DummyRegistry,
    DummyTTSService,
    FailingTTSService,
    MissingModeTTSService,
    MissingModelTTSService,
    SlowTTSService,
    WorkerFailingTTSService,
    extract_json_logs,
    make_wav_bytes,
)


pytestmark = pytest.mark.integration


def _state(client: TestClient) -> Any:
    return cast(Any, client.app).state


class ContentionTTSService(DummyTTSService):
    def __init__(
        self,
        settings: ServerSettings,
        *,
        release_after_calls: int,
        sleep_seconds: float = 0.0,
    ):
        super().__init__(settings)
        self.release_after_calls = release_after_calls
        self.sleep_seconds = sleep_seconds
        self.started = Event()
        self.release = Event()
        self._lock = Lock()
        self.calls_started = 0

    def synthesize_custom(self, request):
        with self._lock:
            self.calls_started += 1
            current_calls = self.calls_started
            self.started.set()
            if current_calls >= self.release_after_calls:
                self.release.set()
        self.release.wait(timeout=2.0)
        if self.sleep_seconds > 0:
            sleep(self.sleep_seconds)
        return super().synthesize_custom(request)


class StubRegistry:
    @property
    def backend(self):
        return type("BackendStub", (), {"key": "torch", "label": "PyTorch + Transformers"})()

    def get_model_spec(self, model_name=None, mode=None):
        from core.models.catalog import MODEL_SPECS

        if model_name is not None:
            return next(
                spec
                for spec in MODEL_SPECS.values()
                if model_name in {spec.api_name, spec.folder, spec.key, spec.model_id}
            )
        return next(spec for spec in MODEL_SPECS.values() if spec.mode == (mode or "clone"))

    def get_model(self, model_name=None, mode=None):
        spec = self.get_model_spec(model_name=model_name, mode=mode)
        return spec, type("HandleStub", (), {"backend_key": "torch", "spec": spec})()

    def backend_for_spec(self, spec):
        return self.backend

    def backend_route_for_spec(self, spec):
        return {
            "route_reason": "registry_model_resolution",
            "execution_backend": self.backend.key,
        }


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        upload_staging_dir=tmp_path / ".uploads",
        enable_streaming=True,
        default_save_output=False,
        max_input_text_chars=32,
    )
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    settings.voices_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    app = create_app(settings)
    app.state.registry = DummyRegistry(settings)
    app.state.tts_service = DummyTTSService(settings)
    app.state.application = DummyTTSService(settings)
    object.__setattr__(
        app.state.job_execution.manager.executor,
        "application_service",
        app.state.application,
    )

    def reset_admission_state() -> None:
        app.state.rate_limiter = build_rate_limiter(app.state.settings)
        app.state.quota_guard = build_quota_guard(
            app.state.settings, store=app.state.job_store
        )

    app.state.reset_admission_state = reset_admission_state

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
    assert len(payload["data"]) == 4
    assert payload["data"][0]["available"] is True
    qwen_model = next(
        item for item in payload["data"] if item["id"] == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    )
    design_model = next(
        item for item in payload["data"] if item["id"] == "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"
    )
    piper_model = next(
        item for item in payload["data"] if item["id"] == "Piper-en_US-lessac-medium"
    )
    assert qwen_model["profile"]["key"] == "qwen"
    assert qwen_model["profile"]["pack_refs"]["family"] == ["qwen"]
    assert design_model["profile"]["key"] == "qwen"
    assert piper_model["profile"]["key"] == "piper"
    assert piper_model["profile"]["isolated_env_name"] == "piper"


def test_ready_endpoint_exposes_family_profile_metadata(client: TestClient):
    response = client.get("/health/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["checks"]["models"]["family_profiles"]["qwen3_tts"]["key"] == "qwen"
    assert payload["checks"]["models"]["family_profiles"]["piper"]["key"] == "piper"


def test_design_tts_rejects_model_without_design_capability(client: TestClient):
    response = client.post(
        "/api/v1/tts/design",
        json={
            "text": "Hello design",
            "voice_description": "calm narrator",
            "model": "Piper-en_US-lessac-medium",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "model_capability_not_supported"
    assert (
        payload["details"]["reason"]
        == "Model 'Piper-en_US-lessac-medium' does not support capability 'voice_description_tts'"
    )
    assert payload["details"]["model"] == "Piper-en_US-lessac-medium"
    assert payload["details"]["capability"] == "voice_description_tts"


def test_clone_async_submit_rejects_model_without_clone_capability(
    client: TestClient,
):
    files = {"ref_audio": ("reference.wav", make_wav_bytes(), "audio/wav")}
    data = {
        "text": "Clone this",
        "ref_text": "Clone this",
        "model": "Piper-en_US-lessac-medium",
    }

    response = client.post("/api/v1/tts/clone/jobs", data=data, files=files)

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "model_capability_not_supported"
    assert (
        payload["details"]["reason"]
        == "Model 'Piper-en_US-lessac-medium' does not support capability 'reference_voice_clone'"
    )
    assert payload["details"]["model"] == "Piper-en_US-lessac-medium"
    assert payload["details"]["capability"] == "reference_voice_clone"


def test_sync_rate_limit_returns_unified_error_with_retry_after(client: TestClient):
    object.__setattr__(_state(client).settings, "rate_limit_enabled", True)
    object.__setattr__(_state(client).settings, "rate_limit_sync_tts_per_minute", 1)
    _state(client).reset_admission_state()

    first = client.post(
        "/v1/audio/speech",
        json={
            "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "input": "Hello world",
            "voice": "Vivian",
            "response_format": "wav",
            "speed": 1.0,
        },
    )
    second = client.post(
        "/v1/audio/speech",
        json={
            "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "input": "Hello world again",
            "voice": "Vivian",
            "response_format": "wav",
            "speed": 1.0,
        },
    )

    assert first.status_code == 200
    assert second.status_code == 429
    payload = second.json()
    assert payload["code"] == "rate_limit_exceeded"
    assert payload["details"]["reason"] == "Request rate limit was exceeded"
    assert payload["details"]["policy"] == "sync_tts"
    assert second.headers["Retry-After"]


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
    assert response.headers["content-length"] == str(len(response.content))
    assert "transfer-encoding" not in response.headers
    assert response.content.startswith(b"RIFF")


def test_openai_speech_uses_runtime_binding_when_model_is_omitted(client: TestClient):
    object.__setattr__(
        _state(client).settings,
        "default_custom_model",
        "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    )

    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "Hello world",
            "voice": "Vivian",
            "response_format": "wav",
            "speed": 1.0,
        },
    )

    assert response.status_code == 200
    assert response.headers["x-model-id"] == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"


def test_clone_endpoint_rejects_when_runtime_clone_capability_is_unbound(client: TestClient):
    object.__setattr__(_state(client).settings, "default_clone_model", None)
    _state(client).tts_service = TTSService(
        registry=StubRegistry(), settings=_state(client).settings
    )
    _state(client).application = TTSApplicationService(tts_service=_state(client).tts_service)
    object.__setattr__(
        _state(client).job_execution.manager.executor,
        "application_service",
        _state(client).application,
    )
    files = {"ref_audio": ("reference.wav", make_wav_bytes(), "audio/wav")}
    data = {"text": "Clone this", "ref_text": "Clone this"}

    response = client.post("/api/v1/tts/clone", data=data, files=files)

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "runtime_capability_not_configured"
    assert payload["details"]["capability"] == "reference_voice_clone"
    assert payload["details"]["execution_mode"] == "clone"


def test_openai_speech_passes_language_to_application(client: TestClient):
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "input": "Hello world",
            "voice": "Vivian",
            "language": "RU",
            "response_format": "wav",
            "speed": 1.0,
        },
    )
    assert response.status_code == 200
    assert _state(client).application.last_clone_request is None


def test_clone_tts_passes_language_to_application(client: TestClient):
    files = {"ref_audio": ("reference.wav", make_wav_bytes(), "audio/wav")}
    data = {"text": "Clone this", "ref_text": "Clone this", "language": "Ru"}
    response = client.post("/api/v1/tts/clone", data=data, files=files)
    assert response.status_code == 200
    assert _state(client).application.last_clone_request is not None
    assert _state(client).application.last_clone_request.language == "ru"


def test_openai_speech_ignores_streaming_flag_for_materialized_audio(
    client: TestClient,
):
    assert _state(client).settings.enable_streaming is True

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
    assert response.headers["content-length"] == str(len(response.content))
    assert "transfer-encoding" not in response.headers
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
    assert response.headers["x-saved-output-file"] == "saved_custom.wav"
    assert "x-saved-output-path" not in response.headers


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
    object.__setattr__(_state(client).settings, "max_upload_size_bytes", 8)
    files = {"ref_audio": ("reference.wav", make_wav_bytes(), "audio/wav")}
    data = {"text": "Clone this"}
    response = client.post("/api/v1/tts/clone", data=data, files=files)
    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == "upload_too_large"
    assert payload["request_id"]


def test_clone_upload_rejects_unsupported_media_type(client: TestClient):
    files = {"ref_audio": ("reference.txt", b"not-audio", "text/plain")}
    data = {"text": "Clone this"}
    response = client.post("/api/v1/tts/clone", data=data, files=files)
    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == "unsupported_upload_media_type"
    assert payload["details"]["field"] == "ref_audio"


def test_clone_upload_rejects_empty_audio_body(client: TestClient):
    files = {"ref_audio": ("reference.wav", b"", "audio/wav")}
    data = {"text": "Clone this"}
    response = client.post("/api/v1/tts/clone", data=data, files=files)
    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == "invalid_upload_audio"
    assert payload["details"]["field"] == "ref_audio"


def test_openai_speech_overlong_text_returns_unified_error(client: TestClient):
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "input": "x" * 33,
            "voice": "Vivian",
            "response_format": "wav",
            "speed": 1.0,
        },
    )
    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "validation_error"
    assert payload["details"]["errors"][0]["loc"] == ["body", "input"]
    assert (
        payload["details"]["errors"][0]["msg"] == "input must be at most 32 characters"
    )


def test_custom_tts_overlong_text_returns_unified_error(client: TestClient):
    response = client.post(
        "/api/v1/tts/custom",
        json={
            "text": "x" * 33,
            "speaker": "Vivian",
        },
    )
    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "validation_error"
    assert payload["details"]["errors"][0]["loc"] == ["body", "text"]
    assert (
        payload["details"]["errors"][0]["msg"] == "text must be at most 32 characters"
    )


def test_clone_tts_overlong_text_returns_unified_error(client: TestClient):
    files = {"ref_audio": ("reference.wav", make_wav_bytes(), "audio/wav")}
    data = {"text": "x" * 33, "ref_text": "Clone this"}
    response = client.post("/api/v1/tts/clone", data=data, files=files)
    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "validation_error"
    assert payload["details"]["errors"][0]["loc"] == ["body", "text"]
    assert (
        payload["details"]["errors"][0]["msg"] == "text must be at most 32 characters"
    )


def test_validation_error_uses_unified_error_format(client: TestClient):
    response = client.post(
        "/api/v1/tts/custom",
        json={"text": "   ", "speaker": "Vivian"},
    )
    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "validation_error"
    assert payload["request_id"]


def test_model_load_error_uses_centralized_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
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
    object.__setattr__(
        app.state.job_execution.manager.executor,
        "application_service",
        app.state.application,
    )

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


def test_model_not_available_error_uses_centralized_mapping_for_unknown_identifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
    )
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    app = create_app(settings)
    app.state.registry = DummyRegistry(settings)
    app.state.tts_service = MissingModelTTSService(settings)
    app.state.application = MissingModelTTSService(settings)
    object.__setattr__(
        app.state.job_execution.manager.executor,
        "application_service",
        app.state.application,
    )

    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/audio/speech",
            json={
                "model": "unknown-model",
                "input": "Hello world",
                "voice": "Vivian",
                "response_format": "wav",
                "speed": 1.0,
            },
        )

    assert response.status_code == 404
    payload = response.json()
    assert payload["code"] == "model_not_available"
    assert payload["message"] == "Requested model is not available"
    assert (
        payload["details"]["reason"]
        == "Requested model is not available: unknown-model"
    )
    assert payload["details"]["model"] == "unknown-model"
    assert payload["details"]["backend"] == "mlx"


def test_model_not_available_error_uses_centralized_mapping_for_missing_mode_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
    )
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    app = create_app(settings)
    app.state.registry = DummyRegistry(settings)
    app.state.tts_service = MissingModeTTSService(settings)
    app.state.application = MissingModeTTSService(settings)
    object.__setattr__(
        app.state.job_execution.manager.executor,
        "application_service",
        app.state.application,
    )

    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/v1/tts/design",
            json={
                "text": "Hello design",
                "voice_description": "calm narrator",
            },
        )

    assert response.status_code == 404
    payload = response.json()
    assert payload["code"] == "model_not_available"
    assert payload["message"] == "Requested model is not available"
    assert (
        payload["details"]["reason"] == "No local model is available for mode: design"
    )
    assert payload["details"]["mode"] == "design"
    assert payload["details"]["backend"] == "mlx"


def test_inference_busy_error_preserves_status_and_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
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
    object.__setattr__(
        app.state.job_execution.manager.executor,
        "application_service",
        app.state.application,
    )

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


def test_request_timeout_returns_unified_error_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        request_timeout_seconds=0,
    )
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    app = create_app(settings)
    app.state.registry = DummyRegistry(settings)
    app.state.tts_service = SlowTTSService(settings, sleep_seconds=0.05)
    app.state.application = app.state.tts_service
    object.__setattr__(
        app.state.job_execution.manager.executor,
        "application_service",
        app.state.application,
    )

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

    assert response.status_code == 504
    payload = response.json()
    assert payload["code"] == "request_timeout"
    assert payload["message"] == "Inference request timed out"
    assert payload["details"]["reason"] == "Inference request timed out"
    assert payload["details"]["operation"] == "synthesize_custom"
    assert payload["details"]["timeout_seconds"] == 0
    assert payload["request_id"]


def test_async_custom_job_submit_status_result_flow(client: TestClient):
    submit = client.post(
        "/api/v1/tts/custom/jobs",
        json={"text": "Hello async", "speaker": "Vivian", "save_output": True},
    )
    assert submit.status_code == 202
    submit_payload = submit.json()
    assert submit_payload["job_id"]
    assert submit_payload["status"] in {"queued", "running", "succeeded"}
    assert submit_payload["operation"] == "synthesize_custom"
    assert submit_payload["mode"] == "custom"
    assert submit_payload["request_id"]
    assert submit_payload["submit_request_id"]
    assert submit_payload["response_format"] == "wav"
    assert submit_payload["save_output"] is True
    assert submit_payload["backend"] in {None, "mlx"}
    assert submit_payload["created_at"]
    assert submit_payload["status_url"].endswith(
        f"/api/v1/tts/jobs/{submit_payload['job_id']}"
    )
    assert submit_payload["result_url"].endswith(
        f"/api/v1/tts/jobs/{submit_payload['job_id']}/result"
    )
    assert submit_payload["cancel_url"].endswith(
        f"/api/v1/tts/jobs/{submit_payload['job_id']}/cancel"
    )

    job_id = submit_payload["job_id"]
    status_payload = None
    for _ in range(50):
        status = client.get(f"/api/v1/tts/jobs/{job_id}")
        assert status.status_code == 200
        status_payload = status.json()
        if status_payload["status"] == "succeeded":
            break
    assert status_payload is not None
    assert status_payload["status"] == "succeeded"
    assert status_payload["request_id"]
    assert status_payload["submit_request_id"] == submit_payload["submit_request_id"]
    assert status_payload["backend"] == "mlx"
    assert status_payload["response_format"] == "wav"
    assert status_payload["save_output"] is True
    assert status_payload["terminal_error"] is None
    assert status_payload["created_at"] == submit_payload["created_at"]
    assert status_payload["started_at"] is not None
    assert status_payload["completed_at"] is not None
    assert status_payload["saved_path"].endswith("saved_custom.wav")

    result = client.get(f"/api/v1/tts/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.headers["x-request-id"]
    assert result.headers["x-job-id"] == job_id
    assert result.headers["x-model-id"] == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    assert result.headers["x-tts-mode"] == "custom"
    assert result.headers["x-backend-id"] == "mlx"
    assert result.headers["x-saved-output-file"] == "saved_custom.wav"
    assert result.content.startswith(b"RIFF")


def test_async_openai_job_submit_supports_pcm_result(client: TestClient):
    submit = client.post(
        "/v1/audio/speech/jobs",
        json={
            "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "input": "Hello async openai",
            "voice": "Vivian",
            "response_format": "pcm",
            "speed": 1.0,
        },
    )
    assert submit.status_code == 202
    submit_payload = submit.json()
    assert submit_payload["mode"] == "custom"
    assert submit_payload["operation"] == "synthesize_custom"
    assert submit_payload["response_format"] == "pcm"
    assert submit_payload["save_output"] is False
    job_id = submit_payload["job_id"]

    for _ in range(50):
        status = client.get(f"/api/v1/tts/jobs/{job_id}")
        if status.json()["status"] == "succeeded":
            break

    result = client.get(f"/api/v1/tts/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.headers["content-type"].startswith("audio/pcm")
    assert result.headers["x-request-id"]
    assert result.headers["x-model-id"] == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    assert result.headers["x-tts-mode"] == "custom"
    assert result.headers["x-backend-id"] == "mlx"
    assert result.headers["x-job-id"] == job_id
    assert not result.content.startswith(b"RIFF")


def test_async_custom_job_submit_supports_idempotent_replay(client: TestClient):
    headers = {"Idempotency-Key": "idem-custom-1"}
    first = client.post(
        "/api/v1/tts/custom/jobs",
        headers=headers,
        json={"text": "Hello async idem", "speaker": "Vivian", "save_output": True},
    )
    second = client.post(
        "/api/v1/tts/custom/jobs",
        headers=headers,
        json={"text": "Hello async idem", "speaker": "Vivian", "save_output": True},
    )

    assert first.status_code == 202
    assert second.status_code == 202
    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["job_id"] == second_payload["job_id"]
    assert second_payload["idempotency_key"] == "idem-custom-1"
    assert first_payload["submit_request_id"] == second_payload["submit_request_id"]


def test_async_job_endpoints_remain_compatible_with_auth_disabled_local_principal(
    client: TestClient,
):
    submit = client.post(
        "/api/v1/tts/custom/jobs", json={"text": "Hello owner", "speaker": "Vivian"}
    )

    assert submit.status_code == 202
    job_id = submit.json()["job_id"]
    snapshot = _state(client).job_execution.get_job(job_id)
    assert snapshot is not None
    assert snapshot.owner_principal_id == "local-default"

    status = client.get(f"/api/v1/tts/jobs/{job_id}")
    assert status.status_code == 200


def test_async_job_submit_idempotency_is_principal_scoped(client: TestClient):
    headers_a = {
        "Authorization": "Bearer token-a",
        "Idempotency-Key": "idem-principal-scope",
    }
    headers_b = {
        "Authorization": "Bearer token-b",
        "Idempotency-Key": "idem-principal-scope",
    }
    object.__setattr__(
        _state(client).settings,
        "auth_mode",
        "static_bearer",
    )
    object.__setattr__(_state(client).settings, "auth_static_bearer_token", "token-a")
    object.__setattr__(
        _state(client).settings, "auth_static_bearer_principal_id", "principal-a"
    )

    first = client.post(
        "/api/v1/tts/custom/jobs",
        headers=headers_a,
        json={"text": "Hello scoped idem", "speaker": "Vivian"},
    )
    replay = client.post(
        "/api/v1/tts/custom/jobs",
        headers=headers_a,
        json={"text": "Hello scoped idem", "speaker": "Vivian"},
    )

    object.__setattr__(_state(client).settings, "auth_static_bearer_token", "token-b")
    object.__setattr__(
        _state(client).settings, "auth_static_bearer_principal_id", "principal-b"
    )
    second_principal = client.post(
        "/api/v1/tts/custom/jobs",
        headers=headers_b,
        json={"text": "Hello scoped idem", "speaker": "Vivian"},
    )

    assert first.status_code == 202
    assert replay.status_code == 202
    assert second_principal.status_code == 202
    assert first.json()["job_id"] == replay.json()["job_id"]
    assert first.json()["job_id"] != second_principal.json()["job_id"]


def test_static_bearer_auth_requires_authorization_for_protected_routes(
    client: TestClient,
):
    object.__setattr__(_state(client).settings, "auth_mode", "static_bearer")
    object.__setattr__(
        _state(client).settings, "auth_static_bearer_token", "secret-token"
    )
    object.__setattr__(
        _state(client).settings, "auth_static_bearer_principal_id", "principal-auth"
    )

    models = client.get("/api/v1/models")
    live = client.get("/health/live")

    assert models.status_code == 401
    assert models.json()["code"] == "unauthorized"
    assert live.status_code == 200


def test_static_bearer_job_access_is_owner_bound(client: TestClient):
    object.__setattr__(_state(client).settings, "auth_mode", "static_bearer")
    object.__setattr__(_state(client).settings, "auth_static_bearer_token", "token-a")
    object.__setattr__(
        _state(client).settings, "auth_static_bearer_principal_id", "principal-a"
    )

    submit = client.post(
        "/api/v1/tts/custom/jobs",
        headers={"Authorization": "Bearer token-a"},
        json={"text": "Hello owner bound", "speaker": "Vivian"},
    )
    assert submit.status_code == 202
    job_id = submit.json()["job_id"]

    object.__setattr__(_state(client).settings, "auth_static_bearer_token", "token-b")
    object.__setattr__(
        _state(client).settings, "auth_static_bearer_principal_id", "principal-b"
    )

    status = client.get(
        f"/api/v1/tts/jobs/{job_id}", headers={"Authorization": "Bearer token-b"}
    )
    result = client.get(
        f"/api/v1/tts/jobs/{job_id}/result", headers={"Authorization": "Bearer token-b"}
    )
    cancel = client.post(
        f"/api/v1/tts/jobs/{job_id}/cancel", headers={"Authorization": "Bearer token-b"}
    )

    assert status.status_code == 403
    assert status.json()["code"] == "forbidden"
    assert result.status_code == 403
    assert result.json()["code"] == "forbidden"
    assert cancel.status_code == 403
    assert cancel.json()["code"] == "forbidden"


def test_async_design_job_submit_status_and_result_flow(client: TestClient):
    submit = client.post(
        "/api/v1/tts/design/jobs",
        json={
            "text": "Hello async design",
            "voice_description": "calm narrator",
            "save_output": True,
        },
    )
    assert submit.status_code == 202
    submit_payload = submit.json()
    assert submit_payload["mode"] == "design"
    assert submit_payload["operation"] == "synthesize_design"
    assert submit_payload["save_output"] is True
    job_id = submit_payload["job_id"]

    status_payload = None
    for _ in range(50):
        status = client.get(f"/api/v1/tts/jobs/{job_id}")
        assert status.status_code == 200
        status_payload = status.json()
        if status_payload["status"] == "succeeded":
            break

    assert status_payload is not None
    assert status_payload["status"] == "succeeded"
    assert status_payload["mode"] == "design"
    assert status_payload["saved_path"].endswith("saved_design.wav")

    result = client.get(f"/api/v1/tts/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.headers["x-job-id"] == job_id
    assert result.headers["x-tts-mode"] == "design"
    assert result.content.startswith(b"RIFF")


def test_async_openai_job_submit_rejects_idempotency_conflict(client: TestClient):
    headers = {"Idempotency-Key": "idem-openai-conflict"}
    first = client.post(
        "/v1/audio/speech/jobs",
        headers=headers,
        json={
            "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "input": "Hello first payload",
            "voice": "Vivian",
            "response_format": "wav",
            "speed": 1.0,
        },
    )
    conflict = client.post(
        "/v1/audio/speech/jobs",
        headers=headers,
        json={
            "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "input": "Hello second payload",
            "voice": "Vivian",
            "response_format": "wav",
            "speed": 1.0,
        },
    )

    assert first.status_code == 202
    assert conflict.status_code == 409
    payload = conflict.json()
    assert payload["code"] == "job_idempotency_conflict"
    assert payload["request_id"]
    assert payload["details"]["idempotency_key"] == "idem-openai-conflict"
    assert payload["details"]["job_id"] == first.json()["job_id"]


def test_sync_endpoints_remain_compatible_after_async_job_activity(client: TestClient):
    async_submit = client.post(
        "/api/v1/tts/custom/jobs",
        json={"text": "Hello async first", "speaker": "Vivian"},
    )
    assert async_submit.status_code == 202
    async_job_id = async_submit.json()["job_id"]

    for _ in range(50):
        async_status = client.get(f"/api/v1/tts/jobs/{async_job_id}")
        assert async_status.status_code == 200
        if async_status.json()["status"] == "succeeded":
            break

    sync_response = client.post(
        "/api/v1/tts/custom",
        json={
            "text": "Hello sync after async",
            "speaker": "Vivian",
            "emotion": "Happy",
            "save_output": True,
        },
    )
    assert sync_response.status_code == 200
    assert sync_response.headers["content-type"].startswith("audio/wav")
    assert sync_response.headers["x-saved-output-file"] == "saved_custom.wav"
    assert "x-saved-output-path" not in sync_response.headers
    assert sync_response.content.startswith(b"RIFF")

    openai_sync_response = client.post(
        "/v1/audio/speech",
        json={
            "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "input": "Hello sync openai after async",
            "voice": "Vivian",
            "response_format": "pcm",
            "speed": 1.0,
        },
    )
    assert openai_sync_response.status_code == 200
    assert openai_sync_response.headers["content-type"].startswith("audio/pcm")
    assert not openai_sync_response.content.startswith(b"RIFF")


def test_async_clone_job_submit_preserves_staged_input_until_completion(
    client: TestClient,
):
    files = {"ref_audio": ("reference.wav", make_wav_bytes(), "audio/wav")}
    response = client.post(
        "/api/v1/tts/clone/jobs", data={"text": "Clone async"}, files=files
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    status_payload = None
    for _ in range(50):
        status = client.get(f"/api/v1/tts/jobs/{job_id}")
        assert status.status_code == 200
        status_payload = status.json()
        if status_payload["status"] == "succeeded":
            break
    assert status_payload is not None
    assert status_payload["status"] == "succeeded"

    clone_request = _state(client).application.last_clone_request
    assert clone_request is not None
    assert (
        clone_request.ref_audio_path.parent
        == _state(client).settings.upload_staging_dir
    )
    assert not clone_request.ref_audio_path.exists()


def test_async_clone_job_submit_supports_idempotent_replay(client: TestClient):
    first = client.post(
        "/api/v1/tts/clone/jobs",
        headers={"Idempotency-Key": "idem-clone-1"},
        data={"text": "Clone async idem", "ref_text": "Clone async idem"},
        files={"ref_audio": ("reference.wav", make_wav_bytes(), "audio/wav")},
    )
    second = client.post(
        "/api/v1/tts/clone/jobs",
        headers={"Idempotency-Key": "idem-clone-1"},
        data={"text": "Clone async idem", "ref_text": "Clone async idem"},
        files={"ref_audio": ("reference.wav", make_wav_bytes(), "audio/wav")},
    )

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["job_id"] == second.json()["job_id"]
    assert second.json()["idempotency_key"] == "idem-clone-1"


def test_async_clone_job_submit_language_affects_idempotency(client: TestClient):
    first = client.post(
        "/api/v1/tts/clone/jobs",
        headers={"Idempotency-Key": "idem-clone-language"},
        data={
            "text": "Clone async idem",
            "ref_text": "Clone async idem",
            "language": "auto",
        },
        files={"ref_audio": ("reference.wav", make_wav_bytes(), "audio/wav")},
    )
    second = client.post(
        "/api/v1/tts/clone/jobs",
        headers={"Idempotency-Key": "idem-clone-language"},
        data={
            "text": "Clone async idem",
            "ref_text": "Clone async idem",
            "language": "ru",
        },
        files={"ref_audio": ("reference.wav", make_wav_bytes(), "audio/wav")},
    )

    assert first.status_code == 202
    assert second.status_code == 409


def test_async_submit_quota_exceeded_returns_controlled_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        request_timeout_seconds=5,
        quota_enabled=True,
        quota_max_active_jobs_per_principal=1,
    )
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    app = create_app(settings)
    slow = SlowTTSService(settings, sleep_seconds=0.3)
    app.state.registry = DummyRegistry(settings)
    app.state.tts_service = slow
    app.state.application = slow
    object.__setattr__(
        app.state.job_execution.manager.executor,
        "application_service",
        app.state.application,
    )
    app.state.quota_guard = build_quota_guard(
        app.state.settings, store=app.state.job_store
    )

    with TestClient(app) as test_client:
        first = test_client.post(
            "/api/v1/tts/custom/jobs", json={"text": "first quota", "speaker": "Vivian"}
        )
        second = test_client.post(
            "/api/v1/tts/custom/jobs",
            json={"text": "second quota", "speaker": "Vivian"},
        )

    assert first.status_code == 202
    assert second.status_code == 429
    payload = second.json()
    assert payload["code"] == "quota_exceeded"
    assert payload["details"]["reason"] == "Quota policy was exceeded"
    assert payload["details"]["policy"] == "active_async_jobs"
    assert payload["details"]["limit"] == 1


def test_async_job_result_returns_not_ready_for_queued_or_running_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        request_timeout_seconds=5,
    )
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    app = create_app(settings)
    slow = SlowTTSService(settings, sleep_seconds=0.2)
    app.state.registry = DummyRegistry(settings)
    app.state.tts_service = slow
    app.state.application = slow
    object.__setattr__(
        app.state.job_execution.manager.executor,
        "application_service",
        app.state.application,
    )

    with TestClient(app) as test_client:
        submit = test_client.post(
            "/api/v1/tts/custom/jobs", json={"text": "wait", "speaker": "Vivian"}
        )
        assert submit.status_code == 202
        result = test_client.get(f"/api/v1/tts/jobs/{submit.json()['job_id']}/result")

    assert result.status_code == 409
    payload = result.json()
    assert payload["code"] == "job_not_ready"
    assert payload["request_id"]
    assert payload["details"]["job_id"]
    assert payload["details"]["status"] in {"queued", "running"}


def test_async_job_result_returns_not_succeeded_for_failed_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
    )
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    app = create_app(settings)
    failing = WorkerFailingTTSService(settings)
    app.state.registry = DummyRegistry(settings)
    app.state.tts_service = failing
    app.state.application = failing
    object.__setattr__(
        app.state.job_execution.manager.executor,
        "application_service",
        app.state.application,
    )

    with TestClient(app) as test_client:
        submit = test_client.post(
            "/api/v1/tts/custom/jobs", json={"text": "boom", "speaker": "Vivian"}
        )
        job_id = submit.json()["job_id"]
        status_payload = None
        for _ in range(50):
            status = test_client.get(f"/api/v1/tts/jobs/{job_id}")
            status_payload = status.json()
            if status_payload["status"] == "failed":
                break
        result = test_client.get(f"/api/v1/tts/jobs/{job_id}/result")

    assert status_payload is not None
    assert status_payload["status"] == "failed"
    assert status_payload["terminal_error"]["code"] == "job_execution_failed"
    assert status_payload["terminal_error"]["message"] == "Job execution failed"
    assert (
        status_payload["terminal_error"]["details"]["reason"]
        == "worker execution failed"
    )
    assert (
        status_payload["terminal_error"]["details"]["error_type"]
        == "TTSGenerationError"
    )
    assert status_payload["completed_at"] is not None
    assert result.status_code == 409
    payload = result.json()
    assert payload["code"] == "job_not_succeeded"
    assert payload["request_id"]
    assert payload["details"]["status"] == "failed"
    assert payload["details"]["terminal_error"]["code"] == "job_execution_failed"
    assert (
        payload["details"]["terminal_error"]["message"]
        == "Job execution failed"
    )
    assert (
        payload["details"]["terminal_error"]["details"]["reason"]
        == "worker execution failed"
    )
    assert (
        payload["details"]["terminal_error"]["details"]["error_type"]
        == "TTSGenerationError"
    )


def test_async_job_cancel_queued_job_returns_cancelled_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        request_timeout_seconds=5,
    )
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    app = create_app(settings)
    slow = SlowTTSService(settings, sleep_seconds=0.2)
    app.state.registry = DummyRegistry(settings)
    app.state.tts_service = slow
    app.state.application = slow
    object.__setattr__(
        app.state.job_execution.manager.executor,
        "application_service",
        app.state.application,
    )

    with TestClient(app) as test_client:
        first = test_client.post(
            "/api/v1/tts/custom/jobs", json={"text": "first", "speaker": "Vivian"}
        )
        second = test_client.post(
            "/api/v1/tts/custom/jobs", json={"text": "second", "speaker": "Vivian"}
        )
        cancelled = test_client.post(
            f"/api/v1/tts/jobs/{second.json()['job_id']}/cancel"
        )
        status = test_client.get(f"/api/v1/tts/jobs/{second.json()['job_id']}")

    assert first.status_code == 202
    assert second.status_code == 202
    assert cancelled.status_code == 202
    payload = cancelled.json()
    assert payload["request_id"]
    assert payload["job_id"] == second.json()["job_id"]
    assert payload["status"] == "cancelled"
    assert payload["completed_at"] is not None
    assert payload["terminal_error"]["code"] == "job_cancelled"
    assert (
        payload["terminal_error"]["message"]
        == "Job was cancelled before execution started"
    )
    assert payload["terminal_error"]["details"] is None
    assert status.json()["status"] == "cancelled"


def test_async_job_cancel_rejects_running_or_terminal_job(client: TestClient):
    submit = client.post(
        "/api/v1/tts/custom/jobs", json={"text": "done", "speaker": "Vivian"}
    )
    job_id = submit.json()["job_id"]
    for _ in range(50):
        status = client.get(f"/api/v1/tts/jobs/{job_id}")
        if status.json()["status"] == "succeeded":
            break

    cancel = client.post(f"/api/v1/tts/jobs/{job_id}/cancel")
    assert cancel.status_code == 409
    payload = cancel.json()
    assert payload["code"] == "job_not_cancellable"
    assert payload["request_id"]
    assert payload["details"]["job_id"] == job_id
    assert payload["details"]["status"] == "succeeded"


def test_async_job_endpoints_return_job_not_found_for_missing_job(client: TestClient):
    missing_id = "missing-job-id"
    status = client.get(f"/api/v1/tts/jobs/{missing_id}")
    result = client.get(f"/api/v1/tts/jobs/{missing_id}/result")
    cancel = client.post(f"/api/v1/tts/jobs/{missing_id}/cancel")

    assert status.status_code == 404
    assert status.json()["code"] == "job_not_found"
    assert status.json()["request_id"]
    assert result.status_code == 404
    assert result.json()["code"] == "job_not_found"
    assert result.json()["request_id"]
    assert cancel.status_code == 404
    assert cancel.json()["code"] == "job_not_found"
    assert cancel.json()["request_id"]


def test_async_job_queue_full_returns_controlled_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        request_timeout_seconds=5,
        inference_busy_status_code=429,
    )
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    app = create_app(settings)
    slow = SlowTTSService(settings, sleep_seconds=0.4)
    app.state.registry = DummyRegistry(settings)
    app.state.tts_service = slow
    app.state.application = slow
    object.__setattr__(
        app.state.job_execution.manager.executor,
        "application_service",
        app.state.application,
    )
    app.state.job_execution.manager.queue_capacity = 1
    app.state.job_execution.manager.__post_init__()

    with TestClient(app) as test_client:
        first = test_client.post(
            "/api/v1/tts/custom/jobs", json={"text": "first", "speaker": "Vivian"}
        )
        second = test_client.post(
            "/api/v1/tts/custom/jobs", json={"text": "second", "speaker": "Vivian"}
        )

    assert first.status_code == 202
    assert second.status_code == 429
    payload = second.json()
    assert payload["code"] == "job_queue_full"
    assert payload["request_id"]
    assert payload["details"]["reason"] == "Local job queue is full"


def test_async_job_endpoints_handle_concurrent_reads_and_cancel_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        request_timeout_seconds=5,
    )
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    app = create_app(settings)
    contention_service = ContentionTTSService(
        settings, release_after_calls=1, sleep_seconds=0.05
    )
    app.state.registry = DummyRegistry(settings)
    app.state.tts_service = contention_service
    app.state.application = contention_service
    object.__setattr__(
        app.state.job_execution.manager.executor,
        "application_service",
        app.state.application,
    )

    with TestClient(app) as test_client:
        submit = test_client.post(
            "/api/v1/tts/custom/jobs",
            json={"text": "concurrent reads", "speaker": "Vivian"},
        )
        assert submit.status_code == 202
        job_id = submit.json()["job_id"]

        assert contention_service.started.wait(timeout=1.0), (
            "Expected worker to start before concurrent status/result reads"
        )

        def fetch_status() -> tuple[int, str]:
            response = test_client.get(f"/api/v1/tts/jobs/{job_id}")
            payload = response.json()
            return response.status_code, payload["status"]

        def fetch_result_status() -> tuple[int, str]:
            response = test_client.get(f"/api/v1/tts/jobs/{job_id}/result")
            if response.status_code == 200:
                return response.status_code, response.headers["content-type"]
            payload = response.json()
            return response.status_code, payload["code"]

        def cancel_job() -> tuple[int, str]:
            response = test_client.post(f"/api/v1/tts/jobs/{job_id}/cancel")
            payload = response.json()
            return response.status_code, payload["code"]

        with ThreadPoolExecutor(max_workers=7) as executor:
            futures = [executor.submit(fetch_status) for _ in range(3)]
            futures.extend(executor.submit(fetch_result_status) for _ in range(3))
            futures.append(executor.submit(cancel_job))
            outcomes = [future.result(timeout=2.0) for future in futures]

        status_outcomes = [outcome for outcome in outcomes if outcome[0] == 200]
        result_not_ready_outcomes = [
            outcome
            for outcome in outcomes
            if outcome[0] == 409 and outcome[1] == "job_not_ready"
        ]
        result_ready_outcomes = [
            outcome
            for outcome in outcomes
            if outcome[0] == 200 and str(outcome[1]).startswith("audio/wav")
        ]
        cancel_outcomes = [
            outcome
            for outcome in outcomes
            if outcome[0] == 409 and outcome[1] == "job_not_cancellable"
        ]

        assert len(status_outcomes) == 3, (
            f"Expected all concurrent status reads to succeed, got: {outcomes}"
        )
        assert {status for _, status in status_outcomes} <= {"running", "succeeded"}
        assert len(result_not_ready_outcomes) + len(result_ready_outcomes) == 3, (
            f"Expected all concurrent result reads to stay controlled, got: {outcomes}"
        )
        assert len(cancel_outcomes) == 1, (
            f"Expected running job cancel to be rejected once, got: {outcomes}"
        )

        contention_service.release.set()
        final_status = None
        for _ in range(50):
            response = test_client.get(f"/api/v1/tts/jobs/{job_id}")
            final_status = response.json()
            if final_status["status"] == "succeeded":
                break
            sleep(0.01)

        assert final_status is not None
        assert final_status["status"] == "succeeded"
        result = test_client.get(f"/api/v1/tts/jobs/{job_id}/result")
        assert result.status_code == 200
        assert result.headers["x-job-id"] == job_id


def test_async_owner_isolation_remains_forbidden_under_parallel_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        request_timeout_seconds=5,
        auth_mode="static_bearer",
        auth_static_bearer_token="token-a",
        auth_static_bearer_principal_id="principal-a",
    )
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    app = create_app(settings)
    contention_service = ContentionTTSService(
        settings, release_after_calls=1, sleep_seconds=0.05
    )
    app.state.registry = DummyRegistry(settings)
    app.state.tts_service = contention_service
    app.state.application = contention_service
    object.__setattr__(
        app.state.job_execution.manager.executor,
        "application_service",
        app.state.application,
    )

    with TestClient(app) as test_client:
        submit = test_client.post(
            "/api/v1/tts/custom/jobs",
            headers={"Authorization": "Bearer token-a"},
            json={"text": "owner contention", "speaker": "Vivian"},
        )
        assert submit.status_code == 202
        job_id = submit.json()["job_id"]
        assert contention_service.started.wait(timeout=1.0), (
            "Expected worker to start before forbidden access checks"
        )

        object.__setattr__(
            _state(test_client).settings, "auth_static_bearer_token", "token-b"
        )
        object.__setattr__(
            _state(test_client).settings,
            "auth_static_bearer_principal_id",
            "principal-b",
        )

        def fetch_status_forbidden() -> tuple[int, str]:
            response = test_client.get(
                f"/api/v1/tts/jobs/{job_id}",
                headers={"Authorization": "Bearer token-b"},
            )
            return response.status_code, response.json()["code"]

        def fetch_result_forbidden() -> tuple[int, str]:
            response = test_client.get(
                f"/api/v1/tts/jobs/{job_id}/result",
                headers={"Authorization": "Bearer token-b"},
            )
            return response.status_code, response.json()["code"]

        def cancel_forbidden() -> tuple[int, str]:
            response = test_client.post(
                f"/api/v1/tts/jobs/{job_id}/cancel",
                headers={"Authorization": "Bearer token-b"},
            )
            return response.status_code, response.json()["code"]

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(fetch_status_forbidden) for _ in range(2)]
            futures.extend(executor.submit(fetch_result_forbidden) for _ in range(2))
            futures.extend(executor.submit(cancel_forbidden) for _ in range(2))
            outcomes = [future.result(timeout=2.0) for future in futures]

        assert all(status_code == 403 for status_code, _ in outcomes), (
            f"Expected forbidden responses for non-owner access, got: {outcomes}"
        )
        assert all(code == "forbidden" for _, code in outcomes), (
            f"Expected forbidden error codes, got: {outcomes}"
        )

        contention_service.release.set()
        owner_result_deadline = time() + 2.0
        object.__setattr__(
            _state(test_client).settings, "auth_static_bearer_token", "token-a"
        )
        object.__setattr__(
            _state(test_client).settings,
            "auth_static_bearer_principal_id",
            "principal-a",
        )
        owner_result = None
        while time() < owner_result_deadline:
            owner_result = test_client.get(
                f"/api/v1/tts/jobs/{job_id}/result",
                headers={"Authorization": "Bearer token-a"},
            )
            if owner_result.status_code == 200:
                break
            sleep(0.01)

        assert owner_result is not None
        assert owner_result.status_code == 200


def test_async_submit_quota_enforces_local_default_limit_under_parallel_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        request_timeout_seconds=5,
        quota_enabled=True,
        quota_max_active_jobs_per_principal=1,
    )
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    app = create_app(settings)
    slow = SlowTTSService(settings, sleep_seconds=0.25)
    app.state.registry = DummyRegistry(settings)
    app.state.tts_service = slow
    app.state.application = slow
    object.__setattr__(
        app.state.job_execution.manager.executor,
        "application_service",
        app.state.application,
    )
    app.state.quota_guard = build_quota_guard(
        app.state.settings, store=app.state.job_store
    )

    with TestClient(app) as test_client:

        def submit_job(text: str) -> tuple[int, dict]:
            response = test_client.post(
                "/api/v1/tts/custom/jobs",
                json={"text": text, "speaker": "Vivian"},
            )
            return response.status_code, response.json()

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(submit_job, "local default first")
            deadline = time() + 1.0
            while (
                app.state.job_store.count_active_jobs_for_principal("local-default")
                == 0
                and time() < deadline
            ):
                sleep(0.01)
            second_future = executor.submit(submit_job, "local default second")
            first = first_future.result(timeout=2.0)
            second = second_future.result(timeout=2.0)

        assert first[0] == 202, (
            f"Expected first local-default submit to pass, got: {first}"
        )
        assert second[0] == 429, (
            f"Expected second local-default submit to be quota-limited, got: {second}"
        )
        assert second[1]["code"] == "quota_exceeded"
        assert second[1]["details"]["policy"] == "active_async_jobs"
        assert second[1]["details"]["limit"] == 1


def test_inference_execution_logging_includes_offload_and_timeout_context(
    client: TestClient, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO)

    response = client.post(
        "/v1/audio/speech",
        headers={"x-request-id": "req-observe-success"},
        json={
            "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "input": "Hello world",
            "voice": "Vivian",
            "response_format": "wav",
            "speed": 1.0,
        },
    )

    assert response.status_code == 200

    started_logs = extract_json_logs(
        caplog, "[RoutesTTS][run_inference_with_timeout][BLOCK_EXECUTE_SYNTHESIS]"
    )
    worker_logs = extract_json_logs(
        caplog, "[RoutesTTS][run_inference_with_timeout][BLOCK_EXECUTE_SYNTHESIS]"
    )
    completed_logs = extract_json_logs(
        caplog,
        "[RoutesTTS][run_inference_with_timeout][BLOCK_LOG_INFERENCE_COMPLETION]",
    )

    assert any(
        item["request_id"] == "req-observe-success"
        and item["inference_operation"] == "synthesize_custom"
        and item["execution_mode"] == "thread_offload"
        and item["offloaded_from_event_loop"] is True
        and item["timeout_seconds"] == _state(client).settings.request_timeout_seconds
        for item in started_logs
    )
    assert any(
        item["request_id"] == "req-observe-success"
        and item["inference_operation"] == "synthesize_custom"
        and item["execution_mode"] == "thread_offload"
        and item["offloaded_from_event_loop"] is True
        and item["timeout_seconds"] == _state(client).settings.request_timeout_seconds
        for item in worker_logs
    )
    assert any(
        item["request_id"] == "req-observe-success"
        and item["inference_operation"] == "synthesize_custom"
        and item["execution_mode"] == "thread_offload"
        and item["offloaded_from_event_loop"] is True
        and item["timeout_seconds"] == _state(client).settings.request_timeout_seconds
        and item["duration_ms"] >= 0
        for item in completed_logs
    )


def test_request_timeout_logs_observable_timeout_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        request_timeout_seconds=0,
    )
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    app = create_app(settings)
    app.state.registry = DummyRegistry(settings)
    app.state.tts_service = SlowTTSService(settings, sleep_seconds=0.05)
    app.state.application = app.state.tts_service
    object.__setattr__(
        app.state.job_execution.manager.executor,
        "application_service",
        app.state.application,
    )

    caplog.set_level(logging.INFO)
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/audio/speech",
            headers={"x-request-id": "req-observe-timeout"},
            json={
                "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                "input": "Hello world",
                "voice": "Vivian",
                "response_format": "wav",
                "speed": 1.0,
            },
        )

    assert response.status_code == 504
    timeout_logs = extract_json_logs(
        caplog,
        "[RoutesTTS][run_inference_with_timeout][BLOCK_HANDLE_INFERENCE_TIMEOUT]",
    )
    assert any(
        item["request_id"] == "req-observe-timeout"
        and item["inference_operation"] == "synthesize_custom"
        and item["execution_mode"] == "thread_offload"
        and item["offloaded_from_event_loop"] is True
        and item["timeout_seconds"] == 0
        and item["duration_ms"] >= 0
        for item in timeout_logs
    )


def test_inference_worker_failure_logs_controlled_failure_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
    )
    settings.ensure_directories()
    monkeypatch.setattr("server.api.routes_health.check_ffmpeg_available", lambda: True)

    app = create_app(settings)
    app.state.registry = DummyRegistry(settings)
    app.state.tts_service = WorkerFailingTTSService(settings)
    app.state.application = app.state.tts_service
    object.__setattr__(
        app.state.job_execution.manager.executor,
        "application_service",
        app.state.application,
    )

    caplog.set_level(logging.INFO)
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/audio/speech",
            headers={"x-request-id": "req-observe-failure"},
            json={
                "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                "input": "Hello world",
                "voice": "Vivian",
                "response_format": "wav",
                "speed": 1.0,
            },
        )

    assert response.status_code == 500
    failed_logs = extract_json_logs(
        caplog,
        "[RoutesTTS][run_inference_with_timeout][BLOCK_HANDLE_INFERENCE_FAILURE]",
    )
    assert any(
        item["request_id"] == "req-observe-failure"
        and item["inference_operation"] == "synthesize_custom"
        and item["execution_mode"] == "thread_offload"
        and item["offloaded_from_event_loop"] is True
        and item["timeout_seconds"] == settings.request_timeout_seconds
        and item["error_type"] == "TTSGenerationError"
        and item["error"] == "worker execution failed"
        and item["duration_ms"] >= 0
        for item in failed_logs
    )


def test_request_logging_includes_request_id_and_endpoint_context(
    client: TestClient, caplog: pytest.LogCaptureFixture
):
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
    started_logs = extract_json_logs(
        caplog, "[ServerApp][request_context_middleware][BLOCK_LOG_REQUEST_START]"
    )
    completed_logs = extract_json_logs(
        caplog, "[ServerApp][request_context_middleware][BLOCK_LOG_SUCCESS_RESPONSE]"
    )
    endpoint_logs = extract_json_logs(
        caplog, "[RoutesTTS][openai_speech][BLOCK_LOG_OPENAI_REQUEST]"
    )
    audio_logs = extract_json_logs(
        caplog, "[Responses][build_audio_response][BUILD_AUDIO_RESPONSE]"
    )

    assert any(
        item["request_id"] == "req-123" and item["path"] == "/v1/audio/speech"
        for item in started_logs
    )
    assert any(
        item["request_id"] == "req-123" and item["status_code"] == 200
        for item in completed_logs
    )
    assert any(
        item["request_id"] == "req-123" and item["endpoint"] == "/v1/audio/speech"
        for item in endpoint_logs
    )
    assert any(
        item["request_id"] == "req-123"
        and item["model"] == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
        for item in audio_logs
    )


def test_clone_upload_uses_isolated_staging_dir_and_cleans_up(client: TestClient):
    files = {"ref_audio": ("reference.wav", make_wav_bytes(), "audio/wav")}
    data = {"text": "Clone this", "ref_text": "Clone this"}
    response = client.post("/api/v1/tts/clone", data=data, files=files)

    assert response.status_code == 200
    clone_request = _state(client).application.last_clone_request
    assert clone_request is not None
    assert (
        clone_request.ref_audio_path.parent
        == _state(client).settings.upload_staging_dir
    )
    assert clone_request.ref_audio_path.parent != _state(client).settings.outputs_dir
    assert not clone_request.ref_audio_path.exists()


def test_clone_endpoint_returns_controlled_error_when_generation_artifact_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        upload_staging_dir=tmp_path / ".uploads",
        enable_streaming=True,
        default_save_output=False,
        default_clone_model="Qwen3-TTS-12Hz-1.7B-Base-8bit",
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
    object.__setattr__(
        app.state.job_execution.manager.executor,
        "application_service",
        app.state.application,
    )
    with TestClient(app) as test_client:
        files = {"ref_audio": ("reference.wav", make_wav_bytes(), "audio/wav")}
        data = {"text": "Clone this", "ref_text": "Clone this"}
        response = test_client.post("/api/v1/tts/clone", data=data, files=files)

    assert response.status_code == 500
    payload = response.json()
    assert payload["code"] == "generation_failed"
    assert payload["message"] == "Audio generation failed"
    assert payload["details"]["reason"].startswith("Generated audio file not found in ")
    assert "/private/" not in payload["details"]["reason"]
    assert "/var/" not in payload["details"]["reason"]
    assert payload["details"]["failure_kind"] == "missing_artifact"
    assert payload["request_id"]
