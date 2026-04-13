# FILE: tests/smoke/test_local_runtime.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Smoke tests for the local runtime HTTP service.
#   SCOPE: Health endpoints, sync/async synthesis smoke checks
#   DEPENDS: M-SERVER
#   LINKS: V-M-SERVER
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   require_smoke_prerequisites - Autouse smoke gate for local runtime dependencies and selected smoke-model readiness
#   resolve_smoke_target - Resolve smoke model/backend expectations used by sync and async smoke requests
#   request_json - JSON HTTP helper for smoke endpoint checks
#   request_binary - Binary HTTP helper for smoke audio/result checks
#   skip_if_missing_runtime_feature - Skip smoke checks when optional runtime endpoints are absent
#   wait_for_terminal_job_status - Poll async jobs until a terminal snapshot is returned
#   assert_audio_response - Shared smoke assertion helper for audio responses
#   test_health_live_smoke - Verifies live probe response from the local server
#   test_health_ready_smoke - Verifies ready probe response and runtime checks
#   test_custom_tts_endpoint_smoke - Verifies synchronous custom TTS endpoint end-to-end
#   test_async_custom_job_flow_smoke - Verifies async custom submit, status, and result flow
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.2.0 - Extended smoke target resolution for OmniVoice and VoxCPM2 while preserving the existing Qwen and Piper request paths]
# END_CHANGE_SUMMARY

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


pytestmark = pytest.mark.smoke


SMOKE_FLAG = "QWEN_TTS_RUN_SMOKE"
SMOKE_BASE_URL = os.getenv("QWEN_TTS_SMOKE_BASE_URL", "http://127.0.0.1:8001").rstrip(
    "/"
)
MODELS_DIR = Path(os.getenv("QWEN_TTS_MODELS_DIR", ".models"))
CUSTOM_MODEL_DIR = MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
OMNIVOICE_MODEL_DIR = MODELS_DIR / "OmniVoice"
VOXCPM_MODEL_DIR = MODELS_DIR / "VoxCPM2"
PIPER_MODEL_DIR = MODELS_DIR / "Piper-en_US-lessac-medium"
SMOKE_MODEL_ID = os.getenv("QWEN_TTS_SMOKE_MODEL_ID", "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit")
EXPECTED_BACKEND = os.getenv("QWEN_TTS_SMOKE_EXPECTED_BACKEND")
DEFAULT_CUSTOM_SMOKE_TEXT = "Smoke test for local custom voice endpoint."
ASYNC_CUSTOM_SMOKE_TEXT = "Smoke async job test for local custom voice endpoint."
SMOKE_TARGETS: dict[str, dict[str, str | Path | bool | dict[str, object]]] = {
    "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit": {
        "model_id": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "model_dir": CUSTOM_MODEL_DIR,
        "speaker": "Vivian",
        "sync_path": "/api/v1/tts/custom",
        "sync_payload": {
            "text": DEFAULT_CUSTOM_SMOKE_TEXT,
            "speaker": "Vivian",
            "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "save_output": False,
        },
        "expected_backend": EXPECTED_BACKEND,
        "supports_async_custom_jobs": True,
        "async_submit_payload": {
            "text": ASYNC_CUSTOM_SMOKE_TEXT,
            "speaker": "Vivian",
            "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "save_output": False,
        },
    },
    "OmniVoice-Custom": {
        "model_id": "OmniVoice-Custom",
        "model_dir": OMNIVOICE_MODEL_DIR,
        "speaker": "Vivian",
        "sync_path": "/api/v1/tts/custom",
        "sync_payload": {
            "text": DEFAULT_CUSTOM_SMOKE_TEXT,
            "speaker": "Vivian",
            "model": "OmniVoice-Custom",
            "save_output": False,
        },
        "expected_backend": EXPECTED_BACKEND or "torch",
        "supports_async_custom_jobs": True,
        "async_submit_payload": {
            "text": ASYNC_CUSTOM_SMOKE_TEXT,
            "speaker": "Vivian",
            "model": "OmniVoice-Custom",
            "save_output": False,
        },
    },
    "VoxCPM2-Custom": {
        "model_id": "VoxCPM2-Custom",
        "model_dir": VOXCPM_MODEL_DIR,
        "speaker": "Vivian",
        "sync_path": "/api/v1/tts/custom",
        "sync_payload": {
            "text": DEFAULT_CUSTOM_SMOKE_TEXT,
            "speaker": "Vivian",
            "model": "VoxCPM2-Custom",
            "save_output": False,
        },
        "expected_backend": EXPECTED_BACKEND or "torch",
        "supports_async_custom_jobs": True,
        "async_submit_payload": {
            "text": ASYNC_CUSTOM_SMOKE_TEXT,
            "speaker": "Vivian",
            "model": "VoxCPM2-Custom",
            "save_output": False,
        },
    },
    "Piper-en_US-lessac-medium": {
        "model_id": "Piper-en_US-lessac-medium",
        "model_dir": PIPER_MODEL_DIR,
        "speaker": "en_US-lessac-medium",
        "sync_path": "/v1/audio/speech",
        "sync_payload": {
            "model": "Piper-en_US-lessac-medium",
            "input": DEFAULT_CUSTOM_SMOKE_TEXT,
            "voice": "en_US-lessac-medium",
            "response_format": "wav",
            "speed": 1.0,
        },
        "expected_backend": "onnx",
        "supports_async_custom_jobs": False,
    },
}
SUPPORTED_SMOKE_MODEL_IDS = set(SMOKE_TARGETS)
ASYNC_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "timeout"}
ASYNC_POLL_ATTEMPTS = int(os.getenv("QWEN_TTS_SMOKE_ASYNC_POLL_ATTEMPTS", "60"))
ASYNC_POLL_INTERVAL_SECONDS = float(
    os.getenv("QWEN_TTS_SMOKE_ASYNC_POLL_INTERVAL_SECONDS", "1.0")
)


@pytest.fixture(scope="module", autouse=True)
def require_smoke_prerequisites():
    if os.getenv(SMOKE_FLAG) != "1":
        pytest.skip(
            f"smoke suite is disabled; set {SMOKE_FLAG}=1 and run pytest tests/smoke"
        )
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not available in PATH")
    smoke_target = resolve_smoke_target()
    if not smoke_target["model_dir"].exists():
        pytest.skip(f"required local model is missing: {smoke_target['model_dir']}")
    try:
        live_response = request_json("GET", "/health/live")
    except Exception as exc:  # pragma: no cover - environment-dependent gate
        pytest.skip(f"local server is not reachable at {SMOKE_BASE_URL}: {exc}")
    if live_response["status"] != 200:
        pytest.skip(
            f"local server health probe returned unexpected status: {live_response['status']}"
        )

    try:
        ready_response = request_json("GET", "/health/ready")
    except Exception as exc:  # pragma: no cover - environment-dependent gate
        pytest.skip(f"local server readiness probe failed at {SMOKE_BASE_URL}: {exc}")

    checks = ready_response["json"].get("checks", {})
    ffmpeg_check = checks.get("ffmpeg", {})
    models_check = checks.get("models", {})
    if ffmpeg_check.get("available") is not True:
        pytest.skip("local runtime ffmpeg readiness check is not passing")
    if models_check.get("available_models", 0) < 1:
        pytest.skip("local runtime reports zero available models")
    runtime_ready_models = models_check.get("runtime_ready_models")
    if runtime_ready_models is not None and runtime_ready_models < 1:
        pytest.skip("local runtime reports zero runtime-ready models")


def resolve_smoke_target() -> dict[str, str | Path | bool]:
    smoke_model_id = SMOKE_MODEL_ID.strip()
    if smoke_model_id not in SUPPORTED_SMOKE_MODEL_IDS:
        pytest.skip(
            "unsupported smoke model id "
            f"'{smoke_model_id}'; expected one of {sorted(SUPPORTED_SMOKE_MODEL_IDS)}"
        )
    return dict(SMOKE_TARGETS[smoke_model_id])


def request_json(method: str, path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{SMOKE_BASE_URL}{path}", data=data, method=method, headers=headers
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read().decode("utf-8")
        return {
            "status": response.status,
            "headers": {key.lower(): value for key, value in response.headers.items()},
            "json": json.loads(body),
        }


def request_binary(method: str, path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "audio/wav"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{SMOKE_BASE_URL}{path}",
        data=data,
        method=method,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return {
            "status": response.status,
            "headers": {key.lower(): value for key, value in response.headers.items()},
            "body": response.read(),
        }


def skip_if_missing_runtime_feature(
    exc: urllib.error.HTTPError, *, feature: str
) -> None:
    if exc.code in {404, 405, 501}:  # pragma: no cover - environment-dependent gate
        pytest.skip(
            f"local runtime does not expose required {feature} endpoint at {SMOKE_BASE_URL}: HTTP {exc.code}"
        )
    raise exc


def wait_for_terminal_job_status(job_id: str) -> dict:
    last_status_payload = None
    for _ in range(ASYNC_POLL_ATTEMPTS):
        try:
            status_response = request_json("GET", f"/api/v1/tts/jobs/{job_id}")
        except urllib.error.HTTPError as exc:
            skip_if_missing_runtime_feature(exc, feature="async job status")
        assert status_response["status"] == 200, (
            f"Expected async status polling to return HTTP 200 for job {job_id}"
        )
        status_payload = status_response["json"]
        last_status_payload = status_payload
        if status_payload["status"] in ASYNC_TERMINAL_STATUSES:
            return status_payload
        time.sleep(ASYNC_POLL_INTERVAL_SECONDS)
    pytest.fail(
        "Async smoke job did not reach a terminal state within the safe polling timeout. "
        f"job_id={job_id}, last_status={last_status_payload}"
    )


def assert_audio_response(response: dict, *, expected_model: str) -> None:
    assert response["status"] == 200
    assert response["headers"]["content-type"].startswith("audio/wav")
    assert response["headers"].get("x-model-id") == expected_model
    assert response["headers"].get("x-request-id")
    assert response["headers"].get("x-backend-id")
    if EXPECTED_BACKEND:
        assert response["headers"].get("x-backend-id") == EXPECTED_BACKEND
    assert response["body"].startswith(b"RIFF")


def test_health_live_smoke():
    response = request_json("GET", "/health/live")

    assert response["status"] == 200
    assert response["json"]["status"] == "ok"


def test_health_ready_smoke():
    smoke_target = resolve_smoke_target()
    response = request_json("GET", "/health/ready")

    assert response["status"] == 200
    assert response["json"]["status"] in {"ok", "degraded"}
    assert response["json"]["checks"]["ffmpeg"]["available"] is True
    assert response["json"]["checks"]["models"]["available_models"] >= 1
    runtime_ready_models = response["json"]["checks"]["models"].get(
        "runtime_ready_models"
    )
    if runtime_ready_models is not None:
        assert runtime_ready_models >= 1
    if EXPECTED_BACKEND and smoke_target["supports_async_custom_jobs"] is True:
        models_check = response["json"]["checks"]["models"]
        selected_backend = models_check.get("selected_backend")
        mixed_backend_routing = models_check.get("routing", {}).get(
            "mixed_backend_routing"
        )
        assert selected_backend == EXPECTED_BACKEND or mixed_backend_routing is True


def test_custom_tts_endpoint_smoke():
    smoke_target = resolve_smoke_target()
    try:
        response = request_binary(
            "POST",
            str(smoke_target["sync_path"]),
            payload=dict(smoke_target["sync_payload"]),
        )
    except urllib.error.HTTPError as exc:
        skip_if_missing_runtime_feature(exc, feature="sync custom TTS")

    assert_audio_response(response, expected_model=str(smoke_target["model_id"]))
    expected_backend = str(smoke_target["expected_backend"] or "")
    if expected_backend:
        assert response["headers"].get("x-backend-id") == expected_backend


def test_async_custom_job_flow_smoke():
    smoke_target = resolve_smoke_target()
    if smoke_target["supports_async_custom_jobs"] is not True:
        pytest.skip(
            "async custom smoke flow is only applicable to custom-family smoke targets"
        )

    try:
        submit_response = request_json(
            "POST",
            "/api/v1/tts/custom/jobs",
            payload=dict(smoke_target["async_submit_payload"]),
        )
    except urllib.error.HTTPError as exc:
        skip_if_missing_runtime_feature(exc, feature="async custom job submit")

    assert submit_response["status"] == 202
    submit_payload = submit_response["json"]
    job_id = submit_payload["job_id"]
    assert job_id, "Expected async submit response to include a job_id"
    assert submit_payload["operation"] == "synthesize_custom"
    assert submit_payload["mode"] == "custom"
    assert submit_payload["status"] in {"queued", "running", "succeeded"}
    assert submit_payload["status_url"].endswith(f"/api/v1/tts/jobs/{job_id}")
    assert submit_payload["result_url"].endswith(f"/api/v1/tts/jobs/{job_id}/result")

    terminal_status = wait_for_terminal_job_status(job_id)

    assert terminal_status["job_id"] == job_id
    assert terminal_status["status"] == "succeeded", (
        f"Expected async smoke job to succeed, got {terminal_status}"
    )
    assert terminal_status["backend"], (
        "Expected terminal async smoke snapshot to report backend"
    )
    assert terminal_status["response_format"] == "wav"
    assert terminal_status["save_output"] is False
    if EXPECTED_BACKEND:
        assert terminal_status["backend"] == EXPECTED_BACKEND

    try:
        result_response = request_binary("GET", f"/api/v1/tts/jobs/{job_id}/result")
    except urllib.error.HTTPError as exc:
        skip_if_missing_runtime_feature(exc, feature="async job result")

    assert_audio_response(
        result_response, expected_model=str(smoke_target["model_id"])
    )
    assert result_response["headers"].get("x-job-id") == job_id
    assert result_response["headers"].get("x-tts-mode") == "custom"


def test_piper_model_is_visible_when_installed():
    if not PIPER_MODEL_DIR.exists():
        pytest.skip(f"optional Piper model is missing: {PIPER_MODEL_DIR}")

    response = request_json("GET", "/api/v1/models")

    assert response["status"] == 200
    assert any(
        item["id"] == "Piper-en_US-lessac-medium" or item.get("family_key") == "piper"
        for item in response["json"]["data"]
    )
