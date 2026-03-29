from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import pytest


pytestmark = pytest.mark.smoke


SMOKE_FLAG = "QWEN_TTS_RUN_SMOKE"
SMOKE_BASE_URL = os.getenv("QWEN_TTS_SMOKE_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
MODELS_DIR = Path(os.getenv("QWEN_TTS_MODELS_DIR", ".models"))
CUSTOM_MODEL_DIR = MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
EXPECTED_BACKEND = os.getenv("QWEN_TTS_SMOKE_EXPECTED_BACKEND")


@pytest.fixture(scope="module", autouse=True)
def require_smoke_prerequisites():
    if os.getenv(SMOKE_FLAG) != "1":
        pytest.skip(f"smoke suite is disabled; set {SMOKE_FLAG}=1 and run pytest tests/smoke")
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not available in PATH")
    if not CUSTOM_MODEL_DIR.exists():
        pytest.skip(f"required local model is missing: {CUSTOM_MODEL_DIR}")
    try:
        response = request_json("GET", "/health/live")
    except Exception as exc:  # pragma: no cover - environment-dependent gate
        pytest.skip(f"local server is not reachable at {SMOKE_BASE_URL}: {exc}")
    if response["status"] != 200:
        pytest.skip(f"local server health probe returned unexpected status: {response['status']}")


def request_json(method: str, path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(f"{SMOKE_BASE_URL}{path}", data=data, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read().decode("utf-8")
        return {
            "status": response.status,
            "headers": {key.lower(): value for key, value in response.headers.items()},
            "json": json.loads(body),
        }


def request_binary(method: str, path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{SMOKE_BASE_URL}{path}",
        data=data,
        method=method,
        headers={
            "Accept": "audio/wav",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return {
            "status": response.status,
            "headers": {key.lower(): value for key, value in response.headers.items()},
            "body": response.read(),
        }


def test_health_live_smoke():
    response = request_json("GET", "/health/live")

    assert response["status"] == 200
    assert response["json"]["status"] == "ok"


def test_health_ready_smoke():
    response = request_json("GET", "/health/ready")

    assert response["status"] == 200
    assert response["json"]["status"] in {"ok", "degraded"}
    assert response["json"]["checks"]["ffmpeg"]["available"] is True
    assert response["json"]["checks"]["models"]["available_models"] >= 1
    runtime_ready_models = response["json"]["checks"]["models"].get("runtime_ready_models")
    if runtime_ready_models is not None:
        assert runtime_ready_models >= 1
    if EXPECTED_BACKEND:
        assert response["json"]["checks"]["models"]["selected_backend"] == EXPECTED_BACKEND


def test_custom_tts_endpoint_smoke():
    response = request_binary(
        "POST",
        "/api/v1/tts/custom",
        payload={
            "text": "Smoke test for local custom voice endpoint.",
            "speaker": "Vivian",
            "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "save_output": False,
        },
    )

    assert response["status"] == 200
    assert response["headers"]["content-type"].startswith("audio/wav")
    assert response["headers"].get("x-model-id") == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    assert response["headers"].get("x-request-id")
    assert response["headers"].get("x-backend-id")
    if EXPECTED_BACKEND:
        assert response["headers"].get("x-backend-id") == EXPECTED_BACKEND
    assert response["body"].startswith(b"RIFF")
