"""Integration-style tests for the Telegram canonical remote HTTP client."""

# pyright: reportMissingImports=false

# FILE: tests/integration/test_remote_telegram_client.py
# VERSION: 1.2.0
# START_MODULE_CONTRACT
#   PURPOSE: Verify the Telegram-side canonical remote HTTP client against stable server contract semantics.
#   SCOPE: Readiness, model discovery, sync canonical speech checks, async submit/status/result, unreachable transport, public error envelope decoding, failed-job result decoding, split-topology process proof
#   DEPENDS: M-TELEGRAM, M-SERVER
#   LINKS: V-M-TELEGRAM
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _make_client - Build a RemoteServerClient backed by an injected MockTransport client
#   test_get_readiness_returns_payload_and_request_correlation - Verifies readiness payload decoding and x-request-id capture
#   test_list_models_returns_discovery_payload - Verifies model discovery payload decoding
#   test_submit_and_follow_job_lifecycle_capture_headers - Verifies submit/status/result flows capture job and submit correlation headers
#   test_submit_design_job_uses_design_async_endpoint - Verifies design submit requests use the dedicated async route
#   test_submit_clone_job_uses_clone_async_endpoint_and_multipart_payload - Verifies clone submit requests use the dedicated async route and multipart file upload
#   test_unreachable_base_url_raises_transport_error - Verifies unreachable remote server decoding
#   test_validation_error_envelope_decodes_into_api_error - Verifies controlled validation envelope decoding
#   test_failed_job_result_response_decodes_job_not_succeeded_error - Verifies failed-job result envelope decoding with terminal error details
#   test_failed_job_result_preserves_requested_job_correlation_without_headers - Verifies live-style failed result responses still preserve request-known job correlation when async headers are absent
#   test_failed_job_status_preserves_requested_job_and_known_submit_correlation - Verifies failed status responses preserve request-known job and submit correlation when headers are absent
#   _wait_for_remote_server - Poll the launched canonical HTTP stub until it publishes readiness metadata and becomes probeable
#   _launch_remote_server_process - Start and stop the separate-process canonical HTTP stub used for split-topology proof without a parent-side port reservation race
#   _read_request_log - Decode machine-readable request evidence emitted by the child server process
#   test_split_topology_remote_client_crosses_real_process_boundary_and_sync_contract - Verifies Telegram remote-client flows cross a real loopback process boundary and the canonical sync surface remains reachable without in-process mocks
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.2.0 - Added process-separated canonical server proof and sync-contract verification for Telegram remote-client flows]
# END_CHANGE_SUMMARY

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from telegram_bot.remote_client import (
    RemoteServerAPIError,
    RemoteServerClient,
    RemoteServerTransportError,
)

pytestmark = pytest.mark.integration


def _make_client(handler) -> RemoteServerClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    return RemoteServerClient(
        base_url="http://remote.test",
        http_client=http_client,
    )


def _wait_for_remote_server(
    ready_file_path: Path, process: subprocess.Popen[str], timeout_seconds: float = 10.0
) -> tuple[str, int]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = ""
            if process.stderr is not None:
                stderr = process.stderr.read()
            raise AssertionError(
                f"remote stub server exited before readiness metadata was published: {stderr}"
            )
        if not ready_file_path.exists():
            time.sleep(0.1)
            continue
        ready_payload = json.loads(ready_file_path.read_text(encoding="utf-8"))
        base_url = str(ready_payload["base_url"])
        published_pid = int(ready_payload["pid"])
        try:
            with urllib.request.urlopen(f"{base_url}/health/ready", timeout=1.0) as response:
                if response.status == 200:
                    return base_url, published_pid
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.1)

    raise AssertionError("timed out waiting for remote stub server readiness metadata")


@contextmanager
def _launch_remote_server_process(request_log_path: Path):
    ready_file_path = request_log_path.with_suffix(".ready.json")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.support.remote_server_stub",
            "--request-log",
            str(request_log_path),
            "--ready-file",
            str(ready_file_path),
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        base_url, server_pid = _wait_for_remote_server(ready_file_path, process)
        yield base_url, server_pid
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if ready_file_path.exists():
            ready_file_path.unlink()


def _read_request_log(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


@pytest.mark.anyio
async def test_get_readiness_returns_payload_and_request_correlation():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health/ready"
        assert request.headers["x-request-id"] == "req-readiness"
        return httpx.Response(
            200,
            json={"status": "ok", "checks": {"ffmpeg": {"available": True}}},
            headers={"x-request-id": "req-ready-response"},
        )

    client = _make_client(handler)

    readiness = await client.get_readiness(request_id="req-readiness")

    assert readiness.status == "ok"
    assert readiness.checks["ffmpeg"]["available"] is True
    assert readiness.correlation.request_id == "req-ready-response"
    await client.close()


@pytest.mark.anyio
async def test_list_models_returns_discovery_payload():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/models"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                        "available": True,
                        "runtime_ready": True,
                    }
                ]
            },
            headers={"x-request-id": "req-models"},
        )

    client = _make_client(handler)

    models = await client.list_models()

    assert models.data[0]["id"] == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    assert models.data[0]["runtime_ready"] is True
    assert models.correlation.request_id == "req-models"
    await client.close()


@pytest.mark.anyio
async def test_submit_and_follow_job_lifecycle_capture_headers():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/audio/speech/jobs":
            assert request.headers["x-request-id"] == "req-submit"
            assert request.headers["Idempotency-Key"] == "idem-123"
            return httpx.Response(
                202,
                json={
                    "job_id": "job-123",
                    "status": "queued",
                    "submit_request_id": "submit-123",
                    "status_url": "/api/v1/tts/jobs/job-123",
                    "result_url": "/api/v1/tts/jobs/job-123/result",
                    "cancel_url": "/api/v1/tts/jobs/job-123/cancel",
                },
                headers={
                    "x-request-id": "req-submit-response",
                    "x-job-id": "job-123",
                    "x-submit-request-id": "submit-123",
                },
            )
        if request.url.path == "/api/v1/tts/jobs/job-123":
            return httpx.Response(
                200,
                json={
                    "job_id": "job-123",
                    "status": "succeeded",
                    "submit_request_id": "submit-123",
                },
                headers={
                    "x-request-id": "req-status-response",
                    "x-job-id": "job-123",
                    "x-submit-request-id": "submit-123",
                },
            )
        if request.url.path == "/api/v1/tts/jobs/job-123/result":
            return httpx.Response(
                200,
                content=b"RIFF....",
                headers={
                    "content-type": "audio/wav",
                    "x-request-id": "req-result-response",
                    "x-job-id": "job-123",
                    "x-submit-request-id": "submit-123",
                    "x-model-id": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                    "x-tts-mode": "custom",
                    "x-backend-id": "mlx",
                    "x-saved-output-file": "saved_custom.wav",
                },
            )
        raise AssertionError(f"Unexpected path: {request.url.path}")

    client = _make_client(handler)

    submit = await client.submit_speech_job(
        {
            "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "input": "Hello",
            "voice": "Vivian",
        },
        request_id="req-submit",
        idempotency_key="idem-123",
    )
    status = await client.get_job_status("job-123")
    result = await client.get_job_result("job-123")

    assert submit.payload["job_id"] == "job-123"
    assert submit.correlation.job_id == "job-123"
    assert submit.correlation.submit_request_id == "submit-123"
    assert status.payload["status"] == "succeeded"
    assert status.correlation.request_id == "req-status-response"
    assert result.audio_bytes.startswith(b"RIFF")
    assert result.content_type.startswith("audio/wav")
    assert result.model_id == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    assert result.tts_mode == "custom"
    assert result.backend_id == "mlx"
    assert result.saved_output_file == "saved_custom.wav"
    assert result.correlation.job_id == "job-123"
    assert result.correlation.submit_request_id == "submit-123"
    await client.close()


@pytest.mark.anyio
async def test_submit_design_job_uses_design_async_endpoint():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/tts/design/jobs"
        assert request.headers["x-request-id"] == "req-design"
        assert request.headers["Idempotency-Key"] == "idem-design"
        payload = await request.aread()
        assert b"voice_description" in payload
        assert b"calm narrator" in payload
        return httpx.Response(
            202,
            json={
                "job_id": "job-design-123",
                "status": "queued",
                "submit_request_id": "submit-design-123",
                "status_url": "/api/v1/tts/jobs/job-design-123",
                "result_url": "/api/v1/tts/jobs/job-design-123/result",
                "cancel_url": "/api/v1/tts/jobs/job-design-123/cancel",
            },
            headers={
                "x-request-id": "req-design-response",
                "x-job-id": "job-design-123",
                "x-submit-request-id": "submit-design-123",
            },
        )

    client = _make_client(handler)

    submit = await client.submit_design_job(
        {
            "model": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
            "text": "Hello",
            "voice_description": "calm narrator",
            "language": "ru",
        },
        request_id="req-design",
        idempotency_key="idem-design",
    )

    assert submit.payload["job_id"] == "job-design-123"
    assert submit.correlation.job_id == "job-design-123"
    assert submit.correlation.submit_request_id == "submit-design-123"
    await client.close()


@pytest.mark.anyio
async def test_submit_clone_job_uses_clone_async_endpoint_and_multipart_payload():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/tts/clone/jobs"
        assert request.headers["x-request-id"] == "req-clone"
        assert request.headers["Idempotency-Key"] == "idem-clone"
        body = await request.aread()
        assert b'name="text"' in body
        assert b'name="ref_text"' in body
        assert b'name="ref_audio"; filename="reference.ogg"' in body
        assert b"audio/ogg" in body
        assert b"sample transcript" in body
        return httpx.Response(
            202,
            json={
                "job_id": "job-clone-123",
                "status": "queued",
                "submit_request_id": "submit-clone-123",
                "status_url": "/api/v1/tts/jobs/job-clone-123",
                "result_url": "/api/v1/tts/jobs/job-clone-123/result",
                "cancel_url": "/api/v1/tts/jobs/job-clone-123/cancel",
            },
            headers={
                "x-request-id": "req-clone-response",
                "x-job-id": "job-clone-123",
                "x-submit-request-id": "submit-clone-123",
            },
        )

    client = _make_client(handler)

    submit = await client.submit_clone_job(
        text="Speak like me",
        ref_audio_bytes=b"OggSfake",
        ref_audio_filename="reference.ogg",
        ref_audio_content_type="audio/ogg",
        ref_text="sample transcript",
        language="en",
        model="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        request_id="req-clone",
        idempotency_key="idem-clone",
    )

    assert submit.payload["job_id"] == "job-clone-123"
    assert submit.correlation.job_id == "job-clone-123"
    assert submit.correlation.submit_request_id == "submit-clone-123"
    await client.close()


@pytest.mark.anyio
async def test_unreachable_base_url_raises_transport_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _make_client(handler)

    with pytest.raises(RemoteServerTransportError) as exc_info:
        await client.get_readiness()

    assert "connection failed" in str(exc_info.value).lower()
    await client.close()


@pytest.mark.anyio
async def test_validation_error_envelope_decodes_into_api_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "code": "validation_error",
                "message": "Input text must not be empty",
                "details": {"field": "input"},
                "request_id": "req-validation",
            },
            headers={"x-request-id": "req-validation"},
        )

    client = _make_client(handler)

    with pytest.raises(RemoteServerAPIError) as exc_info:
        await client.submit_speech_job({"input": "", "voice": "Vivian"})

    error = exc_info.value
    assert error.code == "validation_error"
    assert error.details == {"field": "input"}
    assert error.request_id == "req-validation"
    assert error.correlation.request_id == "req-validation"
    await client.close()


@pytest.mark.anyio
async def test_failed_job_result_response_decodes_job_not_succeeded_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "code": "job_not_succeeded",
                "message": "Async job is not in succeeded state",
                "details": {
                    "terminal_error": {
                        "code": "job_execution_timeout",
                        "message": "Async job timed out",
                        "details": {"status": "failed"},
                    }
                },
                "request_id": "req-result-error",
            },
            headers={
                "x-request-id": "req-result-error",
                "x-job-id": "job-failed",
                "x-submit-request-id": "submit-failed",
            },
        )

    client = _make_client(handler)

    with pytest.raises(RemoteServerAPIError) as exc_info:
        await client.get_job_result("job-failed")

    error = exc_info.value
    assert error.code == "job_not_succeeded"
    assert error.details["terminal_error"]["code"] == "job_execution_timeout"
    assert error.correlation.job_id == "job-failed"
    assert error.correlation.submit_request_id == "submit-failed"
    await client.close()


@pytest.mark.anyio
async def test_failed_job_result_preserves_requested_job_correlation_without_headers():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/tts/jobs/job-lived/result"
        return httpx.Response(
            409,
            json={
                "code": "job_not_succeeded",
                "message": "Job did not finish successfully",
                "details": {
                    "job_id": "job-lived",
                    "status": "failed",
                    "terminal_error": {
                        "code": "job_execution_timeout",
                        "message": "Async job timed out",
                        "details": {"status": "failed"},
                    },
                },
                "request_id": "atlas-result-fix",
            },
            headers={
                "x-request-id": "atlas-result-fix",
            },
        )

    client = _make_client(handler)

    with pytest.raises(RemoteServerAPIError) as exc_info:
        await client.get_job_result(
            "job-lived",
            submit_request_id="atlas-submit-fix",
        )

    error = exc_info.value
    assert error.code == "job_not_succeeded"
    assert error.request_id == "atlas-result-fix"
    assert error.correlation.request_id == "atlas-result-fix"
    assert error.correlation.job_id == "job-lived"
    assert error.correlation.submit_request_id == "atlas-submit-fix"
    await client.close()


@pytest.mark.anyio
async def test_failed_job_status_preserves_requested_job_and_known_submit_correlation():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/tts/jobs/job-status-lived"
        return httpx.Response(
            404,
            json={
                "code": "job_not_found",
                "message": "Job was not found",
                "details": {"job_id": "job-status-lived"},
                "request_id": "atlas-status-fix",
            },
            headers={"x-request-id": "atlas-status-fix"},
        )

    client = _make_client(handler)

    with pytest.raises(RemoteServerAPIError) as exc_info:
        await client.get_job_status(
            "job-status-lived",
            submit_request_id="atlas-submit-status",
        )

    error = exc_info.value
    assert error.code == "job_not_found"
    assert error.correlation.request_id == "atlas-status-fix"
    assert error.correlation.job_id == "job-status-lived"
    assert error.correlation.submit_request_id == "atlas-submit-status"
    await client.close()


@pytest.mark.anyio
async def test_split_topology_remote_client_crosses_real_process_boundary_and_sync_contract(
    tmp_path: Path,
):
    request_log_path = tmp_path / "remote-server-requests.jsonl"

    with _launch_remote_server_process(request_log_path) as (base_url, server_pid):
        client = RemoteServerClient(base_url=base_url)

        readiness = await client.get_readiness(request_id="req-split-ready")
        models = await client.list_models(request_id="req-split-models")
        submit = await client.submit_speech_job(
            {
                "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                "input": "Hello from the remote split topology test",
                "voice": "Vivian",
                "response_format": "wav",
            },
            request_id="req-split-submit",
            idempotency_key="idem-split-123",
        )
        status = await client.get_job_status(
            "job-remote-123",
            request_id="req-split-status",
            submit_request_id="submit-remote-123",
        )
        result = await client.get_job_result(
            "job-remote-123",
            request_id="req-split-result",
            submit_request_id="submit-remote-123",
        )

        async with httpx.AsyncClient() as http_client:
            sync_response = await http_client.post(
                f"{base_url}/v1/audio/speech",
                headers={"x-request-id": "req-sync-contract"},
                json={
                    "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                    "input": "Sync contract proof from the split-topology harness",
                    "voice": "Vivian",
                    "response_format": "wav",
                },
            )

        await client.close()

    request_log = _read_request_log(request_log_path)

    assert server_pid != os.getpid()
    assert readiness.status == "ok"
    assert readiness.correlation.request_id == "req-split-ready-ack"
    assert models.data[0]["id"] == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    assert models.data[0]["runtime_ready"] is True
    assert submit.payload["job_id"] == "job-remote-123"
    assert submit.correlation.job_id == "job-remote-123"
    assert submit.correlation.submit_request_id == "submit-remote-123"
    assert status.payload["status"] == "succeeded"
    assert status.correlation.request_id == "req-split-status-ack"
    assert result.audio_bytes.startswith(b"RIFF")
    assert result.content_type.startswith("audio/wav")
    assert result.model_id == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    assert result.backend_id == "stub"

    assert sync_response.status_code == 200
    assert sync_response.content.startswith(b"RIFF")
    assert sync_response.headers["x-request-id"] == "req-sync-contract-ack"
    assert sync_response.headers["x-model-id"] == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    assert sync_response.headers["x-tts-mode"] == "custom"
    assert sync_response.headers["x-backend-id"] == "stub"

    observed_requests = {
        (entry["method"], entry["path"], entry["request_id"]) for entry in request_log
    }
    assert ("GET", "/health/ready", None) in observed_requests
    assert ("GET", "/health/ready", "req-split-ready") in observed_requests
    assert ("GET", "/api/v1/models", "req-split-models") in observed_requests
    assert ("POST", "/v1/audio/speech/jobs", "req-split-submit") in observed_requests
    assert ("GET", "/api/v1/tts/jobs/job-remote-123", "req-split-status") in observed_requests
    assert (
        "GET",
        "/api/v1/tts/jobs/job-remote-123/result",
        "req-split-result",
    ) in observed_requests
    assert ("POST", "/v1/audio/speech", "req-sync-contract") in observed_requests
    assert all(entry["handler_pid"] == server_pid for entry in request_log)
