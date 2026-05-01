"""Separate-process canonical HTTP stub for Telegram remote-client verification."""

# FILE: tests/support/remote_server_stub.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide a process-launched canonical HTTP stub that proves Telegram remote-client flows cross a real network boundary.
#   SCOPE: Readiness, model discovery, canonical sync speech response, async submit/status/result endpoints, request logging for split-topology evidence
#   DEPENDS: M-SERVER, M-TELEGRAM
#   LINKS: V-M-TELEGRAM
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _append_request_log - Persist machine-readable request evidence emitted by the child server process
#   _json_response - Return JSON responses with deterministic canonical headers
#   _audio_response - Return deterministic WAV-like audio bytes with canonical headers
#   _build_job_payload - Build a stable async job snapshot payload for remote-client verification
#   _RemoteServerStubHandler - Serve the minimal canonical HTTP contract required by split-topology tests
#   main - Launch the separate-process HTTP stub server on loopback and publish its selected base URL
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added process-launched canonical HTTP stub for split-topology Telegram remote-client verification]
# END_CHANGE_SUMMARY

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
JOB_ID = "job-remote-123"
SUBMIT_REQUEST_ID = "submit-remote-123"
AUDIO_BYTES = b"RIFFstub-audio"


def _append_request_log(request_log_path: Path | None, record: dict[str, Any]) -> None:
    if request_log_path is None:
        return
    request_log_path.parent.mkdir(parents=True, exist_ok=True)
    with request_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _json_response(
    handler: BaseHTTPRequestHandler,
    *,
    status: int,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    for key, value in (headers or {}).items():
        handler.send_header(key, value)
    handler.end_headers()
    handler.wfile.write(body)


def _audio_response(
    handler: BaseHTTPRequestHandler,
    *,
    status: int,
    body: bytes,
    headers: dict[str, str] | None = None,
) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", "audio/wav")
    handler.send_header("Content-Length", str(len(body)))
    for key, value in (headers or {}).items():
        handler.send_header(key, value)
    handler.end_headers()
    handler.wfile.write(body)


def _build_job_payload() -> dict[str, Any]:
    return {
        "job_id": JOB_ID,
        "status": "queued",
        "submit_request_id": SUBMIT_REQUEST_ID,
        "status_url": f"/api/v1/tts/jobs/{JOB_ID}",
        "result_url": f"/api/v1/tts/jobs/{JOB_ID}/result",
        "cancel_url": f"/api/v1/tts/jobs/{JOB_ID}/cancel",
    }


class _RemoteServerStubHandler(BaseHTTPRequestHandler):
    server_version = "TelegramRemoteServerStub/1.0"
    request_log_path: Path | None = None

    def log_message(self, format: str, *args: object) -> None:  # pragma: no cover
        return

    def _record_request(self, body: bytes) -> None:
        _append_request_log(
            self.request_log_path,
            {
                "handler_pid": os.getpid(),
                "method": self.command,
                "path": self.path,
                "request_id": self.headers.get("x-request-id"),
                "idempotency_key": self.headers.get("Idempotency-Key"),
                "content_type": self.headers.get("Content-Type"),
                "content_length": len(body),
            },
        )

    def _response_headers(self, request_id: str | None, **extra: str) -> dict[str, str]:
        response_request_id = f"{request_id}-ack" if request_id else "stub-request-ack"
        headers = {
            "x-request-id": response_request_id,
            "x-stub-pid": str(os.getpid()),
        }
        headers.update(extra)
        return headers

    def do_GET(self) -> None:  # noqa: N802
        body = b""
        self._record_request(body)
        request_id = self.headers.get("x-request-id")

        if self.path == "/health/live":
            _json_response(
                self,
                status=HTTPStatus.OK,
                payload={"status": "ok"},
                headers=self._response_headers(request_id),
            )
            return

        if self.path == "/health/ready":
            _json_response(
                self,
                status=HTTPStatus.OK,
                payload={
                    "status": "ok",
                    "checks": {
                        "ffmpeg": {"available": True},
                        "models": {"available_models": 1, "runtime_ready_models": 1},
                    },
                },
                headers=self._response_headers(request_id),
            )
            return

        if self.path == "/api/v1/models":
            _json_response(
                self,
                status=HTTPStatus.OK,
                payload={
                    "data": [
                        {
                            "id": MODEL_ID,
                            "available": True,
                            "runtime_ready": True,
                            "capabilities": {
                                "supports_custom": True,
                                "supports_design": True,
                                "supports_clone": True,
                            },
                        }
                    ]
                },
                headers=self._response_headers(request_id),
            )
            return

        if self.path == f"/api/v1/tts/jobs/{JOB_ID}":
            _json_response(
                self,
                status=HTTPStatus.OK,
                payload={
                    "job_id": JOB_ID,
                    "status": "succeeded",
                    "submit_request_id": SUBMIT_REQUEST_ID,
                },
                headers=self._response_headers(
                    request_id,
                    **{
                        "x-job-id": JOB_ID,
                        "x-submit-request-id": SUBMIT_REQUEST_ID,
                    },
                ),
            )
            return

        if self.path == f"/api/v1/tts/jobs/{JOB_ID}/result":
            _audio_response(
                self,
                status=HTTPStatus.OK,
                body=AUDIO_BYTES,
                headers=self._response_headers(
                    request_id,
                    **{
                        "x-job-id": JOB_ID,
                        "x-submit-request-id": SUBMIT_REQUEST_ID,
                        "x-model-id": MODEL_ID,
                        "x-tts-mode": "custom",
                        "x-backend-id": "stub",
                    },
                ),
            )
            return

        _json_response(
            self,
            status=HTTPStatus.NOT_FOUND,
            payload={
                "code": "not_found",
                "message": "Endpoint not found",
                "details": {"path": self.path},
                "request_id": request_id,
            },
            headers=self._response_headers(request_id),
        )

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        self._record_request(body)
        request_id = self.headers.get("x-request-id")

        if self.path == "/v1/audio/speech":
            _audio_response(
                self,
                status=HTTPStatus.OK,
                body=AUDIO_BYTES,
                headers=self._response_headers(
                    request_id,
                    **{
                        "x-model-id": MODEL_ID,
                        "x-tts-mode": "custom",
                        "x-backend-id": "stub",
                    },
                ),
            )
            return

        if self.path == "/v1/audio/speech/jobs":
            _json_response(
                self,
                status=HTTPStatus.ACCEPTED,
                payload=_build_job_payload(),
                headers=self._response_headers(
                    request_id,
                    **{
                        "x-job-id": JOB_ID,
                        "x-submit-request-id": SUBMIT_REQUEST_ID,
                    },
                ),
            )
            return

        _json_response(
            self,
            status=HTTPStatus.NOT_FOUND,
            payload={
                "code": "not_found",
                "message": "Endpoint not found",
                "details": {"path": self.path},
                "request_id": request_id,
            },
            headers=self._response_headers(request_id),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--request-log", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()

    _RemoteServerStubHandler.request_log_path = args.request_log
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _RemoteServerStubHandler)
    selected_host, selected_port = server.server_address[:2]
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    args.ready_file.write_text(
        json.dumps(
            {
                "base_url": f"http://{selected_host}:{selected_port}",
                "pid": os.getpid(),
            }
        ),
        encoding="utf-8",
    )
    try:
        server.serve_forever()
    finally:  # pragma: no cover
        server.server_close()


if __name__ == "__main__":
    main()
