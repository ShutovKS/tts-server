#!/usr/bin/env python3
# FILE: scripts/validate_runtime.py
# VERSION: 1.9.1
# START_MODULE_CONTRACT
#   PURPOSE: Provide operator- and CI-facing validation commands for host-specific backend checks, automated local smoke orchestration, Docker parity probes, optional Telegram live connectivity, and explicit advisory artifact review.
#   SCOPE: CLI subcommands, runtime validation environment helpers, HTTP smoke start/stop orchestration, qwen_fast host-matrix assertions, optional representative real-model validation, server and Telegram Docker parity validation, Telegram Bot API reachability checks, opt-in inbound Telegram update validation, and optional artifact-review of persisted validation evidence
#   DEPENDS: M-CONFIG, M-BOOTSTRAP, M-SERVER, M-TELEGRAM, M-HOST-PROBE, M-RUNTIME-SELF-CHECK
#   LINKS: M-VALIDATION-AUTOMATION
#   ROLE: SCRIPT
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   DEFAULT_SERVER_HOST - Default bind host for automated local smoke runs
#   DEFAULT_SERVER_PORT - Default bind port for automated local smoke runs
#   CUSTOM_SMOKE_MODEL_ID - Default Qwen custom model used by server smoke validation
#   OMNIVOICE_SMOKE_MODEL_ID - OmniVoice custom model used by torch-backed smoke validation
#   SERVER_DOCKER_COMPOSE_FILE - Checked-in compose scenario used for server Docker parity validation
#   TELEGRAM_DOCKER_COMPOSE_FILE - Checked-in compose scenario used for Telegram Docker parity validation
#   resolve_smoke_model_folder - Resolve the local model directory name for a smoke model entry
#   ValidationCommandError - Structured validation failure carrying stable machine-readable semantics
#   _build_result_summary - Build a normalized command summary for success, failure, or advisory outcomes
#   _summary_exit_code - Map normalized validation summaries to CLI exit codes
#   _ensure_evidence_dir - Create the repository-local evidence directory for retained runtime-validation artifacts
#   _resolve_compose_invocation - Resolve the available Docker Compose invocation for parity validation commands
#   _run_compose_command - Execute a Docker Compose command with captured output and structured environment handling
#   _capture_compose_logs - Retain compose logs for a specific service under the evidence directory
#   _wait_for_server_docker_probes - Wait for Dockerized server health/model probes to succeed and parse JSON payloads
#   _wait_for_telegram_docker_startup - Wait for Telegram compose logs to prove startup, API connectivity, and polling activation
#   parse_args - Parse CLI arguments for validation subcommands
#   build_validation_env - Build a normalized environment mapping for validation runs
#   _next_update_offset - Compute the follow-up update offset after a baseline Telegram poll
#   _find_matching_update - Locate an inbound Telegram update that satisfies the validation filters
#   run_host_matrix_validation - Validate baseline and simulated optional-lane readiness expectations
#   run_smoke_server_validation - Start a local server, run smoke pytest, and stop the server cleanly
#   _representative_target_lookup - Resolve representative target metadata from the self-check payload
#   _build_representative_preflight_summary - Build machine-readable representative target preflight evidence
#   run_representative_model_validation - Validate one or all optional representative real-model targets through the existing smoke harness
#   run_server_docker_validation - Validate server Docker parity through compose startup, probes, logs, and teardown
#   run_telegram_live_validation - Validate Telegram Bot API reachability plus optional inbound update visibility with a real token
#   run_telegram_docker_validation - Validate Telegram Docker parity through compose startup, polling-log proof, host-side API proof, and teardown
#   _collect_reviewable_artifacts - Collect persisted runtime-validation artifacts from the evidence directory and explicit paths
#   _review_runtime_artifact - Build an advisory summary for one persisted artifact
#   run_artifact_review_validation - Review persisted runtime-validation evidence and emit advisory summaries only
#   main - Dispatch CLI subcommands and print JSON summaries
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.9.1 - Made CLI exit codes follow the structured validation envelope so advisory or skipped ValidationCommandError outcomes stay non-blocking while hard failures remain non-zero]
# END_CHANGE_SUMMARY

from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any, Mapping


# START_BLOCK_BOOTSTRAP_IMPORT_PATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# END_BLOCK_BOOTSTRAP_IMPORT_PATH

from scripts.runtime_self_check import build_self_check_payload  # noqa: E402
from telegram_bot.client import RetryConfig, TelegramBotClient  # noqa: E402


DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 0
CUSTOM_SMOKE_MODEL_ID = "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
OMNIVOICE_SMOKE_MODEL_ID = "OmniVoice-Custom"
PIPER_SMOKE_MODEL_ID = "Piper-en_US-lessac-medium"
SERVER_DOCKER_COMPOSE_FILE = PROJECT_ROOT / "docker-compose.server.yaml"
TELEGRAM_DOCKER_COMPOSE_FILE = PROJECT_ROOT / "docker-compose.telegram-bot.yaml"
SERVER_DOCKER_COMPOSE_PROJECT = "qwen3_tts_validate_server"
TELEGRAM_DOCKER_COMPOSE_PROJECT = "qwen3_tts_validate_telegram"
SERVER_DOCKER_HEALTH_LIVE_ARTIFACT = "server-docker-health-live.json"
SERVER_DOCKER_HEALTH_READY_ARTIFACT = "server-docker-health-ready.json"
SERVER_DOCKER_MODELS_ARTIFACT = "server-docker-models.json"
SERVER_DOCKER_LOG_ARTIFACT = "server-docker-log.txt"
TELEGRAM_DOCKER_LOG_ARTIFACT = "telegram-docker-log.txt"
TELEGRAM_DOCKER_REQUIRED_LOG_MARKERS = (
    "Telegram API connectivity verified",
    "Telegram bot startup complete, entering polling loop",
    "Polling loop is now active",
)
SUPPORTED_SMOKE_MODEL_IDS = (
    CUSTOM_SMOKE_MODEL_ID,
    OMNIVOICE_SMOKE_MODEL_ID,
    PIPER_SMOKE_MODEL_ID,
)
REPRESENTATIVE_TARGET_TO_MODEL_ID = {
    "qwen": CUSTOM_SMOKE_MODEL_ID,
    "omnivoice": OMNIVOICE_SMOKE_MODEL_ID,
    "piper": PIPER_SMOKE_MODEL_ID,
}
REPRESENTATIVE_TARGETS = tuple(REPRESENTATIVE_TARGET_TO_MODEL_ID)


class ValidationCommandError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        command: str,
        reason: str,
        outcome: str = "failed",
        stage: str | None = None,
        details: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.command = command
        self.reason = reason
        self.outcome = outcome
        self.stage = stage
        self.details = dict(details or {})
        self.artifacts = {
            key: value for key, value in dict(artifacts or {}).items() if value is not None
        }

    def to_summary(self) -> dict[str, Any]:
        status = "advisory" if self.outcome in {"advisory", "skipped"} else "error"
        return _build_result_summary(
            self.command,
            status=status,
            outcome=self.outcome,
            reason=self.reason,
            message=str(self),
            stage=self.stage,
            details=self.details,
            artifacts=self.artifacts,
        )


def _build_result_summary(
    command: str,
    *,
    status: str,
    outcome: str,
    reason: str,
    message: str | None = None,
    stage: str | None = None,
    details: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    advisories: list[dict[str, Any]] | None = None,
    **payload: Any,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "status": status,
        "command": command,
        "outcome": outcome,
        "reason": reason,
        "artifacts": {
            key: value
            for key, value in dict(artifacts or {}).items()
            if value is not None
        },
        "advisories": list(advisories or []),
    }
    if message is not None:
        summary["message"] = message
    if stage is not None:
        summary["stage"] = stage
    if details:
        summary["details"] = dict(details)
    summary.update(payload)
    return summary


def _summary_exit_code(summary: Mapping[str, Any]) -> int:
    status = str(summary.get("status") or "")
    outcome = str(summary.get("outcome") or "")
    if status in {"ok", "advisory"} and outcome in {"passed", "advisory", "skipped"}:
        return 0
    return 1


def _build_self_check_payload_with_diagnostics(
    environ: Mapping[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        payload = build_self_check_payload(environ)

    diagnostics: list[dict[str, Any]] = []
    for source, raw_output in (
        ("stdout", stdout_buffer.getvalue()),
        ("stderr", stderr_buffer.getvalue()),
    ):
        for message in [line.strip() for line in raw_output.splitlines() if line.strip()]:
            diagnostics.append(
                {
                    "kind": "captured_runtime_message",
                    "source": source,
                    "message": message,
                }
            )
    return payload, diagnostics


def _ensure_evidence_dir() -> Path:
    evidence_dir = PROJECT_ROOT / ".sisyphus" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    return evidence_dir


def _trim_command_output(raw_output: str, *, max_lines: int = 40) -> str:
    lines = [line.rstrip() for line in str(raw_output).splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(["...", *lines[-max_lines:]])


def _normalize_artifact_paths(paths: list[str] | None) -> list[Path]:
    normalized: list[Path] = []
    for raw_path in paths or []:
        candidate = Path(raw_path).expanduser()
        if candidate not in normalized:
            normalized.append(candidate)
    return normalized


def _collect_reviewable_artifacts(
    evidence_dir: Path, explicit_paths: list[str] | None = None
) -> list[Path]:
    reviewable_suffixes = {".json", ".txt", ".log"}
    collected: list[Path] = []
    if evidence_dir.exists():
        for artifact in sorted(evidence_dir.iterdir(), key=lambda item: item.name):
            if artifact.is_file() and artifact.suffix.lower() in reviewable_suffixes:
                collected.append(artifact)
    for artifact in _normalize_artifact_paths(explicit_paths):
        if artifact.is_file() and artifact.suffix.lower() in reviewable_suffixes and artifact not in collected:
            collected.append(artifact)
    return collected


def _read_text_artifact(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lower_text = text.lower()
    observed_signals = [
        marker
        for marker in ("traceback", "exception", "error", "failed", "failure")
        if marker in lower_text
    ]
    return {
        "path": path.as_posix(),
        "kind": "text",
        "line_count": len(text.splitlines()),
        "observed_signals": observed_signals,
        "excerpt": _trim_command_output(text, max_lines=12),
    }


def _review_json_artifact(path: Path) -> dict[str, Any]:
    raw_text = path.read_text(encoding="utf-8")
    payload = json.loads(raw_text)
    review: dict[str, Any] = {
        "path": path.as_posix(),
        "kind": "json",
        "top_level_type": type(payload).__name__,
    }
    if isinstance(payload, dict):
        review["top_level_keys"] = sorted(payload)
        authoritative_verdict = {
            key: payload.get(key)
            for key in ("status", "outcome", "reason", "command", "stage")
            if payload.get(key) is not None
        }
        if authoritative_verdict:
            review["authoritative_verdict"] = authoritative_verdict
        if payload.get("status") == "error" or payload.get("outcome") == "failed":
            review["observed_signals"] = ["authoritative_failure"]
        elif payload.get("status") == "advisory" or payload.get("outcome") == "skipped":
            review["observed_signals"] = ["authoritative_advisory"]
        else:
            review["observed_signals"] = ["authoritative_pass"]
        if isinstance(payload.get("artifacts"), dict):
            review["artifact_keys"] = sorted(payload["artifacts"])
    else:
        review["observed_signals"] = ["non_object_json"]
    return review


def _review_runtime_artifact(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        try:
            return _review_json_artifact(path)
        except json.JSONDecodeError as exc:
            return {
                "path": path.as_posix(),
                "kind": "json",
                "parse_error": str(exc),
                "observed_signals": ["invalid_json"],
            }
    return _read_text_artifact(path)


def _merge_error_context(
    error: ValidationCommandError,
    *,
    details: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    message: str | None = None,
) -> ValidationCommandError:
    merged_details = dict(error.details)
    merged_details.update(dict(details or {}))
    merged_artifacts = dict(error.artifacts)
    merged_artifacts.update({
        key: value for key, value in dict(artifacts or {}).items() if value is not None
    })
    return ValidationCommandError(
        message or str(error),
        command=error.command,
        reason=error.reason,
        outcome=error.outcome,
        stage=error.stage,
        details=merged_details,
        artifacts=merged_artifacts,
    )


def _compose_command_display(command: list[str]) -> str:
    return " ".join(command)


def _resolve_compose_invocation() -> tuple[list[str], str]:
    candidates: list[tuple[list[str], str]] = []
    if shutil.which("docker"):
        candidates.append((["docker", "compose"], "docker compose"))
    if shutil.which("docker-compose"):
        candidates.append((["docker-compose"], "docker-compose"))

    for invocation, label in candidates:
        try:
            result = subprocess.run(
                [*invocation, "version"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            continue
        if result.returncode == 0:
            return invocation, label

    raise RuntimeError(
        "Docker Compose is not available. Install Docker Desktop or a compose-capable Docker CLI."
    )


def _run_compose_command(
    compose_command: list[str],
    compose_file: Path,
    project_name: str,
    compose_args: list[str],
    *,
    environ: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            *compose_command,
            "-f",
            compose_file.as_posix(),
            "-p",
            project_name,
            *compose_args,
        ],
        cwd=PROJECT_ROOT,
        env=dict(environ),
        capture_output=True,
        text=True,
        check=False,
    )


def _capture_compose_logs(
    compose_command: list[str],
    compose_file: Path,
    project_name: str,
    service: str,
    artifact_path: Path,
    *,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    result = _run_compose_command(
        compose_command,
        compose_file,
        project_name,
        ["logs", "--no-color", service],
        environ=environ,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to capture compose logs for {service}: {_trim_command_output(result.stderr or result.stdout)}"
        )
    artifact_path.write_text(result.stdout, encoding="utf-8")
    return {
        "path": artifact_path.as_posix(),
        "line_count": len(result.stdout.splitlines()),
    }


def _parse_probe_json(stdout: str, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} returned unexpected payload type: {type(payload).__name__}")
    return payload


def _run_service_http_probe(
    compose_command: list[str],
    compose_file: Path,
    project_name: str,
    service: str,
    url: str,
    *,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    probe_script = (
        "import json, urllib.request; "
        f"response = urllib.request.urlopen({url!r}, timeout=5); "
        "print(response.read().decode('utf-8'))"
    )
    result = _run_compose_command(
        compose_command,
        compose_file,
        project_name,
        ["exec", "-T", service, "python", "-c", probe_script],
        environ=environ,
    )
    if result.returncode != 0:
        raise RuntimeError(_trim_command_output(result.stderr or result.stdout))
    return _parse_probe_json(result.stdout, label=url)


def _wait_for_server_docker_probes(
    compose_command: list[str],
    compose_file: Path,
    project_name: str,
    *,
    timeout_seconds: float,
    environ: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    last_error = "compose service is not ready yet"
    while time.monotonic() < deadline:
        try:
            live = _run_service_http_probe(
                compose_command,
                compose_file,
                project_name,
                "server",
                "http://127.0.0.1:8000/health/live",
                environ=environ,
            )
            ready = _run_service_http_probe(
                compose_command,
                compose_file,
                project_name,
                "server",
                "http://127.0.0.1:8000/health/ready",
                environ=environ,
            )
            models = _run_service_http_probe(
                compose_command,
                compose_file,
                project_name,
                "server",
                "http://127.0.0.1:8000/api/v1/models",
                environ=environ,
            )
        except RuntimeError as exc:
            last_error = str(exc)
            time.sleep(1.0)
            continue

        if not isinstance(models.get("data"), list):
            last_error = "model discovery payload did not contain a data list"
            time.sleep(1.0)
            continue

        return {"live": live, "ready": ready, "models": models}

    raise RuntimeError(
        f"Timed out waiting for Docker server probes to succeed: {last_error}"
    )


def _wait_for_telegram_docker_startup(
    compose_command: list[str],
    compose_file: Path,
    project_name: str,
    *,
    timeout_seconds: float,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_log_excerpt = "compose logs are not available yet"
    observed_markers: list[str] = []
    while time.monotonic() < deadline:
        try:
            result = _run_compose_command(
                compose_command,
                compose_file,
                project_name,
                ["logs", "--no-color", "telegram-bot"],
                environ=environ,
            )
        except OSError as exc:
            last_log_excerpt = str(exc)
            time.sleep(1.0)
            continue

        if result.returncode != 0:
            last_log_excerpt = _trim_command_output(result.stderr or result.stdout)
            time.sleep(1.0)
            continue

        log_output = result.stdout
        observed_markers = [
            marker for marker in TELEGRAM_DOCKER_REQUIRED_LOG_MARKERS if marker in log_output
        ]
        if len(observed_markers) == len(TELEGRAM_DOCKER_REQUIRED_LOG_MARKERS):
            return {
                "required_markers": list(TELEGRAM_DOCKER_REQUIRED_LOG_MARKERS),
                "observed_markers": observed_markers,
                "log_excerpt": _trim_command_output(log_output),
            }

        last_log_excerpt = _trim_command_output(log_output)
        time.sleep(1.0)

    raise RuntimeError(
        "Timed out waiting for Telegram Docker startup proof. "
        f"Observed markers: {observed_markers}. Last logs: {last_log_excerpt}"
    )


# START_CONTRACT: parse_args
#   PURPOSE: Parse CLI arguments for the runtime validation subcommands.
#   INPUTS: { argv: list[str] | None - optional raw CLI arguments }
#   OUTPUTS: { argparse.Namespace - parsed validation command payload }
#   SIDE_EFFECTS: none
#   LINKS: M-VALIDATION-AUTOMATION
# END_CONTRACT: parse_args
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run runtime validation flows for local smoke, host matrices, and Telegram live checks.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used for subprocess-based validation steps.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    host_parser = subparsers.add_parser(
        "host-matrix",
        help="Validate baseline host readiness plus simulated qwen_fast optional-lane scenarios.",
    )
    host_parser.add_argument(
        "--backend",
        default=None,
        help="Optional backend override for the self-check environment.",
    )

    smoke_parser = subparsers.add_parser(
        "smoke-server",
        help="Start a local HTTP server, run smoke pytest, and stop the server automatically.",
    )
    smoke_parser.add_argument("--host", default=DEFAULT_SERVER_HOST)
    smoke_parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help="Bind port for the temporary HTTP server. Use 0 to auto-select a free port.",
    )
    smoke_parser.add_argument(
        "--backend",
        default=None,
        help="Optional backend override for the temporary server process.",
    )
    smoke_parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=90.0,
        help="Maximum time to wait for the HTTP server to become healthy.",
    )
    smoke_parser.add_argument(
        "--expected-backend",
        default=None,
        help="Optional expected backend override passed into the smoke tests.",
    )
    smoke_parser.add_argument(
        "--smoke-model-id",
        default=(os.getenv("TTS_SMOKE_MODEL_ID") or CUSTOM_SMOKE_MODEL_ID),
        help=(
            "Smoke model id to validate through the HTTP API. "
            f"Defaults to TTS_SMOKE_MODEL_ID or {CUSTOM_SMOKE_MODEL_ID}."
        ),
    )
    smoke_parser.add_argument(
        "--strict-runtime",
        action="store_true",
        help="Fail when the self-check reports missing assets, instead of validating only smoke prerequisites.",
    )

    representative_parser = subparsers.add_parser(
        "representative-models",
        help="Run optional bounded representative real-model validation through the existing smoke-server harness.",
    )
    representative_parser.add_argument("--host", default=DEFAULT_SERVER_HOST)
    representative_parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help="Bind port for temporary representative smoke runs. Use 0 to auto-select a free port per target.",
    )
    representative_parser.add_argument(
        "--backend",
        default=None,
        help="Optional backend override for representative target validation.",
    )
    representative_parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=90.0,
        help="Maximum time to wait for each temporary HTTP server to become healthy.",
    )
    representative_parser.add_argument(
        "--strict-runtime",
        action="store_true",
        help="Fail representative runs when the self-check reports missing assets outside the selected target.",
    )
    representative_parser.add_argument(
        "--target",
        choices=REPRESENTATIVE_TARGETS,
        default=None,
        help="Optional representative target to validate. Defaults to all bounded targets.",
    )

    telegram_parser = subparsers.add_parser(
        "telegram-live",
        help="Validate Telegram Bot API reachability with a real bot token.",
    )
    telegram_parser.add_argument(
        "--bot-token",
        default=None,
        help="Telegram bot token override. Falls back to TTS_TELEGRAM_BOT_TOKEN.",
    )
    telegram_parser.add_argument(
        "--chat-id",
        type=int,
        default=None,
        help="Optional chat ID for an explicit sendMessage validation.",
    )
    telegram_parser.add_argument(
        "--message",
        default="Qwen3-TTS validation ping.",
        help="Optional message body for sendMessage validation when --chat-id is provided.",
    )
    telegram_parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Retry attempts for Telegram API validation requests.",
    )
    telegram_parser.add_argument(
        "--expect-update-chat-id",
        type=int,
        default=None,
        help="Optional inbound update chat ID expected during live validation. When set, the validator polls getUpdates for a matching message update.",
    )
    telegram_parser.add_argument(
        "--expect-update-text",
        default=None,
        help="Optional substring expected in a matching inbound Telegram message update.",
    )
    telegram_parser.add_argument(
        "--update-timeout-seconds",
        type=int,
        default=30,
        help="Long-poll timeout used when inbound update validation is requested.",
    )

    docker_server_parser = subparsers.add_parser(
        "docker-server",
        help="Validate server Docker parity with compose startup, HTTP probes, retained logs, and teardown.",
    )
    docker_server_parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=120.0,
        help="Maximum time to wait for the Dockerized server probes to succeed.",
    )

    docker_telegram_parser = subparsers.add_parser(
        "docker-telegram",
        help="Validate Telegram bot Docker parity with compose startup, polling-log proof, host-side API proof, and teardown.",
    )
    docker_telegram_parser.add_argument(
        "--bot-token",
        default=None,
        help="Telegram bot token override. Falls back to TTS_TELEGRAM_BOT_TOKEN.",
    )
    docker_telegram_parser.add_argument(
        "--chat-id",
        type=int,
        default=None,
        help="Optional chat ID for sendMessage proof during host-side Telegram validation.",
    )
    docker_telegram_parser.add_argument(
        "--message",
        default="Qwen3-TTS validation ping.",
        help="Optional message body for sendMessage validation when --chat-id is provided.",
    )
    docker_telegram_parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Retry attempts for Telegram API validation requests.",
    )
    docker_telegram_parser.add_argument(
        "--expect-update-chat-id",
        type=int,
        default=None,
        help="Optional inbound update chat ID expected during host-side Telegram validation.",
    )
    docker_telegram_parser.add_argument(
        "--expect-update-text",
        default=None,
        help="Optional substring expected in a matching inbound Telegram message update.",
    )
    docker_telegram_parser.add_argument(
        "--update-timeout-seconds",
        type=int,
        default=30,
        help="Long-poll timeout used when inbound update validation is requested.",
    )
    docker_telegram_parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=90.0,
        help="Maximum time to wait for Telegram compose logs to prove startup and polling activation.",
    )

    artifact_review_parser = subparsers.add_parser(
        "artifact-review",
        help="Review persisted runtime-validation evidence and emit advisory summaries only.",
    )
    artifact_review_parser.add_argument(
        "artifact_path",
        nargs="*",
        default=[],
        help="Optional explicit persisted artifact paths to review in addition to .sisyphus/evidence.",
    )

    return parser.parse_args(argv)


# START_CONTRACT: build_validation_env
#   PURPOSE: Build a normalized environment mapping for validation commands with repository-local defaults.
#   INPUTS: { environ: Mapping[str, str] | None - optional base environment, backend: str | None - optional backend override, host: str | None - optional bind host override, port: int | None - optional bind port override }
#   OUTPUTS: { dict[str, str] - normalized environment payload for self-check and subprocess validation }
#   SIDE_EFFECTS: none
#   LINKS: M-CONFIG, M-VALIDATION-AUTOMATION
# END_CONTRACT: build_validation_env
def build_validation_env(
    environ: Mapping[str, str] | None = None,
    *,
    backend: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> dict[str, str]:
    env = dict(os.environ if environ is None else environ)
    sox_dir = Path.home() / "AppData" / "Local" / "Programs" / "sox"
    if sox_dir.exists():
        current_path = env.get("PATH", "")
        path_entries = [entry for entry in current_path.split(os.pathsep) if entry]
        normalized_entries = {entry.lower() for entry in path_entries}
        if str(sox_dir).lower() not in normalized_entries:
            env["PATH"] = os.pathsep.join([str(sox_dir), *path_entries])
    env.setdefault(
        "TTS_MODELS_DIR", (PROJECT_ROOT / ".models").resolve().as_posix()
    )
    env.setdefault(
        "TTS_MLX_MODELS_DIR",
        (PROJECT_ROOT / ".models" / "mlx").resolve().as_posix(),
    )
    env.setdefault(
        "TTS_OUTPUTS_DIR", (PROJECT_ROOT / ".outputs").resolve().as_posix()
    )
    env.setdefault(
        "TTS_VOICES_DIR", (PROJECT_ROOT / ".voices").resolve().as_posix()
    )
    env.setdefault(
        "TTS_UPLOAD_STAGING_DIR",
        (PROJECT_ROOT / ".uploads").resolve().as_posix(),
    )
    if backend is not None:
        env["TTS_BACKEND"] = backend
    if host is not None:
        env["TTS_HOST"] = host
    if port is not None:
        env["TTS_PORT"] = str(port)
    return env


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _choose_server_port(host: str, requested_port: int) -> int:
    if requested_port > 0:
        _require(
            _is_port_available(host, requested_port),
            f"Requested smoke-server port is already in use: {requested_port}",
        )
        return requested_port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _backend_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    backends = payload["readiness"].get("available_backends", [])
    return {item["key"]: item for item in backends}


def _model_entry(payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    for item in payload["readiness"].get("items", []):
        if item.get("id") == model_id:
            return item
    raise RuntimeError(f"Model entry not found in readiness payload: {model_id}")


def resolve_smoke_model_folder(model_entry: Mapping[str, Any]) -> str:
    folder = str(model_entry.get("folder") or "").strip()
    _require(
        bool(folder),
        f"Smoke model entry is missing folder metadata: {model_entry.get('id')}",
    )
    return folder


def _route_candidate(model_entry: dict[str, Any], backend_key: str) -> dict[str, Any]:
    candidates = model_entry.get("route", {}).get("candidates", [])
    for candidate in candidates:
        if candidate.get("key") == backend_key:
            return candidate
    raise RuntimeError(
        f"Route candidate '{backend_key}' not found for model {model_entry.get('id')}"
    )


def _request_json(url: str, *, timeout: float = 10.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return {
            "status": response.status,
            "json": json.loads(response.read().decode("utf-8")),
        }


# START_CONTRACT: _next_update_offset
#   PURPOSE: Compute the next Telegram update offset after observing a baseline batch.
#   INPUTS: { updates: list[dict[str, Any]] - baseline Telegram updates }
#   OUTPUTS: { int - next update offset suitable for long polling newer updates }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM, M-VALIDATION-AUTOMATION
# END_CONTRACT: _next_update_offset
def _next_update_offset(updates: list[dict[str, Any]]) -> int:
    max_update_id = max(
        (int(item.get("update_id", 0)) for item in updates if isinstance(item, dict)),
        default=0,
    )
    return max_update_id + 1 if max_update_id > 0 else 0


# START_CONTRACT: _find_matching_update
#   PURPOSE: Locate an inbound Telegram update that matches the requested chat and message filters.
#   INPUTS: { updates: list[dict[str, Any]] - candidate Telegram updates, expected_chat_id: int | None - optional chat constraint, expected_text: str | None - optional message substring constraint }
#   OUTPUTS: { dict[str, Any] | None - the first matching Telegram update or None }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM, M-VALIDATION-AUTOMATION
# END_CONTRACT: _find_matching_update
def _find_matching_update(
    updates: list[dict[str, Any]],
    *,
    expected_chat_id: int | None,
    expected_text: str | None,
) -> dict[str, Any] | None:
    normalized_expected_text = expected_text.strip() if expected_text else None
    for update in updates:
        if not isinstance(update, dict):
            continue
        message = update.get("message") or update.get("edited_message")
        if not isinstance(message, dict):
            continue
        chat = message.get("chat")
        chat_id = chat.get("id") if isinstance(chat, dict) else None
        if expected_chat_id is not None and chat_id != expected_chat_id:
            continue
        message_text = message.get("text") or message.get("caption") or ""
        if normalized_expected_text and normalized_expected_text not in str(
            message_text
        ):
            continue
        return update
    return None


def _wait_for_server(base_url: str, *, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = "server did not respond"
    while time.monotonic() < deadline:
        try:
            live = _request_json(f"{base_url}/health/live")
            ready = _request_json(f"{base_url}/health/ready")
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            time.sleep(1.0)
            continue
        except Exception as exc:  # pragma: no cover - environment-dependent polling
            last_error = str(exc)
            time.sleep(1.0)
            continue
        if live["status"] == 200 and ready["status"] == 200:
            return {"live": live["json"], "ready": ready["json"]}
        last_error = f"live_status={live['status']}, ready_status={ready['status']}"
        time.sleep(1.0)
    raise RuntimeError(
        f"Timed out waiting for server health at {base_url}: {last_error}"
    )


def _raise_validation_error(
    command: str,
    reason: str,
    message: str,
    *,
    outcome: str = "failed",
    stage: str | None = None,
    details: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
) -> None:
    raise ValidationCommandError(
        message,
        command=command,
        reason=reason,
        outcome=outcome,
        stage=stage,
        details=details,
        artifacts=artifacts,
    )


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def _representative_target_lookup(
    payload: dict[str, Any],
    target: str,
) -> dict[str, Any]:
    for item in payload.get("representative_models", {}).get("targets", []):
        if item.get("target") == target:
            return item
    raise RuntimeError(f"Representative target not found in self-check payload: {target}")


def _build_representative_preflight_summary(
    payload: dict[str, Any],
    target: str,
) -> dict[str, Any]:
    target_entry = _representative_target_lookup(payload, target)
    return {
        "target": target,
        "model_id": target_entry.get("model_id"),
        "status": target_entry.get("status"),
        "reason": target_entry.get("reason"),
        "expected_backend": target_entry.get("expected_backend"),
        "selected_backend": target_entry.get("selected_backend"),
        "execution_backend": target_entry.get("execution_backend"),
        "available": target_entry.get("available"),
        "loadable": target_entry.get("loadable"),
        "runtime_ready": target_entry.get("runtime_ready"),
        "missing_artifacts": list(target_entry.get("missing_artifacts") or []),
        "required_artifacts": list(target_entry.get("required_artifacts") or []),
        "route_reason": target_entry.get("route_reason"),
    }


# START_CONTRACT: run_host_matrix_validation
#   PURPOSE: Validate current-host readiness plus simulated qwen_fast optional-lane outcomes used by support and CI claims.
#   INPUTS: { environ: Mapping[str, str] - normalized validation environment }
#   OUTPUTS: { dict[str, Any] - structured validation summary for baseline and simulated host/lane checks }
#   SIDE_EFFECTS: Builds multiple self-check payloads with different qwen_fast test modes
#   LINKS: M-HOST-PROBE, M-BACKENDS, M-RUNTIME-SELF-CHECK
# END_CONTRACT: run_host_matrix_validation
def run_host_matrix_validation(environ: Mapping[str, str]) -> dict[str, Any]:
    # START_BLOCK_BUILD_BASELINE_PAYLOADS
    baseline, baseline_diagnostics = _build_self_check_payload_with_diagnostics(environ)
    eligible_env = dict(environ)
    eligible_env["TTS_QWEN_FAST_TEST_MODE"] = "eligible"
    eligible, eligible_diagnostics = _build_self_check_payload_with_diagnostics(eligible_env)
    cuda_missing_env = dict(environ)
    cuda_missing_env["TTS_QWEN_FAST_TEST_MODE"] = "cuda_missing"
    cuda_missing, cuda_missing_diagnostics = _build_self_check_payload_with_diagnostics(
        cuda_missing_env
    )
    dependency_missing_env = dict(environ)
    dependency_missing_env["TTS_QWEN_FAST_TEST_MODE"] = "dependency_missing"
    dependency_missing, dependency_missing_diagnostics = _build_self_check_payload_with_diagnostics(
        dependency_missing_env
    )
    # END_BLOCK_BUILD_BASELINE_PAYLOADS

    # START_BLOCK_ASSERT_HOST_MATRIX
    host = baseline["readiness"]["host"]
    baseline_backends = _backend_map(baseline)
    eligible_backends = _backend_map(eligible)
    cuda_missing_backends = _backend_map(cuda_missing)
    dependency_missing_backends = _backend_map(dependency_missing)

    _require(
        host["platform_system"] in {"darwin", "linux", "windows"},
        f"Unexpected platform in host probe: {host['platform_system']}",
    )
    _require("qwen_fast" in baseline_backends, "qwen_fast backend is missing")
    _require("torch" in baseline_backends, "torch backend is missing")
    _require("mlx" in baseline_backends, "mlx backend is missing")
    _require("onnx" in baseline_backends, "onnx backend is missing")

    baseline_qwen_fast = baseline_backends["qwen_fast"]
    qwen_fast_enabled = bool(
        baseline["settings"].get("qwen_fast_enabled", True)
    )
    if host["platform_system"] == "darwin":
        _require(
            baseline_qwen_fast["diagnostics"]["reason"] == "platform_unsupported",
            "Expected qwen_fast to report platform_unsupported on Darwin baseline",
        )

    eligible_qwen_fast = eligible_backends["qwen_fast"]
    _require(
        eligible_qwen_fast["diagnostics"]["details"].get("test_mode") == "eligible",
        "Expected qwen_fast eligible simulation to report eligible test mode",
    )

    eligible_custom_route = _route_candidate(
        _model_entry(eligible, CUSTOM_SMOKE_MODEL_ID), "qwen_fast"
    )
    eligible_design_route = _route_candidate(
        _model_entry(eligible, "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"), "qwen_fast"
    )
    eligible_clone_route = _route_candidate(
        _model_entry(eligible, "Qwen3-TTS-12Hz-1.7B-Base-8bit"), "qwen_fast"
    )
    if qwen_fast_enabled:
        _require(
            eligible_qwen_fast["diagnostics"]["ready"] is True,
            "Expected qwen_fast eligible simulation to become ready",
        )
        _require(
            eligible_custom_route["ready"] is True,
            "Expected qwen_fast eligible simulation to produce a ready custom route",
        )
        _require(
            eligible_custom_route["route_reason"] == "route_candidate_accepted",
            "Expected qwen_fast eligible simulation to accept the custom route",
        )
        _require(
            eligible_design_route["ready"] is True,
            "Expected qwen_fast eligible simulation to produce a ready design route candidate",
        )
        _require(
            eligible_design_route["route_reason"] == "route_candidate_accepted",
            "Expected qwen_fast eligible simulation to accept the design route candidate",
        )
        _require(
            eligible_clone_route["ready"] is True,
            "Expected qwen_fast eligible simulation to produce a ready clone route candidate",
        )
        _require(
            eligible_clone_route["route_reason"] == "route_candidate_accepted",
            "Expected qwen_fast eligible simulation to accept the clone route candidate",
        )
    else:
        _require(
            eligible_qwen_fast["diagnostics"]["reason"] == "disabled_by_config",
            "Expected qwen_fast eligible simulation to remain disabled when qwen_fast is disabled by config",
        )
        _require(
            eligible_custom_route["route_reason"] == "disabled_by_config",
            "Expected qwen_fast eligible simulation to keep the custom route disabled when qwen_fast is disabled by config",
        )
        _require(
            eligible_design_route["route_reason"] == "disabled_by_config",
            "Expected qwen_fast eligible simulation to keep the design route disabled when qwen_fast is disabled by config",
        )
        _require(
            eligible_clone_route["route_reason"] == "disabled_by_config",
            "Expected qwen_fast eligible simulation to keep the clone route disabled when qwen_fast is disabled by config",
        )

    cuda_missing_qwen_fast = cuda_missing_backends["qwen_fast"]
    if qwen_fast_enabled:
        _require(
            cuda_missing_qwen_fast["diagnostics"]["reason"] == "cuda_required",
            "Expected qwen_fast cuda_missing simulation to report cuda_required",
        )
    else:
        _require(
            cuda_missing_qwen_fast["diagnostics"]["reason"] == "disabled_by_config",
            "Expected qwen_fast cuda_missing simulation to remain disabled when qwen_fast is disabled by config",
        )

    dependency_missing_qwen_fast = dependency_missing_backends["qwen_fast"]
    if qwen_fast_enabled:
        _require(
            dependency_missing_qwen_fast["diagnostics"]["reason"]
            == "runtime_dependency_missing",
            "Expected qwen_fast dependency_missing simulation to report runtime_dependency_missing",
        )
    else:
        _require(
            dependency_missing_qwen_fast["diagnostics"]["reason"] == "disabled_by_config",
            "Expected qwen_fast dependency_missing simulation to remain disabled when qwen_fast is disabled by config",
        )
    # END_BLOCK_ASSERT_HOST_MATRIX

    # START_BLOCK_BUILD_HOST_MATRIX_SUMMARY
    advisories: list[dict[str, Any]] = []
    advisories.extend(baseline_diagnostics)
    advisories.extend(eligible_diagnostics)
    advisories.extend(cuda_missing_diagnostics)
    advisories.extend(dependency_missing_diagnostics)
    if baseline_qwen_fast["diagnostics"]["ready"] is not True:
        advisories.append(
            {
                "kind": "backend_optional_lane",
                "subject": "qwen_fast",
                "reason": baseline_qwen_fast["diagnostics"]["reason"],
                "details": baseline_qwen_fast["diagnostics"].get("details", {}),
            }
        )
    return _build_result_summary(
        "host-matrix",
        status="ok",
        outcome="passed",
        reason="host_matrix_validated",
        host=host,
        baseline={
            "selected_backend": baseline["readiness"]["backend_diagnostics"]["backend"],
            "qwen_fast_reason": baseline_qwen_fast["diagnostics"]["reason"],
            "torch_reason": baseline_backends["torch"]["diagnostics"]["reason"],
        },
        simulated_qwen_fast={
            "eligible": {
                "ready": eligible_qwen_fast["diagnostics"]["ready"],
                "reason": eligible_qwen_fast["diagnostics"]["reason"],
                "custom_route_reason": eligible_custom_route["route_reason"],
                "design_route_reason": eligible_design_route["route_reason"],
                "clone_route_reason": eligible_clone_route["route_reason"],
            },
            "cuda_missing": {
                "ready": cuda_missing_qwen_fast["diagnostics"]["ready"],
                "reason": cuda_missing_qwen_fast["diagnostics"]["reason"],
            },
            "dependency_missing": {
                "ready": dependency_missing_qwen_fast["diagnostics"]["ready"],
                "reason": dependency_missing_qwen_fast["diagnostics"]["reason"],
            },
        },
        advisories=advisories,
    )
    # END_BLOCK_BUILD_HOST_MATRIX_SUMMARY


# START_CONTRACT: run_smoke_server_validation
#   PURPOSE: Start a temporary local HTTP server, run smoke pytest against it, and stop the server regardless of outcome.
#   INPUTS: { args: argparse.Namespace - parsed smoke-server CLI arguments, environ: Mapping[str, str] - normalized validation environment }
#   OUTPUTS: { dict[str, Any] - structured smoke validation summary including base URL and server log path }
#   SIDE_EFFECTS: Starts a uvicorn subprocess, performs HTTP health checks, runs pytest, and terminates the subprocess
#   LINKS: M-SERVER, M-RUNTIME-SELF-CHECK
# END_CONTRACT: run_smoke_server_validation
def run_smoke_server_validation(
    args: argparse.Namespace, environ: Mapping[str, str]
) -> dict[str, Any]:
    # START_BLOCK_PREPARE_SMOKE_ENVIRONMENT
    command = "smoke-server"
    smoke_model_id = str(getattr(args, "smoke_model_id", CUSTOM_SMOKE_MODEL_ID)).strip()
    if smoke_model_id not in SUPPORTED_SMOKE_MODEL_IDS:
        _raise_validation_error(
            command,
            "unsupported_smoke_model_id",
            (
                "Unsupported smoke model id "
                f"'{smoke_model_id}'. Supported values: {', '.join(SUPPORTED_SMOKE_MODEL_IDS)}"
            ),
            stage="preflight",
            details={"smoke_model_id": smoke_model_id},
        )
    resolved_port = _choose_server_port(args.host, int(args.port))
    env = build_validation_env(
        environ,
        backend=args.backend,
        host=args.host,
        port=resolved_port,
    )
    payload, self_check_diagnostics = _build_self_check_payload_with_diagnostics(env)
    host = payload["readiness"]["host"]
    smoke_model = _model_entry(payload, smoke_model_id)
    smoke_model_dir = (
        Path(env["TTS_MODELS_DIR"]) / resolve_smoke_model_folder(smoke_model)
    )
    if not smoke_model_dir.exists():
        _raise_validation_error(
            command,
            "model_directory_missing",
            f"Required smoke model directory is missing: {smoke_model_dir}",
            stage="preflight",
            details={
                "smoke_model_id": smoke_model_id,
                "smoke_model_dir": smoke_model_dir.as_posix(),
            },
        )
    if smoke_model["runtime_ready"] is not True:
        _raise_validation_error(
            command,
            "model_runtime_not_ready",
            f"Required smoke model is not runtime-ready: {smoke_model_id}",
            stage="preflight",
            details={
                "smoke_model_id": smoke_model_id,
                "execution_backend": smoke_model.get("execution_backend"),
            },
        )
    if smoke_model_id == PIPER_SMOKE_MODEL_ID:
        if smoke_model.get("execution_backend") != "onnx":
            _raise_validation_error(
                command,
                "piper_execution_backend_invalid",
                (
                    "Piper smoke validation expects ONNX execution backend, "
                    f"got: {smoke_model.get('execution_backend')}"
                ),
                stage="preflight",
                details={"smoke_model_id": smoke_model_id},
            )
    if not host["ffmpeg_available"]:
        _raise_validation_error(
            command,
            "ffmpeg_missing",
            "ffmpeg is required for smoke validation",
            stage="preflight",
        )
    if args.strict_runtime:
        if payload["assets"]["models_missing_assets"]:
            _raise_validation_error(
                command,
                "runtime_assets_missing",
                "Strict smoke validation requested, but the runtime self-check still reports missing assets",
                stage="preflight",
                details={
                    "missing_assets": payload["assets"]["models_missing_assets"],
                },
            )
    expected_backend = (
        args.expected_backend
        or smoke_model.get("execution_backend")
        or payload["readiness"]["backend_diagnostics"]["backend"]
    )
    if smoke_model_id == PIPER_SMOKE_MODEL_ID:
        if str(expected_backend) != "onnx":
            _raise_validation_error(
                command,
                "expected_backend_mismatch",
                f"Piper smoke validation requires expected backend 'onnx', got '{expected_backend}'",
                stage="preflight",
                details={
                    "smoke_model_id": smoke_model_id,
                    "expected_backend": str(expected_backend),
                },
            )
    base_url = f"http://{args.host}:{resolved_port}"
    advisories: list[dict[str, Any]] = []
    advisories.extend(self_check_diagnostics)
    if payload["assets"]["models_missing_assets"]:
        advisories.append(
            {
                "kind": "runtime_assets_missing",
                "reason": "models_missing_assets",
                "details": {
                    "missing_assets": payload["assets"]["models_missing_assets"],
                },
            }
        )
    # END_BLOCK_PREPARE_SMOKE_ENVIRONMENT

    # START_BLOCK_RUN_SMOKE_SERVER
    log_file = tempfile.NamedTemporaryFile(
        mode="w+",
        prefix="qwen3_tts_smoke_server_",
        suffix=".log",
        delete=False,
    )
    process: subprocess.Popen[Any] | None = None
    try:
        process = subprocess.Popen(
            [
                args.python_executable,
                "-m",
                "uvicorn",
                "server:app",
                "--host",
                args.host,
                "--port",
                str(resolved_port),
            ],
            cwd=PROJECT_ROOT,
            env={**env, "PYTHONUNBUFFERED": "1"},
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        health = _wait_for_server(
            base_url, timeout_seconds=args.startup_timeout_seconds
        )

        smoke_env = dict(env)
        smoke_env["TTS_RUN_SMOKE"] = "1"
        smoke_env["TTS_SMOKE_BASE_URL"] = base_url
        smoke_env["TTS_SMOKE_MODEL_ID"] = smoke_model_id
        smoke_env["TTS_SMOKE_EXPECTED_BACKEND"] = str(expected_backend)
        smoke_result = subprocess.run(
            [
                args.python_executable,
                "-m",
                "pytest",
                "tests/smoke/test_local_runtime.py",
                "-m",
                "smoke",
            ],
            cwd=PROJECT_ROOT,
            env=smoke_env,
            check=False,
        )
        if smoke_result.returncode != 0:
            _raise_validation_error(
                command,
                "smoke_pytest_failed",
                f"Smoke pytest failed with exit code {smoke_result.returncode}. Server log: {log_file.name}",
                stage="smoke_pytest",
                details={"smoke_returncode": smoke_result.returncode},
                artifacts={"server_log_path": log_file.name},
            )
    except ValidationCommandError:
        raise
    except Exception as exc:
        reason = "server_startup_timeout"
        stage = "server_startup"
        if isinstance(exc, RuntimeError) and str(exc).startswith("Timed out waiting for server health"):
            reason = "server_startup_timeout"
        else:
            reason = "smoke_server_runtime_error"
            stage = "server_orchestration"
        _raise_validation_error(
            command,
            reason,
            f"{exc}. Server log: {log_file.name}",
            stage=stage,
            artifacts={"server_log_path": log_file.name},
        )
    finally:
        if process is not None:
            _stop_process(process)
        log_file.close()
    # END_BLOCK_RUN_SMOKE_SERVER

    # START_BLOCK_BUILD_SMOKE_SUMMARY
    return _build_result_summary(
        command,
        status="ok",
        outcome="passed",
        reason="smoke_server_validated",
        stage="completed",
        artifacts={"server_log_path": log_file.name},
        advisories=advisories,
        base_url=base_url,
        smoke_model_id=smoke_model_id,
        expected_backend=expected_backend,
        server_log_path=log_file.name,
        health=health,
        preflight={
            "resolved_port": resolved_port,
            "runtime_strict": bool(args.strict_runtime),
            "host_ffmpeg_available": bool(host["ffmpeg_available"]),
            "smoke_model_dir": smoke_model_dir.as_posix(),
            "self_check_selected_backend": payload["readiness"]["backend_diagnostics"]["backend"],
        },
    )
    # END_BLOCK_BUILD_SMOKE_SUMMARY


def run_representative_model_validation(
    args: argparse.Namespace, environ: Mapping[str, str]
) -> dict[str, Any]:
    command = "representative-models"
    env = build_validation_env(
        environ,
        backend=args.backend,
        host=args.host,
        port=int(args.port),
    )
    payload, self_check_diagnostics = _build_self_check_payload_with_diagnostics(env)
    requested_targets = [args.target] if args.target else list(REPRESENTATIVE_TARGETS)
    target_summaries: list[dict[str, Any]] = []
    advisories: list[dict[str, Any]] = list(self_check_diagnostics)

    for target in requested_targets:
        preflight = _build_representative_preflight_summary(payload, target)
        target_entry = _representative_target_lookup(payload, target)
        target_status = target_entry.get("status")
        target_reason = str(target_entry.get("reason") or "runtime_not_ready")
        model_id = str(target_entry.get("model_id"))
        expected_backend = target_entry.get("execution_backend")

        if target_status != "ready":
            outcome = "failed" if target_status == "failed" else "skipped"
            status = "error" if outcome == "failed" else "advisory"
            target_summary = _build_result_summary(
                command,
                status=status,
                outcome=outcome,
                reason=target_reason,
                message=str(target_entry.get("message") or "Representative target did not satisfy validation preflight."),
                stage="preflight",
                representative_target=target,
                smoke_model_id=model_id,
                expected_backend=expected_backend,
                preflight=preflight,
            )
            target_summaries.append(target_summary)
            if outcome == "skipped":
                advisories.append(
                    {
                        "kind": "representative_model_skipped",
                        "target": target,
                        "reason": target_reason,
                        "details": preflight,
                    }
                )
            continue

        smoke_args = argparse.Namespace(
            command="smoke-server",
            python_executable=args.python_executable,
            host=args.host,
            port=args.port,
            backend=args.backend,
            startup_timeout_seconds=args.startup_timeout_seconds,
            expected_backend=expected_backend,
            smoke_model_id=model_id,
            strict_runtime=args.strict_runtime,
        )
        try:
            smoke_summary = run_smoke_server_validation(smoke_args, env)
            target_summaries.append(
                _build_result_summary(
                    command,
                    status="ok",
                    outcome="passed",
                    reason="representative_model_validated",
                    stage="completed",
                    representative_target=target,
                    smoke_model_id=model_id,
                    expected_backend=expected_backend,
                    preflight=preflight,
                    artifacts=smoke_summary.get("artifacts"),
                    smoke_summary=smoke_summary,
                )
            )
        except ValidationCommandError as exc:
            target_summaries.append(
                _build_result_summary(
                    command,
                    status="error",
                    outcome="failed",
                    reason=exc.reason,
                    message=str(exc),
                    stage=exc.stage,
                    representative_target=target,
                    smoke_model_id=model_id,
                    expected_backend=expected_backend,
                    preflight=preflight,
                    details=exc.details,
                    artifacts=exc.artifacts,
                )
            )

    if any(item["outcome"] == "failed" for item in target_summaries):
        overall_status = "error"
        overall_outcome = "failed"
        overall_reason = "representative_model_validation_failed"
    elif any(item["outcome"] == "passed" for item in target_summaries):
        overall_status = "ok"
        overall_outcome = "passed"
        overall_reason = "representative_model_validation_completed"
    else:
        overall_status = "advisory"
        overall_outcome = "skipped"
        overall_reason = "representative_model_validation_skipped"

    return _build_result_summary(
        command,
        status=overall_status,
        outcome=overall_outcome,
        reason=overall_reason,
        stage="completed",
        advisories=advisories,
        targets=target_summaries,
        requested_targets=requested_targets,
    )


def run_server_docker_validation(
    args: argparse.Namespace, environ: Mapping[str, str]
) -> dict[str, Any]:
    command = "docker-server"
    evidence_dir = _ensure_evidence_dir()
    live_artifact = evidence_dir / SERVER_DOCKER_HEALTH_LIVE_ARTIFACT
    ready_artifact = evidence_dir / SERVER_DOCKER_HEALTH_READY_ARTIFACT
    models_artifact = evidence_dir / SERVER_DOCKER_MODELS_ARTIFACT
    log_artifact = evidence_dir / SERVER_DOCKER_LOG_ARTIFACT
    retained_artifacts = {
        "server_log_path": log_artifact.as_posix(),
        "health_live_path": live_artifact.as_posix(),
        "health_ready_path": ready_artifact.as_posix(),
        "models_path": models_artifact.as_posix(),
    }

    if not SERVER_DOCKER_COMPOSE_FILE.exists():
        _raise_validation_error(
            command,
            "compose_file_missing",
            f"Server Docker parity requires compose file: {SERVER_DOCKER_COMPOSE_FILE.as_posix()}",
            stage="preflight",
            artifacts=retained_artifacts,
        )

    try:
        compose_command, compose_label = _resolve_compose_invocation()
    except RuntimeError as exc:
        _raise_validation_error(
            command,
            "docker_compose_unavailable",
            str(exc),
            outcome="skipped",
            stage="preflight",
            details={"requires": ["docker compose", "docker-compose"]},
            artifacts=retained_artifacts,
        )

    env = build_validation_env(environ)
    try:
        resolved_host_port = _choose_server_port("0.0.0.0", 0)
    except RuntimeError as exc:
        _raise_validation_error(
            command,
            "docker_host_port_unavailable",
            f"Server Docker parity could not reserve a free host port for the compose mapping: {exc}",
            stage="preflight",
            details={"compose_host": "0.0.0.0"},
            artifacts=retained_artifacts,
        )
    env["TTS_SERVER_PORT"] = str(resolved_host_port)
    main_error: ValidationCommandError | None = None
    probes: dict[str, dict[str, Any]] | None = None
    compose_attempted = False
    teardown_details: dict[str, Any] = {
        "attempted": False,
        "succeeded": False,
    }

    try:
        compose_attempted = True
        up_result = _run_compose_command(
            compose_command,
            SERVER_DOCKER_COMPOSE_FILE,
            SERVER_DOCKER_COMPOSE_PROJECT,
            ["up", "--build", "-d", "server"],
            environ=env,
        )
        if up_result.returncode != 0:
            _raise_validation_error(
                command,
                "docker_compose_up_failed",
                "Server Docker parity failed to start the compose service.",
                stage="compose_up",
                details={
                    "returncode": up_result.returncode,
                    "host_port": resolved_host_port,
                    "stderr": _trim_command_output(up_result.stderr),
                    "stdout": _trim_command_output(up_result.stdout),
                },
                artifacts=retained_artifacts,
            )

        probes = _wait_for_server_docker_probes(
            compose_command,
            SERVER_DOCKER_COMPOSE_FILE,
            SERVER_DOCKER_COMPOSE_PROJECT,
            timeout_seconds=args.startup_timeout_seconds,
            environ=env,
        )
        live_artifact.write_text(json.dumps(probes["live"], indent=2, sort_keys=True), encoding="utf-8")
        ready_artifact.write_text(json.dumps(probes["ready"], indent=2, sort_keys=True), encoding="utf-8")
        models_artifact.write_text(json.dumps(probes["models"], indent=2, sort_keys=True), encoding="utf-8")
    except ValidationCommandError as exc:
        main_error = exc
    except Exception as exc:
        main_error = ValidationCommandError(
            str(exc),
            command=command,
            reason="docker_server_probe_failed",
            stage="probe",
            details={"error_type": type(exc).__name__},
            artifacts=retained_artifacts,
        )
    finally:
        if compose_attempted:
            try:
                _capture_compose_logs(
                    compose_command,
                    SERVER_DOCKER_COMPOSE_FILE,
                    SERVER_DOCKER_COMPOSE_PROJECT,
                    "server",
                    log_artifact,
                    environ=env,
                )
            except Exception as exc:
                capture_error = ValidationCommandError(
                    f"Failed to retain server Docker logs: {exc}",
                    command=command,
                    reason="docker_logs_capture_failed",
                    stage="log_capture",
                    details={"error_type": type(exc).__name__},
                    artifacts=retained_artifacts,
                )
                main_error = (
                    capture_error
                    if main_error is None
                    else _merge_error_context(
                        main_error,
                        details={
                            "log_capture_error": str(exc),
                            "log_capture_error_type": type(exc).__name__,
                        },
                        artifacts=retained_artifacts,
                    )
                )

            teardown_details["attempted"] = True
            teardown_result = _run_compose_command(
                compose_command,
                SERVER_DOCKER_COMPOSE_FILE,
                SERVER_DOCKER_COMPOSE_PROJECT,
                ["down", "--remove-orphans"],
                environ=env,
            )
            teardown_details.update(
                {
                    "succeeded": teardown_result.returncode == 0,
                    "returncode": teardown_result.returncode,
                    "stdout": _trim_command_output(teardown_result.stdout),
                    "stderr": _trim_command_output(teardown_result.stderr),
                }
            )
            if teardown_result.returncode != 0:
                teardown_error = ValidationCommandError(
                    "Server Docker parity teardown failed.",
                    command=command,
                    reason="docker_teardown_failed",
                    stage="teardown",
                    details=teardown_details,
                    artifacts=retained_artifacts,
                )
                main_error = (
                    teardown_error
                    if main_error is None
                    else _merge_error_context(
                        main_error,
                        details={"teardown": teardown_details},
                        artifacts=retained_artifacts,
                    )
                )

    if main_error is not None:
        raise _merge_error_context(
            main_error,
            details={"teardown": teardown_details},
            artifacts=retained_artifacts,
        )

    return _build_result_summary(
        command,
        status="ok",
        outcome="passed",
        reason="server_docker_validated",
        stage="completed",
        artifacts=retained_artifacts,
        compose={
            "file": SERVER_DOCKER_COMPOSE_FILE.as_posix(),
            "project": SERVER_DOCKER_COMPOSE_PROJECT,
            "service": "server",
            "host_port": resolved_host_port,
            "invocation": compose_label,
        },
        probes=probes,
        teardown=teardown_details,
    )


# START_CONTRACT: run_telegram_live_validation
#   PURPOSE: Validate Telegram Bot API reachability with a real bot token without entering the polling loop.
#   INPUTS: { args: argparse.Namespace - parsed telegram-live CLI arguments, environ: Mapping[str, str] - normalized validation environment }
#   OUTPUTS: { dict[str, Any] - structured Telegram connectivity summary }
#   SIDE_EFFECTS: Performs live Telegram Bot API requests and may optionally send a message to a target chat
#   LINKS: M-TELEGRAM
# END_CONTRACT: run_telegram_live_validation
async def run_telegram_live_validation(
    args: argparse.Namespace, environ: Mapping[str, str]
) -> dict[str, Any]:
    # START_BLOCK_INIT_TELEGRAM_CLIENT
    command = "telegram-live"
    token = (args.bot_token or environ.get("TTS_TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        _raise_validation_error(
            command,
            "telegram_bot_token_missing",
            "Telegram live validation requires --bot-token or TTS_TELEGRAM_BOT_TOKEN",
            outcome="advisory",
            stage="preflight",
            details={
                "requires": [
                    "--bot-token",
                    "TTS_TELEGRAM_BOT_TOKEN",
                ]
            },
        )
    client = TelegramBotClient(
        bot_token=token,
        retry_config=RetryConfig(max_attempts=args.max_attempts),
    )
    # END_BLOCK_INIT_TELEGRAM_CLIENT

    # START_BLOCK_EXECUTE_TELEGRAM_VALIDATION
    try:
        bot_info = await client.get_me()
        summary = _build_result_summary(
            command,
            status="ok",
            outcome="passed",
            reason="telegram_api_reachable",
            stage="completed",
            bot_id=bot_info.get("id"),
            bot_username=bot_info.get("username"),
            bot_name=bot_info.get("first_name"),
            message_sent=False,
            update_checked=False,
            dedicated_chat_check_requested=False,
            advisories=[],
        )
        if args.chat_id is not None:
            message = await client.send_message(args.chat_id, args.message)
            summary["message_sent"] = True
            summary["message_id"] = message.get("message_id")

        expected_update_text = (
            args.expect_update_text.strip() if args.expect_update_text else None
        )
        update_requested = (
            args.expect_update_chat_id is not None or expected_update_text is not None
        )
        summary["dedicated_chat_check_requested"] = bool(
            args.chat_id is not None or update_requested
        )
        if update_requested:
            if args.update_timeout_seconds < 0:
                _raise_validation_error(
                    command,
                    "telegram_update_timeout_invalid",
                    "Telegram live validation requires a non-negative --update-timeout-seconds",
                    stage="preflight",
                    details={"update_timeout_seconds": args.update_timeout_seconds},
                )
            expected_update_chat_id = args.expect_update_chat_id
            if expected_update_chat_id is None:
                expected_update_chat_id = args.chat_id
            if expected_update_chat_id is None:
                summary["update_checked"] = False
                summary["update_check_status"] = "skipped"
                summary["update_check_reason"] = "expected_update_chat_id_missing"
                summary["advisories"].append(
                    {
                        "kind": "telegram_update_check_skipped",
                        "reason": "expected_update_chat_id_missing",
                        "message": (
                            "Skipping inbound update proof because no dedicated chat id was provided via "
                            "--expect-update-chat-id or --chat-id"
                        ),
                    }
                )
                return summary
            baseline_updates = await client.get_updates(
                timeout=0,
                allowed_updates=["message", "edited_message"],
            )
            next_offset = _next_update_offset(baseline_updates)
            polled_updates = await client.get_updates(
                offset=next_offset,
                timeout=args.update_timeout_seconds,
                allowed_updates=["message", "edited_message"],
            )
            matched_update = _find_matching_update(
                polled_updates,
                expected_chat_id=expected_update_chat_id,
                expected_text=expected_update_text,
            )
            matching_chat_update = _find_matching_update(
                polled_updates,
                expected_chat_id=expected_update_chat_id,
                expected_text=None,
            )
            update_kind = "message"
            if matched_update is None:
                details = {
                    "expected_update_chat_id": expected_update_chat_id,
                    "expected_update_text": expected_update_text,
                    "update_poll_offset": next_offset,
                    "update_poll_count": len(polled_updates),
                }
                if matching_chat_update is not None and expected_update_text:
                    matched_message = (
                        matching_chat_update.get("message")
                        or matching_chat_update.get("edited_message")
                        or {}
                    )
                    details["observed_update_chat_id"] = expected_update_chat_id
                    details["observed_update_text"] = (
                        matched_message.get("text") or matched_message.get("caption") or ""
                    )
                    _raise_validation_error(
                        command,
                        "telegram_matching_update_text_mismatch",
                        "Telegram live validation observed a newer update for the expected chat, but its text did not match the requested proof",
                        outcome="advisory",
                        stage="update_poll",
                        details=details,
                    )
                _raise_validation_error(
                    command,
                    "telegram_matching_update_not_found",
                    "Telegram live validation did not observe a matching inbound update",
                    outcome="advisory",
                    stage="update_poll",
                    details=details,
                )
            matched_message = (
                matched_update.get("message")
                or matched_update.get("edited_message")
                or {}
            )
            if matched_update.get("edited_message"):
                update_kind = "edited_message"
            matched_chat = matched_message.get("chat")
            summary["update_checked"] = True
            summary["update_check_status"] = "matched"
            summary["update_check_reason"] = "telegram_matching_update_observed"
            summary["expected_update_chat_id"] = expected_update_chat_id
            summary["expected_update_text"] = expected_update_text
            summary["update_poll_offset"] = next_offset
            summary["update_poll_count"] = len(polled_updates)
            summary["matched_update_id"] = matched_update.get("update_id")
            summary["matched_update_chat_id"] = (
                matched_chat.get("id") if isinstance(matched_chat, dict) else None
            )
            summary["matched_update_kind"] = update_kind
        return summary
    finally:
        await client.close()
    # END_BLOCK_EXECUTE_TELEGRAM_VALIDATION


async def run_telegram_docker_validation(
    args: argparse.Namespace, environ: Mapping[str, str]
) -> dict[str, Any]:
    command = "docker-telegram"
    evidence_dir = _ensure_evidence_dir()
    log_artifact = evidence_dir / TELEGRAM_DOCKER_LOG_ARTIFACT
    retained_artifacts = {"telegram_log_path": log_artifact.as_posix()}

    if not TELEGRAM_DOCKER_COMPOSE_FILE.exists():
        _raise_validation_error(
            command,
            "compose_file_missing",
            f"Telegram Docker parity requires compose file: {TELEGRAM_DOCKER_COMPOSE_FILE.as_posix()}",
            stage="preflight",
            artifacts=retained_artifacts,
        )

    token = (args.bot_token or environ.get("TTS_TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        _raise_validation_error(
            command,
            "telegram_bot_token_missing",
            "Telegram Docker parity requires --bot-token or TTS_TELEGRAM_BOT_TOKEN.",
            outcome="skipped",
            stage="preflight",
            details={
                "requires": [
                    "--bot-token",
                    "TTS_TELEGRAM_BOT_TOKEN",
                ]
            },
            artifacts=retained_artifacts,
        )

    try:
        compose_command, compose_label = _resolve_compose_invocation()
    except RuntimeError as exc:
        _raise_validation_error(
            command,
            "docker_compose_unavailable",
            str(exc),
            outcome="skipped",
            stage="preflight",
            details={"requires": ["docker compose", "docker-compose"]},
            artifacts=retained_artifacts,
        )

    env = build_validation_env(environ)
    env["TTS_TELEGRAM_BOT_TOKEN"] = token
    main_error: ValidationCommandError | None = None
    startup_proof: dict[str, Any] | None = None
    api_summary: dict[str, Any] | None = None
    advisories: list[dict[str, Any]] = []
    compose_attempted = False
    teardown_details: dict[str, Any] = {
        "attempted": False,
        "succeeded": False,
    }

    try:
        compose_attempted = True
        up_result = _run_compose_command(
            compose_command,
            TELEGRAM_DOCKER_COMPOSE_FILE,
            TELEGRAM_DOCKER_COMPOSE_PROJECT,
            ["up", "--build", "-d", "telegram-bot"],
            environ=env,
        )
        if up_result.returncode != 0:
            _raise_validation_error(
                command,
                "docker_compose_up_failed",
                "Telegram Docker parity failed to start the compose service.",
                stage="compose_up",
                details={
                    "returncode": up_result.returncode,
                    "stderr": _trim_command_output(up_result.stderr),
                    "stdout": _trim_command_output(up_result.stdout),
                },
                artifacts=retained_artifacts,
            )

        startup_proof = _wait_for_telegram_docker_startup(
            compose_command,
            TELEGRAM_DOCKER_COMPOSE_FILE,
            TELEGRAM_DOCKER_COMPOSE_PROJECT,
            timeout_seconds=args.startup_timeout_seconds,
            environ=env,
        )

        live_args = argparse.Namespace(
            command="telegram-live",
            bot_token=token,
            chat_id=args.chat_id,
            message=args.message,
            max_attempts=args.max_attempts,
            expect_update_chat_id=args.expect_update_chat_id,
            expect_update_text=args.expect_update_text,
            update_timeout_seconds=args.update_timeout_seconds,
        )
        try:
            api_summary = await run_telegram_live_validation(live_args, env)
        except ValidationCommandError as exc:
            main_error = ValidationCommandError(
                f"Telegram Docker parity collected startup/polling proof, but the host-side Telegram API proof is {exc.outcome}: {exc}",
                command=command,
                reason=exc.reason,
                outcome=exc.outcome,
                stage="api_proof",
                details={
                    "startup_proof": startup_proof,
                    "telegram_live": exc.to_summary(),
                },
                artifacts=retained_artifacts,
            )

        if args.chat_id is None and args.expect_update_chat_id is None:
            advisories.append(
                {
                    "kind": "telegram_dedicated_chat_check_skipped",
                    "reason": "telegram_validation_chat_id_missing",
                    "message": (
                        "Startup, polling, and Bot API reachability were validated, but dedicated-chat "
                        "sendMessage/getUpdates proof was skipped because no --chat-id or --expect-update-chat-id was provided."
                    ),
                }
            )
        if api_summary is not None:
            advisories.extend(api_summary.get("advisories", []))
    except ValidationCommandError as exc:
        main_error = exc
    except Exception as exc:
        main_error = ValidationCommandError(
            str(exc),
            command=command,
            reason="telegram_docker_startup_proof_failed",
            stage="startup_proof",
            details={"error_type": type(exc).__name__},
            artifacts=retained_artifacts,
        )
    finally:
        if compose_attempted:
            try:
                _capture_compose_logs(
                    compose_command,
                    TELEGRAM_DOCKER_COMPOSE_FILE,
                    TELEGRAM_DOCKER_COMPOSE_PROJECT,
                    "telegram-bot",
                    log_artifact,
                    environ=env,
                )
            except Exception as exc:
                capture_error = ValidationCommandError(
                    f"Failed to retain Telegram Docker logs: {exc}",
                    command=command,
                    reason="docker_logs_capture_failed",
                    stage="log_capture",
                    details={"error_type": type(exc).__name__},
                    artifacts=retained_artifacts,
                )
                main_error = (
                    capture_error
                    if main_error is None
                    else _merge_error_context(
                        main_error,
                        details={
                            "log_capture_error": str(exc),
                            "log_capture_error_type": type(exc).__name__,
                        },
                        artifacts=retained_artifacts,
                    )
                )

            teardown_details["attempted"] = True
            teardown_result = _run_compose_command(
                compose_command,
                TELEGRAM_DOCKER_COMPOSE_FILE,
                TELEGRAM_DOCKER_COMPOSE_PROJECT,
                ["down", "--remove-orphans"],
                environ=env,
            )
            teardown_details.update(
                {
                    "succeeded": teardown_result.returncode == 0,
                    "returncode": teardown_result.returncode,
                    "stdout": _trim_command_output(teardown_result.stdout),
                    "stderr": _trim_command_output(teardown_result.stderr),
                }
            )
            if teardown_result.returncode != 0:
                teardown_error = ValidationCommandError(
                    "Telegram Docker parity teardown failed.",
                    command=command,
                    reason="docker_teardown_failed",
                    stage="teardown",
                    details=teardown_details,
                    artifacts=retained_artifacts,
                )
                main_error = (
                    teardown_error
                    if main_error is None
                    else _merge_error_context(
                        main_error,
                        details={"teardown": teardown_details},
                        artifacts=retained_artifacts,
                    )
                )

    if main_error is not None:
        raise _merge_error_context(
            main_error,
            details={"teardown": teardown_details},
            artifacts=retained_artifacts,
        )

    return _build_result_summary(
        command,
        status="ok",
        outcome="passed",
        reason="telegram_docker_validated",
        stage="completed",
        artifacts=retained_artifacts,
        advisories=advisories,
        compose={
            "file": TELEGRAM_DOCKER_COMPOSE_FILE.as_posix(),
            "project": TELEGRAM_DOCKER_COMPOSE_PROJECT,
            "service": "telegram-bot",
            "invocation": compose_label,
        },
        startup_proof=startup_proof,
        telegram_live=api_summary,
        teardown=teardown_details,
    )


# START_CONTRACT: run_artifact_review_validation
#   PURPOSE: Review already persisted runtime-validation artifacts and emit advisory summaries only.
#   INPUTS: { args: argparse.Namespace - parsed artifact-review CLI arguments, environ: Mapping[str, str] - normalized validation environment }
#   OUTPUTS: { dict[str, Any] - advisory review summary built exclusively from persisted artifacts }
#   SIDE_EFFECTS: Reads repository-local persisted artifact files only, never live process state
#   LINKS: M-VALIDATION-AUTOMATION, M-VALIDATION-EVIDENCE
# END_CONTRACT: run_artifact_review_validation
def run_artifact_review_validation(
    args: argparse.Namespace, environ: Mapping[str, str]
) -> dict[str, Any]:
    command = "artifact-review"
    evidence_dir = _ensure_evidence_dir()
    reviewable_artifacts = _collect_reviewable_artifacts(
        evidence_dir, getattr(args, "artifact_path", None)
    )
    if not reviewable_artifacts:
        return _build_result_summary(
            command,
            status="advisory",
            outcome="skipped",
            reason="artifact_review_evidence_missing",
            stage="preflight",
            artifacts={"evidence_dir": evidence_dir.as_posix()},
            advisories=[
                {
                    "kind": "artifact_review_lane",
                    "message": (
                        "Artifact review was invoked without persisted evidence. No live process state was read."
                    ),
                    "evidence_dir": evidence_dir.as_posix(),
                    "review_count": 0,
                }
            ],
            source="persisted_artifacts_only",
            review_count=0,
            reviewed_artifacts=[],
            reviews=[],
            authoritative_signals=[],
            explicit_paths=[Path(path).expanduser().as_posix() for path in getattr(args, "artifact_path", []) or []],
            contains_authoritative_failure=False,
        )

    reviews = [_review_runtime_artifact(path) for path in reviewable_artifacts]
    authoritative_signals: list[str] = []
    for review in reviews:
        authoritative_signals.extend(review.get("observed_signals", []))

    advisory_summary = _build_result_summary(
        command,
        status="advisory",
        outcome="skipped",
        reason="artifact_review_completed",
        stage="completed",
        artifacts={"evidence_dir": evidence_dir.as_posix()},
        source="persisted_artifacts_only",
        review_count=len(reviews),
        reviewed_artifacts=[review["path"] for review in reviews],
        reviews=reviews,
        advisories=[
            {
                "kind": "artifact_review_lane",
                "message": (
                    "This lane is advisory only. Deterministic/runtime validation remains authoritative over this summary."
                ),
                "evidence_dir": evidence_dir.as_posix(),
                "review_count": len(reviews),
            }
        ],
        authoritative_signals=sorted(set(authoritative_signals)),
        explicit_paths=[Path(path).expanduser().as_posix() for path in getattr(args, "artifact_path", []) or []],
    )

    advisory_summary["contains_authoritative_failure"] = any(
        review.get("observed_signals") == ["authoritative_failure"] for review in reviews
    )

    return advisory_summary


# START_CONTRACT: main
#   PURPOSE: Dispatch the requested validation subcommand and print a structured JSON summary.
#   INPUTS: { argv: list[str] | None - optional raw CLI arguments }
#   OUTPUTS: { int - process exit code }
#   SIDE_EFFECTS: May start subprocesses, perform HTTP requests, and write JSON to stdout
#   LINKS: M-VALIDATION-AUTOMATION
# END_CONTRACT: main
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    environ = build_validation_env()

    try:
        # START_BLOCK_DISPATCH_COMMAND
        if args.command == "host-matrix":
            summary = run_host_matrix_validation(
                build_validation_env(environ, backend=args.backend)
            )
        elif args.command == "smoke-server":
            summary = run_smoke_server_validation(args, environ)
        elif args.command == "representative-models":
            summary = run_representative_model_validation(args, environ)
        elif args.command == "docker-server":
            summary = run_server_docker_validation(args, environ)
        elif args.command == "telegram-live":
            summary = asyncio.run(run_telegram_live_validation(args, environ))
        elif args.command == "docker-telegram":
            summary = asyncio.run(run_telegram_docker_validation(args, environ))
        elif args.command == "artifact-review":
            summary = run_artifact_review_validation(args, environ)
        else:  # pragma: no cover
            raise RuntimeError(f"Unsupported validation command: {args.command}")
        # END_BLOCK_DISPATCH_COMMAND
    except ValidationCommandError as exc:
        summary = exc.to_summary()
    except Exception as exc:
        summary = _build_result_summary(
            args.command,
            status="error",
            outcome="failed",
            reason="unhandled_validation_exception",
            message=str(exc),
            details={"error_type": type(exc).__name__},
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return _summary_exit_code(summary)


if __name__ == "__main__":
    raise SystemExit(main())
