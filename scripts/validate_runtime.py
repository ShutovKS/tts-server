#!/usr/bin/env python3
# FILE: scripts/validate_runtime.py
# VERSION: 1.2.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide operator- and CI-facing validation commands for host-specific backend checks, automated local smoke orchestration, and optional Telegram live connectivity.
#   SCOPE: CLI subcommands, runtime validation environment helpers, HTTP smoke start/stop orchestration, qwen_fast host-matrix assertions, Telegram Bot API reachability checks, and opt-in inbound Telegram update validation
#   DEPENDS: M-CONFIG, M-BOOTSTRAP, M-SERVER, M-TELEGRAM, M-HOST-PROBE, M-RUNTIME-SELF-CHECK
#   LINKS: M-VALIDATION-AUTOMATION
#   ROLE: SCRIPT
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   DEFAULT_SERVER_HOST - Default bind host for automated local smoke runs
#   DEFAULT_SERVER_PORT - Default bind port for automated local smoke runs
#   CUSTOM_SMOKE_MODEL_ID - Required custom model used by server smoke validation
#   parse_args - Parse CLI arguments for validation subcommands
#   build_validation_env - Build a normalized environment mapping for validation runs
#   _next_update_offset - Compute the follow-up update offset after a baseline Telegram poll
#   _find_matching_update - Locate an inbound Telegram update that satisfies the validation filters
#   run_host_matrix_validation - Validate baseline and simulated optional-lane readiness expectations
#   run_smoke_server_validation - Start a local server, run smoke pytest, and stop the server cleanly
#   run_telegram_live_validation - Validate Telegram Bot API reachability plus optional inbound update visibility with a real token
#   main - Dispatch CLI subcommands and print JSON summaries
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.2.0 - Added model-aware smoke-server selection so smoke validation can target default Qwen custom coverage or Piper ONNX routing through the existing HTTP API]
# END_CHANGE_SUMMARY

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
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
PIPER_SMOKE_MODEL_ID = "Piper-en_US-lessac-medium"
SUPPORTED_SMOKE_MODEL_IDS = (CUSTOM_SMOKE_MODEL_ID, PIPER_SMOKE_MODEL_ID)


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
        default=(os.getenv("QWEN_TTS_SMOKE_MODEL_ID") or CUSTOM_SMOKE_MODEL_ID),
        help=(
            "Smoke model id to validate through the HTTP API. "
            f"Defaults to QWEN_TTS_SMOKE_MODEL_ID or {CUSTOM_SMOKE_MODEL_ID}."
        ),
    )
    smoke_parser.add_argument(
        "--strict-runtime",
        action="store_true",
        help="Fail when the self-check reports missing assets, instead of validating only smoke prerequisites.",
    )

    telegram_parser = subparsers.add_parser(
        "telegram-live",
        help="Validate Telegram Bot API reachability with a real bot token.",
    )
    telegram_parser.add_argument(
        "--bot-token",
        default=None,
        help="Telegram bot token override. Falls back to QWEN_TTS_TELEGRAM_BOT_TOKEN.",
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
    env.setdefault(
        "QWEN_TTS_MODELS_DIR", (PROJECT_ROOT / ".models").resolve().as_posix()
    )
    env.setdefault(
        "QWEN_TTS_MLX_MODELS_DIR",
        (PROJECT_ROOT / ".models" / "mlx").resolve().as_posix(),
    )
    env.setdefault(
        "QWEN_TTS_OUTPUTS_DIR", (PROJECT_ROOT / ".outputs").resolve().as_posix()
    )
    env.setdefault(
        "QWEN_TTS_VOICES_DIR", (PROJECT_ROOT / ".voices").resolve().as_posix()
    )
    env.setdefault(
        "QWEN_TTS_UPLOAD_STAGING_DIR",
        (PROJECT_ROOT / ".uploads").resolve().as_posix(),
    )
    if backend is not None:
        env["QWEN_TTS_BACKEND"] = backend
    if host is not None:
        env["QWEN_TTS_HOST"] = host
    if port is not None:
        env["QWEN_TTS_PORT"] = str(port)
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


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


# START_CONTRACT: run_host_matrix_validation
#   PURPOSE: Validate current-host readiness plus simulated qwen_fast optional-lane outcomes used by support and CI claims.
#   INPUTS: { environ: Mapping[str, str] - normalized validation environment }
#   OUTPUTS: { dict[str, Any] - structured validation summary for baseline and simulated host/lane checks }
#   SIDE_EFFECTS: Builds multiple self-check payloads with different qwen_fast test modes
#   LINKS: M-HOST-PROBE, M-BACKENDS, M-RUNTIME-SELF-CHECK
# END_CONTRACT: run_host_matrix_validation
def run_host_matrix_validation(environ: Mapping[str, str]) -> dict[str, Any]:
    # START_BLOCK_BUILD_BASELINE_PAYLOADS
    baseline = build_self_check_payload(environ)
    eligible_env = dict(environ)
    eligible_env["QWEN_TTS_QWEN_FAST_TEST_MODE"] = "eligible"
    eligible = build_self_check_payload(eligible_env)
    cuda_missing_env = dict(environ)
    cuda_missing_env["QWEN_TTS_QWEN_FAST_TEST_MODE"] = "cuda_missing"
    cuda_missing = build_self_check_payload(cuda_missing_env)
    dependency_missing_env = dict(environ)
    dependency_missing_env["QWEN_TTS_QWEN_FAST_TEST_MODE"] = "dependency_missing"
    dependency_missing = build_self_check_payload(dependency_missing_env)
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
    if qwen_fast_enabled:
        _require(
            eligible_qwen_fast["diagnostics"]["ready"] is True,
            "Expected qwen_fast eligible simulation to become ready",
        )
        _require(
            eligible_custom_route["ready"] is True,
            "Expected qwen_fast eligible simulation to produce a ready custom route candidate",
        )
        _require(
            eligible_custom_route["route_reason"] == "route_candidate_accepted",
            "Expected qwen_fast eligible simulation to accept the custom route candidate",
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
    return {
        "status": "ok",
        "host": host,
        "baseline": {
            "selected_backend": baseline["readiness"]["backend_diagnostics"]["backend"],
            "qwen_fast_reason": baseline_qwen_fast["diagnostics"]["reason"],
            "torch_reason": baseline_backends["torch"]["diagnostics"]["reason"],
        },
        "simulated_qwen_fast": {
            "eligible": {
                "ready": eligible_qwen_fast["diagnostics"]["ready"],
                "reason": eligible_qwen_fast["diagnostics"]["reason"],
                "custom_route_reason": eligible_custom_route["route_reason"],
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
    }
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
    smoke_model_id = str(getattr(args, "smoke_model_id", CUSTOM_SMOKE_MODEL_ID)).strip()
    _require(
        smoke_model_id in SUPPORTED_SMOKE_MODEL_IDS,
        (
            "Unsupported smoke model id "
            f"'{smoke_model_id}'. Supported values: {', '.join(SUPPORTED_SMOKE_MODEL_IDS)}"
        ),
    )
    resolved_port = _choose_server_port(args.host, int(args.port))
    env = build_validation_env(
        environ,
        backend=args.backend,
        host=args.host,
        port=resolved_port,
    )
    payload = build_self_check_payload(env)
    host = payload["readiness"]["host"]
    smoke_model_dir = Path(env["QWEN_TTS_MODELS_DIR"]) / smoke_model_id
    smoke_model = _model_entry(payload, smoke_model_id)
    _require(
        smoke_model_dir.exists(),
        f"Required smoke model directory is missing: {smoke_model_dir}",
    )
    _require(
        smoke_model["runtime_ready"] is True,
        f"Required smoke model is not runtime-ready: {smoke_model_id}",
    )
    if smoke_model_id == PIPER_SMOKE_MODEL_ID:
        _require(
            smoke_model.get("execution_backend") == "onnx",
            (
                "Piper smoke validation expects ONNX execution backend, "
                f"got: {smoke_model.get('execution_backend')}"
            ),
        )
    _require(host["ffmpeg_available"], "ffmpeg is required for smoke validation")
    if args.strict_runtime:
        _require(
            not payload["assets"]["models_missing_assets"],
            "Strict smoke validation requested, but the runtime self-check still reports missing assets",
        )
    expected_backend = (
        args.expected_backend
        or smoke_model.get("execution_backend")
        or payload["readiness"]["backend_diagnostics"]["backend"]
    )
    if smoke_model_id == PIPER_SMOKE_MODEL_ID:
        _require(
            str(expected_backend) == "onnx",
            f"Piper smoke validation requires expected backend 'onnx', got '{expected_backend}'",
        )
    base_url = f"http://{args.host}:{resolved_port}"
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
        smoke_env["QWEN_TTS_RUN_SMOKE"] = "1"
        smoke_env["QWEN_TTS_SMOKE_BASE_URL"] = base_url
        smoke_env["QWEN_TTS_SMOKE_MODEL_ID"] = smoke_model_id
        smoke_env["QWEN_TTS_SMOKE_EXPECTED_BACKEND"] = str(expected_backend)
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
        _require(
            smoke_result.returncode == 0,
            f"Smoke pytest failed with exit code {smoke_result.returncode}. Server log: {log_file.name}",
        )
    except Exception as exc:
        raise RuntimeError(f"{exc}. Server log: {log_file.name}") from exc
    finally:
        if process is not None:
            _stop_process(process)
        log_file.close()
    # END_BLOCK_RUN_SMOKE_SERVER

    # START_BLOCK_BUILD_SMOKE_SUMMARY
    return {
        "status": "ok",
        "base_url": base_url,
        "smoke_model_id": smoke_model_id,
        "expected_backend": expected_backend,
        "server_log_path": log_file.name,
        "health": health,
    }
    # END_BLOCK_BUILD_SMOKE_SUMMARY


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
    token = (args.bot_token or environ.get("QWEN_TTS_TELEGRAM_BOT_TOKEN") or "").strip()
    _require(
        bool(token),
        "Telegram live validation requires --bot-token or QWEN_TTS_TELEGRAM_BOT_TOKEN",
    )
    client = TelegramBotClient(
        bot_token=token,
        retry_config=RetryConfig(max_attempts=args.max_attempts),
    )
    # END_BLOCK_INIT_TELEGRAM_CLIENT

    # START_BLOCK_EXECUTE_TELEGRAM_VALIDATION
    try:
        bot_info = await client.get_me()
        summary = {
            "status": "ok",
            "bot_id": bot_info.get("id"),
            "bot_username": bot_info.get("username"),
            "bot_name": bot_info.get("first_name"),
            "message_sent": False,
            "update_checked": False,
        }
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
        if update_requested:
            _require(
                args.update_timeout_seconds >= 0,
                "Telegram live validation requires a non-negative --update-timeout-seconds",
            )
            expected_update_chat_id = args.expect_update_chat_id
            if expected_update_chat_id is None:
                expected_update_chat_id = args.chat_id
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
            _require(
                matched_update is not None,
                "Telegram live validation did not observe a matching inbound update",
            )
            matched_message = (
                matched_update.get("message")
                or matched_update.get("edited_message")
                or {}
            )
            matched_chat = matched_message.get("chat")
            summary["update_checked"] = True
            summary["expected_update_chat_id"] = expected_update_chat_id
            summary["expected_update_text"] = expected_update_text
            summary["update_poll_offset"] = next_offset
            summary["update_poll_count"] = len(polled_updates)
            summary["matched_update_id"] = matched_update.get("update_id")
            summary["matched_update_chat_id"] = (
                matched_chat.get("id") if isinstance(matched_chat, dict) else None
            )
        return summary
    finally:
        await client.close()
    # END_BLOCK_EXECUTE_TELEGRAM_VALIDATION


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
        elif args.command == "telegram-live":
            summary = asyncio.run(run_telegram_live_validation(args, environ))
        else:  # pragma: no cover
            raise RuntimeError(f"Unsupported validation command: {args.command}")
        # END_BLOCK_DISPATCH_COMMAND
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "command": args.command,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
