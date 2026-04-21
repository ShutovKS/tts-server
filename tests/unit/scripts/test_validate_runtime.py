# FILE: tests/unit/scripts/test_validate_runtime.py
# VERSION: 1.8.1
# START_MODULE_CONTRACT
#   PURPOSE: Verify runtime validation automation helpers for host-matrix checks, smoke-server orchestration, Docker parity validation, and Telegram live validation flows.
#   SCOPE: Validation env defaults, qwen_fast simulation assertions, representative-model preflight semantics, smoke preflight/runtime error handling, Docker evidence/teardown semantics, smoke summary evidence payloads, and opt-in Telegram inbound update validation
#   DEPENDS: M-VALIDATION-AUTOMATION, M-RUNTIME-SELF-CHECK
#   LINKS: V-M-VALIDATION-AUTOMATION
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_build_validation_env_sets_repo_defaults - Verifies validation env helper populates repository-local paths
#   test_parse_args_smoke_server_defaults_to_env_smoke_model_id - Verifies smoke-server parser defaults smoke model id from environment
#   test_parse_args_smoke_server_allows_cli_smoke_model_id_override - Verifies smoke-server parser accepts explicit smoke model id overrides
#   test_run_host_matrix_validation_accepts_simulated_qwen_fast_modes - Verifies optional-lane host matrix validation succeeds with simulated qwen_fast states
#   test_run_host_matrix_validation_adds_machine_readable_result_fields - Verifies host-matrix summaries expose command/outcome/reason and advisory entries
#   test_run_representative_model_validation_skips_missing_assets_target - Verifies representative validation returns an explicit skipped summary when a target is missing assets
#   test_run_representative_model_validation_skips_optional_dependency_target - Verifies representative validation reports missing optional dependency packs explicitly
#   test_run_representative_model_validation_fails_for_corrupt_or_incomplete_target - Verifies representative validation distinguishes corrupt/incomplete artifacts from simple skips
#   test_run_representative_model_validation_reuses_smoke_server_for_ready_target - Verifies representative validation reuses the smoke-server harness for runnable targets
#   test_run_smoke_server_validation_returns_summary_with_expected_evidence_payload - Verifies smoke validation returns base URL, expected backend, server log path, and health evidence when orchestration succeeds
#   test_run_smoke_server_validation_returns_machine_readable_preflight_and_artifact_fields - Verifies smoke validation summary includes explicit command/outcome/reason, preflight diagnostics, and retained artifacts
#   test_run_smoke_server_validation_uses_expected_backend_override_in_smoke_env - Verifies smoke validation passes an explicit expected backend override through to smoke pytest
#   test_run_smoke_server_validation_rejects_wrong_expected_backend_for_piper - Verifies Piper smoke validation fails fast when the caller expects a non-ONNX backend
#   test_run_smoke_server_validation_requires_runtime_ready_custom_model - Verifies smoke validation fails fast when the custom smoke model is not runtime-ready
#   test_run_smoke_server_validation_requires_existing_model_directory_before_runtime_ready - Verifies smoke validation distinguishes missing local model directories from runtime-readiness failures
#   test_run_smoke_server_validation_rejects_unsupported_smoke_model_id - Verifies smoke validation fails fast for unknown smoke model ids
#   test_run_smoke_server_validation_rejects_missing_ffmpeg_preflight - Verifies smoke validation fails at host preflight when ffmpeg is unavailable
#   test_run_smoke_server_validation_reports_startup_timeout_with_server_log_artifact - Verifies startup timeouts carry explicit reason, stage, and retained server log artifact
#   test_run_smoke_server_validation_allows_missing_assets_when_not_strict - Verifies smoke validation keeps running when self-check reports missing assets but strict runtime is disabled
#   test_run_smoke_server_validation_rejects_strict_runtime_missing_assets - Verifies smoke validation fails strict runtime requests when self-check still reports missing assets
#   test_parse_args_accepts_docker_server_timeout_override - Verifies docker-server parser accepts explicit startup timeout overrides
#   test_parse_args_accepts_docker_telegram_options - Verifies docker-telegram parser accepts compose and Telegram proof options
#   test_run_server_docker_validation_returns_probe_and_log_artifacts - Verifies server Docker validation returns retained probe/log evidence plus explicit teardown details
#   test_run_server_docker_validation_selects_free_host_port_for_compose_mapping - Verifies docker-server chooses a free host port at runtime and injects it into the compose environment
#   test_run_server_docker_validation_returns_skipped_summary_when_compose_is_unavailable - Verifies server Docker validation downgrades missing Docker Compose to an explicit skipped outcome
#   test_run_server_docker_validation_reports_teardown_failure_with_retained_artifacts - Verifies server Docker validation keeps retained artifacts and surfaces teardown failures explicitly
#   test_next_update_offset_uses_highest_update_id - Verifies Telegram validation offset derivation skips already seen updates
#   test_find_matching_update_filters_by_chat_and_text - Verifies Telegram update matching honors chat and text filters
#   test_run_telegram_live_validation_returns_connectivity_summary_without_update_polling - Verifies baseline Telegram live validation only calls getMe when no optional checks are requested
#   test_run_telegram_live_validation_sends_message_when_chat_id_provided - Verifies Telegram live validation sends an explicit ping when chat_id is supplied
#   test_run_telegram_live_validation_checks_matching_inbound_update - Verifies Telegram live validation can poll for a matching inbound update using getUpdates
#   test_run_telegram_live_validation_returns_advisory_when_update_chat_context_missing - Verifies Telegram live validation explicitly skips inbound update proof when no dedicated chat context is provided
#   test_run_telegram_live_validation_returns_advisory_when_matching_update_text_mismatches - Verifies Telegram live validation reports advisory mismatch semantics when the expected chat updates but the text proof does not match
#   test_run_telegram_live_validation_returns_advisory_when_matching_update_not_found - Verifies Telegram live validation reports advisory semantics when the expected inbound update never appears
#   test_run_telegram_docker_validation_returns_startup_api_and_teardown_evidence - Verifies Telegram Docker validation returns startup-proof, API-proof, retained logs, advisories, and teardown details
#   test_run_telegram_docker_validation_returns_skipped_summary_when_token_missing - Verifies Telegram Docker validation reports an explicit skipped outcome when bot credentials are unavailable
#   test_run_telegram_docker_validation_propagates_advisory_api_outcomes_with_startup_proof - Verifies Telegram Docker validation preserves startup proof while surfacing advisory API outcomes explicitly
#   test_run_artifact_review_validation_returns_advisory_summary_for_persisted_evidence - Verifies artifact review reads persisted evidence only and stays advisory
#   test_run_artifact_review_validation_surfaces_authoritative_failures_in_review_output - Verifies artifact review preserves deterministic failure signals from reviewed artifacts
#   test_main_returns_advisory_summary_for_missing_telegram_token - Verifies CLI output reports an explicit advisory summary with a zero exit code when telegram-live runs without a token
#   test_main_returns_skipped_summary_for_missing_docker_telegram_token - Verifies CLI output reports an explicit skipped summary with a zero exit code when docker-telegram runs without a token
#   test_main_returns_nonzero_exit_code_for_failed_smoke_validation_error - Verifies CLI output keeps hard validation failures non-zero when smoke-server preflight fails
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.9.0 - Locked validation command parsing to canonical TTS_* environment variables only]
# END_CHANGE_SUMMARY

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.support.api_fakes import (
    ManagedProcessDouble,
    make_validation_model_entry,
    make_validation_self_check_payload,
)
from scripts.validate_runtime import (
    CUSTOM_SMOKE_MODEL_ID,
    OMNIVOICE_SMOKE_MODEL_ID,
    PIPER_SMOKE_MODEL_ID,
    _find_matching_update,
    _next_update_offset,
    build_validation_env,
    main,
    parse_args,
    ValidationCommandError,
    run_representative_model_validation,
    run_server_docker_validation,
    run_artifact_review_validation,
    run_host_matrix_validation,
    run_smoke_server_validation,
    run_telegram_docker_validation,
    run_telegram_live_validation,
)


pytestmark = pytest.mark.unit


def _make_smoke_args(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    backend: str | None = None,
    expected_backend: str | None = None,
    smoke_model_id: str = CUSTOM_SMOKE_MODEL_ID,
    strict_runtime: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        command="smoke-server",
        python_executable="python3",
        host=host,
        port=port,
        backend=backend,
        startup_timeout_seconds=1.0,
        expected_backend=expected_backend,
        smoke_model_id=smoke_model_id,
        strict_runtime=strict_runtime,
    )


def _make_representative_args(
    *,
    target: str | None = None,
    backend: str | None = None,
    strict_runtime: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        command="representative-models",
        python_executable="python3",
        host="127.0.0.1",
        port=0,
        backend=backend,
        startup_timeout_seconds=1.0,
        strict_runtime=strict_runtime,
        target=target,
    )


def _make_smoke_env(tmp_path: Path) -> dict[str, str]:
    return build_validation_env(
        {
            "TTS_MODELS_DIR": str(tmp_path / "models"),
            "TTS_MLX_MODELS_DIR": str(tmp_path / "models" / "mlx"),
            "TTS_OUTPUTS_DIR": str(tmp_path / "outputs"),
            "TTS_VOICES_DIR": str(tmp_path / "voices"),
            "TTS_UPLOAD_STAGING_DIR": str(tmp_path / "uploads"),
        }
    )


def _make_server_docker_args(*, startup_timeout_seconds: float = 45.0) -> argparse.Namespace:
    return argparse.Namespace(
        command="docker-server",
        python_executable="python3",
        startup_timeout_seconds=startup_timeout_seconds,
    )


def _make_telegram_docker_args(
    *,
    bot_token: str | None = "token",
    chat_id: int | None = None,
    expect_update_chat_id: int | None = None,
    expect_update_text: str | None = None,
    startup_timeout_seconds: float = 30.0,
) -> argparse.Namespace:
    return argparse.Namespace(
        command="docker-telegram",
        python_executable="python3",
        bot_token=bot_token,
        chat_id=chat_id,
        message="Qwen3-TTS validation ping.",
        max_attempts=3,
        expect_update_chat_id=expect_update_chat_id,
        expect_update_text=expect_update_text,
        update_timeout_seconds=15,
        startup_timeout_seconds=startup_timeout_seconds,
    )


def _make_artifact_review_args(*artifact_path: str) -> argparse.Namespace:
    return argparse.Namespace(
        command="artifact-review",
        artifact_path=list(artifact_path),
    )


def _patch_smoke_server_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: dict,
    resolved_port: int = 8765,
    health: dict | None = None,
    process: ManagedProcessDouble | None = None,
    smoke_returncode: int = 0,
) -> dict[str, object]:
    process_double = process or ManagedProcessDouble()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "scripts.validate_runtime.build_self_check_payload",
        lambda _env: payload,
    )
    monkeypatch.setattr(
        "scripts.validate_runtime._choose_server_port",
        lambda host, requested_port: resolved_port,
    )
    monkeypatch.setattr(
        "scripts.validate_runtime._wait_for_server",
        lambda base_url, timeout_seconds: (
            health
            or {
                "live": {"status": "ok"},
                "ready": {"status": "ok"},
            }
        ),
    )

    def _fake_popen(*args, **kwargs):
        captured["popen_args"] = args
        captured["popen_kwargs"] = kwargs
        return process_double

    def _fake_run(*args, **kwargs):
        captured["run_args"] = args
        captured["run_kwargs"] = kwargs
        return argparse.Namespace(returncode=smoke_returncode)

    monkeypatch.setattr("scripts.validate_runtime.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("scripts.validate_runtime.subprocess.run", _fake_run)
    return captured


def test_build_validation_env_sets_repo_defaults():
    env = build_validation_env({}, backend="mlx", host="127.0.0.1", port=8123)

    assert env["TTS_BACKEND"] == "mlx"
    assert env["TTS_HOST"] == "127.0.0.1"
    assert env["TTS_PORT"] == "8123"
    assert env["TTS_MODELS_DIR"].endswith(".models")
    assert env["TTS_MLX_MODELS_DIR"].endswith(".models/mlx")


def test_parse_args_smoke_server_defaults_to_env_smoke_model_id(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TTS_SMOKE_MODEL_ID", PIPER_SMOKE_MODEL_ID)

    args = parse_args(["smoke-server"])

    assert args.smoke_model_id == PIPER_SMOKE_MODEL_ID


def test_parse_args_smoke_server_allows_cli_smoke_model_id_override(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TTS_SMOKE_MODEL_ID", PIPER_SMOKE_MODEL_ID)

    args = parse_args(
        [
            "smoke-server",
            "--smoke-model-id",
            CUSTOM_SMOKE_MODEL_ID,
        ]
    )

    assert args.smoke_model_id == CUSTOM_SMOKE_MODEL_ID


def test_run_host_matrix_validation_accepts_simulated_qwen_fast_modes():
    summary = run_host_matrix_validation(build_validation_env())

    assert summary["status"] == "ok"
    assert summary["simulated_qwen_fast"]["eligible"]["ready"] is True
    assert summary["simulated_qwen_fast"]["cuda_missing"]["reason"] == "cuda_required"
    assert (
        summary["simulated_qwen_fast"]["dependency_missing"]["reason"]
        == "runtime_dependency_missing"
    )


def test_run_host_matrix_validation_adds_machine_readable_result_fields():
    summary = run_host_matrix_validation(build_validation_env())

    assert summary["command"] == "host-matrix"
    assert summary["outcome"] == "passed"
    assert summary["reason"] == "host_matrix_validated"
    assert isinstance(summary["advisories"], list)
    if summary["baseline"]["qwen_fast_reason"] is None:
        assert summary["advisories"] == []
    else:
        assert any(
            advisory["reason"] == summary["baseline"]["qwen_fast_reason"]
            for advisory in summary["advisories"]
        )


def test_run_host_matrix_validation_respects_disabled_qwen_fast_config():
    summary = run_host_matrix_validation(
        build_validation_env(
            {
                "TTS_QWEN_FAST_ENABLED": "false",
            }
        )
    )

    assert summary["status"] == "ok"
    assert summary["simulated_qwen_fast"]["eligible"]["ready"] is False
    assert summary["simulated_qwen_fast"]["eligible"]["reason"] == "disabled_by_config"
    assert (
        summary["simulated_qwen_fast"]["eligible"]["custom_route_reason"]
        == "disabled_by_config"
    )
    assert (
        summary["simulated_qwen_fast"]["eligible"]["design_route_reason"]
        == "disabled_by_config"
    )
    assert (
        summary["simulated_qwen_fast"]["eligible"]["clone_route_reason"]
        == "disabled_by_config"
    )


def test_run_representative_model_validation_skips_missing_assets_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    env = _make_smoke_env(tmp_path)
    args = _make_representative_args(target="piper")
    payload = make_validation_self_check_payload(
        items=[
            make_validation_model_entry(
                model_id=PIPER_SMOKE_MODEL_ID,
                folder="Piper-en_US-lessac-medium",
                execution_backend="onnx",
                runtime_ready=False,
                available=False,
                loadable=False,
                missing_artifacts=["model.onnx", "model.onnx.json"],
                required_artifacts=["model.onnx", "model.onnx.json"],
                route_reason="selected_backend_incompatible_with_model",
                selected_backend_compatible_with_model=False,
            )
        ]
    )
    payload["representative_models"] = {
        "targets": [
            {
                "target": "piper",
                "model_id": PIPER_SMOKE_MODEL_ID,
                "status": "skipped",
                "reason": "model_assets_missing",
                "message": "Representative model assets are missing.",
                "expected_backend": "onnx",
                "selected_backend": "onnx",
                "execution_backend": "onnx",
                "available": False,
                "loadable": False,
                "runtime_ready": False,
                "missing_artifacts": ["model.onnx", "model.onnx.json"],
                "required_artifacts": ["model.onnx", "model.onnx.json"],
                "route_reason": "selected_backend_incompatible_with_model",
            }
        ],
        "ready_targets": [],
        "skipped_targets": [PIPER_SMOKE_MODEL_ID],
        "failed_targets": [],
    }

    monkeypatch.setattr(
        "scripts.validate_runtime.build_self_check_payload",
        lambda _env: payload,
    )

    summary = run_representative_model_validation(args, env)

    assert summary["status"] == "advisory"
    assert summary["outcome"] == "skipped"
    assert summary["reason"] == "representative_model_validation_skipped"
    assert summary["targets"][0]["representative_target"] == "piper"
    assert summary["targets"][0]["outcome"] == "skipped"
    assert summary["targets"][0]["reason"] == "model_assets_missing"


def test_run_representative_model_validation_skips_optional_dependency_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    env = _make_smoke_env(tmp_path)
    args = _make_representative_args(target="omnivoice")
    payload = make_validation_self_check_payload(
        items=[
            make_validation_model_entry(
                model_id=OMNIVOICE_SMOKE_MODEL_ID,
                folder="OmniVoice",
                execution_backend="torch",
                runtime_ready=False,
                available=True,
                loadable=True,
                selected_backend="torch",
                candidate_diagnostics=[
                    {"reason": "runtime_dependency_missing", "details": {"package": "omnivoice"}}
                ],
            )
        ],
        selected_backend="torch",
    )
    payload["representative_models"] = {
        "targets": [
            {
                "target": "omnivoice",
                "model_id": OMNIVOICE_SMOKE_MODEL_ID,
                "status": "skipped",
                "reason": "optional_dependency_pack_missing",
                "message": "Representative model requires an optional runtime dependency pack that is not installed.",
                "expected_backend": "torch",
                "selected_backend": "torch",
                "execution_backend": "torch",
                "available": True,
                "loadable": True,
                "runtime_ready": False,
                "missing_artifacts": [],
                "required_artifacts": [],
                "route_reason": "runtime_not_ready",
            }
        ],
        "ready_targets": [],
        "skipped_targets": [OMNIVOICE_SMOKE_MODEL_ID],
        "failed_targets": [],
    }

    monkeypatch.setattr(
        "scripts.validate_runtime.build_self_check_payload",
        lambda _env: payload,
    )

    summary = run_representative_model_validation(args, env)

    assert summary["status"] == "advisory"
    assert summary["targets"][0]["reason"] == "optional_dependency_pack_missing"
    assert any(advisory["reason"] == "optional_dependency_pack_missing" for advisory in summary["advisories"])


def test_run_representative_model_validation_reuses_smoke_server_for_ready_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    env = _make_smoke_env(tmp_path)
    args = _make_representative_args(target="qwen")
    payload = make_validation_self_check_payload(
        items=[
            make_validation_model_entry(
                model_id=CUSTOM_SMOKE_MODEL_ID,
                folder="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                execution_backend="mlx",
                runtime_ready=True,
                available=True,
                loadable=True,
                selected_backend="mlx",
            )
        ],
        selected_backend="mlx",
    )
    payload["representative_models"] = {
        "targets": [
            {
                "target": "qwen",
                "model_id": CUSTOM_SMOKE_MODEL_ID,
                "status": "ready",
                "reason": "representative_model_ready",
                "message": "Representative model is runtime-ready for opt-in validation.",
                "expected_backend": "mlx|torch|qwen_fast",
                "selected_backend": "mlx",
                "execution_backend": "mlx",
                "available": True,
                "loadable": True,
                "runtime_ready": True,
                "missing_artifacts": [],
                "required_artifacts": [],
                "route_reason": "selected_backend_supports_model",
            }
        ],
        "ready_targets": [CUSTOM_SMOKE_MODEL_ID],
        "skipped_targets": [],
        "failed_targets": [],
    }

    monkeypatch.setattr(
        "scripts.validate_runtime.build_self_check_payload",
        lambda _env: payload,
    )
    monkeypatch.setattr(
        "scripts.validate_runtime.run_smoke_server_validation",
        lambda smoke_args, smoke_env: {
            "command": "smoke-server",
            "outcome": "passed",
            "reason": "smoke_server_validated",
            "artifacts": {"server_log_path": "temp.log"},
            "smoke_model_id": smoke_args.smoke_model_id,
            "expected_backend": smoke_args.expected_backend,
            "base_url": f"http://{smoke_args.host}:8123",
        },
    )

    summary = run_representative_model_validation(args, env)

    assert summary["status"] == "ok"
    assert summary["outcome"] == "passed"
    assert summary["reason"] == "representative_model_validation_completed"
    assert summary["targets"][0]["reason"] == "representative_model_validated"
    assert summary["targets"][0]["smoke_model_id"] == CUSTOM_SMOKE_MODEL_ID
    assert summary["targets"][0]["smoke_summary"]["expected_backend"] == "mlx"


def test_run_smoke_server_validation_returns_summary_with_expected_evidence_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    smoke_dir = tmp_path / "models" / "OmniVoice"
    smoke_dir.mkdir(parents=True)
    env = _make_smoke_env(tmp_path)
    args = _make_smoke_args(smoke_model_id=OMNIVOICE_SMOKE_MODEL_ID)
    payload = make_validation_self_check_payload(
        items=[
            make_validation_model_entry(
                model_id=OMNIVOICE_SMOKE_MODEL_ID,
                folder="OmniVoice",
                execution_backend="torch",
            )
        ],
        selected_backend="torch",
    )
    health = {
        "live": {"status": "ok", "checks": ["api"]},
        "ready": {"status": "ok", "models": [OMNIVOICE_SMOKE_MODEL_ID]},
    }
    process = ManagedProcessDouble()
    captured = _patch_smoke_server_runtime(
        monkeypatch,
        payload=payload,
        resolved_port=8124,
        health=health,
        process=process,
    )

    summary = run_smoke_server_validation(args, env)

    assert summary["status"] == "ok"
    assert summary["base_url"] == "http://127.0.0.1:8124"
    assert summary["smoke_model_id"] == OMNIVOICE_SMOKE_MODEL_ID
    assert summary["expected_backend"] == "torch"
    assert summary["health"] == health
    assert summary["server_log_path"]
    run_env = captured["run_kwargs"]["env"]
    assert run_env["TTS_SMOKE_BASE_URL"] == "http://127.0.0.1:8124"
    assert run_env["TTS_SMOKE_MODEL_ID"] == OMNIVOICE_SMOKE_MODEL_ID
    assert run_env["TTS_SMOKE_EXPECTED_BACKEND"] == "torch"
    assert process.terminated is True


def test_run_smoke_server_validation_returns_machine_readable_preflight_and_artifact_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    smoke_dir = tmp_path / "models" / "OmniVoice"
    smoke_dir.mkdir(parents=True)
    env = _make_smoke_env(tmp_path)
    args = _make_smoke_args(smoke_model_id=OMNIVOICE_SMOKE_MODEL_ID)
    payload = make_validation_self_check_payload(
        items=[
            make_validation_model_entry(
                model_id=OMNIVOICE_SMOKE_MODEL_ID,
                folder="OmniVoice",
                execution_backend="torch",
            )
        ],
        selected_backend="torch",
        models_missing_assets=["Optional/other-model/config.json"],
    )
    _patch_smoke_server_runtime(monkeypatch, payload=payload, resolved_port=8130)

    summary = run_smoke_server_validation(args, env)

    assert summary["command"] == "smoke-server"
    assert summary["outcome"] == "passed"
    assert summary["reason"] == "smoke_server_validated"
    assert summary["stage"] == "completed"
    assert summary["preflight"]["resolved_port"] == 8130
    assert summary["preflight"]["smoke_model_dir"].endswith("models/OmniVoice")
    assert summary["artifacts"]["server_log_path"] == summary["server_log_path"]
    assert summary["advisories"][0]["reason"] == "models_missing_assets"


def test_run_smoke_server_validation_uses_expected_backend_override_in_smoke_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    smoke_dir = tmp_path / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    smoke_dir.mkdir(parents=True)
    env = _make_smoke_env(tmp_path)
    args = _make_smoke_args(expected_backend="torch")
    payload = make_validation_self_check_payload(
        items=[
            make_validation_model_entry(
                model_id=CUSTOM_SMOKE_MODEL_ID,
                folder="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                execution_backend="mlx",
            )
        ]
    )
    captured = _patch_smoke_server_runtime(monkeypatch, payload=payload)

    summary = run_smoke_server_validation(args, env)

    assert summary["expected_backend"] == "torch"
    assert captured["run_kwargs"]["env"]["TTS_SMOKE_EXPECTED_BACKEND"] == "torch"


def test_run_smoke_server_validation_rejects_wrong_expected_backend_for_piper(
    tmp_path: Path,
):
    smoke_dir = tmp_path / "models" / "Piper-en_US-lessac-medium"
    smoke_dir.mkdir(parents=True)
    args = _make_smoke_args(
        smoke_model_id=PIPER_SMOKE_MODEL_ID,
        expected_backend="torch",
    )
    env = _make_smoke_env(tmp_path)
    payload = make_validation_self_check_payload(
        items=[
            make_validation_model_entry(
                model_id=PIPER_SMOKE_MODEL_ID,
                folder="Piper-en_US-lessac-medium",
                execution_backend="onnx",
            )
        ]
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "scripts.validate_runtime.build_self_check_payload",
            lambda _env: payload,
        )
        monkeypatch.setattr(
            "scripts.validate_runtime._choose_server_port",
            lambda host, requested_port: 8125,
        )

        with pytest.raises(RuntimeError) as exc_info:
            run_smoke_server_validation(args, env)

    assert "requires expected backend 'onnx'" in str(exc_info.value)


def test_run_smoke_server_validation_requires_runtime_ready_custom_model(
    tmp_path: Path,
):
    smoke_dir = tmp_path / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    smoke_dir.mkdir(parents=True)
    args = _make_smoke_args()
    env = _make_smoke_env(tmp_path)
    payload = make_validation_self_check_payload(
        items=[
            make_validation_model_entry(
                model_id=CUSTOM_SMOKE_MODEL_ID,
                folder="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                runtime_ready=False,
            )
        ]
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "scripts.validate_runtime.build_self_check_payload",
            lambda _env: payload,
        )
        monkeypatch.setattr(
            "scripts.validate_runtime._choose_server_port",
            lambda host, requested_port: 8126,
        )

        with pytest.raises(RuntimeError) as exc_info:
            run_smoke_server_validation(args, env)

    assert CUSTOM_SMOKE_MODEL_ID in str(exc_info.value)


def test_run_smoke_server_validation_requires_existing_model_directory_before_runtime_ready(
    tmp_path: Path,
):
    args = _make_smoke_args()
    env = _make_smoke_env(tmp_path)
    payload = make_validation_self_check_payload(
        items=[
            make_validation_model_entry(
                model_id=CUSTOM_SMOKE_MODEL_ID,
                folder="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                runtime_ready=True,
            )
        ]
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "scripts.validate_runtime.build_self_check_payload",
            lambda _env: payload,
        )
        monkeypatch.setattr(
            "scripts.validate_runtime._choose_server_port",
            lambda host, requested_port: 8127,
        )

        with pytest.raises(RuntimeError) as exc_info:
            run_smoke_server_validation(args, env)

    assert "Required smoke model directory is missing" in str(exc_info.value)


def test_run_smoke_server_validation_rejects_unsupported_smoke_model_id(
    tmp_path: Path,
):
    args = _make_smoke_args(smoke_model_id="Unsupported-Model")
    env = _make_smoke_env(tmp_path)

    with pytest.raises(RuntimeError) as exc_info:
        run_smoke_server_validation(args, env)

    assert "Unsupported smoke model id" in str(exc_info.value)


def test_run_smoke_server_validation_rejects_missing_ffmpeg_preflight(
    tmp_path: Path,
):
    smoke_dir = tmp_path / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    smoke_dir.mkdir(parents=True)
    args = _make_smoke_args()
    env = _make_smoke_env(tmp_path)
    payload = make_validation_self_check_payload(
        items=[
            make_validation_model_entry(
                model_id=CUSTOM_SMOKE_MODEL_ID,
                folder="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            )
        ],
        ffmpeg_available=False,
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "scripts.validate_runtime.build_self_check_payload",
            lambda _env: payload,
        )
        monkeypatch.setattr(
            "scripts.validate_runtime._choose_server_port",
            lambda host, requested_port: 8128,
        )

        with pytest.raises(RuntimeError) as exc_info:
            run_smoke_server_validation(args, env)

    assert "ffmpeg is required" in str(exc_info.value)


def test_run_smoke_server_validation_reports_startup_timeout_with_server_log_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    smoke_dir = tmp_path / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    smoke_dir.mkdir(parents=True)
    env = _make_smoke_env(tmp_path)
    args = _make_smoke_args()
    payload = make_validation_self_check_payload(
        items=[
            make_validation_model_entry(
                model_id=CUSTOM_SMOKE_MODEL_ID,
                folder="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                execution_backend="mlx",
            )
        ]
    )

    _patch_smoke_server_runtime(monkeypatch, payload=payload)
    monkeypatch.setattr(
        "scripts.validate_runtime._wait_for_server",
        lambda base_url, timeout_seconds: (_ for _ in ()).throw(
            RuntimeError(
                f"Timed out waiting for server health at {base_url}: server did not respond"
            )
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_smoke_server_validation(args, env)

    error = exc_info.value
    assert error.reason == "server_startup_timeout"
    assert error.stage == "server_startup"
    assert error.artifacts["server_log_path"]
    assert "Server log:" in str(error)


def test_run_smoke_server_validation_allows_missing_assets_when_not_strict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    smoke_dir = tmp_path / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    smoke_dir.mkdir(parents=True)
    args = _make_smoke_args(strict_runtime=False)
    env = _make_smoke_env(tmp_path)
    payload = make_validation_self_check_payload(
        items=[
            make_validation_model_entry(
                model_id=CUSTOM_SMOKE_MODEL_ID,
                folder="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            )
        ],
        models_missing_assets=["Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit/config.json"],
    )
    captured = _patch_smoke_server_runtime(monkeypatch, payload=payload)

    summary = run_smoke_server_validation(args, env)

    assert summary["status"] == "ok"
    assert captured["run_kwargs"]["env"]["TTS_SMOKE_MODEL_ID"] == CUSTOM_SMOKE_MODEL_ID


def test_run_smoke_server_validation_rejects_strict_runtime_missing_assets(
    tmp_path: Path,
):
    smoke_dir = tmp_path / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    smoke_dir.mkdir(parents=True)
    args = _make_smoke_args(strict_runtime=True)
    env = _make_smoke_env(tmp_path)
    payload = make_validation_self_check_payload(
        items=[
            make_validation_model_entry(
                model_id=CUSTOM_SMOKE_MODEL_ID,
                folder="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            )
        ],
        models_missing_assets=["Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit/config.json"],
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "scripts.validate_runtime.build_self_check_payload",
            lambda _env: payload,
        )
        monkeypatch.setattr(
            "scripts.validate_runtime._choose_server_port",
            lambda host, requested_port: 8129,
        )

        with pytest.raises(RuntimeError) as exc_info:
            run_smoke_server_validation(args, env)

    assert "Strict smoke validation requested" in str(exc_info.value)


def test_parse_args_accepts_docker_server_timeout_override():
    args = parse_args(["docker-server", "--startup-timeout-seconds", "33"])

    assert args.command == "docker-server"
    assert args.startup_timeout_seconds == 33.0


def test_parse_args_accepts_docker_telegram_options():
    args = parse_args(
        [
            "docker-telegram",
            "--bot-token",
            "token",
            "--chat-id",
            "555",
            "--expect-update-chat-id",
            "555",
            "--expect-update-text",
            "ack",
            "--startup-timeout-seconds",
            "61",
        ]
    )

    assert args.command == "docker-telegram"
    assert args.bot_token == "token"
    assert args.chat_id == 555
    assert args.expect_update_chat_id == 555
    assert args.expect_update_text == "ack"
    assert args.startup_timeout_seconds == 61.0


def test_run_server_docker_validation_returns_probe_and_log_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    args = _make_server_docker_args(startup_timeout_seconds=12.0)
    env = _make_smoke_env(tmp_path)
    evidence_dir = tmp_path / ".sisyphus" / "evidence"

    monkeypatch.setattr("scripts.validate_runtime.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("scripts.validate_runtime.SERVER_DOCKER_COMPOSE_FILE", tmp_path / "docker-compose.server.yaml")
    (tmp_path / "docker-compose.server.yaml").write_text("services: {}", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.validate_runtime._resolve_compose_invocation",
        lambda: (["docker", "compose"], "docker compose"),
    )
    monkeypatch.setattr(
        "scripts.validate_runtime._run_compose_command",
        lambda *args, **kwargs: argparse.Namespace(returncode=0, stdout="compose ok", stderr=""),
    )
    monkeypatch.setattr(
        "scripts.validate_runtime._wait_for_server_docker_probes",
        lambda *args, **kwargs: {
            "live": {"status": "ok"},
            "ready": {"status": "ok", "routing": {"degraded_routes": 0}},
            "models": {"data": [{"id": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"}]},
        },
    )
    monkeypatch.setattr(
        "scripts.validate_runtime._capture_compose_logs",
        lambda *args, **kwargs: {
            "path": args[4].as_posix(),
            "line_count": 3,
        },
    )

    summary = run_server_docker_validation(args, env)

    assert summary["status"] == "ok"
    assert summary["command"] == "docker-server"
    assert summary["reason"] == "server_docker_validated"
    assert summary["compose"]["service"] == "server"
    assert summary["probes"]["ready"]["routing"]["degraded_routes"] == 0
    assert summary["teardown"]["attempted"] is True
    assert summary["teardown"]["succeeded"] is True
    assert summary["artifacts"]["health_live_path"].endswith("server-docker-health-live.json")
    assert summary["artifacts"]["server_log_path"].endswith("server-docker-log.txt")
    assert json.loads((evidence_dir / "server-docker-health-ready.json").read_text(encoding="utf-8"))["status"] == "ok"


def test_run_server_docker_validation_selects_free_host_port_for_compose_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    args = _make_server_docker_args(startup_timeout_seconds=12.0)
    env = _make_smoke_env(tmp_path)
    choose_port_calls: list[tuple[str, int]] = []
    compose_envs: list[dict[str, str]] = []

    monkeypatch.setattr("scripts.validate_runtime.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "scripts.validate_runtime.SERVER_DOCKER_COMPOSE_FILE",
        tmp_path / "docker-compose.server.yaml",
    )
    (tmp_path / "docker-compose.server.yaml").write_text("services: {}", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.validate_runtime._resolve_compose_invocation",
        lambda: (["docker", "compose"], "docker compose"),
    )

    def _fake_choose_server_port(host: str, requested_port: int) -> int:
        choose_port_calls.append((host, requested_port))
        return 18080

    def _fake_run_compose_command(
        _compose_command,
        _compose_file,
        _project_name,
        _compose_args,
        *,
        environ,
    ):
        compose_envs.append(dict(environ))
        return argparse.Namespace(returncode=0, stdout="compose ok", stderr="")

    def _fake_capture_logs(
        _compose_command,
        _compose_file,
        _project_name,
        _service,
        artifact_path,
        *,
        environ,
    ):
        compose_envs.append(dict(environ))
        artifact_path.write_text("server logs", encoding="utf-8")
        return {"path": artifact_path.as_posix(), "line_count": 1}

    monkeypatch.setattr(
        "scripts.validate_runtime._choose_server_port",
        _fake_choose_server_port,
    )
    monkeypatch.setattr(
        "scripts.validate_runtime._run_compose_command",
        _fake_run_compose_command,
    )
    monkeypatch.setattr(
        "scripts.validate_runtime._wait_for_server_docker_probes",
        lambda *args, **kwargs: {
            "live": {"status": "ok"},
            "ready": {"status": "ok"},
            "models": {"data": []},
        },
    )
    monkeypatch.setattr(
        "scripts.validate_runtime._capture_compose_logs",
        _fake_capture_logs,
    )

    summary = run_server_docker_validation(args, env)

    assert choose_port_calls == [("0.0.0.0", 0)]
    assert compose_envs
    assert all(run_env["TTS_SERVER_PORT"] == "18080" for run_env in compose_envs)
    assert summary["compose"]["host_port"] == 18080


def test_run_server_docker_validation_returns_skipped_summary_when_compose_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    args = _make_server_docker_args()
    env = _make_smoke_env(tmp_path)

    monkeypatch.setattr("scripts.validate_runtime.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("scripts.validate_runtime.SERVER_DOCKER_COMPOSE_FILE", tmp_path / "docker-compose.server.yaml")
    (tmp_path / "docker-compose.server.yaml").write_text("services: {}", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.validate_runtime._resolve_compose_invocation",
        lambda: (_ for _ in ()).throw(RuntimeError("Docker Compose is not available")),
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_server_docker_validation(args, env)

    error = exc_info.value
    assert error.reason == "docker_compose_unavailable"
    assert error.outcome == "skipped"
    assert error.stage == "preflight"


def test_run_server_docker_validation_reports_teardown_failure_with_retained_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    args = _make_server_docker_args()
    env = _make_smoke_env(tmp_path)
    calls: list[list[str]] = []

    monkeypatch.setattr("scripts.validate_runtime.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("scripts.validate_runtime.SERVER_DOCKER_COMPOSE_FILE", tmp_path / "docker-compose.server.yaml")
    (tmp_path / "docker-compose.server.yaml").write_text("services: {}", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.validate_runtime._resolve_compose_invocation",
        lambda: (["docker", "compose"], "docker compose"),
    )

    def _fake_run_compose_command(_compose_command, _compose_file, _project_name, compose_args, **_kwargs):
        calls.append(list(compose_args))
        if compose_args[:1] == ["down"]:
            return argparse.Namespace(returncode=1, stdout="", stderr="teardown failed")
        return argparse.Namespace(returncode=0, stdout="compose ok", stderr="")

    monkeypatch.setattr("scripts.validate_runtime._run_compose_command", _fake_run_compose_command)
    monkeypatch.setattr(
        "scripts.validate_runtime._wait_for_server_docker_probes",
        lambda *args, **kwargs: {
            "live": {"status": "ok"},
            "ready": {"status": "ok"},
            "models": {"data": []},
        },
    )

    def _fake_capture_logs(*_args, **kwargs):
        artifact_path = _args[4]
        artifact_path.write_text("server logs", encoding="utf-8")
        return {"path": artifact_path.as_posix(), "line_count": 1}

    monkeypatch.setattr("scripts.validate_runtime._capture_compose_logs", _fake_capture_logs)

    with pytest.raises(RuntimeError) as exc_info:
        run_server_docker_validation(args, env)

    error = exc_info.value
    assert error.reason == "docker_teardown_failed"
    assert error.stage == "teardown"
    assert error.artifacts["server_log_path"].endswith("server-docker-log.txt")
    assert error.details["teardown"]["attempted"] is True
    assert error.details["teardown"]["succeeded"] is False
    assert any(call[:1] == ["down"] for call in calls)


def test_next_update_offset_uses_highest_update_id():
    updates = [
        {"update_id": 100},
        {"update_id": 105},
        {"update_id": 101},
        {"ignored": True},
    ]

    assert _next_update_offset(updates) == 106
    assert _next_update_offset([]) == 0


def test_find_matching_update_filters_by_chat_and_text():
    updates = [
        {
            "update_id": 11,
            "message": {"chat": {"id": 1001}, "text": "irrelevant"},
        },
        {
            "update_id": 12,
            "edited_message": {"chat": {"id": 2002}, "caption": "voice ping ack"},
        },
    ]

    matched = _find_matching_update(
        updates,
        expected_chat_id=2002,
        expected_text="ping ack",
    )

    assert matched is not None
    assert matched["update_id"] == 12
    assert (
        _find_matching_update(
            updates,
            expected_chat_id=9999,
            expected_text=None,
        )
        is None
    )


def test_run_telegram_live_validation_returns_connectivity_summary_without_update_polling(
    monkeypatch: pytest.MonkeyPatch,
):
    client = MagicMock()
    client.get_me = AsyncMock(
        return_value={
        "id": 123,
        "username": "test_bot",
        "first_name": "Test",
        }
    )
    client.send_message = AsyncMock()
    client.get_updates = AsyncMock()
    client.close = AsyncMock()

    monkeypatch.setattr(
        "scripts.validate_runtime.TelegramBotClient", lambda **_: client
    )

    args = argparse.Namespace(
        command="telegram-live",
        bot_token="token",
        chat_id=None,
        message="ping",
        max_attempts=2,
        expect_update_chat_id=None,
        expect_update_text=None,
        update_timeout_seconds=30,
    )

    summary = asyncio.run(run_telegram_live_validation(args, {}))

    assert summary["status"] == "ok"
    assert summary["command"] == "telegram-live"
    assert summary["outcome"] == "passed"
    assert summary["reason"] == "telegram_api_reachable"
    assert summary["message_sent"] is False
    assert summary["update_checked"] is False
    assert summary["dedicated_chat_check_requested"] is False
    client.get_me.assert_awaited_once()
    client.send_message.assert_not_called()
    client.get_updates.assert_not_called()
    client.close.assert_awaited_once()


def test_run_telegram_live_validation_sends_message_when_chat_id_provided(
    monkeypatch: pytest.MonkeyPatch,
):
    client = MagicMock()
    client.get_me = AsyncMock(
        return_value={
        "id": 123,
        "username": "test_bot",
        "first_name": "Test",
        }
    )
    client.send_message = AsyncMock(return_value={"message_id": 777})
    client.get_updates = AsyncMock()
    client.close = AsyncMock()

    monkeypatch.setattr(
        "scripts.validate_runtime.TelegramBotClient", lambda **_: client
    )

    args = argparse.Namespace(
        command="telegram-live",
        bot_token="token",
        chat_id=555,
        message="validation ping",
        max_attempts=2,
        expect_update_chat_id=None,
        expect_update_text=None,
        update_timeout_seconds=30,
    )

    summary = asyncio.run(run_telegram_live_validation(args, {}))

    assert summary["message_sent"] is True
    assert summary["message_id"] == 777
    assert summary["dedicated_chat_check_requested"] is True
    client.send_message.assert_awaited_once_with(555, "validation ping")
    client.get_updates.assert_not_called()
    client.close.assert_awaited_once()


def test_run_telegram_live_validation_checks_matching_inbound_update(
    monkeypatch: pytest.MonkeyPatch,
):
    client = MagicMock()
    client.get_me = AsyncMock(
        return_value={
        "id": 123,
        "username": "test_bot",
        "first_name": "Test",
        }
    )
    client.send_message = AsyncMock()
    client.get_updates = AsyncMock(
        side_effect=[
            [{"update_id": 40, "message": {"chat": {"id": 111}, "text": "older"}}],
            [
                {
                    "update_id": 41,
                    "message": {"chat": {"id": 222}, "text": "ignore me"},
                },
                {
                    "update_id": 42,
                    "message": {"chat": {"id": 555}, "text": "validation ack"},
                },
            ],
        ]
    )
    client.close = AsyncMock()

    monkeypatch.setattr(
        "scripts.validate_runtime.TelegramBotClient", lambda **_: client
    )

    args = argparse.Namespace(
        command="telegram-live",
        bot_token="token",
        chat_id=None,
        message="unused",
        max_attempts=2,
        expect_update_chat_id=555,
        expect_update_text="ack",
        update_timeout_seconds=15,
    )

    summary = asyncio.run(run_telegram_live_validation(args, {}))

    assert summary["update_checked"] is True
    assert summary["update_check_status"] == "matched"
    assert summary["update_check_reason"] == "telegram_matching_update_observed"
    assert summary["expected_update_chat_id"] == 555
    assert summary["expected_update_text"] == "ack"
    assert summary["update_poll_offset"] == 41
    assert summary["update_poll_count"] == 2
    assert summary["matched_update_id"] == 42
    assert summary["matched_update_chat_id"] == 555
    assert summary["matched_update_kind"] == "message"
    assert client.get_updates.await_args_list[0].kwargs == {
        "timeout": 0,
        "allowed_updates": ["message", "edited_message"],
    }
    assert client.get_updates.await_args_list[1].kwargs == {
        "offset": 41,
        "timeout": 15,
        "allowed_updates": ["message", "edited_message"],
    }
    client.close.assert_awaited_once()


def test_run_telegram_live_validation_returns_advisory_when_update_chat_context_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    client = MagicMock()
    client.get_me = AsyncMock(
        return_value={
        "id": 123,
        "username": "test_bot",
        "first_name": "Test",
        }
    )
    client.send_message = AsyncMock()
    client.get_updates = AsyncMock()
    client.close = AsyncMock()

    monkeypatch.setattr(
        "scripts.validate_runtime.TelegramBotClient", lambda **_: client
    )

    args = argparse.Namespace(
        command="telegram-live",
        bot_token="token",
        chat_id=None,
        message="unused",
        max_attempts=2,
        expect_update_chat_id=None,
        expect_update_text="ack",
        update_timeout_seconds=5,
    )

    summary = asyncio.run(run_telegram_live_validation(args, {}))

    assert summary["status"] == "ok"
    assert summary["update_checked"] is False
    assert summary["update_check_status"] == "skipped"
    assert summary["update_check_reason"] == "expected_update_chat_id_missing"
    assert summary["dedicated_chat_check_requested"] is True
    assert summary["advisories"][0]["reason"] == "expected_update_chat_id_missing"
    client.send_message.assert_not_called()
    client.get_updates.assert_not_called()
    client.close.assert_awaited_once()


def test_run_telegram_live_validation_returns_advisory_when_matching_update_text_mismatches(
    monkeypatch: pytest.MonkeyPatch,
):
    client = MagicMock()
    client.get_me = AsyncMock(
        return_value={
        "id": 123,
        "username": "test_bot",
        "first_name": "Test",
        }
    )
    client.send_message = AsyncMock(return_value={"message_id": 901})
    client.get_updates = AsyncMock(
        side_effect=[
            [],
            [{"update_id": 51, "message": {"chat": {"id": 321}, "text": "wrong ack"}}],
        ]
    )
    client.close = AsyncMock()

    monkeypatch.setattr(
        "scripts.validate_runtime.TelegramBotClient", lambda **_: client
    )

    args = argparse.Namespace(
        command="telegram-live",
        bot_token="token",
        chat_id=321,
        message="validation ping",
        max_attempts=2,
        expect_update_chat_id=None,
        expect_update_text="expected ack",
        update_timeout_seconds=5,
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(run_telegram_live_validation(args, {}))

    error = exc_info.value
    assert error.reason == "telegram_matching_update_text_mismatch"
    assert error.outcome == "advisory"
    assert error.stage == "update_poll"
    assert error.details["expected_update_chat_id"] == 321
    assert error.details["observed_update_chat_id"] == 321
    assert error.details["observed_update_text"] == "wrong ack"
    client.close.assert_awaited_once()


def test_run_telegram_live_validation_returns_advisory_when_matching_update_not_found(
    monkeypatch: pytest.MonkeyPatch,
):
    client = MagicMock()
    client.get_me = AsyncMock(
        return_value={
        "id": 123,
        "username": "test_bot",
        "first_name": "Test",
        }
    )
    client.send_message = AsyncMock(return_value={"message_id": 901})
    client.get_updates = AsyncMock(side_effect=[[], []])
    client.close = AsyncMock()

    monkeypatch.setattr(
        "scripts.validate_runtime.TelegramBotClient", lambda **_: client
    )

    args = argparse.Namespace(
        command="telegram-live",
        bot_token="token",
        chat_id=321,
        message="validation ping",
        max_attempts=2,
        expect_update_chat_id=None,
        expect_update_text="ack",
        update_timeout_seconds=5,
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(run_telegram_live_validation(args, {}))

    error = exc_info.value
    assert error.reason == "telegram_matching_update_not_found"
    assert error.outcome == "advisory"
    assert error.stage == "update_poll"
    assert error.details["expected_update_chat_id"] == 321
    assert error.details["expected_update_text"] == "ack"
    assert error.details["update_poll_count"] == 0
    client.send_message.assert_awaited_once_with(321, "validation ping")
    assert client.get_updates.await_count == 2
    client.close.assert_awaited_once()


def test_run_telegram_docker_validation_returns_startup_api_and_teardown_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    args = _make_telegram_docker_args(bot_token="token")
    env = _make_smoke_env(tmp_path)

    monkeypatch.setattr("scripts.validate_runtime.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "scripts.validate_runtime.TELEGRAM_DOCKER_COMPOSE_FILE",
        tmp_path / "docker-compose.telegram-bot.yaml",
    )
    (tmp_path / "docker-compose.telegram-bot.yaml").write_text("services: {}", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.validate_runtime._resolve_compose_invocation",
        lambda: (["docker", "compose"], "docker compose"),
    )
    monkeypatch.setattr(
        "scripts.validate_runtime._run_compose_command",
        lambda *args, **kwargs: argparse.Namespace(returncode=0, stdout="compose ok", stderr=""),
    )
    monkeypatch.setattr(
        "scripts.validate_runtime._wait_for_telegram_docker_startup",
        lambda *args, **kwargs: {
            "required_markers": ["Telegram API connectivity verified"],
            "observed_markers": ["Telegram API connectivity verified"],
            "log_excerpt": "Polling loop is now active",
        },
    )

    def _fake_capture_logs(*_args, **kwargs):
        artifact_path = _args[4]
        artifact_path.write_text("telegram logs", encoding="utf-8")
        return {"path": artifact_path.as_posix(), "line_count": 1}

    monkeypatch.setattr("scripts.validate_runtime._capture_compose_logs", _fake_capture_logs)

    async def _fake_run_telegram_live_validation(live_args, _environ):
        assert live_args.command == "telegram-live"
        return {
            "status": "ok",
            "command": "telegram-live",
            "outcome": "passed",
            "reason": "telegram_api_reachable",
            "advisories": [],
            "message_sent": False,
            "update_checked": False,
        }

    monkeypatch.setattr(
        "scripts.validate_runtime.run_telegram_live_validation",
        _fake_run_telegram_live_validation,
    )

    summary = asyncio.run(run_telegram_docker_validation(args, env))

    assert summary["status"] == "ok"
    assert summary["command"] == "docker-telegram"
    assert summary["reason"] == "telegram_docker_validated"
    assert summary["startup_proof"]["log_excerpt"] == "Polling loop is now active"
    assert summary["telegram_live"]["reason"] == "telegram_api_reachable"
    assert summary["teardown"]["attempted"] is True
    assert summary["teardown"]["succeeded"] is True
    assert summary["artifacts"]["telegram_log_path"].endswith("telegram-docker-log.txt")
    assert summary["advisories"][0]["reason"] == "telegram_validation_chat_id_missing"


def test_run_telegram_docker_validation_returns_skipped_summary_when_token_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    args = _make_telegram_docker_args(bot_token=None)
    env = _make_smoke_env(tmp_path)

    monkeypatch.setattr("scripts.validate_runtime.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "scripts.validate_runtime.TELEGRAM_DOCKER_COMPOSE_FILE",
        tmp_path / "docker-compose.telegram-bot.yaml",
    )
    (tmp_path / "docker-compose.telegram-bot.yaml").write_text("services: {}", encoding="utf-8")

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(run_telegram_docker_validation(args, env))

    error = exc_info.value
    assert error.reason == "telegram_bot_token_missing"
    assert error.outcome == "skipped"
    assert error.stage == "preflight"


def test_run_telegram_docker_validation_propagates_advisory_api_outcomes_with_startup_proof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    args = _make_telegram_docker_args(bot_token="token", chat_id=321, expect_update_text="ack")
    env = _make_smoke_env(tmp_path)

    monkeypatch.setattr("scripts.validate_runtime.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "scripts.validate_runtime.TELEGRAM_DOCKER_COMPOSE_FILE",
        tmp_path / "docker-compose.telegram-bot.yaml",
    )
    (tmp_path / "docker-compose.telegram-bot.yaml").write_text("services: {}", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.validate_runtime._resolve_compose_invocation",
        lambda: (["docker", "compose"], "docker compose"),
    )
    monkeypatch.setattr(
        "scripts.validate_runtime._run_compose_command",
        lambda *args, **kwargs: argparse.Namespace(returncode=0, stdout="compose ok", stderr=""),
    )
    monkeypatch.setattr(
        "scripts.validate_runtime._wait_for_telegram_docker_startup",
        lambda *args, **kwargs: {
            "required_markers": ["Telegram API connectivity verified"],
            "observed_markers": ["Telegram API connectivity verified"],
            "log_excerpt": "Polling loop is now active",
        },
    )

    def _fake_capture_logs(*_args, **kwargs):
        artifact_path = _args[4]
        artifact_path.write_text("telegram logs", encoding="utf-8")
        return {"path": artifact_path.as_posix(), "line_count": 1}

    monkeypatch.setattr("scripts.validate_runtime._capture_compose_logs", _fake_capture_logs)

    async def _fake_run_telegram_live_validation(_live_args, _environ):
        raise ValidationCommandError(
            "Telegram live validation did not observe a matching inbound update",
            command="telegram-live",
            reason="telegram_matching_update_not_found",
            outcome="advisory",
            stage="update_poll",
            details={"expected_update_chat_id": 321},
        )

    monkeypatch.setattr(
        "scripts.validate_runtime.run_telegram_live_validation",
        _fake_run_telegram_live_validation,
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(run_telegram_docker_validation(args, env))

    error = exc_info.value
    assert error.reason == "telegram_matching_update_not_found"
    assert error.outcome == "advisory"
    assert error.stage == "api_proof"
    assert error.details["startup_proof"]["log_excerpt"] == "Polling loop is now active"
    assert error.artifacts["telegram_log_path"].endswith("telegram-docker-log.txt")


def test_main_returns_advisory_summary_for_missing_telegram_token(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.delenv("TTS_TELEGRAM_BOT_TOKEN", raising=False)

    exit_code = main(["telegram-live"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "advisory"
    assert payload["command"] == "telegram-live"
    assert payload["outcome"] == "advisory"
    assert payload["reason"] == "telegram_bot_token_missing"
    assert payload["stage"] == "preflight"
    assert "TTS_TELEGRAM_BOT_TOKEN" in payload["message"]


def test_main_returns_skipped_summary_for_missing_docker_telegram_token(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.delenv("TTS_TELEGRAM_BOT_TOKEN", raising=False)

    exit_code = main(["docker-telegram"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "advisory"
    assert payload["command"] == "docker-telegram"
    assert payload["outcome"] == "skipped"
    assert payload["reason"] == "telegram_bot_token_missing"
    assert payload["stage"] == "preflight"


def test_main_returns_nonzero_exit_code_for_failed_smoke_validation_error(
    capsys: pytest.CaptureFixture[str],
):
    exit_code = main(["smoke-server", "--smoke-model-id", "Unsupported-Model"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["command"] == "smoke-server"
    assert payload["outcome"] == "failed"
    assert payload["reason"] == "unsupported_smoke_model_id"
    assert payload["stage"] == "preflight"


def test_run_artifact_review_validation_returns_advisory_summary_for_persisted_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr("scripts.validate_runtime.PROJECT_ROOT", tmp_path)
    evidence_dir = tmp_path / ".sisyphus" / "evidence"
    evidence_dir.mkdir(parents=True)
    reviewed = evidence_dir / "server-docker-log.txt"
    reviewed.write_text("startup complete\n", encoding="utf-8")
    explicit = tmp_path / "custom-summary.json"
    explicit.write_text(
        json.dumps(
            {
                "status": "ok",
                "outcome": "passed",
                "reason": "smoke_server_validated",
                "command": "smoke-server",
            }
        ),
        encoding="utf-8",
    )

    summary = run_artifact_review_validation(
        _make_artifact_review_args(explicit.as_posix()),
            build_validation_env({"TTS_MODELS_DIR": str(tmp_path / "models")}),
    )

    assert summary["status"] == "advisory"
    assert summary["outcome"] == "skipped"
    assert summary["reason"] == "artifact_review_completed"
    assert summary["source"] == "persisted_artifacts_only"
    assert summary["review_count"] >= 2
    assert reviewed.as_posix() in summary["reviewed_artifacts"]
    assert explicit.as_posix() in summary["reviewed_artifacts"]
    assert summary["contains_authoritative_failure"] is False
    assert summary["advisories"][0]["kind"] == "artifact_review_lane"


def test_run_artifact_review_validation_surfaces_authoritative_failures_in_review_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr("scripts.validate_runtime.PROJECT_ROOT", tmp_path)
    evidence_dir = tmp_path / ".sisyphus" / "evidence"
    evidence_dir.mkdir(parents=True)
    failure_artifact = evidence_dir / "server-docker-health-ready.json"
    failure_artifact.write_text(
        json.dumps(
            {
                "status": "error",
                "outcome": "failed",
                "reason": "docker_teardown_failed",
                "command": "docker-server",
                "stage": "teardown",
            }
        ),
        encoding="utf-8",
    )

    summary = run_artifact_review_validation(
        _make_artifact_review_args(),
            build_validation_env({"TTS_MODELS_DIR": str(tmp_path / "models")}),
    )

    assert summary["status"] == "advisory"
    assert summary["contains_authoritative_failure"] is True
    assert any(
        review["path"] == failure_artifact.as_posix()
        and review["observed_signals"] == ["authoritative_failure"]
        for review in summary["reviews"]
    )
