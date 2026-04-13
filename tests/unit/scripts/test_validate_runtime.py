# FILE: tests/unit/scripts/test_validate_runtime.py
# VERSION: 1.2.0
# START_MODULE_CONTRACT
#   PURPOSE: Verify runtime validation automation helpers for host-matrix checks, smoke-server orchestration, and Telegram live validation flows.
#   SCOPE: Validation env defaults, qwen_fast simulation assertions, smoke preflight error handling, and opt-in Telegram inbound update validation
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
#   test_run_smoke_server_validation_requires_runtime_ready_custom_model - Verifies smoke validation fails fast when the custom smoke model is not runtime-ready
#   test_run_smoke_server_validation_rejects_unsupported_smoke_model_id - Verifies smoke validation fails fast for unknown smoke model ids
#   test_next_update_offset_uses_highest_update_id - Verifies Telegram validation offset derivation skips already seen updates
#   test_find_matching_update_filters_by_chat_and_text - Verifies Telegram update matching honors chat and text filters
#   test_run_telegram_live_validation_returns_connectivity_summary_without_update_polling - Verifies baseline Telegram live validation only calls getMe when no optional checks are requested
#   test_run_telegram_live_validation_sends_message_when_chat_id_provided - Verifies Telegram live validation sends an explicit ping when chat_id is supplied
#   test_run_telegram_live_validation_checks_matching_inbound_update - Verifies Telegram live validation can poll for a matching inbound update using getUpdates
#   test_run_telegram_live_validation_fails_when_matching_update_not_found - Verifies Telegram live validation fails when the expected inbound update never appears
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.2.0 - Added regression coverage for model-aware smoke-server argument and validation behavior]
# END_CHANGE_SUMMARY

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from scripts.validate_runtime import (
    CUSTOM_SMOKE_MODEL_ID,
    PIPER_SMOKE_MODEL_ID,
    _find_matching_update,
    _next_update_offset,
    build_validation_env,
    parse_args,
    run_host_matrix_validation,
    run_smoke_server_validation,
    run_telegram_live_validation,
)


pytestmark = pytest.mark.unit


def test_build_validation_env_sets_repo_defaults():
    env = build_validation_env({}, backend="mlx", host="127.0.0.1", port=8123)

    assert env["QWEN_TTS_BACKEND"] == "mlx"
    assert env["QWEN_TTS_HOST"] == "127.0.0.1"
    assert env["QWEN_TTS_PORT"] == "8123"
    assert env["QWEN_TTS_MODELS_DIR"].endswith(".models")
    assert env["QWEN_TTS_MLX_MODELS_DIR"].endswith(".models/mlx")


def test_parse_args_smoke_server_defaults_to_env_smoke_model_id(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("QWEN_TTS_SMOKE_MODEL_ID", PIPER_SMOKE_MODEL_ID)

    args = parse_args(["smoke-server"])

    assert args.smoke_model_id == PIPER_SMOKE_MODEL_ID


def test_parse_args_smoke_server_allows_cli_smoke_model_id_override(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("QWEN_TTS_SMOKE_MODEL_ID", PIPER_SMOKE_MODEL_ID)

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


def test_run_host_matrix_validation_respects_disabled_qwen_fast_config():
    summary = run_host_matrix_validation(
        build_validation_env(
            {
                "QWEN_TTS_QWEN_FAST_ENABLED": "false",
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


def test_run_smoke_server_validation_requires_runtime_ready_custom_model(
    tmp_path: Path,
):
    args = argparse.Namespace(
        command="smoke-server",
        python_executable="python3",
        host="127.0.0.1",
        port=0,
        backend=None,
        startup_timeout_seconds=1.0,
        expected_backend=None,
        smoke_model_id=CUSTOM_SMOKE_MODEL_ID,
        strict_runtime=False,
    )
    env = build_validation_env(
        {
            "QWEN_TTS_MODELS_DIR": str(tmp_path / "models"),
            "QWEN_TTS_MLX_MODELS_DIR": str(tmp_path / "models" / "mlx"),
            "QWEN_TTS_OUTPUTS_DIR": str(tmp_path / "outputs"),
            "QWEN_TTS_VOICES_DIR": str(tmp_path / "voices"),
            "QWEN_TTS_UPLOAD_STAGING_DIR": str(tmp_path / "uploads"),
        }
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_smoke_server_validation(args, env)

    assert CUSTOM_SMOKE_MODEL_ID in str(exc_info.value)


def test_run_smoke_server_validation_rejects_unsupported_smoke_model_id(
    tmp_path: Path,
):
    args = argparse.Namespace(
        command="smoke-server",
        python_executable="python3",
        host="127.0.0.1",
        port=0,
        backend=None,
        startup_timeout_seconds=1.0,
        expected_backend=None,
        smoke_model_id="Unsupported-Model",
        strict_runtime=False,
    )
    env = build_validation_env(
        {
            "QWEN_TTS_MODELS_DIR": str(tmp_path / "models"),
            "QWEN_TTS_MLX_MODELS_DIR": str(tmp_path / "models" / "mlx"),
            "QWEN_TTS_OUTPUTS_DIR": str(tmp_path / "outputs"),
            "QWEN_TTS_VOICES_DIR": str(tmp_path / "voices"),
            "QWEN_TTS_UPLOAD_STAGING_DIR": str(tmp_path / "uploads"),
        }
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_smoke_server_validation(args, env)

    assert "Unsupported smoke model id" in str(exc_info.value)


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


@pytest.mark.asyncio
async def test_run_telegram_live_validation_returns_connectivity_summary_without_update_polling(
    monkeypatch: pytest.MonkeyPatch,
):
    client = AsyncMock()
    client.get_me.return_value = {
        "id": 123,
        "username": "test_bot",
        "first_name": "Test",
    }

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

    summary = await run_telegram_live_validation(args, {})

    assert summary["status"] == "ok"
    assert summary["message_sent"] is False
    assert summary["update_checked"] is False
    client.get_me.assert_awaited_once()
    client.send_message.assert_not_called()
    client.get_updates.assert_not_called()
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_telegram_live_validation_sends_message_when_chat_id_provided(
    monkeypatch: pytest.MonkeyPatch,
):
    client = AsyncMock()
    client.get_me.return_value = {
        "id": 123,
        "username": "test_bot",
        "first_name": "Test",
    }
    client.send_message.return_value = {"message_id": 777}

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

    summary = await run_telegram_live_validation(args, {})

    assert summary["message_sent"] is True
    assert summary["message_id"] == 777
    client.send_message.assert_awaited_once_with(555, "validation ping")
    client.get_updates.assert_not_called()
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_telegram_live_validation_checks_matching_inbound_update(
    monkeypatch: pytest.MonkeyPatch,
):
    client = AsyncMock()
    client.get_me.return_value = {
        "id": 123,
        "username": "test_bot",
        "first_name": "Test",
    }
    client.get_updates.side_effect = [
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

    summary = await run_telegram_live_validation(args, {})

    assert summary["update_checked"] is True
    assert summary["expected_update_chat_id"] == 555
    assert summary["expected_update_text"] == "ack"
    assert summary["update_poll_offset"] == 41
    assert summary["update_poll_count"] == 2
    assert summary["matched_update_id"] == 42
    assert summary["matched_update_chat_id"] == 555
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


@pytest.mark.asyncio
async def test_run_telegram_live_validation_fails_when_matching_update_not_found(
    monkeypatch: pytest.MonkeyPatch,
):
    client = AsyncMock()
    client.get_me.return_value = {
        "id": 123,
        "username": "test_bot",
        "first_name": "Test",
    }
    client.send_message.return_value = {"message_id": 901}
    client.get_updates.side_effect = [[], []]

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
        await run_telegram_live_validation(args, {})

    assert "matching inbound update" in str(exc_info.value)
    client.send_message.assert_awaited_once_with(321, "validation ping")
    assert client.get_updates.await_count == 2
    client.close.assert_awaited_once()
