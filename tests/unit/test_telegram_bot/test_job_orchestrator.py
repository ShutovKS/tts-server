"""Unit tests for Telegram remote job orchestrator."""

# FILE: tests/unit/test_telegram_bot/test_job_orchestrator.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for Telegram remote async job orchestration and delivery metadata.
#   SCOPE: Remote submit payloads, remote status/result polling, failure shaping, metadata persistence
#   DEPENDS: M-TELEGRAM, M-SERVER
#   LINKS: V-M-TELEGRAM
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   TestDeliveryMetadataStoreAsync - Verifies async delivery metadata storage and persistence semantics
#   TestJobSubmissionResult - Verifies submission result DTO semantics including submit correlation
#   TestTelegramJobOrchestratorRemoteSubmission - Verifies remote submit payloads for custom, design, and clone jobs
#   TestTelegramJobOrchestratorRemotePolling - Verifies remote status/result polling and transport/error shaping
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Replaced local core job-execution unit coverage with remote async orchestrator coverage]
# END_CHANGE_SUMMARY

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.contracts.jobs import JobStatus
from telegram_bot.job_orchestrator import (
    DeliveryMetadataStore,
    JobCompletionResult,
    JobSubmissionResult,
    TelegramJobOrchestrator,
)
from telegram_bot.remote_client import (
    RemoteAsyncJobResponse,
    RemoteServerAPIError,
    RemoteServerCorrelation,
    RemoteServerErrorEnvelope,
    RemoteServerTransportError,
)


def _make_settings():
    settings = MagicMock()
    settings.resolve_runtime_model_binding.side_effect = lambda mode: {
        "custom": "model-custom",
        "design": "model-design",
        "clone": "model-clone",
    }.get(mode)
    return settings


def _make_submit_response(job_id: str, submit_request_id: str) -> RemoteAsyncJobResponse:
    return RemoteAsyncJobResponse(
        payload={
            "job_id": job_id,
            "status": "queued",
            "submit_request_id": submit_request_id,
            "status_url": f"/api/v1/tts/jobs/{job_id}",
            "result_url": f"/api/v1/tts/jobs/{job_id}/result",
            "cancel_url": f"/api/v1/tts/jobs/{job_id}/cancel",
        },
        correlation=RemoteServerCorrelation(
            request_id=f"req-{job_id}",
            job_id=job_id,
            submit_request_id=submit_request_id,
        ),
    )


class TestDeliveryMetadataStoreAsync:
    @pytest.fixture
    def temp_storage(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()
        tmp_path = temp_path.with_suffix(".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

    @pytest.fixture
    def store(self, temp_storage):
        return DeliveryMetadataStore(temp_storage)

    @pytest.mark.asyncio
    async def test_create_delivery_metadata_includes_submit_request_id(self, store):
        metadata = await store.create(
            chat_id=12345,
            message_id=67890,
            job_id="job-abc-123",
            submit_request_id="submit-abc-123",
        )

        assert metadata["job_id"] == "job-abc-123"
        assert metadata["submit_request_id"] == "submit-abc-123"
        assert metadata["status"] == "pending"

    @pytest.mark.asyncio
    async def test_persistence_round_trip_preserves_submit_request_id(self, temp_storage):
        store1 = DeliveryMetadataStore(temp_storage)
        await store1.create(
            chat_id=10,
            message_id=20,
            job_id="job-persist",
            submit_request_id="submit-persist",
        )

        store2 = DeliveryMetadataStore(temp_storage)
        metadata = await store2.get(chat_id=10, message_id=20)

        assert metadata is not None
        assert metadata["submit_request_id"] == "submit-persist"


class TestJobSubmissionResult:
    def test_success_result_carries_submit_request_id(self):
        result = JobSubmissionResult(
            success=True,
            job_id="job-1",
            submit_request_id="submit-1",
        )

        assert result.success is True
        assert result.job_id == "job-1"
        assert result.submit_request_id == "submit-1"

    def test_failed_result_has_no_submit_request_id(self):
        result = JobSubmissionResult(success=False, error_message="Error")

        assert result.success is False
        assert result.submit_request_id is None
        assert result.error_message == "Error"


class TestTelegramJobOrchestratorRemoteSubmission:
    @pytest.fixture
    def temp_storage(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()
        tmp_path = temp_path.with_suffix(".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

    @pytest.fixture
    def delivery_store(self, temp_storage):
        return DeliveryMetadataStore(temp_storage)

    @pytest.fixture
    def remote_client(self):
        client = MagicMock()
        client.submit_speech_job = AsyncMock(
            return_value=_make_submit_response("job-new-123", "submit-new-123")
        )
        client.submit_design_job = AsyncMock(
            return_value=_make_submit_response("job-design-123", "submit-design-123")
        )
        client.submit_clone_job = AsyncMock(
            return_value=_make_submit_response("job-clone-123", "submit-clone-123")
        )
        return client

    @pytest.fixture
    def orchestrator(self, delivery_store, remote_client):
        return TelegramJobOrchestrator(
            remote_client=remote_client,
            delivery_store=delivery_store,
            settings=_make_settings(),
        )

    @pytest.mark.asyncio
    async def test_submit_new_job_uses_remote_speech_submit(self, orchestrator, remote_client):
        result = await orchestrator.submit_tts_job(
            text="Hello world",
            speaker="Vivian",
            speed=1.0,
            chat_id=12345,
            message_id=67890,
        )

        assert result.success is True
        assert result.job_id == "job-new-123"
        assert result.submit_request_id == "submit-new-123"
        remote_client.submit_speech_job.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_submit_design_job_uses_design_specific_idempotency_key(
        self, orchestrator, remote_client
    ):
        result = await orchestrator.submit_design_job(
            voice_description="calm narrator",
            text="Hello",
            chat_id=12345,
            message_id=67890,
        )

        assert result.success is True
        kwargs = remote_client.submit_design_job.await_args.kwargs
        payload = remote_client.submit_design_job.await_args.args[0]
        assert kwargs["idempotency_key"] == "telegram:design:12345:67890"
        assert payload["voice_description"] == "calm narrator"

    @pytest.mark.asyncio
    async def test_submit_clone_job_uses_clone_specific_idempotency_key(
        self, orchestrator, remote_client, temp_storage
    ):
        ref_audio = temp_storage.with_suffix(".wav")
        ref_audio.write_bytes(b"RIFFfake")

        result = await orchestrator.submit_clone_job(
            text="Hello",
            ref_text="Sample",
            chat_id=12345,
            message_id=67890,
            ref_audio_path=str(ref_audio),
            ref_audio_content_type="audio/wav",
        )

        assert result.success is True
        kwargs = remote_client.submit_clone_job.await_args.kwargs
        assert kwargs["idempotency_key"] == "telegram:clone:12345:67890"
        assert kwargs["ref_audio_bytes"] == b"RIFFfake"

    @pytest.mark.asyncio
    async def test_submit_transport_failure_returns_failed_result(self, delivery_store):
        remote_client = MagicMock()
        remote_client.submit_speech_job = AsyncMock(
            side_effect=RemoteServerTransportError("connection failed")
        )
        orchestrator = TelegramJobOrchestrator(
            remote_client=remote_client,
            delivery_store=delivery_store,
            settings=_make_settings(),
        )

        result = await orchestrator.submit_tts_job(
            text="Hello",
            speaker="Vivian",
            speed=1.0,
            chat_id=1,
            message_id=2,
        )

        assert result.success is False
        assert result.job_id is None
        assert "connection failed" in (result.error_message or "")


class TestTelegramJobOrchestratorRemotePolling:
    @pytest.fixture
    def temp_storage(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()
        tmp_path = temp_path.with_suffix(".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

    @pytest.fixture
    def delivery_store(self, temp_storage):
        return DeliveryMetadataStore(temp_storage)

    @pytest.mark.asyncio
    async def test_check_job_completion_returns_successful_audio_result(self, delivery_store):
        remote_client = MagicMock()
        remote_client.get_job_status = AsyncMock(
            return_value=RemoteAsyncJobResponse(
                payload={
                    "job_id": "job-ok",
                    "status": "succeeded",
                    "submit_request_id": "submit-ok",
                    "started_at": "2026-04-22T18:00:00+00:00",
                    "completed_at": "2026-04-22T18:00:01+00:00",
                },
                correlation=RemoteServerCorrelation(
                    request_id="req-status",
                    job_id="job-ok",
                    submit_request_id="submit-ok",
                ),
            )
        )
        remote_client.get_job_result = AsyncMock(return_value=MagicMock(audio_bytes=b"RIFFaudio"))
        orchestrator = TelegramJobOrchestrator(
            remote_client=remote_client,
            delivery_store=delivery_store,
            settings=_make_settings(),
        )

        result = await orchestrator.check_job_completion("job-ok", "submit-ok")

        assert result.status is JobStatus.SUCCEEDED
        assert result.success is True
        assert result.audio_bytes == b"RIFFaudio"
        assert result.duration_ms == 1000.0

    @pytest.mark.asyncio
    async def test_check_job_completion_returns_failed_terminal_result(self, delivery_store):
        remote_client = MagicMock()
        remote_client.get_job_status = AsyncMock(
            return_value=RemoteAsyncJobResponse(
                payload={
                    "job_id": "job-failed",
                    "status": "failed",
                    "submit_request_id": "submit-failed",
                    "terminal_error": {
                        "code": "job_execution_timeout",
                        "message": "Async job timed out",
                    },
                },
                correlation=RemoteServerCorrelation(
                    request_id="req-failed",
                    job_id="job-failed",
                    submit_request_id="submit-failed",
                ),
            )
        )
        orchestrator = TelegramJobOrchestrator(
            remote_client=remote_client,
            delivery_store=delivery_store,
            settings=_make_settings(),
        )

        result = await orchestrator.check_job_completion("job-failed", "submit-failed")

        assert result.status is JobStatus.FAILED
        assert result.success is False
        assert result.error_code == "job_execution_timeout"

    @pytest.mark.asyncio
    async def test_check_job_completion_result_fetch_api_error_becomes_failed(self, delivery_store):
        remote_client = MagicMock()
        remote_client.get_job_status = AsyncMock(
            return_value=RemoteAsyncJobResponse(
                payload={
                    "job_id": "job-api-error",
                    "status": "succeeded",
                    "submit_request_id": "submit-api-error",
                },
                correlation=RemoteServerCorrelation(
                    request_id="req-api",
                    job_id="job-api-error",
                    submit_request_id="submit-api-error",
                ),
            )
        )
        remote_client.get_job_result = AsyncMock(
            side_effect=RemoteServerAPIError(
                RemoteServerErrorEnvelope(
                    code="job_not_succeeded",
                    message="Job result fetch failed",
                    details={},
                    request_id="req-api-result",
                )
            )
        )
        orchestrator = TelegramJobOrchestrator(
            remote_client=remote_client,
            delivery_store=delivery_store,
            settings=_make_settings(),
        )

        result = await orchestrator.check_job_completion("job-api-error", "submit-api-error")

        assert result.status is JobStatus.FAILED
        assert result.success is False
        assert result.error_code == "job_not_succeeded"

    @pytest.mark.asyncio
    async def test_check_job_completion_transport_failure_stays_non_terminal(self, delivery_store):
        remote_client = MagicMock()
        remote_client.get_job_status = AsyncMock(
            side_effect=RemoteServerTransportError("connection failed")
        )
        orchestrator = TelegramJobOrchestrator(
            remote_client=remote_client,
            delivery_store=delivery_store,
            settings=_make_settings(),
        )

        result = await orchestrator.check_job_completion("job-pending", "submit-pending")

        assert result.status is JobStatus.QUEUED
        assert result.success is None

    @pytest.mark.asyncio
    async def test_check_job_completion_job_not_found_is_terminal_failure(self, delivery_store):
        remote_client = MagicMock()
        remote_client.get_job_status = AsyncMock(
            side_effect=RemoteServerAPIError(
                RemoteServerErrorEnvelope(
                    code="job_not_found",
                    message="Job was not found",
                    details={"job_id": "job-missing"},
                    request_id="req-missing",
                )
            )
        )
        orchestrator = TelegramJobOrchestrator(
            remote_client=remote_client,
            delivery_store=delivery_store,
            settings=_make_settings(),
        )

        result = await orchestrator.check_job_completion("job-missing", "submit-missing")

        assert result.status is JobStatus.FAILED
        assert result.is_terminal is True
        assert result.success is False
        assert result.error_code == "job_not_found"


class TestJobCompletionResult:
    def test_completed_result_fields(self):
        result = JobCompletionResult(
            status=JobStatus.SUCCEEDED,
            is_terminal=True,
            success=True,
            audio_bytes=b"fake_audio",
            duration_ms=1500.0,
        )

        assert result.is_terminal is True
        assert result.success is True
        assert result.audio_bytes == b"fake_audio"
        assert result.duration_ms == 1500.0

    def test_failed_result_fields(self):
        result = JobCompletionResult(
            status=JobStatus.FAILED,
            is_terminal=True,
            success=False,
            error_message="Backend unavailable",
            error_code="E_BACKEND_DOWN",
        )

        assert result.is_terminal is True
        assert result.success is False
        assert result.error_message == "Backend unavailable"
        assert result.error_code == "E_BACKEND_DOWN"
