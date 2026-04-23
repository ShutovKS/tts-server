"""Integration tests for Telegram remote async job integration."""

# FILE: tests/integration/test_telegram_adapter/test_job_integration.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Integration tests for Telegram remote async job submission, polling, delivery, and no-local-fallback behavior.
#   SCOPE: Remote orchestrator submission, delivery metadata persistence, async poll/result flows, dispatcher remote-only command routing
#   DEPENDS: M-TELEGRAM, M-SERVER
#   LINKS: V-M-TELEGRAM
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _make_settings - Build Telegram settings doubles with runtime bindings for remote-job tests
#   _make_remote_submit_response - Build remote async submit snapshots with stable correlation metadata
#   TestTelegramRemoteJobSubmissionIntegration - Verifies remote submit flows for /tts, /design, and /clone orchestration
#   TestTelegramRemoteJobLifecycleIntegration - Verifies remote status/result polling, terminal failures, result fetch failure, and transport failure behavior
#   TestTelegramDispatcherRemoteOnlyIntegration - Verifies migrated dispatcher command paths require remote async orchestration and do not fall back to local synthesis
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Replaced local job-execution integration coverage with remote async Telegram cutover coverage]
# END_CHANGE_SUMMARY

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.contracts.jobs import JobStatus
from telegram_bot.handlers.dispatcher import CommandDispatcher
from telegram_bot.job_orchestrator import (
    DeliveryMetadataStore,
    JobCompletionResult,
    TelegramJobOrchestrator,
    TelegramJobPoller,
)
from telegram_bot.remote_client import (
    RemoteAsyncJobResponse,
    RemoteServerAPIError,
    RemoteServerCorrelation,
    RemoteServerErrorEnvelope,
    RemoteServerTransportError,
)


pytestmark = pytest.mark.integration


class _SenderDouble:
    def __init__(self):
        self.send_text = AsyncMock()
        self.send_voice = AsyncMock()


def _make_settings(**overrides):
    defaults = {
        "telegram_default_speaker": "Vivian",
        "telegram_max_text_length": 1000,
        "telegram_server_base_url": "http://server.internal:8000",
        "active_family": "qwen",
        "is_user_allowed": lambda user_id: True,
    }
    defaults.update(overrides)

    def resolve_runtime_model_binding(mode: str) -> str | None:
        bindings = {
            "custom": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "design": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
            "clone": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        }
        return bindings.get(mode)

    settings = SimpleNamespace(**defaults)
    settings.resolve_runtime_model_binding = resolve_runtime_model_binding
    return settings


def _make_remote_submit_response(
    *,
    job_id: str,
    submit_request_id: str,
    status: str = "queued",
    created: bool | None = None,
) -> RemoteAsyncJobResponse:
    payload: dict[str, object] = {
        "job_id": job_id,
        "status": status,
        "submit_request_id": submit_request_id,
        "status_url": f"/api/v1/tts/jobs/{job_id}",
        "result_url": f"/api/v1/tts/jobs/{job_id}/result",
        "cancel_url": f"/api/v1/tts/jobs/{job_id}/cancel",
    }
    if created is not None:
        payload["created"] = created
    return RemoteAsyncJobResponse(
        payload=payload,
        correlation=RemoteServerCorrelation(
            request_id=f"req-{job_id}",
            job_id=job_id,
            submit_request_id=submit_request_id,
        ),
    )


class TestTelegramRemoteJobSubmissionIntegration:
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
    def settings(self):
        return _make_settings()

    @pytest.fixture
    def remote_client(self):
        client = MagicMock()
        client.submit_speech_job = AsyncMock(
            return_value=_make_remote_submit_response(
                job_id="job-tts-123",
                submit_request_id="submit-tts-123",
            )
        )
        client.submit_design_job = AsyncMock(
            return_value=_make_remote_submit_response(
                job_id="job-design-123",
                submit_request_id="submit-design-123",
            )
        )
        client.submit_clone_job = AsyncMock(
            return_value=_make_remote_submit_response(
                job_id="job-clone-123",
                submit_request_id="submit-clone-123",
            )
        )
        return client

    @pytest.fixture
    def orchestrator(self, remote_client, delivery_store, settings):
        return TelegramJobOrchestrator(
            remote_client=remote_client,
            delivery_store=delivery_store,
            settings=settings,
        )

    @pytest.mark.anyio
    async def test_submit_tts_job_uses_remote_async_http_client(self, orchestrator, remote_client):
        result = await orchestrator.submit_tts_job(
            text="Hello remote",
            speaker="Vivian",
            speed=1.0,
            chat_id=12345,
            message_id=67890,
            language="ru",
        )

        assert result.success is True
        assert result.job_id == "job-tts-123"
        assert result.submit_request_id == "submit-tts-123"
        remote_client.submit_speech_job.assert_awaited_once()
        kwargs = remote_client.submit_speech_job.await_args.kwargs
        payload = remote_client.submit_speech_job.await_args.args[0]
        assert kwargs["request_id"] == "telegram:12345:67890"
        assert kwargs["idempotency_key"] == "telegram:12345:67890"
        assert payload["model"] == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"

    @pytest.mark.anyio
    async def test_submit_design_job_uses_dedicated_remote_async_endpoint(self, orchestrator, remote_client):
        result = await orchestrator.submit_design_job(
            voice_description="calm narrator",
            text="Hello design",
            chat_id=12345,
            message_id=67891,
            language="en",
        )

        assert result.success is True
        assert result.job_id == "job-design-123"
        assert result.submit_request_id == "submit-design-123"
        remote_client.submit_design_job.assert_awaited_once()
        kwargs = remote_client.submit_design_job.await_args.kwargs
        payload = remote_client.submit_design_job.await_args.args[0]
        assert kwargs["idempotency_key"] == "telegram:design:12345:67891"
        assert payload["voice_description"] == "calm narrator"

    @pytest.mark.anyio
    async def test_submit_clone_job_uses_remote_async_clone_endpoint(self, orchestrator, remote_client, temp_storage):
        ref_audio = temp_storage.with_suffix(".wav")
        ref_audio.write_bytes(b"RIFFfake")

        result = await orchestrator.submit_clone_job(
            text="Hello clone",
            ref_text="sample transcript",
            chat_id=12345,
            message_id=67892,
            ref_audio_path=str(ref_audio),
            ref_audio_content_type="audio/wav",
            language="auto",
        )

        assert result.success is True
        assert result.job_id == "job-clone-123"
        assert result.submit_request_id == "submit-clone-123"
        remote_client.submit_clone_job.assert_awaited_once()
        kwargs = remote_client.submit_clone_job.await_args.kwargs
        assert kwargs["idempotency_key"] == "telegram:clone:12345:67892"
        assert kwargs["ref_audio_filename"] == ref_audio.name
        assert kwargs["ref_audio_content_type"] == "audio/wav"
        assert kwargs["ref_audio_bytes"] == b"RIFFfake"

    @pytest.mark.anyio
    async def test_dispatcher_persists_submit_request_id_for_pending_delivery(
        self,
        remote_client,
        delivery_store,
        settings,
    ):
        orchestrator = TelegramJobOrchestrator(
            remote_client=remote_client,
            delivery_store=delivery_store,
            settings=settings,
        )
        dispatcher = CommandDispatcher(
            synthesizer=MagicMock(),
            settings=settings,
            sender=_SenderDouble(),
            job_orchestrator=orchestrator,
            delivery_store=delivery_store,
        )

        await dispatcher._handle_tts_via_job(
            chat_id=12345,
            message_id=67890,
            text="Hello",
            speaker="Vivian",
            speed=1.0,
            language="ru",
        )

        metadata = await delivery_store.get(12345, 67890)
        assert metadata is not None
        assert metadata["job_id"] == "job-tts-123"
        assert metadata["submit_request_id"] == "submit-tts-123"


class TestTelegramRemoteJobLifecycleIntegration:
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
    def settings(self):
        return _make_settings()

    @pytest.mark.anyio
    async def test_check_job_completion_fetches_remote_result_when_succeeded(self, delivery_store, settings):
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
        remote_client.get_job_result = AsyncMock(
            return_value=SimpleNamespace(audio_bytes=b"RIFFaudio")
        )
        orchestrator = TelegramJobOrchestrator(
            remote_client=remote_client,
            delivery_store=delivery_store,
            settings=settings,
        )

        result = await orchestrator.check_job_completion("job-ok", "submit-ok")

        assert result.is_terminal is True
        assert result.success is True
        assert result.status is JobStatus.SUCCEEDED
        assert result.audio_bytes == b"RIFFaudio"
        assert result.duration_ms == 1000.0
        remote_client.get_job_result.assert_awaited_once()

    @pytest.mark.anyio
    async def test_check_job_completion_reports_remote_failed_job(self, delivery_store, settings):
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
            settings=settings,
        )

        result = await orchestrator.check_job_completion("job-failed", "submit-failed")

        assert result.is_terminal is True
        assert result.success is False
        assert result.status is JobStatus.FAILED
        assert result.error_code == "job_execution_timeout"
        assert result.error_message == "Async job timed out"

    @pytest.mark.anyio
    async def test_check_job_completion_reports_result_fetch_failure(self, delivery_store, settings):
        remote_client = MagicMock()
        remote_client.get_job_status = AsyncMock(
            return_value=RemoteAsyncJobResponse(
                payload={
                    "job_id": "job-result-fail",
                    "status": "succeeded",
                    "submit_request_id": "submit-result-fail",
                },
                correlation=RemoteServerCorrelation(
                    request_id="req-status-fail",
                    job_id="job-result-fail",
                    submit_request_id="submit-result-fail",
                ),
            )
        )
        remote_client.get_job_result = AsyncMock(
            side_effect=RemoteServerAPIError(
                RemoteServerErrorEnvelope(
                    code="job_not_succeeded",
                    message="Job result fetch failed",
                    details={},
                    request_id="req-result-fail",
                )
            )
        )
        orchestrator = TelegramJobOrchestrator(
            remote_client=remote_client,
            delivery_store=delivery_store,
            settings=settings,
        )

        result = await orchestrator.check_job_completion(
            "job-result-fail", "submit-result-fail"
        )

        assert result.is_terminal is True
        assert result.success is False
        assert result.status is JobStatus.FAILED
        assert result.error_code == "job_not_succeeded"
        assert result.error_message == "Job result fetch failed"

    @pytest.mark.anyio
    async def test_check_job_completion_treats_unreachable_server_as_non_terminal(self, delivery_store, settings):
        remote_client = MagicMock()
        remote_client.get_job_status = AsyncMock(
            side_effect=RemoteServerTransportError("Remote server connection failed")
        )
        orchestrator = TelegramJobOrchestrator(
            remote_client=remote_client,
            delivery_store=delivery_store,
            settings=settings,
        )

        result = await orchestrator.check_job_completion("job-pending", "submit-pending")

        assert result.is_terminal is False
        assert result.success is None
        assert result.status is JobStatus.QUEUED

    @pytest.mark.anyio
    async def test_check_job_completion_treats_missing_remote_job_as_terminal_failure(self, delivery_store, settings):
        remote_client = MagicMock()
        remote_client.get_job_status = AsyncMock(
            side_effect=RemoteServerAPIError(
                RemoteServerErrorEnvelope(
                    code="job_not_found",
                    message="Job was not found",
                    details={"job_id": "job-missing"},
                    request_id="req-job-missing",
                )
            )
        )
        orchestrator = TelegramJobOrchestrator(
            remote_client=remote_client,
            delivery_store=delivery_store,
            settings=settings,
        )

        result = await orchestrator.check_job_completion("job-missing", "submit-missing")

        assert result.is_terminal is True
        assert result.success is False
        assert result.status is JobStatus.FAILED
        assert result.error_code == "job_not_found"

    @pytest.mark.anyio
    async def test_poller_uses_submit_request_id_when_polling_pending_deliveries(self, delivery_store, settings):
        remote_client = MagicMock()
        remote_client.get_job_status = AsyncMock(
            return_value=RemoteAsyncJobResponse(
                payload={
                    "job_id": "job-poller",
                    "status": "succeeded",
                    "submit_request_id": "submit-poller",
                },
                correlation=RemoteServerCorrelation(
                    request_id="req-poller-status",
                    job_id="job-poller",
                    submit_request_id="submit-poller",
                ),
            )
        )
        remote_client.get_job_result = AsyncMock(
            return_value=SimpleNamespace(audio_bytes=b"RIFFpoller")
        )
        orchestrator = TelegramJobOrchestrator(
            remote_client=remote_client,
            delivery_store=delivery_store,
            settings=settings,
        )
        sender = _SenderDouble()
        sender.send_voice.return_value = SimpleNamespace(success=True)
        poller = TelegramJobPoller(
            orchestrator=orchestrator,
            sender=sender,
            delivery_store=delivery_store,
            settings=settings,
        )
        await delivery_store.create(
            chat_id=1,
            message_id=2,
            job_id="job-poller",
            submit_request_id="submit-poller",
        )

        await poller._check_pending_deliveries()

        remote_client.get_job_status.assert_awaited_once_with(
            "job-poller",
            request_id="job-poller",
            submit_request_id="submit-poller",
        )
        remote_client.get_job_result.assert_awaited_once_with(
            "job-poller",
            request_id="job-poller",
            submit_request_id="submit-poller",
        )


class TestTelegramDispatcherRemoteOnlyIntegration:
    @pytest.fixture
    def settings(self):
        return _make_settings()

    @pytest.mark.anyio
    async def test_tts_requires_remote_job_orchestrator_and_has_no_local_fallback(self, settings):
        synthesizer = MagicMock()
        synthesizer.synthesize = AsyncMock()
        sender = _SenderDouble()
        dispatcher = CommandDispatcher(
            synthesizer=synthesizer,
            settings=settings,
            sender=sender,
        )

        await dispatcher.handle_update(
            text="/tts -- Hello world",
            user_id=1,
            chat_id=2,
            message_id=3,
            chat_type="private",
        )

        synthesizer.synthesize.assert_not_called()
        sender.send_text.assert_awaited()
        assert sender.send_text.await_args is not None
        sent_text = sender.send_text.await_args.args[1]
        assert "async server" in sent_text.lower()

    @pytest.mark.anyio
    async def test_design_requires_remote_job_orchestrator_and_has_no_local_fallback(self, settings):
        synthesizer = MagicMock()
        synthesizer.synthesize_design = AsyncMock()
        sender = _SenderDouble()
        dispatcher = CommandDispatcher(
            synthesizer=synthesizer,
            settings=settings,
            sender=sender,
        )

        await dispatcher.handle_update(
            text="/design calm narrator -- Hello world",
            user_id=1,
            chat_id=2,
            message_id=3,
            chat_type="private",
        )

        synthesizer.synthesize_design.assert_not_called()
        sender.send_text.assert_awaited()
        assert sender.send_text.await_args is not None
        sent_text = sender.send_text.await_args.args[1]
        assert "async server" in sent_text.lower()

    @pytest.mark.anyio
    async def test_tts_remote_submit_failure_surfaces_error_without_local_fallback(self, settings):
        remote_client = MagicMock()
        remote_client.submit_speech_job = AsyncMock(
            side_effect=RemoteServerTransportError("Remote server connection failed")
        )
        remote_client.submit_design_job = AsyncMock()
        remote_client.submit_clone_job = AsyncMock()
        delivery_store = DeliveryMetadataStore(Path(tempfile.mkstemp(suffix=".json")[1]))
        orchestrator = TelegramJobOrchestrator(
            remote_client=remote_client,
            delivery_store=delivery_store,
            settings=settings,
        )
        synthesizer = MagicMock()
        synthesizer.synthesize = AsyncMock()
        sender = _SenderDouble()
        dispatcher = CommandDispatcher(
            synthesizer=synthesizer,
            settings=settings,
            sender=sender,
            job_orchestrator=orchestrator,
            delivery_store=delivery_store,
        )

        try:
            await dispatcher.handle_update(
                text="/tts -- Hello world",
                user_id=1,
                chat_id=2,
                message_id=3,
                chat_type="private",
            )
        finally:
            delivery_store_path = Path(delivery_store._storage_path)
            if delivery_store_path.exists():
                delivery_store_path.unlink()
            tmp_path = delivery_store_path.with_suffix(".tmp")
            if tmp_path.exists():
                tmp_path.unlink()

        synthesizer.synthesize.assert_not_called()
        sent_messages = [call.args[1] for call in sender.send_text.await_args_list]
        assert any("failed to submit job" in message.lower() for message in sent_messages)
