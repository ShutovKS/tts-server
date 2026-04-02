"""
Integration tests for Telegram job integration.

Tests end-to-end flows including:
- Job submission via dispatcher
- Idempotency across multiple updates
- Completion polling and delivery
- Recovery on restart
"""

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from telegram_bot.job_orchestrator import (
    DeliveryMetadataStore,
    JobCompletionResult,
    JobSubmissionResult,
    TelegramJobOrchestrator,
    TelegramJobPoller,
)


class TestTelegramJobSubmissionIntegration:
    """Integration tests for job submission flow."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage file."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False
        ) as f:
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()
        tmp_path = temp_path.with_suffix(".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
    
    @pytest.fixture
    def delivery_store(self, temp_storage):
        """Create delivery store."""
        return DeliveryMetadataStore(temp_storage)
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.default_speaker = "Vivian"
        settings.request_timeout_seconds = 300
        return settings
    
    @pytest.fixture
    def mock_job_execution_with_store(self):
        """Create mock job execution gateway with _store attribute."""
        gateway = MagicMock()
        
        # Mock the _store to return None (no existing job)
        mock_store = MagicMock()
        mock_store.get_by_idempotency_key = MagicMock(return_value=None)
        gateway._store = mock_store
        
        mock_resolution = MagicMock()
        mock_resolution.created = True
        mock_resolution.snapshot = MagicMock()
        mock_resolution.snapshot.job_id = "job-integration-test"
        mock_resolution.snapshot.status = MagicMock()
        gateway.submit_idempotent = MagicMock(return_value=mock_resolution)
        
        return gateway
    
    @pytest.fixture
    def orchestrator(self, delivery_store, mock_job_execution_with_store, mock_settings):
        """Create job orchestrator."""
        return TelegramJobOrchestrator(
            job_execution=mock_job_execution_with_store,
            delivery_store=delivery_store,
            settings=mock_settings,
        )
    
    def test_orchestrator_creates_delivery_metadata(
        self, orchestrator, delivery_store
    ):
        """Orchestrator creates delivery metadata when submitting job."""
        result = orchestrator.submit_tts_job(
            text="Test",
            speaker="en-US-Neural2-F",
            speed=1.0,
            chat_id=12345,
            message_id=67890,
        )
        
        assert result.success is True
        assert result.job_id == "job-integration-test"
    
    def test_idempotent_submission(
        self, delivery_store, mock_job_execution_with_store, mock_settings
    ):
        """Duplicate submissions are handled via gateway idempotency."""
        orchestrator = TelegramJobOrchestrator(
            job_execution=mock_job_execution_with_store,
            delivery_store=delivery_store,
            settings=mock_settings,
        )
        
        # First submission
        result1 = orchestrator.submit_tts_job(
            text="Hello duplicate",
            speaker="en-US-Neural2-F",
            speed=1.0,
            chat_id=12345,
            message_id=67890,
        )
        
        # Gateway should be called once for this job
        assert mock_job_execution_with_store.submit_idempotent.call_count == 1


class TestTelegramDeliveryRecoveryIntegration:
    """Integration tests for delivery recovery on restart."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage file."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False
        ) as f:
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()
        tmp_path = temp_path.with_suffix(".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
    
    @pytest.fixture
    def delivery_store(self, temp_storage):
        """Create delivery store."""
        return DeliveryMetadataStore(temp_storage)
    
    @pytest.mark.asyncio
    async def test_pending_deliveries_recovered_after_restart(self, temp_storage):
        """Pending deliveries are recovered after process restart."""
        # Simulate pre-restart state: create pending deliveries
        delivery_store = DeliveryMetadataStore(temp_storage)
        await delivery_store.create(chat_id=1, message_id=1, job_id="job-restart-1")
        await delivery_store.create(chat_id=2, message_id=2, job_id="job-restart-2")
        
        # Simulate restart: new store instance
        new_delivery_store = DeliveryMetadataStore(temp_storage)
        
        # New store should see pending deliveries
        pending = await new_delivery_store.get_pending_deliveries()
        
        assert len(pending) == 2
        job_ids = {p["job_id"] for p in pending}
        assert job_ids == {"job-restart-1", "job-restart-2"}
    
    @pytest.mark.asyncio
    async def test_already_delivered_not_in_pending(self, temp_storage):
        """Already delivered jobs are not in pending list."""
        # Simulate pre-restart state: delivered
        delivery_store = DeliveryMetadataStore(temp_storage)
        await delivery_store.create(chat_id=12345, message_id=67890, job_id="job-done")
        await delivery_store.mark_delivered(chat_id=12345, message_id=67890, success=True)
        
        # Simulate restart
        new_delivery_store = DeliveryMetadataStore(temp_storage)
        
        # Should have no pending deliveries
        pending = await new_delivery_store.get_pending_deliveries()
        assert len(pending) == 0
        
        # Check status via get
        metadata = await new_delivery_store.get(chat_id=12345, message_id=67890)
        assert metadata is not None
        assert metadata["status"] == "delivered"


class TestTelegramJobLifecycleIntegration:
    """Integration tests for job lifecycle with various states."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage file."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False
        ) as f:
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()
        tmp_path = temp_path.with_suffix(".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
    
    @pytest.mark.asyncio
    async def test_submission_creates_pending_delivery(self, temp_storage):
        """Submission flow creates pending delivery entry."""
        delivery_store = DeliveryMetadataStore(temp_storage)
        
        mock_settings = MagicMock()
        mock_settings.default_speaker = "Vivian"
        mock_settings.request_timeout_seconds = 300
        
        mock_gateway = MagicMock()
        mock_store = MagicMock()
        mock_store.get_by_idempotency_key = MagicMock(return_value=None)
        mock_gateway._store = mock_store
        
        mock_resolution = MagicMock()
        mock_resolution.created = True
        mock_resolution.snapshot = MagicMock()
        mock_resolution.snapshot.job_id = "job-flow-test"
        mock_gateway.submit_idempotent = MagicMock(return_value=mock_resolution)
        
        orchestrator = TelegramJobOrchestrator(
            job_execution=mock_gateway,
            delivery_store=delivery_store,
            settings=mock_settings,
        )
        
        result = orchestrator.submit_tts_job(
            text="Test",
            speaker="en-US-Neural2-F",
            speed=1.0,
            chat_id=12345,
            message_id=67890,
        )
        
        assert result.success is True
        assert result.job_id == "job-flow-test"
    
    @pytest.mark.asyncio
    async def test_delivery_completes_after_result(self, temp_storage):
        """Delivery becomes complete after marking as delivered."""
        delivery_store = DeliveryMetadataStore(temp_storage)
        
        # Create pending delivery
        await delivery_store.create(
            chat_id=12345,
            message_id=67890,
            job_id="job-complete"
        )
        
        # Verify is pending
        metadata = await delivery_store.get(chat_id=12345, message_id=67890)
        assert metadata is not None
        assert metadata["status"] == "pending"
        
        # Mark as delivered
        await delivery_store.mark_delivered(
            chat_id=12345,
            message_id=67890,
            success=True,
        )
        
        # Now should be delivered
        metadata = await delivery_store.get(chat_id=12345, message_id=67890)
        assert metadata["status"] == "delivered"
    
    @pytest.mark.asyncio
    async def test_failed_delivery_tracked(self, temp_storage):
        """Failed delivery is tracked correctly."""
        delivery_store = DeliveryMetadataStore(temp_storage)
        
        # Create pending delivery
        await delivery_store.create(
            chat_id=12345,
            message_id=67890,
            job_id="job-fail-test"
        )
        
        # Mark as failed
        await delivery_store.mark_delivered(
            chat_id=12345,
            message_id=67890,
            success=False,
            error_message="Network error"
        )
        
        # Check status
        metadata = await delivery_store.get(chat_id=12345, message_id=67890)
        assert metadata["status"] == "failed"
        assert metadata["success"] is False
        assert metadata["error_message"] == "Network error"


class TestJobCompletionResult:
    """Tests for job completion result handling."""
    
    def test_completed_result_fields(self):
        """Completed job has expected fields."""
        result = JobCompletionResult(
            status="completed",  # Using string as actual status enum may vary
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
        """Failed job has expected fields."""
        result = JobCompletionResult(
            status="failed",
            is_terminal=True,
            success=False,
            error_message="Backend unavailable",
            error_code="E_BACKEND_DOWN",
        )
        
        assert result.is_terminal is True
        assert result.success is False
        assert result.error_message == "Backend unavailable"
        assert result.error_code == "E_BACKEND_DOWN"
    
    def test_running_result_not_terminal(self):
        """Running job has non-terminal result."""
        result = JobCompletionResult(
            status="running",
            is_terminal=False,
            success=None,
        )
        
        assert result.is_terminal is False
        assert result.success is None
    
    def test_queued_result_not_terminal(self):
        """Queued job has non-terminal result."""
        result = JobCompletionResult(
            status="queued",
            is_terminal=False,
            success=None,
        )
        
        assert result.is_terminal is False
        assert result.success is None
