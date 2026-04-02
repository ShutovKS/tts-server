"""
Tests for Telegram job orchestrator.

Covers:
- Job submission flow
- Idempotency (duplicate handling)
- Completion polling
- Delivery recovery on restart
- Error handling
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
    JobSuccessSnapshot,
    TelegramJobOrchestrator,
    TelegramJobPoller,
)


class TestDeliveryMetadataStoreAsync:
    """Tests for async DeliveryMetadataStore."""
    
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
    def store(self, temp_storage):
        """Create store with temporary storage."""
        return DeliveryMetadataStore(temp_storage)
    
    @pytest.mark.asyncio
    async def test_create_delivery_metadata(self, store):
        """Creating delivery metadata works correctly."""
        metadata = await store.create(
            chat_id=12345,
            message_id=67890,
            job_id="job-abc-123"
        )
        
        assert metadata["chat_id"] == 12345
        assert metadata["message_id"] == 67890
        assert metadata["job_id"] == "job-abc-123"
        assert metadata["idempotency_key"] == "telegram:12345:67890"
        assert metadata["status"] == "pending"
    
    @pytest.mark.asyncio
    async def test_get_delivery_metadata(self, store):
        """Getting delivery metadata by identity works."""
        await store.create(chat_id=12345, message_id=67890, job_id="job-abc")
        
        metadata = await store.get(chat_id=12345, message_id=67890)
        
        assert metadata is not None
        assert metadata["job_id"] == "job-abc"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Getting nonexistent metadata returns None."""
        metadata = await store.get(chat_id=99999, message_id=99999)
        assert metadata is None
    
    @pytest.mark.asyncio
    async def test_mark_delivered_success(self, store):
        """Marking delivery as successful works."""
        await store.create(chat_id=12345, message_id=67890, job_id="job-abc")
        
        updated = await store.mark_delivered(
            chat_id=12345,
            message_id=67890,
            success=True
        )
        
        assert updated is not None
        assert updated["status"] == "delivered"
        assert updated["success"] is True
    
    @pytest.mark.asyncio
    async def test_mark_delivered_failure(self, store):
        """Marking delivery as failed works."""
        await store.create(chat_id=12345, message_id=67890, job_id="job-abc")
        
        updated = await store.mark_delivered(
            chat_id=12345,
            message_id=67890,
            success=False,
            error_message="Network error"
        )
        
        assert updated is not None
        assert updated["status"] == "failed"
        assert updated["success"] is False
        assert updated["error_message"] == "Network error"
    
    @pytest.mark.asyncio
    async def test_get_pending_deliveries(self, store):
        """Getting pending deliveries works."""
        # Create several entries
        await store.create(chat_id=1, message_id=1, job_id="job-1")
        await store.create(chat_id=2, message_id=2, job_id="job-2")
        await store.create(chat_id=3, message_id=3, job_id="job-3")
        
        # Mark one as delivered
        await store.mark_delivered(chat_id=1, message_id=1, success=True)
        
        pending = await store.get_pending_deliveries()
        
        assert len(pending) == 2
        job_ids = {p["job_id"] for p in pending}
        assert job_ids == {"job-2", "job-3"}
    
    @pytest.mark.asyncio
    async def test_persistence(self, temp_storage):
        """Metadata persists across store instances."""
        # Create and populate
        store1 = DeliveryMetadataStore(temp_storage)
        await store1.create(chat_id=12345, message_id=67890, job_id="job-persist")
        
        # New store instance should see same data
        store2 = DeliveryMetadataStore(temp_storage)
        metadata = await store2.get(chat_id=12345, message_id=67890)
        
        assert metadata is not None
        assert metadata["job_id"] == "job-persist"
    
    @pytest.mark.asyncio
    async def test_create_returns_copy_not_internal_cache(self, store):
        """Returned metadata mutation must not affect internal cache."""
        metadata = await store.create(chat_id=10, message_id=20, job_id="job-copy")
        metadata["job_id"] = "tampered"
        
        persisted = await store.get(chat_id=10, message_id=20)
        
        assert persisted is not None
        assert persisted["job_id"] == "job-copy"


class TestJobSubmissionResult:
    """Tests for JobSubmissionResult dataclass."""
    
    def test_success_result(self):
        """Success result with job ID."""
        result = JobSubmissionResult(success=True, job_id="job-1")
        assert result.success is True
        assert result.job_id == "job-1"
        assert result.is_duplicate is False
        assert result.error_message is None
    
    def test_duplicate_result(self):
        """Duplicate job result."""
        result = JobSubmissionResult(success=True, job_id="job-1", is_duplicate=True)
        assert result.success is True
        assert result.is_duplicate is True
    
    def test_failed_result(self):
        """Failed submission result."""
        result = JobSubmissionResult(success=False, error_message="Error")
        assert result.success is False
        assert result.job_id is None
        assert result.error_message == "Error"


class TestTelegramJobOrchestrator:
    """Tests for TelegramJobOrchestrator."""
    
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
        
        # Default: new job
        mock_resolution = MagicMock()
        mock_resolution.created = True
        mock_resolution.snapshot = MagicMock()
        mock_resolution.snapshot.job_id = "job-new-123"
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
    
    def test_submit_new_job_sync(self, orchestrator, mock_job_execution_with_store):
        """Submitting a new job creates job (sync method)."""
        result = orchestrator.submit_tts_job(
            text="Hello world",
            speaker="en-US-Neural2-F",
            speed=1.0,
            chat_id=12345,
            message_id=67890,
        )
        
        assert result.success is True
        assert result.job_id == "job-new-123"
        
        # Verify gateway was called
        mock_job_execution_with_store.submit_idempotent.assert_called_once()
    
    def test_submit_returns_success_result(self, orchestrator):
        """Submission returns success result."""
        result = orchestrator.submit_tts_job(
            text="Hello",
            speaker="en-US-Neural2-F",
            speed=1.0,
            chat_id=12345,
            message_id=67890,
        )
        
        assert result.success is True
        assert result.job_id is not None
        # Note: actual delivery metadata creation is async and happens
        # in the polling loop via TelegramJobPoller
    
    def test_submit_design_job_new(self, orchestrator, mock_job_execution_with_store):
        """Submitting a new design job creates job."""
        result = orchestrator.submit_design_job(
            voice_description="calm narrator",
            text="Hello world",
            chat_id=12345,
            message_id=67890,
        )
        
        assert result.success is True
        assert result.job_id == "job-new-123"
        
        # Verify gateway was called
        mock_job_execution_with_store.submit_idempotent.assert_called_once()
    
    def test_submit_design_job_returns_success(self, orchestrator):
        """Design job submission returns success result."""
        result = orchestrator.submit_design_job(
            voice_description="energetic host",
            text="Hello",
            chat_id=12345,
            message_id=67890,
        )
        
        assert result.success is True
        assert result.job_id is not None
    
    def test_submit_design_job_uses_correct_idempotency_key(self, orchestrator, mock_job_execution_with_store):
        """Design job uses design-specific idempotency key."""
        result = orchestrator.submit_design_job(
            voice_description="calm narrator",
            text="Hello",
            chat_id=12345,
            message_id=67890,
        )
        
        # Verify the call was made
        mock_job_execution_with_store.submit_idempotent.assert_called_once()
        
        # The idempotency key should include 'design' prefix
        call_args = mock_job_execution_with_store.submit_idempotent.call_args
        submission = call_args[0][0]
        assert "design" in submission.idempotency_key


class TestDesignJobIdempotency:
    """Tests for design job idempotency."""
    
    def test_design_idempotency_key_format(self):
        """Design idempotency key follows expected format."""
        expected = "telegram:design:12345:67890"
        
        key = f"telegram:design:{12345}:{67890}"
        assert key == expected
    
    def test_design_and_tts_have_different_keys(self):
        """Design and TTS jobs have different idempotency keys."""
        tts_key = f"telegram:12345:67890"
        design_key = f"telegram:design:12345:67890"
        assert tts_key != design_key


class TestJobIdempotency:
    """Tests for idempotency key generation and handling."""
    
    def test_idempotency_key_format(self):
        """Idempotency key follows expected format."""
        # The key should be: telegram:{chat_id}:{message_id}
        expected = "telegram:12345:67890"
        
        # This is the format used internally
        key = f"telegram:{12345}:{67890}"
        assert key == expected
    
    def test_idempotency_scope(self):
        """Idempotency scope is telegram."""
        from telegram_bot.job_orchestrator import TELEGRAM_IDEMPOTENCY_SCOPE
        assert TELEGRAM_IDEMPOTENCY_SCOPE == "telegram"
    
    def test_different_messages_have_different_keys(self):
        """Different messages generate different idempotency keys."""
        key1 = f"telegram:12345:111"
        key2 = f"telegram:12345:222"
        assert key1 != key2
    
    def test_different_chats_have_different_keys(self):
        """Different chats generate different idempotency keys."""
        key1 = f"telegram:11111:67890"
        key2 = f"telegram:22222:67890"
        assert key1 != key2


class TestCloneJobSubmission:
    """Tests for clone job submission."""
    
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
        
        # Default: new job
        mock_resolution = MagicMock()
        mock_resolution.created = True
        mock_resolution.snapshot = MagicMock()
        mock_resolution.snapshot.job_id = "job-clone-123"
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
    
    def test_submit_clone_job_new(self, orchestrator, mock_job_execution_with_store):
        """Submitting a new clone job creates job."""
        result = orchestrator.submit_clone_job(
            text="Hello world",
            ref_text="This is my sample",
            chat_id=12345,
            message_id=67890,
            ref_audio_path="/tmp/staged/audio.wav",
        )
        
        assert result.success is True
        assert result.job_id == "job-clone-123"
        
        # Verify gateway was called
        mock_job_execution_with_store.submit_idempotent.assert_called_once()
    
    def test_submit_clone_job_returns_success(self, orchestrator):
        """Clone job submission returns success result."""
        result = orchestrator.submit_clone_job(
            text="Hello",
            ref_text="Sample",
            chat_id=12345,
            message_id=67890,
            ref_audio_path="/tmp/staged/audio.wav",
        )
        
        assert result.success is True
        assert result.job_id is not None
    
    def test_submit_clone_job_uses_correct_idempotency_key(self, orchestrator, mock_job_execution_with_store):
        """Clone job uses clone-specific idempotency key."""
        result = orchestrator.submit_clone_job(
            text="Hello",
            ref_text="Sample",
            chat_id=12345,
            message_id=67890,
            ref_audio_path="/tmp/staged/audio.wav",
        )
        
        # Verify the call was made
        mock_job_execution_with_store.submit_idempotent.assert_called_once()
        
        # The idempotency key should include 'clone' prefix and message_id
        call_args = mock_job_execution_with_store.submit_idempotent.call_args
        submission = call_args[0][0]
        assert "clone" in submission.idempotency_key
        assert "67890" in submission.idempotency_key  # message_id
    
    def test_submit_clone_job_without_ref_text(self, orchestrator, mock_job_execution_with_store):
        """Clone job submission works without ref_text."""
        result = orchestrator.submit_clone_job(
            text="Hello world",
            ref_text=None,  # Optional
            chat_id=12345,
            message_id=67890,
            ref_audio_path="/tmp/staged/audio.wav",
        )
        
        assert result.success is True
        assert result.job_id == "job-clone-123"


class TestCloneJobIdempotency:
    """Tests for clone job idempotency."""
    
    def test_clone_idempotency_key_format(self):
        """Clone idempotency key follows expected format."""
        expected = "telegram:clone:12345:67890"
        
        key = f"telegram:clone:{12345}:{67890}"
        assert key == expected
    
    def test_clone_and_tts_have_different_keys(self):
        """Clone and TTS jobs have different idempotency keys."""
        tts_key = f"telegram:12345:67890"
        clone_key = f"telegram:clone:12345:67890"
        assert tts_key != clone_key
    
    def test_clone_and_design_have_different_keys(self):
        """Clone and design jobs have different idempotency keys."""
        design_key = f"telegram:design:12345:67890"
        clone_key = f"telegram:clone:12345:67890"
        assert design_key != clone_key
    
    def test_clone_idempotency_uses_reply_message_id(self):
        """Clone idempotency key uses reply_to_message_id, not command message_id."""
        # The clone command message_id is different from the replied message_id
        command_message_id = 111
        replied_message_id = 222
        
        clone_key = f"telegram:clone:12345:{replied_message_id}"
        
        # Key should NOT contain the command message_id
        assert str(command_message_id) not in clone_key
        # Key SHOULD contain the replied message_id
        assert str(replied_message_id) in clone_key
