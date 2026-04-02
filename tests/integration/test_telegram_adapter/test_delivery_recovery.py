"""
Integration tests for Telegram delivery recovery.

Tests the ability to recover pending deliveries after restart
using the DeliveryMetadataStore.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from telegram_bot.job_orchestrator import (
    DeliveryMetadataStore,
    TelegramJobOrchestrator,
    JobCompletionResult,
)
from core.contracts.jobs import JobStatus


class TestDeliveryMetadataStoreRecovery:
    """Tests for DeliveryMetadataStore persistence and recovery."""
    
    @pytest.fixture
    def temp_store_path(self):
        """Create a temporary path for the store."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        yield path
        # Cleanup
        if path.exists():
            path.unlink()
    
    @pytest.fixture
    def store(self, temp_store_path):
        """Create a fresh DeliveryMetadataStore."""
        return DeliveryMetadataStore(temp_store_path)
    
    @pytest.mark.asyncio
    async def test_create_pending_delivery(self, store):
        """Test creating a pending delivery entry."""
        metadata = await store.create(
            chat_id=12345,
            message_id=67890,
            job_id="job_001",
        )
        
        assert metadata["chat_id"] == 12345
        assert metadata["message_id"] == 67890
        assert metadata["job_id"] == "job_001"
        assert metadata["status"] == "pending"
        assert "created_at" in metadata
    
    @pytest.mark.asyncio
    async def test_mark_delivered_success(self, store):
        """Test marking a delivery as successfully delivered."""
        await store.create(chat_id=123, message_id=456, job_id="job_001")
        
        metadata = await store.mark_delivered(
            chat_id=123,
            message_id=456,
            success=True,
            job_id="job_001",
        )
        
        assert metadata["success"] is True
        assert metadata["status"] == "delivered"
        assert "delivered_at" in metadata
    
    @pytest.mark.asyncio
    async def test_mark_delivered_failure(self, store):
        """Test marking a delivery as failed."""
        await store.create(chat_id=123, message_id=456, job_id="job_001")
        
        metadata = await store.mark_delivered(
            chat_id=123,
            message_id=456,
            success=False,
            error_message="Telegram API error",
        )
        
        assert metadata["success"] is False
        assert metadata["status"] == "failed"
        assert metadata["error_message"] == "Telegram API error"
    
    @pytest.mark.asyncio
    async def test_is_delivered_after_mark(self, store):
        """Test is_delivered returns True after marking."""
        await store.create(chat_id=123, message_id=456, job_id="job_001")
        
        assert await store.is_delivered(123, 456) is False
        
        await store.mark_delivered(chat_id=123, message_id=456, success=True)
        
        assert await store.is_delivered(123, 456) is True
    
    @pytest.mark.asyncio
    async def test_get_pending_deliveries(self, store):
        """Test getting all pending deliveries."""
        # Create multiple deliveries
        await store.create(chat_id=1, message_id=1, job_id="job_1")
        await store.create(chat_id=2, message_id=2, job_id="job_2")
        await store.create(chat_id=3, message_id=3, job_id="job_3")
        
        pending = await store.get_pending_deliveries()
        
        assert len(pending) == 3
    
    @pytest.mark.asyncio
    async def test_get_pending_excludes_delivered(self, store):
        """Test that get_pending excludes already delivered."""
        await store.create(chat_id=1, message_id=1, job_id="job_1")
        await store.create(chat_id=2, message_id=2, job_id="job_2")
        
        # Mark one as delivered
        await store.mark_delivered(chat_id=1, message_id=1, success=True)
        
        pending = await store.get_pending_deliveries()
        
        assert len(pending) == 1
        assert pending[0]["message_id"] == 2
    
    @pytest.mark.asyncio
    async def test_recovery_after_restart(self, temp_store_path):
        """Test that pending deliveries persist across restarts."""
        # First session: create pending deliveries
        store1 = DeliveryMetadataStore(temp_store_path)
        await store1.create(chat_id=111, message_id=222, job_id="job_recovery")
        await store1.create(chat_id=333, message_id=444, job_id="job_recovery_2")
        
        # Simulate restart
        store2 = DeliveryMetadataStore(temp_store_path)
        
        pending = await store2.get_pending_deliveries()
        
        assert len(pending) == 2
    
    @pytest.mark.asyncio
    async def test_recovery_preserves_job_id(self, temp_store_path):
        """Test that job_id is preserved in recovered deliveries."""
        store1 = DeliveryMetadataStore(temp_store_path)
        await store1.create(chat_id=999, message_id=888, job_id="unique_job_id")
        
        store2 = DeliveryMetadataStore(temp_store_path)
        metadata = await store2.get_delivery_metadata(chat_id=999, message_id=888)
        
        assert metadata["job_id"] == "unique_job_id"
    
    @pytest.mark.asyncio
    async def test_get_delivery_metadata(self, store):
        """Test getting full delivery metadata."""
        await store.create(chat_id=100, message_id=200, job_id="job_check")
        
        metadata = await store.get_delivery_metadata(chat_id=100, message_id=200)
        
        assert metadata is not None
        assert metadata["job_id"] == "job_check"
        assert metadata["status"] == "pending"
    
    @pytest.mark.asyncio
    async def test_get_delivery_metadata_not_found(self, store):
        """Test getting metadata for non-existent delivery."""
        metadata = await store.get_delivery_metadata(chat_id=999, message_id=999)
        
        assert metadata is None
    
    @pytest.mark.asyncio
    async def test_idempotent_mark_delivered(self, store):
        """Test that marking delivered is idempotent."""
        await store.create(chat_id=555, message_id=666, job_id="job_idem")
        
        # Mark twice
        await store.mark_delivered(chat_id=555, message_id=666, success=True)
        await store.mark_delivered(chat_id=555, message_id=666, success=False, error_message="Updated")
        
        # Second mark should update, not fail
        metadata = await store.get_delivery_metadata(chat_id=555, message_id=666)
        assert metadata["status"] == "failed"


class TestTelegramJobPollerRecovery:
    """Tests for TelegramJobPoller recovery behavior."""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orchestrator = MagicMock()
        orchestrator.check_job_completion = MagicMock()
        return orchestrator
    
    @pytest.fixture
    def mock_sender(self):
        """Create mock sender."""
        sender = MagicMock()
        sender.send_voice = AsyncMock()
        sender.send_text = AsyncMock()
        return sender
    
    @pytest.mark.asyncio
    async def test_recovery_loads_pending_jobs(self, mock_orchestrator, mock_sender):
        """Test that recovery loads pending jobs from store."""
        from telegram_bot.job_orchestrator import TelegramJobPoller, DeliveryMetadataStore
        from telegram_bot.config import TelegramSettings
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        
        store = DeliveryMetadataStore(path)
        await store.create(chat_id=123, message_id=456, job_id="pending_job")
        
        # Create completed job result
        mock_orchestrator.check_job_completion.return_value = JobCompletionResult(
            status=JobStatus.SUCCEEDED,
            is_terminal=True,
            success=True,
            audio_bytes=b"fake_audio",
            duration_ms=1000.0,
        )
        
        mock_sender.send_voice.return_value = MagicMock(success=True)
        
        settings = MagicMock(spec=TelegramSettings)
        settings.telegram_dev_mode = True
        
        poller = TelegramJobPoller(
            orchestrator=mock_orchestrator,
            sender=mock_sender,
            delivery_store=store,
            settings=settings,
        )
        
        # Trigger recovery
        await poller._recover_pending_jobs()
        
        # Should have checked the job
        mock_orchestrator.check_job_completion.assert_called_once_with("pending_job")
        
        # Cleanup
        path.unlink()
    
    @pytest.mark.asyncio
    async def test_recovery_skips_already_delivered(self, mock_orchestrator, mock_sender):
        """Test that recovery skips already delivered jobs."""
        from telegram_bot.job_orchestrator import TelegramJobPoller, DeliveryMetadataStore
        from telegram_bot.config import TelegramSettings
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        
        store = DeliveryMetadataStore(path)
        
        # Create and mark as delivered
        await store.create(chat_id=999, message_id=888, job_id="already_done")
        await store.mark_delivered(chat_id=999, message_id=888, success=True)
        
        settings = MagicMock(spec=TelegramSettings)
        settings.telegram_dev_mode = True
        
        poller = TelegramJobPoller(
            orchestrator=mock_orchestrator,
            sender=mock_sender,
            delivery_store=store,
            settings=settings,
        )
        
        await poller._recover_pending_jobs()
        
        # Should NOT have checked the job (already delivered)
        mock_orchestrator.check_job_completion.assert_not_called()
        
        # Cleanup
        path.unlink()
