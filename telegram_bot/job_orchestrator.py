"""
Telegram job orchestrator for Stage 2 job integration.

This module provides:
- Job submission for TTS and Voice Design commands
- Job completion checking
- Delivery metadata management
- Background polling and result delivery

Features:
- Idempotency via idempotency_key to prevent duplicate submissions
- Async UX with acknowledgment and result delivery
- Structured logging with operation tracking
- Job integration with core job model
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.contracts.jobs import (
    JobSnapshot,
    JobStatus,
    JobOperation,
    create_job_submission,
)
from core.contracts.jobs import JobOperation as JobOp
from core.contracts.commands import CustomVoiceCommand, VoiceDesignCommand, VoiceCloneCommand
from core.observability import log_event

if TYPE_CHECKING:
    from telegram_bot.config import TelegramSettings

TELEGRAM_IDEMPOTENCY_SCOPE = "telegram"


@dataclass
class JobSubmissionResult:
    """Result of job submission."""
    
    success: bool
    job_id: str | None = None
    is_duplicate: bool = False
    error_message: str | None = None


@dataclass
class JobCompletionResult:
    """Result of job completion check."""
    
    status: JobStatus
    is_terminal: bool
    success: bool | None  # None if not terminal yet
    audio_bytes: bytes | None = None
    duration_ms: float | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass
class JobSuccessSnapshot:
    """Snapshot of successful job result."""
    
    job_id: str
    status: str


class DeliveryMetadataStore:
    """
    Async store for delivery metadata with atomic writes.
    
    This store tracks which jobs have been delivered to prevent
    duplicate deliveries after restarts.
    """
    
    def __init__(self, storage_path: Path | str):
        """Initialize store with path to storage file."""
        self._storage_path = Path(storage_path)
        self._lock = asyncio.Lock()
        self._cache: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self._loaded = False
    
    async def _load_unlocked(self) -> None:
        """Load metadata from storage file while caller holds the store lock."""
        if self._loaded:
            return
        
        if not self._storage_path.exists():
            self._cache = {}
            self._loaded = True
            return
        
        try:
            with open(self._storage_path, "r") as f:
                self._cache = json.load(f)
        except (json.JSONDecodeError, IOError):
            self._cache = {}
        
        self._loaded = True
    
    async def _save_unlocked(self) -> None:
        """Save metadata to storage file atomically while caller holds the store lock."""
        if not self._dirty:
            return
        
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._storage_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(self._cache, f)
        
        temp_path.replace(self._storage_path)
        self._dirty = False
    
    def _key(self, chat_id: int, message_id: int) -> str:
        """Generate storage key."""
        return f"{chat_id}:{message_id}"
    

    
    async def is_delivered(self, chat_id: int, message_id: int) -> bool:
        """Check if message has been fully delivered."""
        async with self._lock:
            await self._load_unlocked()
            key = self._key(chat_id, message_id)
            metadata = self._cache.get(key)
            if metadata is None:
                return False
            return "delivered_at" in metadata
    
    async def get_pending_deliveries(self) -> list[dict[str, Any]]:
        """Get all pending deliveries."""
        async with self._lock:
            await self._load_unlocked()
            return [
                metadata.copy()
                for metadata in self._cache.values()
                if "delivered_at" not in metadata
            ]
    
    async def get_delivery_metadata(
        self,
        chat_id: int,
        message_id: int,
    ) -> dict[str, Any] | None:
        """Get delivery metadata for a message."""
        async with self._lock:
            await self._load_unlocked()
            key = self._key(chat_id, message_id)
            metadata = self._cache.get(key)
            return metadata.copy() if metadata is not None else None
    
    async def create(
        self,
        chat_id: int,
        message_id: int,
        job_id: str,
    ) -> dict[str, Any]:
        """
        Create a new delivery metadata entry for a pending job.
        
        Args:
            chat_id: Telegram chat ID
            message_id: Telegram message ID
            job_id: Core job ID
            
        Returns:
            Created metadata dict
        """
        async with self._lock:
            await self._load_unlocked()
            key = self._key(chat_id, message_id)
            now = datetime.now(timezone.utc).isoformat()
            
            metadata = {
                "chat_id": chat_id,
                "message_id": message_id,
                "job_id": job_id,
                "idempotency_key": f"telegram:{chat_id}:{message_id}",
                "status": "pending",
                "created_at": now,
            }
            
            self._cache[key] = metadata
            self._dirty = True
            await self._save_unlocked()
            
            return metadata.copy()
    
    async def get(
        self,
        chat_id: int,
        message_id: int,
    ) -> dict[str, Any] | None:
        """Get delivery metadata for a message (alias for get_delivery_metadata)."""
        return await self.get_delivery_metadata(chat_id, message_id)
    
    async def mark_delivered(
        self,
        chat_id: int,
        message_id: int,
        success: bool,
        error_message: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Mark a message as delivered."""
        async with self._lock:
            await self._load_unlocked()
            key = self._key(chat_id, message_id)
            now = datetime.now(timezone.utc).isoformat()
            
            # Start with existing metadata if present, or create new
            if key in self._cache:
                metadata = self._cache[key].copy()
            else:
                metadata = {
                    "chat_id": chat_id,
                    "message_id": message_id,
                }
            
            # Update metadata
            metadata["success"] = success
            metadata["error_message"] = error_message
            metadata["delivered_at"] = now
            metadata["status"] = "delivered" if success else "failed"
            
            # Update job_id if provided
            if job_id is not None:
                metadata["job_id"] = job_id
            
            self._cache[key] = metadata
            self._dirty = True
            await self._save_unlocked()
            
            return metadata.copy()


class TelegramJobOrchestrator:
    """
    Orchestrates TTS and Voice Design job submission and completion.
    
    This class provides:
    - Synchronous job submission with idempotency
    - Job completion checking
    - Integration with core job execution gateway
    """
    
    def __init__(
        self,
        job_execution: Any,
        delivery_store: DeliveryMetadataStore,
        settings: TelegramSettings,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize orchestrator."""
        self._job_execution = job_execution
        self._delivery_store = delivery_store
        self._settings = settings
        self._logger = logger or logging.getLogger(__name__)

    @contextmanager
    def _get_store(self):
        """Get the core job store if available."""
        if hasattr(self._job_execution, '_store') and self._job_execution._store is not None:
            yield self._job_execution._store
        else:
            yield None
    
    def submit_tts_job(
        self,
        text: str,
        speaker: str,
        speed: float,
        chat_id: int,
        message_id: int,
    ) -> JobSubmissionResult:
        """
        Submit TTS job for a Telegram message.
        
        Uses idempotency to prevent duplicate submissions when the same
        Telegram message is processed multiple times.
        
        Args:
            text: Text to synthesize
            speaker: Speaker name
            speed: Speed multiplier (0.5-2.0)
            chat_id: Telegram chat ID
            message_id: Telegram message ID
            
        Returns:
            JobSubmissionResult indicating success/duplicate/error
        """
        idempotency_key = f"telegram:{chat_id}:{message_id}"
        
        # Check if job already exists via idempotency in core store
        with self._get_store() as store:
            if store is not None:
                existing_by_idem = store.get_by_idempotency_key(
                    idempotency_key,
                    scope=TELEGRAM_IDEMPOTENCY_SCOPE,
                )
                if existing_by_idem is not None:
                    log_event(
                        self._logger,
                        level=logging.INFO,
                        event="telegram.job.idempotent_reuse",
                        message="Reusing existing job by idempotency key",
                        chat_id=chat_id,
                        message_id=message_id,
                        job_id=existing_by_idem.snapshot.job_id,
                        job_status=existing_by_idem.snapshot.status.value,
                    )
                    
                    return JobSubmissionResult(
                        success=True,
                        job_id=existing_by_idem.snapshot.job_id,
                        is_duplicate=True,
                    )
        
        # Create new job submission
        try:
            submission = create_job_submission(
                operation=JobOperation.SYNTHESIZE_CUSTOM,
                command=CustomVoiceCommand(
                    text=text,
                    speaker=speaker,
                    speed=speed,
                    save_output=False,
                ),
                submit_request_id=idempotency_key,
                owner_principal_id=str(chat_id),
                response_format=None,
                save_output=False,
                execution_timeout_seconds=self._settings.request_timeout_seconds,
                idempotency_key=idempotency_key,
                idempotency_scope=TELEGRAM_IDEMPOTENCY_SCOPE,
                idempotency_fingerprint=None,
            )
            
            # Submit job through gateway
            resolution = self._job_execution.submit_idempotent(submission)
            
            log_event(
                self._logger,
                level=logging.INFO,
                event="telegram.job.submitted",
                message="TTS job submitted",
                chat_id=chat_id,
                message_id=message_id,
                job_id=resolution.snapshot.job_id,
                speaker=speaker,
                speed=speed,
                text_length=len(text),
                created=resolution.created,
            )
            
            return JobSubmissionResult(
                success=True,
                job_id=resolution.snapshot.job_id,
                is_duplicate=not resolution.created,
            )
            
        except Exception as exc:
            log_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.job.submit_failed",
                message=f"Job submission failed: {exc}",
                chat_id=chat_id,
                message_id=message_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            
            return JobSubmissionResult(
                success=False,
                job_id=None,
                is_duplicate=False,
                error_message=str(exc),
            )
    
    def submit_design_job(
        self,
        voice_description: str,
        text: str,
        chat_id: int,
        message_id: int,
    ) -> JobSubmissionResult:
        """
        Submit Voice Design job for a Telegram message.
        
        Uses idempotency to prevent duplicate submissions when the same
        Telegram message is processed multiple times.
        
        Args:
            voice_description: Description of the voice to create
            text: Text to synthesize
            chat_id: Telegram chat ID
            message_id: Telegram message ID
            
        Returns:
            JobSubmissionResult indicating success/duplicate/error
        """
        # Design jobs use "design" prefix for idempotency to separate from TTS jobs
        idempotency_key = f"telegram:design:{chat_id}:{message_id}"
        
        # Check if job already exists via idempotency in core store
        with self._get_store() as store:
            if store is not None:
                existing_by_idem = store.get_by_idempotency_key(
                    idempotency_key,
                    scope=TELEGRAM_IDEMPOTENCY_SCOPE,
                )
                if existing_by_idem is not None:
                    log_event(
                        self._logger,
                        level=logging.INFO,
                        event="telegram.job.design.idempotent_reuse",
                        message="Reusing existing design job by idempotency key",
                        chat_id=chat_id,
                        message_id=message_id,
                        job_id=existing_by_idem.snapshot.job_id,
                        job_status=existing_by_idem.snapshot.status.value,
                    )
                    
                    return JobSubmissionResult(
                        success=True,
                        job_id=existing_by_idem.snapshot.job_id,
                        is_duplicate=True,
                    )
        
        # Create new job submission
        try:
            submission = create_job_submission(
                operation=JobOperation.SYNTHESIZE_DESIGN,
                command=VoiceDesignCommand(
                    text=text,
                    voice_description=voice_description,
                    save_output=False,
                ),
                submit_request_id=idempotency_key,
                owner_principal_id=str(chat_id),
                response_format=None,
                save_output=False,
                execution_timeout_seconds=self._settings.request_timeout_seconds,
                idempotency_key=idempotency_key,
                idempotency_scope=TELEGRAM_IDEMPOTENCY_SCOPE,
                idempotency_fingerprint=None,
            )
            
            # Submit job through gateway
            resolution = self._job_execution.submit_idempotent(submission)
            
            log_event(
                self._logger,
                level=logging.INFO,
                event="telegram.job.design.submitted",
                message="Voice Design job submitted",
                chat_id=chat_id,
                message_id=message_id,
                job_id=resolution.snapshot.job_id,
                voice_description_length=len(voice_description),
                text_length=len(text),
                created=resolution.created,
            )
            
            return JobSubmissionResult(
                success=True,
                job_id=resolution.snapshot.job_id,
                is_duplicate=not resolution.created,
            )
            
        except Exception as exc:
            log_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.job.design.submit_failed",
                message=f"Voice Design job submission failed: {exc}",
                chat_id=chat_id,
                message_id=message_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            
            return JobSubmissionResult(
                success=False,
                job_id=None,
                is_duplicate=False,
                error_message=str(exc),
            )

    def submit_clone_job(
        self,
        text: str,
        ref_text: str | None,
        chat_id: int,
        message_id: int,
        ref_audio_path: str | None = None,
    ) -> JobSubmissionResult:
        """
        Submit Voice Clone job for a Telegram message.
        
        Uses idempotency to prevent duplicate submissions when the same
        Telegram message is processed multiple times.
        
        Args:
            text: Text to synthesize
            ref_text: Optional reference text transcript
            chat_id: Telegram chat ID
            message_id: Telegram message ID
            ref_audio_path: Optional path to staged reference audio file
            
        Returns:
            JobSubmissionResult indicating success/duplicate/error
        """
        # Clone jobs use "clone" prefix for idempotency to separate from TTS/design jobs
        idempotency_key = f"telegram:clone:{chat_id}:{message_id}"
        
        # Check if job already exists via idempotency in core store
        with self._get_store() as store:
            if store is not None:
                existing_by_idem = store.get_by_idempotency_key(
                    idempotency_key,
                    scope=TELEGRAM_IDEMPOTENCY_SCOPE,
                )
                if existing_by_idem is not None:
                    log_event(
                        self._logger,
                        level=logging.INFO,
                        event="telegram.job.clone.idempotent_reuse",
                        message="Reusing existing clone job by idempotency key",
                        chat_id=chat_id,
                        message_id=message_id,
                        job_id=existing_by_idem.snapshot.job_id,
                        job_status=existing_by_idem.snapshot.status.value,
                    )
                    
                    return JobSubmissionResult(
                        success=True,
                        job_id=existing_by_idem.snapshot.job_id,
                        is_duplicate=True,
                    )
        
        # Create new job submission
        try:
            # Import here to avoid circular import issues
            from pathlib import Path as PathLib
            
            submission = create_job_submission(
                operation=JobOperation.SYNTHESIZE_CLONE,
                command=VoiceCloneCommand(
                    text=text,
                    ref_audio_path=PathLib(ref_audio_path) if ref_audio_path else None,
                    ref_text=ref_text,
                    save_output=False,
                ),
                submit_request_id=idempotency_key,
                owner_principal_id=str(chat_id),
                response_format=None,
                save_output=False,
                execution_timeout_seconds=self._settings.request_timeout_seconds,
                idempotency_key=idempotency_key,
                idempotency_scope=TELEGRAM_IDEMPOTENCY_SCOPE,
                idempotency_fingerprint=None,
            )
            
            # Submit job through gateway
            resolution = self._job_execution.submit_idempotent(submission)
            
            log_event(
                self._logger,
                level=logging.INFO,
                event="telegram.job.clone.submitted",
                message="Voice Clone job submitted",
                chat_id=chat_id,
                message_id=message_id,
                job_id=resolution.snapshot.job_id,
                ref_text_provided=ref_text is not None,
                text_length=len(text),
                created=resolution.created,
            )
            
            return JobSubmissionResult(
                success=True,
                job_id=resolution.snapshot.job_id,
                is_duplicate=not resolution.created,
            )
            
        except Exception as exc:
            log_event(
                self._logger,
                level=logging.ERROR,
                event="telegram.job.clone.submit_failed",
                message=f"Voice Clone job submission failed: {exc}",
                chat_id=chat_id,
                message_id=message_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            
            return JobSubmissionResult(
                success=False,
                job_id=None,
                is_duplicate=False,
                error_message=str(exc),
            )

    def check_job_completion(self, job_id: str) -> JobCompletionResult:
        """
        Check if job has completed and get result.
        
        Args:
            job_id: Core job ID
            
        Returns:
            JobCompletionResult with terminal state if completed
        """
        snapshot = self._job_execution.get_job(job_id)
        if snapshot is None:
            return JobCompletionResult(
                status=JobStatus.QUEUED,
                is_terminal=False,
                success=None,
            )
        
        is_terminal = snapshot.status.is_terminal
        
        if snapshot.status == JobStatus.SUCCEEDED:
            result = self._job_execution.get_result(job_id)
            if result is not None and result.success is not None:
                return JobCompletionResult(
                    status=snapshot.status,
                    is_terminal=is_terminal,
                    success=True,
                    audio_bytes=result.success.generation.audio.bytes_data if result.success.generation else None,
                    duration_ms=self._calculate_duration_ms(snapshot),
                )
            return JobCompletionResult(
                status=snapshot.status,
                is_terminal=is_terminal,
                success=True,
            )
        
        if snapshot.status in {JobStatus.FAILED, JobStatus.TIMEOUT, JobStatus.CANCELLED}:
            error_message = None
            error_code = None
            if snapshot.terminal_error:
                error_code = snapshot.terminal_error.code
                error_message = snapshot.terminal_error.message
            
            return JobCompletionResult(
                status=snapshot.status,
                is_terminal=is_terminal,
                success=False,
                error_message=error_message,
                error_code=error_code,
                duration_ms=self._calculate_duration_ms(snapshot),
            )
        
        # QUEUED or RUNNING
        return JobCompletionResult(
            status=snapshot.status,
            is_terminal=is_terminal,
            success=None,
        )
    
    @staticmethod
    def _calculate_duration_ms(snapshot: JobSnapshot) -> float | None:
        """Calculate job duration in milliseconds."""
        if snapshot.started_at and snapshot.completed_at:
            delta = snapshot.completed_at - snapshot.started_at
            return delta.total_seconds() * 1000
        return None


@dataclass
class TelegramJobPoller:
    """
    Polls job completion and handles delivery.
    
    This class runs in the background and checks for completed jobs,
    delivering results to users via the Telegram sender.
    """
    
    orchestrator: TelegramJobOrchestrator
    sender: TelegramSender
    delivery_store: DeliveryMetadataStore
    settings: TelegramSettings
    poll_interval_seconds: float = 1.0
    _running: bool = field(default=False, init=False)
    _task: asyncio.Task | None = field(default=None, init=False)
    
    async def start(self) -> None:
        """Start the job poller and keep running until stopped or cancelled."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.current_task()
        try:
            await self._poll_loop()
        finally:
            self._running = False
            if self._task is asyncio.current_task():
                self._task = None
    
    async def stop(self) -> None:
        """Stop the job poller."""
        self._running = False
        task = self._task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self._task is task:
            self._task = None
    
    async def _poll_loop(self) -> None:
        """Main polling loop for job completion."""
        # First, recover pending jobs from previous run
        await self._recover_pending_jobs()
        
        while self._running:
            try:
                await self._check_pending_deliveries()
                await asyncio.sleep(self.poll_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logging.getLogger(__name__).error(f"Error in job poller: {exc}")
                await asyncio.sleep(5.0)  # Back off on error
    
    async def _recover_pending_jobs(self) -> None:
        """Recover and deliver results from previous session."""
        from telegram_bot.observability import METRICS
        
        pending = await self.delivery_store.get_pending_deliveries()
        
        if not pending:
            return
        
        logging.getLogger(__name__).info(
            f"Recovering {len(pending)} pending jobs from previous session"
        )
        
        for metadata in pending:
            job_id = metadata.get("job_id")
            chat_id = metadata.get("chat_id")
            message_id = metadata.get("message_id")
            
            if not all([job_id, chat_id, message_id]):
                continue
            
            result = self.orchestrator.check_job_completion(job_id)
            
            if result.is_terminal:
                await self._deliver_job_result(
                    job_id,
                    result,
                    chat_id,
                    message_id,
                )
                METRICS.job_delivery_recovered()
    
    async def _check_pending_deliveries(self) -> None:
        """Check pending deliveries and deliver completed jobs."""
        from telegram_bot.observability import METRICS
        
        pending = await self.delivery_store.get_pending_deliveries()
        
        for metadata in pending:
            job_id = metadata.get("job_id")
            chat_id = metadata.get("chat_id")
            message_id = metadata.get("message_id")
            
            if not all([job_id, chat_id, message_id]):
                continue
            
            result = self.orchestrator.check_job_completion(job_id)
            
            if result.is_terminal:
                await self._deliver_job_result(job_id, result, chat_id, message_id)
    
    async def _deliver_job_result(
        self,
        job_id: str,
        result: JobCompletionResult,
        chat_id: int,
        message_id: int,
    ) -> None:
        """Deliver job result to user."""
        from telegram_bot.observability import METRICS
        
        try:
            if result.success:
                # Send voice message
                if result.audio_bytes:
                    caption = self._build_success_caption(result.duration_ms)
                    delivery_result = await self.sender.send_voice(
                        chat_id,
                        result.audio_bytes,
                        caption=caption,
                    )
                    
                    if delivery_result.success:
                        await self.delivery_store.mark_delivered(
                            chat_id,
                            message_id,
                            True,
                        )
                        METRICS.voice_sent()
                        METRICS.job_delivery_completed()
                    else:
                        await self.delivery_store.mark_delivered(
                            chat_id,
                            message_id,
                            False,
                            delivery_result.error_message,
                        )
                        METRICS.voice_send_failed()
                else:
                    # No audio bytes, mark as delivered with warning
                    await self.delivery_store.mark_delivered(
                        chat_id,
                        message_id,
                        False,
                        "No audio data in job result",
                    )
                    METRICS.job_delivery_completed()
            else:
                # Send error message
                error_text = self._build_error_text(result.error_message)
                await self.sender.send_text(chat_id, error_text)
                await self.delivery_store.mark_delivered(
                    chat_id,
                    message_id,
                    True,  # Mark as delivered even on failure
                    result.error_message,
                )
                METRICS.job_delivery_completed()
                
        except Exception as exc:
            LOGGER.error(
                f"Failed to deliver job result: {exc}",
                extra={
                    "job_id": job_id,
                    "chat_id": chat_id,
                    "message_id": message_id,
                }
            )
            await self.delivery_store.mark_delivered(
                chat_id,
                message_id,
                False,
                str(exc),
            )
    
    def _build_success_caption(self, duration_ms: float | None) -> str:
        """Build success caption for voice message."""
        duration = (duration_ms / 1000) if duration_ms else 0
        return f"✅ *Готово*\n\nVoice сообщение успешно подготовлено. Длительность около *{duration:.1f} с*."
    
    def _build_error_text(self, error_message: str | None) -> str:
        """Build error text for failure message."""
        error = error_message or "Во время генерации произошла неизвестная ошибка"
        return (
            "❌ *Ошибка*\n"
            f"{error}\n\n"
            "Откройте `/help`, чтобы проверить синтаксис команды и примеры использования."
        )
