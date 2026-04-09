# FILE: telegram_bot/job_orchestrator.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Manage async job lifecycle for Telegram: submit, poll, deliver results.
#   SCOPE: Job submission, polling, delivery coordination
#   DEPENDS: M-APPLICATION, M-CONTRACTS
#   LINKS: M-TELEGRAM
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LOGGER - Module logger for Telegram job orchestration
#   TelegramSenderProtocol - Protocol for Telegram delivery used by orchestrator flows
#   TELEGRAM_IDEMPOTENCY_SCOPE - Idempotency scope key for Telegram submissions
#   JobSubmissionResult - Outcome payload for Telegram job submission attempts
#   JobCompletionResult - Completion status payload for Telegram job polling
#   JobSuccessSnapshot - Minimal successful job snapshot for Telegram delivery
#   DeliveryMetadataStore - Persistent store for Telegram delivery metadata
#   TelegramJobOrchestrator - Telegram service for job submission and completion checks
#   TelegramJobPoller - Background poller for Telegram job result delivery
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, function contracts, semantic blocks, and migrated log events to block-reference format]
# END_CHANGE_SUMMARY

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
from typing import TYPE_CHECKING, Any, Protocol, cast

from core.contracts.jobs import (
    JobSnapshot,
    JobStatus,
    JobOperation,
    create_job_submission,
)
from core.contracts.jobs import JobOperation as JobOp
from core.contracts.commands import (
    CustomVoiceCommand,
    VoiceDesignCommand,
    VoiceCloneCommand,
)
from core.observability import log_event


LOGGER = logging.getLogger(__name__)


# START_CONTRACT: TelegramSenderProtocol
#   PURPOSE: Define the Telegram delivery capabilities required by job orchestration and polling.
#   INPUTS: {}
#   OUTPUTS: { TelegramSenderProtocol - protocol for text and voice delivery }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: TelegramSenderProtocol
class TelegramSenderProtocol(Protocol):
    # START_CONTRACT: send_text
    #   PURPOSE: Send a Telegram text message through the job delivery interface.
    #   INPUTS: { chat_id: int - target Telegram chat identifier, text: str - message body }
    #   OUTPUTS: { Any - transport-specific send result }
    #   SIDE_EFFECTS: Sends a Telegram API message in concrete implementations.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: send_text
    async def send_text(self, chat_id: int, text: str) -> Any: ...

    # START_CONTRACT: send_voice
    #   PURPOSE: Send a Telegram voice message through the job delivery interface.
    #   INPUTS: { chat_id: int - target Telegram chat identifier, audio_bytes: bytes - voice payload bytes, caption: str | None - optional caption }
    #   OUTPUTS: { Any - transport-specific send result }
    #   SIDE_EFFECTS: Sends a Telegram voice message in concrete implementations.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: send_voice
    async def send_voice(
        self, chat_id: int, audio_bytes: bytes, caption: str | None = None
    ) -> Any: ...


if TYPE_CHECKING:
    from telegram_bot.config import TelegramSettings

TELEGRAM_IDEMPOTENCY_SCOPE = "telegram"


# START_CONTRACT: JobSubmissionResult
#   PURPOSE: Describe the outcome of submitting a Telegram-backed synthesis job.
#   INPUTS: { success: bool - submission result flag, job_id: str | None - core job identifier, is_duplicate: bool - idempotency reuse flag, error_message: str | None - failure detail }
#   OUTPUTS: { JobSubmissionResult - immutable submission outcome }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: JobSubmissionResult
@dataclass
class JobSubmissionResult:
    """Result of job submission."""

    success: bool
    job_id: str | None = None
    is_duplicate: bool = False
    error_message: str | None = None


# START_CONTRACT: JobCompletionResult
#   PURPOSE: Describe the completion status and payload of a Telegram-linked synthesis job.
#   INPUTS: { status: JobStatus - current job status, is_terminal: bool - terminal state flag, success: bool | None - success state when terminal, audio_bytes: bytes | None - generated audio payload, duration_ms: float | None - execution time, error_message: str | None - failure detail, error_code: str | None - failure code }
#   OUTPUTS: { JobCompletionResult - immutable completion status payload }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: JobCompletionResult
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


# START_CONTRACT: JobSuccessSnapshot
#   PURPOSE: Capture a minimal successful job snapshot for Telegram delivery flows.
#   INPUTS: { job_id: str - core job identifier, status: str - serialized job status }
#   OUTPUTS: { JobSuccessSnapshot - immutable successful job summary }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM
# END_CONTRACT: JobSuccessSnapshot
@dataclass
class JobSuccessSnapshot:
    """Snapshot of successful job result."""

    job_id: str
    status: str


# START_CONTRACT: DeliveryMetadataStore
#   PURPOSE: Persist Telegram delivery metadata so completed jobs can be delivered exactly once.
#   INPUTS: { storage_path: Path | str - metadata storage file path }
#   OUTPUTS: { DeliveryMetadataStore - asynchronous metadata store }
#   SIDE_EFFECTS: Reads and writes delivery metadata on disk.
#   LINKS: M-TELEGRAM
# END_CONTRACT: DeliveryMetadataStore
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

    # START_CONTRACT: is_delivered
    #   PURPOSE: Check whether a Telegram message has already received its final delivery.
    #   INPUTS: { chat_id: int - Telegram chat identifier, message_id: int - Telegram message identifier }
    #   OUTPUTS: { bool - True when delivery metadata marks the message as delivered }
    #   SIDE_EFFECTS: Reads persisted delivery metadata from disk when needed.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: is_delivered
    async def is_delivered(self, chat_id: int, message_id: int) -> bool:
        """Check if message has been fully delivered."""
        async with self._lock:
            await self._load_unlocked()
            key = self._key(chat_id, message_id)
            metadata = self._cache.get(key)
            if metadata is None:
                return False
            return "delivered_at" in metadata

    # START_CONTRACT: get_pending_deliveries
    #   PURPOSE: List all Telegram deliveries that are still awaiting final result delivery.
    #   INPUTS: {}
    #   OUTPUTS: { list[dict[str, Any]] - pending delivery metadata records }
    #   SIDE_EFFECTS: Reads persisted delivery metadata from disk when needed.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: get_pending_deliveries
    async def get_pending_deliveries(self) -> list[dict[str, Any]]:
        """Get all pending deliveries."""
        async with self._lock:
            await self._load_unlocked()
            return [
                metadata.copy()
                for metadata in self._cache.values()
                if "delivered_at" not in metadata
            ]

    # START_CONTRACT: get_delivery_metadata
    #   PURPOSE: Retrieve stored Telegram delivery metadata for a specific message.
    #   INPUTS: { chat_id: int - Telegram chat identifier, message_id: int - Telegram message identifier }
    #   OUTPUTS: { dict[str, Any] | None - delivery metadata record when present }
    #   SIDE_EFFECTS: Reads persisted delivery metadata from disk when needed.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: get_delivery_metadata
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

    # START_CONTRACT: create
    #   PURPOSE: Create a pending delivery metadata record for a newly queued Telegram job.
    #   INPUTS: { chat_id: int - Telegram chat identifier, message_id: int - Telegram message identifier, job_id: str - core job identifier }
    #   OUTPUTS: { dict[str, Any] - created delivery metadata record }
    #   SIDE_EFFECTS: Writes delivery metadata to disk.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: create
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

    # START_CONTRACT: get
    #   PURPOSE: Retrieve stored Telegram delivery metadata using a convenience alias.
    #   INPUTS: { chat_id: int - Telegram chat identifier, message_id: int - Telegram message identifier }
    #   OUTPUTS: { dict[str, Any] | None - delivery metadata record when present }
    #   SIDE_EFFECTS: Reads persisted delivery metadata from disk when needed.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: get
    async def get(
        self,
        chat_id: int,
        message_id: int,
    ) -> dict[str, Any] | None:
        """Get delivery metadata for a message (alias for get_delivery_metadata)."""
        return await self.get_delivery_metadata(chat_id, message_id)

    # START_CONTRACT: mark_delivered
    #   PURPOSE: Mark Telegram job delivery as completed or failed and persist the outcome.
    #   INPUTS: { chat_id: int - Telegram chat identifier, message_id: int - Telegram message identifier, success: bool - delivery success flag, error_message: str | None - optional failure detail, job_id: str | None - optional job identifier }
    #   OUTPUTS: { dict[str, Any] - updated delivery metadata record }
    #   SIDE_EFFECTS: Writes delivery metadata to disk.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: mark_delivered
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
            metadata: dict[str, Any]
            if key in self._cache:
                metadata = cast(dict[str, Any], self._cache[key].copy())
            else:
                metadata = {
                    "chat_id": chat_id,
                    "message_id": message_id,
                }

            # Update metadata
            metadata_any = cast(dict[str, Any], metadata)
            metadata_any["success"] = success
            metadata_any["error_message"] = error_message
            metadata_any["delivered_at"] = now
            metadata_any["status"] = "delivered" if success else "failed"

            # Update job_id if provided
            if job_id is not None:
                metadata_any["job_id"] = job_id

            self._cache[key] = metadata
            self._dirty = True
            await self._save_unlocked()

            return metadata.copy()


# START_CONTRACT: TelegramJobOrchestrator
#   PURPOSE: Submit Telegram synthesis jobs to the core execution layer and inspect their completion state.
#   INPUTS: { job_execution: Any - core job execution gateway, delivery_store: DeliveryMetadataStore - delivery metadata store, settings: TelegramSettings - Telegram runtime settings, logger: logging.Logger | None - optional logger }
#   OUTPUTS: { TelegramJobOrchestrator - Telegram job orchestration service }
#   SIDE_EFFECTS: Submits core jobs and emits orchestration logs.
#   LINKS: M-TELEGRAM
# END_CONTRACT: TelegramJobOrchestrator
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
        if (
            hasattr(self._job_execution, "_store")
            and self._job_execution._store is not None
        ):
            yield self._job_execution._store
        else:
            yield None

    # START_CONTRACT: submit_tts_job
    #   PURPOSE: Submit a custom-voice Telegram synthesis request as an idempotent core job.
    #   INPUTS: { text: str - synthesis text, speaker: str - speaker name, speed: float - speed multiplier, chat_id: int - Telegram chat identifier, message_id: int - Telegram message identifier, language: str - requested language code }
    #   OUTPUTS: { JobSubmissionResult - submission outcome for the TTS job }
    #   SIDE_EFFECTS: Submits a core job and emits submission logs.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: submit_tts_job
    def submit_tts_job(
        self,
        text: str,
        speaker: str,
        speed: float,
        chat_id: int,
        message_id: int,
        language: str = "auto",
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
        # START_BLOCK_SUBMIT_JOB
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
                        event="[JobOrchestrator][submit_tts_job][BLOCK_SUBMIT_JOB]",
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
        # END_BLOCK_SUBMIT_JOB

        # START_BLOCK_CREATE_TTS_SUBMISSION
        # Create new job submission
        try:
            submission = create_job_submission(
                operation=JobOperation.SYNTHESIZE_CUSTOM,
                command=CustomVoiceCommand(
                    text=text,
                    speaker=speaker,
                    speed=speed,
                    language=language,
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
                event="[JobOrchestrator][submit_tts_job][BLOCK_CREATE_TTS_SUBMISSION]",
                message="TTS job submitted",
                chat_id=chat_id,
                message_id=message_id,
                job_id=resolution.snapshot.job_id,
                speaker=speaker,
                speed=speed,
                language=language,
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
                event="[JobOrchestrator][submit_tts_job][BLOCK_CREATE_TTS_SUBMISSION]",
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
        # END_BLOCK_CREATE_TTS_SUBMISSION

    # START_CONTRACT: submit_design_job
    #   PURPOSE: Submit a voice-design Telegram synthesis request as an idempotent core job.
    #   INPUTS: { voice_description: str - voice description prompt, text: str - synthesis text, chat_id: int - Telegram chat identifier, message_id: int - Telegram message identifier, language: str - requested language code }
    #   OUTPUTS: { JobSubmissionResult - submission outcome for the design job }
    #   SIDE_EFFECTS: Submits a core job and emits submission logs.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: submit_design_job
    def submit_design_job(
        self,
        voice_description: str,
        text: str,
        chat_id: int,
        message_id: int,
        language: str = "auto",
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
        # START_BLOCK_SUBMIT_DESIGN_JOB
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
                        event="[JobOrchestrator][submit_design_job][BLOCK_SUBMIT_DESIGN_JOB]",
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
        # END_BLOCK_SUBMIT_DESIGN_JOB

        # START_BLOCK_CREATE_DESIGN_SUBMISSION
        # Create new job submission
        try:
            submission = create_job_submission(
                operation=JobOperation.SYNTHESIZE_DESIGN,
                command=VoiceDesignCommand(
                    text=text,
                    voice_description=voice_description,
                    language=language,
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
                event="[JobOrchestrator][submit_design_job][BLOCK_CREATE_DESIGN_SUBMISSION]",
                message="Voice Design job submitted",
                chat_id=chat_id,
                message_id=message_id,
                job_id=resolution.snapshot.job_id,
                voice_description_length=len(voice_description),
                text_length=len(text),
                language=language,
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
                event="[JobOrchestrator][submit_design_job][BLOCK_CREATE_DESIGN_SUBMISSION]",
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
        # END_BLOCK_CREATE_DESIGN_SUBMISSION

    # START_CONTRACT: submit_clone_job
    #   PURPOSE: Submit a voice-clone Telegram synthesis request as an idempotent core job.
    #   INPUTS: { text: str - synthesis text, ref_text: str | None - optional reference transcript, chat_id: int - Telegram chat identifier, message_id: int - Telegram message identifier, ref_audio_path: str | None - staged reference audio path, language: str - requested language code }
    #   OUTPUTS: { JobSubmissionResult - submission outcome for the clone job }
    #   SIDE_EFFECTS: Submits a core job and emits submission logs.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: submit_clone_job
    def submit_clone_job(
        self,
        text: str,
        ref_text: str | None,
        chat_id: int,
        message_id: int,
        ref_audio_path: str | None = None,
        language: str = "auto",
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
        # START_BLOCK_SUBMIT_CLONE_JOB
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
                        event="[JobOrchestrator][submit_clone_job][BLOCK_SUBMIT_CLONE_JOB]",
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
        # END_BLOCK_SUBMIT_CLONE_JOB

        # START_BLOCK_CREATE_CLONE_SUBMISSION
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
                    language=language,
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
                event="[JobOrchestrator][submit_clone_job][BLOCK_CREATE_CLONE_SUBMISSION]",
                message="Voice Clone job submitted",
                chat_id=chat_id,
                message_id=message_id,
                job_id=resolution.snapshot.job_id,
                ref_text_provided=ref_text is not None,
                language=language,
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
                event="[JobOrchestrator][submit_clone_job][BLOCK_CREATE_CLONE_SUBMISSION]",
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
        # END_BLOCK_CREATE_CLONE_SUBMISSION

    # START_CONTRACT: check_job_completion
    #   PURPOSE: Inspect the current completion state and result payload for a Telegram-linked job.
    #   INPUTS: { job_id: str - core job identifier }
    #   OUTPUTS: { JobCompletionResult - current completion snapshot for the job }
    #   SIDE_EFFECTS: Reads job state and result data from the core execution layer.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: check_job_completion
    def check_job_completion(self, job_id: str) -> JobCompletionResult:
        """
        Check if job has completed and get result.

        Args:
            job_id: Core job ID

        Returns:
            JobCompletionResult with terminal state if completed
        """
        # START_BLOCK_POLL_JOB_STATUS
        snapshot = self._job_execution.get_job(job_id)
        if snapshot is None:
            return JobCompletionResult(
                status=JobStatus.QUEUED,
                is_terminal=False,
                success=None,
            )
        # END_BLOCK_POLL_JOB_STATUS

        # START_BLOCK_BUILD_COMPLETION_RESULT
        is_terminal = snapshot.status.is_terminal

        if snapshot.status == JobStatus.SUCCEEDED:
            result = self._job_execution.get_result(job_id)
            if result is not None and result.success is not None:
                return JobCompletionResult(
                    status=snapshot.status,
                    is_terminal=is_terminal,
                    success=True,
                    audio_bytes=result.success.generation.audio.bytes_data
                    if result.success.generation
                    else None,
                    duration_ms=self._calculate_duration_ms(snapshot),
                )
            return JobCompletionResult(
                status=snapshot.status,
                is_terminal=is_terminal,
                success=True,
            )

        if snapshot.status in {
            JobStatus.FAILED,
            JobStatus.TIMEOUT,
            JobStatus.CANCELLED,
        }:
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
        # END_BLOCK_BUILD_COMPLETION_RESULT

    @staticmethod
    def _calculate_duration_ms(snapshot: JobSnapshot) -> float | None:
        """Calculate job duration in milliseconds."""
        if snapshot.started_at and snapshot.completed_at:
            delta = snapshot.completed_at - snapshot.started_at
            return delta.total_seconds() * 1000
        return None


# START_CONTRACT: TelegramJobPoller
#   PURPOSE: Poll queued Telegram jobs for completion and deliver their results back to users.
#   INPUTS: { orchestrator: TelegramJobOrchestrator - job orchestration service, sender: TelegramSenderProtocol - Telegram delivery service, delivery_store: DeliveryMetadataStore - persisted delivery metadata store, settings: TelegramSettings - Telegram runtime settings, poll_interval_seconds: float - polling cadence }
#   OUTPUTS: { TelegramJobPoller - asynchronous Telegram job delivery poller }
#   SIDE_EFFECTS: Polls job state, sends Telegram messages, and updates delivery metadata.
#   LINKS: M-TELEGRAM
# END_CONTRACT: TelegramJobPoller
@dataclass
class TelegramJobPoller:
    """
    Polls job completion and handles delivery.

    This class runs in the background and checks for completed jobs,
    delivering results to users via the Telegram sender.
    """

    orchestrator: TelegramJobOrchestrator
    sender: TelegramSenderProtocol
    delivery_store: DeliveryMetadataStore
    settings: TelegramSettings
    poll_interval_seconds: float = 1.0
    _running: bool = field(default=False, init=False)
    _task: asyncio.Task | None = field(default=None, init=False)

    # START_CONTRACT: start
    #   PURPOSE: Start the Telegram job poller loop until stopped or cancelled.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Starts background polling of delivery metadata and job state.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: start
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

    # START_CONTRACT: stop
    #   PURPOSE: Stop the Telegram job poller and cancel any running task safely.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Cancels the poller task when it is running.
    #   LINKS: M-TELEGRAM
    # END_CONTRACT: stop
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
        # START_BLOCK_RECOVER_PENDING_DELIVERIES
        # First, recover pending jobs from previous run
        await self._recover_pending_jobs()
        # END_BLOCK_RECOVER_PENDING_DELIVERIES

        # START_BLOCK_POLL_JOB_STATUS_LOOP
        while self._running:
            try:
                await self._check_pending_deliveries()
                await asyncio.sleep(self.poll_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logging.getLogger(__name__).error(f"Error in job poller: {exc}")
                await asyncio.sleep(5.0)  # Back off on error
        # END_BLOCK_POLL_JOB_STATUS_LOOP

    async def _recover_pending_jobs(self) -> None:
        """Recover and deliver results from previous session."""
        from telegram_bot.observability import METRICS

        # START_BLOCK_LOAD_PENDING_DELIVERIES
        pending = await self.delivery_store.get_pending_deliveries()

        if not pending:
            return

        logging.getLogger(__name__).info(
            f"Recovering {len(pending)} pending jobs from previous session"
        )
        # END_BLOCK_LOAD_PENDING_DELIVERIES

        # START_BLOCK_DELIVER_RECOVERED_RESULTS
        for metadata in pending:
            job_id = metadata.get("job_id")
            chat_id = metadata.get("chat_id")
            message_id = metadata.get("message_id")

            if not all([job_id, chat_id, message_id]):
                continue

            resolved_job_id = cast(str, job_id)
            resolved_chat_id = cast(int, chat_id)
            resolved_message_id = cast(int, message_id)

            result = self.orchestrator.check_job_completion(resolved_job_id)

            if result.is_terminal:
                await self._deliver_job_result(
                    resolved_job_id,
                    result,
                    resolved_chat_id,
                    resolved_message_id,
                )
                METRICS.job_delivery_recovered()
        # END_BLOCK_DELIVER_RECOVERED_RESULTS

    async def _check_pending_deliveries(self) -> None:
        """Check pending deliveries and deliver completed jobs."""
        from telegram_bot.observability import METRICS

        # START_BLOCK_LOAD_PENDING_DELIVERY_BATCH
        pending = await self.delivery_store.get_pending_deliveries()
        # END_BLOCK_LOAD_PENDING_DELIVERY_BATCH

        # START_BLOCK_POLL_JOB_STATUS_BATCH
        for metadata in pending:
            job_id = metadata.get("job_id")
            chat_id = metadata.get("chat_id")
            message_id = metadata.get("message_id")

            if not all([job_id, chat_id, message_id]):
                continue

            resolved_job_id = cast(str, job_id)
            resolved_chat_id = cast(int, chat_id)
            resolved_message_id = cast(int, message_id)

            result = self.orchestrator.check_job_completion(resolved_job_id)

            if result.is_terminal:
                await self._deliver_job_result(
                    resolved_job_id, result, resolved_chat_id, resolved_message_id
                )
        # END_BLOCK_POLL_JOB_STATUS_BATCH

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
            # START_BLOCK_DELIVER_RESULT
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
            # END_BLOCK_DELIVER_RESULT

        except Exception as exc:
            # START_BLOCK_HANDLE_JOB_FAILURE
            LOGGER.error(
                f"Failed to deliver job result: {exc}",
                extra={
                    "job_id": job_id,
                    "chat_id": chat_id,
                    "message_id": message_id,
                },
            )
            await self.delivery_store.mark_delivered(
                chat_id,
                message_id,
                False,
                str(exc),
            )
            # END_BLOCK_HANDLE_JOB_FAILURE

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

__all__ = [
    "LOGGER",
    "TelegramSenderProtocol",
    "TELEGRAM_IDEMPOTENCY_SCOPE",
    "JobSubmissionResult",
    "JobCompletionResult",
    "JobSuccessSnapshot",
    "DeliveryMetadataStore",
    "TelegramJobOrchestrator",
    "TelegramJobPoller",
]
