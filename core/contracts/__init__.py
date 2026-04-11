# FILE: core/contracts/__init__.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Re-export public contract types for commands, synthesis planning, results, and jobs.
#   SCOPE: barrel re-exports
#   DEPENDS: none
#   LINKS: M-CONTRACTS
#   ROLE: BARREL
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   Command contracts - Re-export custom, clone, and design request command types
#   Job contracts - Re-export async job snapshots, statuses, operations, and submission helpers
#   Synthesis planning contracts - Re-export normalized request and execution plan DTOs
#   Result contracts - Re-export generated audio/result DTOs used across adapters
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from core.contracts.commands import (
    CustomVoiceCommand,
    VoiceCloneCommand,
    VoiceDesignCommand,
)
from core.contracts.jobs import (
    JobFailureSnapshot,
    JobOperation,
    JobResultResolution,
    JobSnapshot,
    JobStatus,
    JobStatusTransition,
    JobSubmission,
    JobSuccessSnapshot,
    StoredJob,
    create_job_submission,
    create_queued_job,
)
from core.contracts.results import AudioResult, GenerationResult
from core.contracts.synthesis import (
    ExecutionPlan,
    PresetSpeakerPayload,
    SynthesisRequest,
    VoiceClonePayload,
    VoiceDesignPayload,
)

__all__ = [
    "AudioResult",
    "CustomVoiceCommand",
    "ExecutionPlan",
    "GenerationResult",
    "JobFailureSnapshot",
    "JobOperation",
    "JobResultResolution",
    "JobSnapshot",
    "JobStatus",
    "JobStatusTransition",
    "JobSubmission",
    "JobSuccessSnapshot",
    "StoredJob",
    "SynthesisRequest",
    "PresetSpeakerPayload",
    "VoiceCloneCommand",
    "VoiceClonePayload",
    "VoiceDesignCommand",
    "VoiceDesignPayload",
    "create_job_submission",
    "create_queued_job",
]
