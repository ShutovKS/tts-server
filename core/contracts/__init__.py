# FILE: core/contracts/__init__.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Re-export public contract types for commands, synthesis planning, results, and jobs, including the runtime CapabilityRegistry seam introduced in Phase 3.10.
#   SCOPE: barrel re-exports
#   DEPENDS: none
#   LINKS: M-CONTRACTS, M-CAPABILITIES
#   ROLE: BARREL
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   Capability contracts - Re-export the dynamic CapabilityRegistry seam (Phase 3.10)
#   Command contracts - Re-export custom, clone, and design request command types
#   Job contracts - Re-export async job snapshots, statuses, operations, and submission helpers
#   Runtime seam contracts - Re-export typed runtime registry protocols and backend-route payloads
#   Synthesis planning contracts - Re-export normalized request and execution plan DTOs
#   Result contracts - Re-export generated audio/result DTOs used across adapters
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Phase 3.10: re-exported CapabilityRegistry, CapabilitySpec, and DEFAULT_CAPABILITY_REGISTRY so transports and tests can register new capabilities without reaching into the contract submodules]
# END_CHANGE_SUMMARY

from core.contracts.capabilities import (
    DEFAULT_CAPABILITY_REGISTRY,
    CapabilityRegistry,
    CapabilitySpec,
    is_supported_capability,
    register_default_capability,
)
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
from core.contracts.runtime import (
    BackendRouteInfo,
    RuntimeExecutionRegistry,
    RuntimePlanningRegistry,
)
from core.contracts.synthesis import (
    ExecutionPlan,
    PresetSpeakerPayload,
    SynthesisRequest,
    VoiceClonePayload,
    VoiceDesignPayload,
)

__all__ = [
    "DEFAULT_CAPABILITY_REGISTRY",
    "AudioResult",
    "BackendRouteInfo",
    "CapabilityRegistry",
    "CapabilitySpec",
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
    "PresetSpeakerPayload",
    "RuntimeExecutionRegistry",
    "RuntimePlanningRegistry",
    "StoredJob",
    "SynthesisRequest",
    "VoiceCloneCommand",
    "VoiceClonePayload",
    "VoiceDesignCommand",
    "VoiceDesignPayload",
    "create_job_submission",
    "create_queued_job",
    "is_supported_capability",
    "register_default_capability",
]
