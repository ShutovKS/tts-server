# FILE: core/infrastructure/__init__.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Re-export public infrastructure implementations.
#   SCOPE: barrel re-exports for local backends
#   DEPENDS: none
#   LINKS: M-INFRASTRUCTURE
#   ROLE: BARREL
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   Admission control implementations - Re-export local quota/rate-limit builders
#   Audio I/O helpers - Re-export ffmpeg checks, audio normalization, persistence, and temp output helpers
#   Runtime adapters - Re-export local async job execution implementations
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Removed InferenceGuard from the public infrastructure barrel after runtime wiring moved fully to EngineScheduler]
# END_CHANGE_SUMMARY

from core.infrastructure.admission_control_local import (
    build_quota_guard,
    build_rate_limiter,
)
from core.infrastructure.audio_io import (
    check_ffmpeg_available,
    convert_audio_to_wav_if_needed,
    persist_output,
    read_generated_wav,
    temporary_output_dir,
)
from core.infrastructure.job_execution_local import (
    LocalBoundedExecutionManager,
    LocalInMemoryJobStore,
    LocalJobArtifactHandler,
    LocalJobArtifactStore,
)

__all__ = [
    "LocalBoundedExecutionManager",
    "LocalInMemoryJobStore",
    "LocalJobArtifactHandler",
    "LocalJobArtifactStore",
    "build_quota_guard",
    "build_rate_limiter",
    "check_ffmpeg_available",
    "convert_audio_to_wav_if_needed",
    "persist_output",
    "read_generated_wav",
    "temporary_output_dir",
]
