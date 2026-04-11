# FILE: core/config.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Parse and validate environment-based runtime configuration for all components.
#   SCOPE: CoreSettings dataclass, environment parsing helpers, typed settings dict
#   DEPENDS: none
#   LINKS: M-CONFIG
#   ROLE: CONFIG
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   PROJECT_ROOT - Repository root directory used for default path resolution
#   DEFAULT_MODELS_DIR - Default local models directory
#   DEFAULT_OUTPUTS_DIR - Default generated outputs directory
#   DEFAULT_VOICES_DIR - Default saved voices directory
#   DEFAULT_UPLOAD_STAGING_DIR - Default upload staging directory
#   LOCAL_JOB_EXECUTION_BACKEND - Default local async execution backend key
#   LOCAL_JOB_METADATA_BACKEND - Default local job metadata backend key
#   LOCAL_JOB_ARTIFACT_BACKEND - Default local job artifact backend key
#   LOCAL_RATE_LIMIT_BACKEND - Default local rate limit backend key
#   LOCAL_QUOTA_BACKEND - Default local quota backend key
#   CoreSettings - Frozen dataclass holding all shared runtime settings
#   CoreSettingsEnv - TypedDict describing parsed settings shape
#   AuthMode - Literal type for authentication mode
#   parse_core_settings_from_env - Parse environment variables into typed settings dict
#   env_text - Read string from environment with default
#   env_int - Read integer from environment with default
#   env_bool - Read boolean from environment with default
#   env_path - Read Path from environment with default
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, TypedDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / ".models"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / ".outputs"
DEFAULT_VOICES_DIR = PROJECT_ROOT / ".voices"
DEFAULT_UPLOAD_STAGING_DIR = PROJECT_ROOT / ".uploads"


LOCAL_JOB_EXECUTION_BACKEND = "local"
LOCAL_JOB_METADATA_BACKEND = "local"
LOCAL_JOB_ARTIFACT_BACKEND = "local"
LOCAL_RATE_LIMIT_BACKEND = "local"
LOCAL_QUOTA_BACKEND = "local"


AuthMode = Literal["off", "static_bearer"]


class CoreSettingsEnv(TypedDict):
    models_dir: Path
    mlx_models_dir: Path
    outputs_dir: Path
    voices_dir: Path
    upload_staging_dir: Path
    model_manifest_path: Path
    backend: str | None
    backend_autoselect: bool
    qwen_fast_enabled: bool
    model_preload_policy: str
    model_preload_ids: tuple[str, ...]
    job_execution_backend: str
    job_metadata_backend: str
    job_artifact_backend: str
    auth_mode: AuthMode
    auth_static_bearer_token: str | None
    auth_static_bearer_principal_id: str | None
    auth_static_bearer_credential_id: str | None
    rate_limit_enabled: bool
    rate_limit_backend: str
    rate_limit_sync_tts_per_minute: int
    rate_limit_async_submit_per_minute: int
    rate_limit_job_read_per_minute: int
    rate_limit_job_cancel_per_minute: int
    rate_limit_control_plane_per_minute: int
    quota_enabled: bool
    quota_backend: str
    quota_compute_requests_per_window: int
    quota_compute_window_seconds: int
    quota_max_active_jobs_per_principal: int
    sample_rate: int
    filename_max_len: int
    default_save_output: bool
    enable_streaming: bool
    max_upload_size_bytes: int
    max_input_text_chars: int
    request_timeout_seconds: int
    inference_busy_status_code: int
    auto_play_cli: bool


# START_CONTRACT: CoreSettings
#   PURPOSE: Hold normalized shared runtime settings resolved from environment configuration.
#   INPUTS: { models_dir: Path - Root directory for backend model assets, outputs_dir: Path - Directory for persisted generated audio, voices_dir: Path - Directory for reusable voice assets, mlx_models_dir: Path - MLX-specific model directory override, upload_staging_dir: Path - Temporary upload staging directory, model_manifest_path: Path - Manifest file describing enabled models, backend: str | None - Requested backend key override, backend_autoselect: bool - Whether backend selection may fall back automatically, model_preload_policy: str - Preload strategy for runtime model warming, model_preload_ids: tuple[str, ...] - Explicit model identifiers to preload, job_execution_backend: str - Selected async execution backend, job_metadata_backend: str - Selected job metadata store backend, job_artifact_backend: str - Selected job artifact store backend, auth_mode: AuthMode - Authentication policy selector, auth_static_bearer_token: str | None - Static bearer token secret, auth_static_bearer_principal_id: str | None - Principal bound to static bearer auth, auth_static_bearer_credential_id: str | None - Credential identifier for static bearer auth, rate_limit_enabled: bool - Enables request throttling policies, rate_limit_backend: str - Selected rate limiter backend, rate_limit_sync_tts_per_minute: int - Per-minute sync synthesis allowance, rate_limit_async_submit_per_minute: int - Per-minute async submission allowance, rate_limit_job_read_per_minute: int - Per-minute async job read allowance, rate_limit_job_cancel_per_minute: int - Per-minute async job cancel allowance, rate_limit_control_plane_per_minute: int - Per-minute control plane allowance, quota_enabled: bool - Enables quota enforcement, quota_backend: str - Selected quota backend, quota_compute_requests_per_window: int - Compute quota limit per window, quota_compute_window_seconds: int - Compute quota window size, quota_max_active_jobs_per_principal: int - Max active async jobs per principal, sample_rate: int - Target sample rate for normalized audio, filename_max_len: int - Max text snippet length in saved filenames, default_save_output: bool - Default persistence behavior for generated output, enable_streaming: bool - Streaming capability toggle, max_upload_size_bytes: int - Maximum accepted upload size, max_input_text_chars: int - Maximum request text length, request_timeout_seconds: int - Request timeout budget, inference_busy_status_code: int - HTTP status code for busy inference responses, auto_play_cli: bool - CLI auto playback toggle }
#   OUTPUTS: { instance - Immutable runtime settings container }
#   SIDE_EFFECTS: none
#   LINKS: M-CONFIG
# END_CONTRACT: CoreSettings
@dataclass(frozen=True)
class CoreSettings:
    models_dir: Path
    outputs_dir: Path
    voices_dir: Path
    mlx_models_dir: Path = field(default_factory=lambda: DEFAULT_MODELS_DIR / "mlx")
    upload_staging_dir: Path = field(default_factory=lambda: DEFAULT_UPLOAD_STAGING_DIR)
    model_manifest_path: Path = field(
        default_factory=lambda: (PROJECT_ROOT / "core" / "models" / "manifest.v1.json")
    )
    backend: str | None = None
    backend_autoselect: bool = True
    qwen_fast_enabled: bool = True
    model_preload_policy: str = "none"
    model_preload_ids: tuple[str, ...] = ()
    job_execution_backend: str = LOCAL_JOB_EXECUTION_BACKEND
    job_metadata_backend: str = LOCAL_JOB_METADATA_BACKEND
    job_artifact_backend: str = LOCAL_JOB_ARTIFACT_BACKEND
    auth_mode: AuthMode = "off"
    auth_static_bearer_token: str | None = None
    auth_static_bearer_principal_id: str | None = None
    auth_static_bearer_credential_id: str | None = None
    rate_limit_enabled: bool = False
    rate_limit_backend: str = LOCAL_RATE_LIMIT_BACKEND
    rate_limit_sync_tts_per_minute: int = 0
    rate_limit_async_submit_per_minute: int = 0
    rate_limit_job_read_per_minute: int = 0
    rate_limit_job_cancel_per_minute: int = 0
    rate_limit_control_plane_per_minute: int = 0
    quota_enabled: bool = False
    quota_backend: str = LOCAL_QUOTA_BACKEND
    quota_compute_requests_per_window: int = 0
    quota_compute_window_seconds: int = 60
    quota_max_active_jobs_per_principal: int = 0
    sample_rate: int = 24000
    filename_max_len: int = 20
    default_save_output: bool = False
    enable_streaming: bool = True
    max_upload_size_bytes: int = 25 * 1024 * 1024
    max_input_text_chars: int = 5_000
    request_timeout_seconds: int = 300
    inference_busy_status_code: int = 503
    auto_play_cli: bool = True

    def ensure_directories(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.upload_staging_dir.mkdir(parents=True, exist_ok=True)


# START_CONTRACT: env_text
#   PURPOSE: Read a string environment variable with a provided default fallback.
#   INPUTS: { name: str - Environment variable name to read, default: str - Fallback value when the variable is unset, environ: Mapping[str, str] | None - Optional environment mapping override }
#   OUTPUTS: { str - Resolved string value from the environment or default }
#   SIDE_EFFECTS: none
#   LINKS: M-CONFIG
# END_CONTRACT: env_text
def env_text(name: str, default: str, environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return env.get(name, default)


# START_CONTRACT: env_int
#   PURPOSE: Read an integer environment variable while reusing shared text parsing behavior.
#   INPUTS: { name: str - Environment variable name to read, default: int - Fallback integer when unset, environ: Mapping[str, str] | None - Optional environment mapping override }
#   OUTPUTS: { int - Parsed integer value }
#   SIDE_EFFECTS: none
#   LINKS: M-CONFIG
# END_CONTRACT: env_int
def env_int(name: str, default: int, environ: Mapping[str, str] | None = None) -> int:
    return int(env_text(name, str(default), environ))


# START_CONTRACT: env_bool
#   PURPOSE: Read a boolean environment variable using common truthy string coercion rules.
#   INPUTS: { name: str - Environment variable name to read, default: bool - Fallback value when unset, environ: Mapping[str, str] | None - Optional environment mapping override }
#   OUTPUTS: { bool - Parsed boolean value }
#   SIDE_EFFECTS: none
#   LINKS: M-CONFIG
# END_CONTRACT: env_bool
def env_bool(
    name: str, default: bool, environ: Mapping[str, str] | None = None
) -> bool:
    env = os.environ if environ is None else environ
    value = env.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# START_CONTRACT: env_path
#   PURPOSE: Read and resolve a filesystem path environment variable with a default path.
#   INPUTS: { name: str - Environment variable name to read, default: Path - Fallback path when unset, environ: Mapping[str, str] | None - Optional environment mapping override }
#   OUTPUTS: { Path - Absolute resolved filesystem path }
#   SIDE_EFFECTS: none
#   LINKS: M-CONFIG
# END_CONTRACT: env_path
def env_path(
    name: str, default: Path, environ: Mapping[str, str] | None = None
) -> Path:
    env = os.environ if environ is None else environ
    return Path(env.get(name, str(default))).resolve()


def _parse_csv_env(
    name: str, environ: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    raw = env_text(name, "", environ)
    values: list[str] = []
    for part in raw.split(","):
        value = part.strip()
        if value and value not in values:
            values.append(value)
    return tuple(values)


# START_CONTRACT: parse_core_settings_from_env
#   PURPOSE: Parse supported core environment variables into a typed settings payload.
#   INPUTS: { environ: Mapping[str, str] | None - Optional environment mapping to parse instead of process env }
#   OUTPUTS: { CoreSettingsEnv - Typed settings dictionary ready for CoreSettings construction }
#   SIDE_EFFECTS: none
#   LINKS: M-CONFIG
# END_CONTRACT: parse_core_settings_from_env
def parse_core_settings_from_env(
    environ: Mapping[str, str] | None = None,
) -> CoreSettingsEnv:
    # START_BLOCK_PARSE_AUTH_SETTINGS
    backend = env_text("QWEN_TTS_BACKEND", "", environ).strip() or None
    auth_mode = env_text("QWEN_TTS_AUTH_MODE", "off", environ).strip().lower() or "off"
    if auth_mode not in {"off", "static_bearer"}:
        raise ValueError(f"Unsupported auth mode: {auth_mode}")
    auth_static_bearer_token = (
        env_text("QWEN_TTS_AUTH_STATIC_BEARER_TOKEN", "", environ).strip() or None
    )
    auth_static_bearer_principal_id = (
        env_text("QWEN_TTS_AUTH_STATIC_BEARER_PRINCIPAL_ID", "", environ).strip()
        or None
    )
    auth_static_bearer_credential_id = (
        env_text("QWEN_TTS_AUTH_STATIC_BEARER_CREDENTIAL_ID", "", environ).strip()
        or None
    )
    # END_BLOCK_PARSE_AUTH_SETTINGS
    # START_BLOCK_PARSE_PATH_SETTINGS
    return {
        "models_dir": env_path("QWEN_TTS_MODELS_DIR", DEFAULT_MODELS_DIR, environ),
        "mlx_models_dir": env_path(
            "QWEN_TTS_MLX_MODELS_DIR", DEFAULT_MODELS_DIR / "mlx", environ
        ),
        "outputs_dir": env_path("QWEN_TTS_OUTPUTS_DIR", DEFAULT_OUTPUTS_DIR, environ),
        "voices_dir": env_path("QWEN_TTS_VOICES_DIR", DEFAULT_VOICES_DIR, environ),
        "upload_staging_dir": env_path(
            "QWEN_TTS_UPLOAD_STAGING_DIR", DEFAULT_UPLOAD_STAGING_DIR, environ
        ),
        "model_manifest_path": env_path(
            "QWEN_TTS_MODEL_MANIFEST_PATH",
            PROJECT_ROOT / "core" / "models" / "manifest.v1.json",
            environ,
        ),
        # END_BLOCK_PARSE_PATH_SETTINGS
        # START_BLOCK_PARSE_RUNTIME_SETTINGS
        "backend": backend,
        "backend_autoselect": env_bool("QWEN_TTS_BACKEND_AUTOSELECT", True, environ),
        "qwen_fast_enabled": env_bool("QWEN_TTS_QWEN_FAST_ENABLED", True, environ),
        "model_preload_policy": env_text(
            "QWEN_TTS_MODEL_PRELOAD_POLICY", "none", environ
        )
        .strip()
        .lower()
        or "none",
        "model_preload_ids": _parse_csv_env("QWEN_TTS_MODEL_PRELOAD_IDS", environ),
        "job_execution_backend": env_text(
            "QWEN_TTS_JOB_EXECUTION_BACKEND",
            LOCAL_JOB_EXECUTION_BACKEND,
            environ,
        ).strip()
        or LOCAL_JOB_EXECUTION_BACKEND,
        "job_metadata_backend": env_text(
            "QWEN_TTS_JOB_METADATA_BACKEND",
            LOCAL_JOB_METADATA_BACKEND,
            environ,
        ).strip()
        or LOCAL_JOB_METADATA_BACKEND,
        "job_artifact_backend": env_text(
            "QWEN_TTS_JOB_ARTIFACT_BACKEND",
            LOCAL_JOB_ARTIFACT_BACKEND,
            environ,
        ).strip()
        or LOCAL_JOB_ARTIFACT_BACKEND,
        "auth_mode": auth_mode,
        "auth_static_bearer_token": auth_static_bearer_token,
        "auth_static_bearer_principal_id": auth_static_bearer_principal_id,
        "auth_static_bearer_credential_id": auth_static_bearer_credential_id,
        "rate_limit_enabled": env_bool("QWEN_TTS_RATE_LIMIT_ENABLED", False, environ),
        "rate_limit_backend": env_text(
            "QWEN_TTS_RATE_LIMIT_BACKEND", LOCAL_RATE_LIMIT_BACKEND, environ
        ).strip()
        or LOCAL_RATE_LIMIT_BACKEND,
        "rate_limit_sync_tts_per_minute": env_int(
            "QWEN_TTS_RATE_LIMIT_SYNC_TTS_PER_MINUTE", 0, environ
        ),
        "rate_limit_async_submit_per_minute": env_int(
            "QWEN_TTS_RATE_LIMIT_ASYNC_SUBMIT_PER_MINUTE", 0, environ
        ),
        "rate_limit_job_read_per_minute": env_int(
            "QWEN_TTS_RATE_LIMIT_JOB_READ_PER_MINUTE", 0, environ
        ),
        "rate_limit_job_cancel_per_minute": env_int(
            "QWEN_TTS_RATE_LIMIT_JOB_CANCEL_PER_MINUTE", 0, environ
        ),
        "rate_limit_control_plane_per_minute": env_int(
            "QWEN_TTS_RATE_LIMIT_CONTROL_PLANE_PER_MINUTE", 0, environ
        ),
        "quota_enabled": env_bool("QWEN_TTS_QUOTA_ENABLED", False, environ),
        "quota_backend": env_text(
            "QWEN_TTS_QUOTA_BACKEND", LOCAL_QUOTA_BACKEND, environ
        ).strip()
        or LOCAL_QUOTA_BACKEND,
        "quota_compute_requests_per_window": env_int(
            "QWEN_TTS_QUOTA_COMPUTE_REQUESTS_PER_WINDOW", 0, environ
        ),
        "quota_compute_window_seconds": env_int(
            "QWEN_TTS_QUOTA_COMPUTE_WINDOW_SECONDS", 60, environ
        ),
        "quota_max_active_jobs_per_principal": env_int(
            "QWEN_TTS_QUOTA_MAX_ACTIVE_JOBS_PER_PRINCIPAL", 0, environ
        ),
        "default_save_output": env_bool("QWEN_TTS_DEFAULT_SAVE_OUTPUT", False, environ),
        "enable_streaming": env_bool("QWEN_TTS_ENABLE_STREAMING", True, environ),
        "max_upload_size_bytes": env_int(
            "QWEN_TTS_MAX_UPLOAD_SIZE_BYTES", 25 * 1024 * 1024, environ
        ),
        "max_input_text_chars": env_int(
            "QWEN_TTS_MAX_INPUT_TEXT_CHARS", 5_000, environ
        ),
        "request_timeout_seconds": env_int(
            "QWEN_TTS_REQUEST_TIMEOUT_SECONDS", 300, environ
        ),
        "inference_busy_status_code": env_int(
            "QWEN_TTS_INFERENCE_BUSY_STATUS_CODE", 503, environ
        ),
        "sample_rate": env_int("QWEN_TTS_SAMPLE_RATE", 24000, environ),
        "filename_max_len": env_int("QWEN_TTS_FILENAME_MAX_LEN", 20, environ),
        "auto_play_cli": env_bool("QWEN_TTS_AUTO_PLAY_CLI", True, environ),
        # END_BLOCK_PARSE_RUNTIME_SETTINGS
    }


__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_MODELS_DIR",
    "DEFAULT_OUTPUTS_DIR",
    "DEFAULT_VOICES_DIR",
    "DEFAULT_UPLOAD_STAGING_DIR",
    "LOCAL_JOB_EXECUTION_BACKEND",
    "LOCAL_JOB_METADATA_BACKEND",
    "LOCAL_JOB_ARTIFACT_BACKEND",
    "LOCAL_RATE_LIMIT_BACKEND",
    "LOCAL_QUOTA_BACKEND",
    "AuthMode",
    "CoreSettingsEnv",
    "CoreSettings",
    "env_text",
    "env_int",
    "env_bool",
    "env_path",
    "parse_core_settings_from_env",
]
