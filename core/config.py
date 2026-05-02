# FILE: core/config.py
# VERSION: 1.5.0
# START_MODULE_CONTRACT
#   PURPOSE: Parse and validate environment-based runtime configuration for all components, including explicit opt-in engine-route toggles.
#   SCOPE: CoreSettings dataclass, pydantic-settings env source, typed settings dict, env helpers, and engine-route feature flags
#   DEPENDS: pydantic, pydantic-settings
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
#   CoreSettings - Frozen dataclass holding all shared runtime settings including active capability bindings
#   CoreSettingsEnv - TypedDict describing parsed settings shape
#   AuthMode - Literal type for authentication mode
#   CoreEnvSettings - pydantic-settings model that owns the canonical TTS_* env contract
#   env_value - Resolve an environment variable by exact canonical name only
#   parse_core_settings_from_env - Parse environment variables into typed settings dict
#   env_text - Read string from environment with default
#   env_int - Read integer from environment with default
#   env_bool - Read boolean from environment with default
#   env_path - Read Path from environment with default
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.5.0 - Task 10: added an explicit Piper engine-route feature flag so the first production TTSEngine can be enabled without changing default runtime behavior]
# END_CHANGE_SUMMARY

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from pydantic import Field, ValidationError, field_validator  # pyright: ignore[reportMissingImports]
from pydantic_settings import BaseSettings, SettingsConfigDict  # pyright: ignore[reportMissingImports]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / ".models"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / ".outputs"
DEFAULT_VOICES_DIR = PROJECT_ROOT / ".voices"
DEFAULT_UPLOAD_STAGING_DIR = PROJECT_ROOT / ".uploads"
DEFAULT_RESULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "results"
DEFAULT_MODEL_MANIFEST_PATH = PROJECT_ROOT / "core" / "models" / "manifest.v1.json"


LOCAL_JOB_EXECUTION_BACKEND = "local"
LOCAL_JOB_METADATA_BACKEND = "local"
LOCAL_JOB_ARTIFACT_BACKEND = "local"
LOCAL_RATE_LIMIT_BACKEND = "local"
LOCAL_QUOTA_BACKEND = "local"


_TRUTHY_STRINGS = {"1", "true", "yes", "on"}
_AUTH_MODE_VALUES = {"off", "static_bearer"}


AuthMode = Literal["off", "static_bearer"]


class CoreSettingsEnv(TypedDict):
    models_dir: Path
    mlx_models_dir: Path
    outputs_dir: Path
    voices_dir: Path
    upload_staging_dir: Path
    model_manifest_path: Path
    active_family: str | None
    default_custom_model: str | None
    default_design_model: str | None
    default_clone_model: str | None
    piper_engine_enabled: bool
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
    cors_allowed_origins: tuple[str, ...]
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
    max_upload_size_bytes: int
    max_input_text_chars: int
    request_timeout_seconds: int
    inference_busy_status_code: int
    auto_play_cli: bool
    result_cache_enabled: bool
    result_cache_dir: Path
    result_cache_max_entries: int
    otel_enabled: bool
    otel_service_name: str
    otel_exporter_endpoint: str | None


# START_CONTRACT: CoreSettings
#   PURPOSE: Hold normalized shared runtime settings resolved from environment configuration.
#   INPUTS: { all CoreSettingsEnv fields - see TypedDict }
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
    model_manifest_path: Path = field(default_factory=lambda: DEFAULT_MODEL_MANIFEST_PATH)
    active_family: str | None = None
    default_custom_model: str | None = None
    default_design_model: str | None = None
    default_clone_model: str | None = None
    piper_engine_enabled: bool = False
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
    cors_allowed_origins: tuple[str, ...] = ()
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
    max_upload_size_bytes: int = 25 * 1024 * 1024
    max_input_text_chars: int = 5_000
    request_timeout_seconds: int = 300
    inference_busy_status_code: int = 503
    auto_play_cli: bool = True
    result_cache_enabled: bool = False
    result_cache_dir: Path = field(default_factory=lambda: DEFAULT_RESULT_CACHE_DIR)
    result_cache_max_entries: int = 1024
    otel_enabled: bool = False
    otel_service_name: str = "tts-server"
    otel_exporter_endpoint: str | None = None

    def ensure_directories(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.upload_staging_dir.mkdir(parents=True, exist_ok=True)

    def runtime_capability_map(self) -> dict[str, str | None]:
        return {
            "family": self.active_family,
            "custom_model": self.default_custom_model,
            "design_model": self.default_design_model,
            "clone_model": self.default_clone_model,
        }

    def resolve_runtime_model_binding(self, execution_mode: str) -> str | None:
        normalized_mode = (execution_mode or "").strip().lower()
        if normalized_mode == "custom":
            return self.default_custom_model
        if normalized_mode == "design":
            return self.default_design_model
        if normalized_mode == "clone":
            return self.default_clone_model
        return None


# START_BLOCK_PYDANTIC_SETTINGS_HELPERS
def _coerce_csv_tuple(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        values: list[str] = []
        for entry in raw:
            value = str(entry).strip()
            if value and value not in values:
                values.append(value)
        return tuple(values)
    if not isinstance(raw, str):
        return ()
    values = []
    for part in raw.split(","):
        value = part.strip()
        if value and value not in values:
            values.append(value)
    return tuple(values)


# END_BLOCK_PYDANTIC_SETTINGS_HELPERS


# START_CONTRACT: CoreEnvSettings
#   PURPOSE: Pydantic-settings model that owns the canonical TTS_* environment contract for the runtime core.
#   INPUTS: { **kwargs - canonical field names matching env vars after stripping the TTS_ prefix }
#   OUTPUTS: { instance - validated and type-coerced settings model }
#   SIDE_EFFECTS: none
#   LINKS: M-CONFIG
# END_CONTRACT: CoreEnvSettings
class CoreEnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TTS_",
        case_sensitive=False,
        extra="ignore",
        env_file=None,
    )

    models_dir: Path = Field(default_factory=lambda: DEFAULT_MODELS_DIR.resolve())
    mlx_models_dir: Path = Field(default_factory=lambda: (DEFAULT_MODELS_DIR / "mlx").resolve())
    outputs_dir: Path = Field(default_factory=lambda: DEFAULT_OUTPUTS_DIR.resolve())
    voices_dir: Path = Field(default_factory=lambda: DEFAULT_VOICES_DIR.resolve())
    upload_staging_dir: Path = Field(default_factory=lambda: DEFAULT_UPLOAD_STAGING_DIR.resolve())
    model_manifest_path: Path = Field(default_factory=lambda: DEFAULT_MODEL_MANIFEST_PATH.resolve())

    active_family: str | None = None
    default_custom_model: str | None = None
    default_design_model: str | None = None
    default_clone_model: str | None = None
    piper_engine_enabled: bool = False
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

    cors_allowed_origins: tuple[str, ...] = ()

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
    max_upload_size_bytes: int = 25 * 1024 * 1024
    max_input_text_chars: int = 5_000
    request_timeout_seconds: int = 300
    inference_busy_status_code: int = 503
    auto_play_cli: bool = True
    result_cache_enabled: bool = False
    result_cache_dir: Path = Field(default_factory=lambda: DEFAULT_RESULT_CACHE_DIR.resolve())
    result_cache_max_entries: int = 1024
    otel_enabled: bool = False
    otel_service_name: str = "tts-server"
    otel_exporter_endpoint: str | None = None

    @field_validator(
        "models_dir",
        "mlx_models_dir",
        "outputs_dir",
        "voices_dir",
        "upload_staging_dir",
        "model_manifest_path",
        "result_cache_dir",
        mode="before",
    )
    @classmethod
    def _normalize_path(cls, value: Any) -> Any:
        if value is None or value == "":
            return value
        if isinstance(value, Path):
            return value.resolve()
        return Path(str(value)).resolve()

    @field_validator(
        "active_family",
        "default_custom_model",
        "default_design_model",
        "default_clone_model",
        "backend",
        "auth_static_bearer_token",
        "auth_static_bearer_principal_id",
        "auth_static_bearer_credential_id",
        mode="before",
    )
    @classmethod
    def _empty_string_to_none(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator(
        "job_execution_backend",
        "job_metadata_backend",
        "job_artifact_backend",
        "rate_limit_backend",
        "quota_backend",
        mode="before",
    )
    @classmethod
    def _strip_required_str(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("model_preload_policy", mode="before")
    @classmethod
    def _normalize_preload_policy(cls, value: Any) -> Any:
        if value is None:
            return "none"
        if isinstance(value, str):
            normalized = value.strip().lower()
            return normalized or "none"
        return value

    @field_validator("auth_mode", mode="before")
    @classmethod
    def _normalize_auth_mode(cls, value: Any) -> Any:
        if value is None:
            return "off"
        if isinstance(value, str):
            normalized = value.strip().lower() or "off"
        else:
            normalized = value
        if normalized not in _AUTH_MODE_VALUES:
            raise ValueError(f"Unsupported auth mode: {normalized}")
        return normalized

    @field_validator(
        "model_preload_ids",
        "cors_allowed_origins",
        mode="before",
    )
    @classmethod
    def _parse_csv(cls, value: Any) -> Any:
        return _coerce_csv_tuple(value)

    @field_validator(
        "backend_autoselect",
        "qwen_fast_enabled",
        "piper_engine_enabled",
        "rate_limit_enabled",
        "quota_enabled",
        "default_save_output",
        "auto_play_cli",
        "result_cache_enabled",
        "otel_enabled",
        mode="before",
    )
    @classmethod
    def _coerce_bool_field(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in _TRUTHY_STRINGS
        return bool(value)


# START_CONTRACT: env_value
#   PURPOSE: Resolve an environment variable value using its exact canonical name.
#   INPUTS: { name: str - Canonical environment variable name, environ: Mapping[str, str] | None - Optional environment mapping override }
#   OUTPUTS: { str | None - Resolved environment value or None when unset }
#   SIDE_EFFECTS: none
#   LINKS: M-CONFIG
# END_CONTRACT: env_value
def env_value(name: str, environ: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    return env.get(name)


# START_CONTRACT: env_text
#   PURPOSE: Read a string environment variable with a provided default fallback.
#   INPUTS: { name: str - Environment variable name to read, default: str - Fallback value when the variable is unset, environ: Mapping[str, str] | None - Optional environment mapping override }
#   OUTPUTS: { str - Resolved string value from the environment or default }
#   SIDE_EFFECTS: none
#   LINKS: M-CONFIG
# END_CONTRACT: env_text
def env_text(name: str, default: str, environ: Mapping[str, str] | None = None) -> str:
    value = env_value(name, environ)
    return default if value is None else value


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
def env_bool(name: str, default: bool, environ: Mapping[str, str] | None = None) -> bool:
    value = env_value(name, environ)
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY_STRINGS


# START_CONTRACT: _parse_csv_env
#   PURPOSE: Backward-compatible CSV env helper preserved for transport adapters that read their own canonical TTS_* lists.
#   INPUTS: { name: str - Environment variable name to read, environ: Mapping[str, str] | None - Optional environment mapping override }
#   OUTPUTS: { tuple[str, ...] - Deduplicated, order-preserving comma-split values }
#   SIDE_EFFECTS: none
#   LINKS: M-CONFIG
# END_CONTRACT: _parse_csv_env
def _parse_csv_env(name: str, environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    return _coerce_csv_tuple(env_text(name, "", environ))


# START_CONTRACT: env_path
#   PURPOSE: Read and resolve a filesystem path environment variable with a default path.
#   INPUTS: { name: str - Environment variable name to read, default: Path - Fallback path when unset, environ: Mapping[str, str] | None - Optional environment mapping override }
#   OUTPUTS: { Path - Absolute resolved filesystem path }
#   SIDE_EFFECTS: none
#   LINKS: M-CONFIG
# END_CONTRACT: env_path
def env_path(name: str, default: Path, environ: Mapping[str, str] | None = None) -> Path:
    value = env_value(name, environ)
    return Path(str(default) if value is None else value).resolve()


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
    # START_BLOCK_COLLECT_TTS_VARS
    env_map = os.environ if environ is None else environ
    raw_kwargs: dict[str, Any] = {}
    for raw_key, value in env_map.items():
        if not isinstance(raw_key, str):
            continue
        upper = raw_key.upper()
        if not upper.startswith("TTS_"):
            continue
        field_name = upper[4:].lower()
        raw_kwargs[field_name] = value
    # END_BLOCK_COLLECT_TTS_VARS

    # START_BLOCK_BUILD_PYDANTIC_SETTINGS
    try:
        settings = CoreEnvSettings.model_validate(raw_kwargs)
    except ValidationError as exc:
        raise ValueError(_format_first_validation_error(exc)) from exc
    # END_BLOCK_BUILD_PYDANTIC_SETTINGS

    # START_BLOCK_PROJECT_TO_TYPED_DICT
    # CoreEnvSettings field names align with CoreSettingsEnv keys, so a plain
    # model_dump() returns the legacy-shaped TypedDict payload directly.
    return cast(CoreSettingsEnv, settings.model_dump())
    # END_BLOCK_PROJECT_TO_TYPED_DICT


def _format_first_validation_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "Invalid core runtime configuration"
    first = errors[0]
    message = first.get("msg") or "Invalid value"
    return message.replace("Value error, ", "")


__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_MODELS_DIR",
    "DEFAULT_OUTPUTS_DIR",
    "DEFAULT_VOICES_DIR",
    "DEFAULT_UPLOAD_STAGING_DIR",
    "DEFAULT_RESULT_CACHE_DIR",
    "DEFAULT_MODEL_MANIFEST_PATH",
    "LOCAL_JOB_EXECUTION_BACKEND",
    "LOCAL_JOB_METADATA_BACKEND",
    "LOCAL_JOB_ARTIFACT_BACKEND",
    "LOCAL_RATE_LIMIT_BACKEND",
    "LOCAL_QUOTA_BACKEND",
    "AuthMode",
    "CoreEnvSettings",
    "CoreSettingsEnv",
    "CoreSettings",
    "env_value",
    "env_text",
    "env_int",
    "env_bool",
    "env_path",
    "parse_core_settings_from_env",
]
