# FILE: server/bootstrap.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Extend CoreSettings with server-specific configuration and build server runtime.
#   SCOPE: ServerSettings, build_server_runtime factory
#   DEPENDS: M-BOOTSTRAP, M-CONFIG
#   LINKS: M-SERVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   ServerSettings - Server-specific configuration extending CoreSettings
#   get_server_settings - Load and cache server settings from environment
#   ServerRuntime - Runtime bundle for the HTTP server adapter
#   build_server_runtime - Factory for server runtime assembly
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Optional

from core.bootstrap import CoreRuntime, build_runtime
from core.config import CoreSettings, env_int, env_text, parse_core_settings_from_env
from core.observability import get_logger, log_event


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
# START_CONTRACT: ServerSettings
#   PURPOSE: Represent server-specific configuration layered on top of shared core settings.
#   INPUTS: { host: str - bind host, port: int - bind port, log_level: str - uvicorn log verbosity }
#   OUTPUTS: { ServerSettings - immutable server settings object }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER, M-CONFIG
# END_CONTRACT: ServerSettings
class ServerSettings(CoreSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    @classmethod
    # START_CONTRACT: from_env
    #   PURPOSE: Build server settings from environment variables and shared core defaults.
    #   INPUTS: { environ: Mapping[str, str] | None - optional environment mapping override }
    #   OUTPUTS: { ServerSettings - parsed server settings instance }
    #   SIDE_EFFECTS: Reads environment-provided configuration values
    #   LINKS: M-SERVER, M-CONFIG
    # END_CONTRACT: from_env
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ServerSettings":
        return cls(
            **parse_core_settings_from_env(environ),
            host=env_text("QWEN_TTS_HOST", "0.0.0.0", environ),
            port=env_int("QWEN_TTS_PORT", 8000, environ),
            log_level=env_text("QWEN_TTS_LOG_LEVEL", "info", environ),
        )


@lru_cache(maxsize=1)
# START_CONTRACT: get_server_settings
#   PURPOSE: Resolve and cache the server settings instance for repeated runtime use.
#   INPUTS: { none: None - no explicit inputs }
#   OUTPUTS: { ServerSettings - cached server settings with directories ensured }
#   SIDE_EFFECTS: Reads environment configuration, creates configured directories, and caches the result
#   LINKS: M-SERVER, M-CONFIG
# END_CONTRACT: get_server_settings
def get_server_settings() -> ServerSettings:
    settings = ServerSettings.from_env()
    settings.ensure_directories()
    return settings


@dataclass(frozen=True)
# START_CONTRACT: ServerRuntime
#   PURPOSE: Bundle resolved server settings with the shared core runtime instance.
#   INPUTS: { settings: ServerSettings - resolved server configuration, core: CoreRuntime - shared runtime bundle }
#   OUTPUTS: { ServerRuntime - immutable runtime container for server composition }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER, M-BOOTSTRAP
# END_CONTRACT: ServerRuntime
class ServerRuntime:
    settings: ServerSettings
    core: CoreRuntime


# START_CONTRACT: build_server_runtime
#   PURPOSE: Assemble the server runtime from provided or cached server settings.
#   INPUTS: { settings: Optional[ServerSettings] - optional settings override for runtime creation }
#   OUTPUTS: { ServerRuntime - composed runtime bundle for the HTTP adapter }
#   SIDE_EFFECTS: May read environment configuration and initialize shared runtime dependencies
#   LINKS: M-SERVER, M-BOOTSTRAP
# END_CONTRACT: build_server_runtime
def build_server_runtime(settings: Optional[ServerSettings] = None) -> ServerRuntime:
    # START_BLOCK_PARSE_SERVER_SETTINGS
    resolved_settings = settings or get_server_settings()
    # END_BLOCK_PARSE_SERVER_SETTINGS
    # START_BLOCK_BUILD_SERVER_RUNTIME
    core_runtime = build_runtime(resolved_settings)
    log_event(
        LOGGER,
        level=20,
        event="[ServerBootstrap][build_server_runtime][BUILD_SERVER_RUNTIME]",
        message="Server runtime bindings resolved",
        active_family=resolved_settings.active_family,
        runtime_capability_map=resolved_settings.runtime_capability_map(),
    )
    return ServerRuntime(settings=resolved_settings, core=core_runtime)
    # END_BLOCK_BUILD_SERVER_RUNTIME

__all__ = [
    "ServerSettings",
    "get_server_settings",
    "ServerRuntime",
    "build_server_runtime",
]
