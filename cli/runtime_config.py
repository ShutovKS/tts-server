# FILE: cli/runtime_config.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define CLI-specific runtime configuration.
#   SCOPE: CLI config dataclass with audio playback settings
#   DEPENDS: M-CONFIG
#   LINKS: M-CLI
#   ROLE: CONFIG
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   CliSettings - CLI-specific runtime settings derived from shared core config
#   get_cli_settings - Load and cache CLI settings from environment
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

from core.config import CoreSettings, parse_core_settings_from_env


# START_CONTRACT: CliSettings
#   PURPOSE: Represent CLI runtime settings built on top of shared core configuration.
#   INPUTS: {}
#   OUTPUTS: { CliSettings - CLI configuration object }
#   SIDE_EFFECTS: none
#   LINKS: M-CLI
# END_CONTRACT: CliSettings
class CliSettings(CoreSettings):
    # START_CONTRACT: from_env
    #   PURPOSE: Parse CLI settings from environment variables using shared core parsing.
    #   INPUTS: { environ: Mapping[str, str] | None - optional environment mapping override }
    #   OUTPUTS: { CliSettings - parsed CLI configuration }
    #   SIDE_EFFECTS: Reads process environment variables when no mapping is provided.
    #   LINKS: M-CLI
    # END_CONTRACT: from_env
    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> CliSettings:
        return cls(**parse_core_settings_from_env(environ))


# START_CONTRACT: get_cli_settings
#   PURPOSE: Load and cache CLI settings with ensured working directories.
#   INPUTS: {}
#   OUTPUTS: { CliSettings - cached CLI settings instance }
#   SIDE_EFFECTS: Ensures configured CLI directories exist on disk.
#   LINKS: M-CLI
# END_CONTRACT: get_cli_settings
@lru_cache(maxsize=1)
def get_cli_settings() -> CliSettings:
    settings = CliSettings.from_env()
    settings.ensure_directories()
    return settings


__all__ = [
    "CliSettings",
    "get_cli_settings",
]
