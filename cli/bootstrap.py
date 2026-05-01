# FILE: cli/bootstrap.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Build CLI runtime by wiring core runtime with CLI-specific configuration.
#   SCOPE: CLI runtime assembly
#   DEPENDS: M-BOOTSTRAP, M-CONFIG
#   LINKS: M-CLI
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   CliRuntimeBootstrap - Runtime container for CLI settings and core services
#   build_cli_runtime - Factory for CLI runtime assembly
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import dataclass

from cli.runtime_config import CliSettings, get_cli_settings
from core.bootstrap import CoreRuntime, build_runtime
from core.observability import get_logger, log_event

LOGGER = get_logger(__name__)


# START_CONTRACT: CliRuntimeBootstrap
#   PURPOSE: Hold resolved CLI settings together with the shared core runtime.
#   INPUTS: { settings: CliSettings - resolved CLI configuration, core: CoreRuntime - shared runtime services }
#   OUTPUTS: { CliRuntimeBootstrap - immutable CLI runtime container }
#   SIDE_EFFECTS: none
#   LINKS: M-CLI
# END_CONTRACT: CliRuntimeBootstrap
@dataclass(frozen=True)
class CliRuntimeBootstrap:
    settings: CliSettings
    core: CoreRuntime


# START_CONTRACT: build_cli_runtime
#   PURPOSE: Assemble the CLI runtime from settings and shared core services.
#   INPUTS: { settings: Optional[CliSettings] - optional prebuilt CLI settings }
#   OUTPUTS: { CliRuntimeBootstrap - CLI runtime container }
#   SIDE_EFFECTS: Builds the shared core runtime for interactive CLI use.
#   LINKS: M-CLI
# END_CONTRACT: build_cli_runtime
def build_cli_runtime(settings: CliSettings | None = None) -> CliRuntimeBootstrap:
    resolved_settings = settings or get_cli_settings()
    core_runtime = build_runtime(resolved_settings)
    log_event(
        LOGGER,
        level=20,
        event="[CliBootstrap][build_cli_runtime][BUILD_CLI_RUNTIME]",
        message="CLI runtime bindings resolved",
        active_family=resolved_settings.active_family,
        runtime_capability_map=resolved_settings.runtime_capability_map(),
    )
    return CliRuntimeBootstrap(settings=resolved_settings, core=core_runtime)


__all__ = [
    "CliRuntimeBootstrap",
    "build_cli_runtime",
]
