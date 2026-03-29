from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cli.runtime_config import CliSettings, get_cli_settings
from core.bootstrap import CoreRuntime, build_runtime


@dataclass(frozen=True)
class CliRuntimeBootstrap:
    settings: CliSettings
    core: CoreRuntime



def build_cli_runtime(settings: Optional[CliSettings] = None) -> CliRuntimeBootstrap:
    resolved_settings = settings or get_cli_settings()
    core_runtime = build_runtime(resolved_settings)
    return CliRuntimeBootstrap(settings=resolved_settings, core=core_runtime)
