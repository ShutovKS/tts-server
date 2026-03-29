from __future__ import annotations

from functools import lru_cache
from typing import Mapping

from core.config import CoreSettings, parse_core_settings_from_env


class CliSettings(CoreSettings):
    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "CliSettings":
        return cls(**parse_core_settings_from_env(environ))


@lru_cache(maxsize=1)
def get_cli_settings() -> CliSettings:
    settings = CliSettings.from_env()
    settings.ensure_directories()
    return settings
