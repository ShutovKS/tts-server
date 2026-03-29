from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Optional

from core.bootstrap import CoreRuntime, build_runtime
from core.config import CoreSettings, env_int, env_text, parse_core_settings_from_env


@dataclass(frozen=True)
class ServerSettings(CoreSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ServerSettings":
        return cls(
            **parse_core_settings_from_env(environ),
            host=env_text("QWEN_TTS_HOST", "0.0.0.0", environ),
            port=env_int("QWEN_TTS_PORT", 8000, environ),
            log_level=env_text("QWEN_TTS_LOG_LEVEL", "info", environ),
        )


@lru_cache(maxsize=1)
def get_server_settings() -> ServerSettings:
    settings = ServerSettings.from_env()
    settings.ensure_directories()
    return settings


@dataclass(frozen=True)
class ServerRuntime:
    settings: ServerSettings
    core: CoreRuntime



def build_server_runtime(settings: Optional[ServerSettings] = None) -> ServerRuntime:
    resolved_settings = settings or get_server_settings()
    core_runtime = build_runtime(resolved_settings)
    return ServerRuntime(settings=resolved_settings, core=core_runtime)
