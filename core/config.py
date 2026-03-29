from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / ".models"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / ".outputs"
DEFAULT_VOICES_DIR = PROJECT_ROOT / ".voices"


@dataclass(frozen=True)
class CoreSettings:
    models_dir: Path
    outputs_dir: Path
    voices_dir: Path
    backend: str | None = None
    backend_autoselect: bool = True
    sample_rate: int = 24000
    filename_max_len: int = 20
    default_save_output: bool = False
    enable_streaming: bool = True
    max_upload_size_bytes: int = 25 * 1024 * 1024
    request_timeout_seconds: int = 300
    inference_busy_status_code: int = 503
    auto_play_cli: bool = True

    def ensure_directories(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)


def env_text(name: str, default: str, environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return env.get(name, default)



def env_int(name: str, default: int, environ: Mapping[str, str] | None = None) -> int:
    return int(env_text(name, str(default), environ))



def env_bool(name: str, default: bool, environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    value = env.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}



def env_path(name: str, default: Path, environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    return Path(env.get(name, str(default))).resolve()



def parse_core_settings_from_env(environ: Mapping[str, str] | None = None) -> dict[str, object]:
    backend = env_text("QWEN_TTS_BACKEND", "", environ).strip() or None
    return {
        "models_dir": env_path("QWEN_TTS_MODELS_DIR", DEFAULT_MODELS_DIR, environ),
        "outputs_dir": env_path("QWEN_TTS_OUTPUTS_DIR", DEFAULT_OUTPUTS_DIR, environ),
        "voices_dir": env_path("QWEN_TTS_VOICES_DIR", DEFAULT_VOICES_DIR, environ),
        "backend": backend,
        "backend_autoselect": env_bool("QWEN_TTS_BACKEND_AUTOSELECT", True, environ),
        "default_save_output": env_bool("QWEN_TTS_DEFAULT_SAVE_OUTPUT", False, environ),
        "enable_streaming": env_bool("QWEN_TTS_ENABLE_STREAMING", True, environ),
        "max_upload_size_bytes": env_int("QWEN_TTS_MAX_UPLOAD_SIZE_BYTES", 25 * 1024 * 1024, environ),
        "request_timeout_seconds": env_int("QWEN_TTS_REQUEST_TIMEOUT_SECONDS", 300, environ),
        "inference_busy_status_code": env_int("QWEN_TTS_INFERENCE_BUSY_STATUS_CODE", 503, environ),
        "sample_rate": env_int("QWEN_TTS_SAMPLE_RATE", 24000, environ),
        "filename_max_len": env_int("QWEN_TTS_FILENAME_MAX_LEN", 20, environ),
        "auto_play_cli": env_bool("QWEN_TTS_AUTO_PLAY_CLI", True, environ),
    }
