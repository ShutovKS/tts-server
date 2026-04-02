# Core module — shared TTS runtime

Русская версия: [README.ru.md](README.ru.md)

## Purpose

[core/](core/) is the shared runtime layer used by the HTTP server, Telegram bot, and CLI. It contains:

- domain-level TTS application services
- model catalog and manifest handling
- backend registry and backend selection
- local job execution primitives
- admission control, quotas, and rate limiting
- observability and metrics helpers

Transport adapters should depend on [core/](core/) instead of duplicating synthesis logic.

## High-level structure

- [core/application/](application/) — application services and job orchestration contracts
- [core/backends/](backends/) — MLX and Torch backends
- [core/contracts/](contracts/) — DTOs, commands, and job/result contracts
- [core/infrastructure/](infrastructure/) — local implementations for storage, execution, and I/O
- [core/models/](models/) — model metadata and manifest files
- [core/services/](services/) — high-level TTS and model registry services
- [core/config.py](config.py) — shared environment-based settings
- [core/errors.py](errors.py) — typed domain errors
- [core/observability.py](observability.py) — request context and structured logging helpers
- [core/metrics.py](metrics.py) — operational metrics registry

## Runtime assembly

The shared runtime is built through [`build_runtime()`](bootstrap.py:76).

```python
from core.bootstrap import build_runtime
from core.config import CoreSettings

settings = CoreSettings.from_env()
runtime = build_runtime(settings)
```

The resulting runtime exposes the shared services consumed by transport adapters.

## Key components

### Application layer

- [`TTSApplicationService`](application/tts_app_service.py:11) — façade for custom, design, and clone synthesis
- [`JobExecutionGateway`](application/job_execution.py:20) — async job submission and retrieval contract
- [`QuotaGuard`](application/admission_control.py:14) and [`RateLimiter`](application/admission_control.py:34) — admission control abstractions

### Service layer

- [`TTSService`](services/tts_service.py:22) — coordinates inference, model resolution, and backend execution
- [`ModelRegistry`](services/model_registry.py:20) — discovers and validates local models

### Backend layer

- [`MLXBackend`](backends/mlx_backend.py:20) — Apple Silicon focused backend
- [`TorchBackend`](backends/torch_backend.py:15) — PyTorch backend for CPU/CUDA-compatible setups
- [`BackendRegistry`](backends/registry.py:14) — backend registration and selection

### Infrastructure layer

- [`LocalBoundedExecutionManager`](infrastructure/job_execution_local.py:15) — local async job execution manager
- [`LocalJobArtifactStore`](infrastructure/job_execution_local.py:14) — job artifact persistence
- [`convert_audio_to_wav_if_needed()`](infrastructure/audio_io.py:40) — shared audio normalization helper

## Configuration

Shared settings are defined by [`CoreSettings`](config.py:27).

Common environment variables:

- `QWEN_TTS_MODELS_DIR`
- `QWEN_TTS_OUTPUTS_DIR`
- `QWEN_TTS_VOICES_DIR`
- `QWEN_TTS_UPLOAD_STAGING_DIR`
- `QWEN_TTS_BACKEND`
- `QWEN_TTS_BACKEND_AUTOSELECT`
- `QWEN_TTS_MODEL_PRELOAD_POLICY`
- `QWEN_TTS_MODEL_PRELOAD_IDS`
- `QWEN_TTS_AUTH_MODE`
- `QWEN_TTS_RATE_LIMIT_ENABLED`
- `QWEN_TTS_QUOTA_ENABLED`
- `QWEN_TTS_SAMPLE_RATE`
- `QWEN_TTS_MAX_INPUT_TEXT_CHARS`

Transport-specific settings are documented in [../server/README.md](../server/README.md), [../telegram_bot/README.md](../telegram_bot/README.md), and [../cli/README.md](../cli/README.md).

## Model assets

The manifest lives in [models/manifest.v1.json](models/manifest.v1.json). Local model directories are resolved through [`ModelRegistry`](services/model_registry.py:20) relative to `QWEN_TTS_MODELS_DIR`.

## Operational notes

- Backend autoselection prefers MLX and then Torch when enabled.
- Local job execution is repository-local and intended for a single-node runtime.
- Temporary clone uploads use `QWEN_TTS_UPLOAD_STAGING_DIR`; adapters should not write temporary clone files into [../.outputs](../.outputs).
- All transport adapters reuse the same core request and error model.

## Related docs

- [../README.md](../README.md) — repository overview
- [../server/README.md](../server/README.md) — HTTP adapter
- [../telegram_bot/README.md](../telegram_bot/README.md) — Telegram adapter
- [../cli/README.md](../cli/README.md) — CLI adapter
