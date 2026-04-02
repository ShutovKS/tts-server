# Модуль Core — общий runtime для TTS

English version: [README.md](README.md)

## Назначение

[core/](core/) — общий runtime-слой, который используют HTTP-сервер, Telegram-бот и CLI. Здесь находятся:

- application-level сервисы синтеза
- каталог моделей и работа с манифестом
- реестр бэкендов и логика выбора backend
- локальные примитивы выполнения jobs
- admission control, quotas и rate limiting
- observability и metrics helper'ы

Транспортные адаптеры должны зависеть от [core/](core/), а не дублировать TTS-логику.

## Структура верхнего уровня

- [core/application/](core/application/) — application services и job contracts
- [core/backends/](core/backends/) — MLX и Torch backends
- [core/contracts/](core/contracts/) — DTO, команды и контракты jobs/results
- [core/infrastructure/](core/infrastructure/) — локальные реализации storage, execution и I/O
- [core/models/](core/models/) — метаданные моделей и manifest files
- [core/services/](core/services/) — высокоуровневые сервисы TTS и model registry
- [core/config.py](core/config.py) — общие настройки из env
- [core/errors.py](core/errors.py) — типизированные доменные ошибки
- [core/observability.py](core/observability.py) — request context и structured logging helpers
- [core/metrics.py](core/metrics.py) — registry операционных метрик

## Сборка runtime

Общий runtime собирается через [`build_runtime()`](bootstrap.py:76).

```python
from core.bootstrap import build_runtime
from core.config import CoreSettings

settings = CoreSettings.from_env()
runtime = build_runtime(settings)
```

Полученный runtime предоставляет общие сервисы, которые используют транспортные адаптеры.

## Ключевые компоненты

### Application layer

- [`TTSApplicationService`](application/tts_app_service.py:11) — фасад для custom, design и clone synthesis
- [`JobExecutionGateway`](application/job_execution.py:20) — контракт отправки и получения async jobs
- [`QuotaGuard`](application/admission_control.py:14) и [`RateLimiter`](application/admission_control.py:34) — абстракции admission control

### Service layer

- [`TTSService`](services/tts_service.py:22) — координирует inference, выбор модели и вызов backend
- [`ModelRegistry`](services/model_registry.py:20) — обнаруживает и валидирует локальные модели

### Backend layer

- [`MLXBackend`](backends/mlx_backend.py:20) — backend для Apple Silicon
- [`TorchBackend`](backends/torch_backend.py:15) — PyTorch backend для CPU/CUDA-совместимых окружений
- [`BackendRegistry`](backends/registry.py:14) — регистрация и выбор backend

### Infrastructure layer

- [`LocalBoundedExecutionManager`](infrastructure/job_execution_local.py:15) — локальный менеджер выполнения async jobs
- [`LocalJobArtifactStore`](infrastructure/job_execution_local.py:14) — хранение job artifacts
- [`convert_audio_to_wav_if_needed()`](infrastructure/audio_io.py:40) — общий helper нормализации аудио

## Конфигурация

Общие настройки определены в [`CoreSettings`](config.py:27).

Основные переменные окружения:

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

Transport-specific настройки описаны в [../server/README.ru.md](../server/README.ru.md), [../telegram_bot/README.ru.md](../telegram_bot/README.ru.md) и [../cli/README.ru.md](../cli/README.ru.md).

## Модельные артефакты

Манифест расположен в [models/manifest.v1.json](models/manifest.v1.json). Локальные директории моделей разрешаются через [`ModelRegistry`](services/model_registry.py:20) относительно `QWEN_TTS_MODELS_DIR`.

## Эксплуатационные замечания

- При включённом autoselect сначала предпочитается MLX, затем Torch.
- Локальное выполнение jobs рассчитано на single-node runtime внутри репозитория.
- Временные clone uploads используют `QWEN_TTS_UPLOAD_STAGING_DIR`; транспортные адаптеры не должны писать временные clone-файлы в [../.outputs](../.outputs).
- Все транспортные адаптеры используют общую модель запросов и ошибок из core.

## Связанные документы

- [../README.md](../README.md) — обзор репозитория
- [../server/README.ru.md](../server/README.ru.md) — HTTP-адаптер
- [../telegram_bot/README.ru.md](../telegram_bot/README.ru.md) — Telegram-адаптер
- [../cli/README.ru.md](../cli/README.ru.md) — CLI-адаптер
