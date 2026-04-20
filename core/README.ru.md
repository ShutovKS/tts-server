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
- [core/backends/](core/backends/) — MLX, ускоренный Qwen fast, Torch и ONNX/Piper backends
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

- [`MLXBackend`](backends/mlx_backend.py:20) — backend для Qwen на Apple Silicon
- [`QwenFastBackend`](backends/qwen_fast_backend.py) — optional ускоренный backend для Qwen с явным CUDA/runtime gating
- [`TorchBackend`](backends/torch_backend.py:15) — PyTorch backend для Qwen на CPU/CUDA-совместимых окружениях
- [`ONNXBackend`](backends/onnx_backend.py) — backend для Piper local voice inference через ONNX
- [`BackendRegistry`](backends/registry.py:14) — регистрация и выбор backend

### Family и planning layer

- [`SynthesisCoordinator`](services/tts_service.py) — внутренний coordinator над planning, family preparation и runtime execution
- [`SynthesisPlanner`](planning/planner.py) — planner на registry contract для normalized request resolution
- [`Qwen3FamilyAdapter`](model_families/qwen3.py) — текущая family semantics для Qwen3
- [`PiperFamilyAdapter`](model_families/piper.py) — family semantics для Piper local ONNX voices
- [`HostProbe`](planning/host_probe.py) и [`CapabilityResolver`](planning/capability_resolver.py) — surfaces для объяснимого выбора backend

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
- `QWEN_TTS_ACTIVE_FAMILY`
- `QWEN_TTS_DEFAULT_CUSTOM_MODEL`
- `QWEN_TTS_DEFAULT_DESIGN_MODEL`
- `QWEN_TTS_DEFAULT_CLONE_MODEL`
- `QWEN_TTS_BACKEND`
- `QWEN_TTS_BACKEND_AUTOSELECT`
- `QWEN_TTS_QWEN_FAST_ENABLED`
- `QWEN_TTS_MODEL_PRELOAD_POLICY`
- `QWEN_TTS_MODEL_PRELOAD_IDS`
- `QWEN_TTS_AUTH_MODE`
- `QWEN_TTS_RATE_LIMIT_ENABLED`
- `QWEN_TTS_QUOTA_ENABLED`
- `QWEN_TTS_SAMPLE_RATE`
- `QWEN_TTS_MAX_INPUT_TEXT_CHARS`

Transport-specific настройки описаны в [../server/README.ru.md](../server/README.ru.md), [../telegram_bot/README.ru.md](../telegram_bot/README.ru.md) и [../cli/README.ru.md](../cli/README.ru.md).

Shared runtime contract теперь также включает явную runtime capability map, доступную через `CoreSettings.runtime_capability_map()`:

- `family`
- `custom_model`
- `design_model`
- `clone_model`

Эти значения являются активными runtime-привязками, а не прокси для того, какие артефакты просто лежат на диске. Само наличие папки модели в `.models/` не делает режим доступным. Режим становится доступным только тогда, когда runtime запущен с соответствующей capability binding.

## Runtime readiness и self-check

- [`ModelRegistry.readiness_report()`](services/model_registry.py:218) теперь публикует selected backend, per-model execution backend, mixed-backend routing summary, host snapshot, family summary и per-candidate route diagnostics для optional fast-lane решений.
- Operator-утилита [`scripts/runtime_self_check.py`](../scripts/runtime_self_check.py) печатает JSON snapshot для локальной setup-проверки и CI evidence.
- В mixed-family deployment model discovery и readiness специально различают `selected_backend` и `execution_backend`, потому что Piper может маршрутизироваться в ONNX, даже если глобально выбран MLX, а Qwen models могут одновременно показывать `qwen_fast` как выбранный или отклонённый route candidate для конкретного режима без влияния на другие family.

## Модельные артефакты

Манифест расположен в [models/manifest.v1.json](models/manifest.v1.json). Локальные директории моделей разрешаются через [`ModelRegistry`](services/model_registry.py:20) относительно `QWEN_TTS_MODELS_DIR`.

Runtime сейчас поддерживает следующие model families:

- `Qwen3-TTS` через `mlx` и `torch`
- `Qwen3-TTS` accelerated lane через `qwen_fast` с явной диагностикой маршрута, когда fast lane недоступен
- `OmniVoice` через `torch`
- `Piper` через `onnx`

Для `qwen_fast` нужен optional runtime faster-qwen3-tts; pinned upstream README документирует установку через `pip install faster-qwen3-tts` и указывает prerequisites: Python 3.10+, PyTorch 2.5.1+ и NVIDIA CUDA.

Для операторов репозиторий теперь собирает runtime dependency contours из `profiles/packs/`. `requirements.txt` остаётся тонким convenience-manifest для стабильного общего окружения Qwen + Piper, а `profiles/packs/family/omnivoice.txt` используется как отдельный family-pack файл для изолированных окружений, когда этому upstream family нужна изоляция зависимостей относительно Qwen.

## Эксплуатационные замечания

- Autoselect backend теперь учитывает host/runtime compatibility и даёт объяснимый результат: MLX предпочитается на совместимых macOS-хостах, Torch — на Linux/Windows CPU/CUDA-хостах, а ONNX может выбираться для Piper voices.
- Unsupported family operations теперь завершаются явными capability errors вместо неоднозначных runtime failure.
- Локальное выполнение jobs рассчитано на single-node runtime внутри репозитория.
- Временные clone uploads используют `QWEN_TTS_UPLOAD_STAGING_DIR`; транспортные адаптеры не должны писать временные clone-файлы в [../.outputs](../.outputs).
- Все транспортные адаптеры используют общую модель запросов и ошибок из core.

## Связанные документы

- [../README.md](../README.md) — обзор репозитория
- [../server/README.ru.md](../server/README.ru.md) — HTTP-адаптер
- [../telegram_bot/README.ru.md](../telegram_bot/README.ru.md) — Telegram-адаптер
- [../cli/README.ru.md](../cli/README.ru.md) — CLI-адаптер
