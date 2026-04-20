# Модуль Server — FastAPI HTTP API

English version: [README.md](README.md)

## Назначение

[server/](./) — HTTP transport adapter над общим TTS runtime из [../core/README.ru.md](../core/README.ru.md). Он публикует синхронные и асинхронные API endpoint'ы, health probes, список моделей и единый формат ошибок.

## Точки входа

- [__main__.py](__main__.py) — package entry point
- [app.py](app.py) — composition root FastAPI-приложения
- [`build_server_runtime()`](bootstrap.py:41) в [bootstrap.py](bootstrap.py)

## Локальный запуск

```bash
source .venv311/bin/activate
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

```powershell
.\.venv311\Scripts\Activate.ps1
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

## Запуск через Docker Compose

```bash
docker compose -f docker-compose.server.yaml up --build
```

Файл [../docker-compose.server.yaml](../docker-compose.server.yaml) собирает образ из [Dockerfile](Dockerfile) с корневым build context `.` и монтирует:

- [../.models](../.models)
- [../.outputs](../.outputs)
- [../.uploads](../.uploads)
- [../.voices](../.voices)

Эта схема заменила старую корневую compose-конфигурацию.

## Устройство контейнерного образа

[Dockerfile](Dockerfile) устанавливает Python-зависимости, `ffmpeg` и копирует общие исходники, необходимые адаптеру:

- [../core/](../core/)
- [./](./)
- [../cli/](../cli/)
- [../requirements.txt](../requirements.txt)
- [../.env.example](../.env.example)

Образ публикует порт `8000` и по умолчанию запускает `python -m uvicorn server:app --host 0.0.0.0 --port 8000`.

## API surface

### Health endpoint'ы

- `GET /health/live`
- `GET /health/ready`

### Получение списка моделей

- `GET /api/v1/models`

Model discovery теперь отдаёт family metadata, supported capabilities, selected backend, per-model execution backend, missing artifacts и route explanations для mixed-family deployment. Для Qwen models здесь также может появляться optional route candidate `qwen_fast`, который при невыполненных fast-path prerequisites остаётся rejected/unresolved route candidate.

### Синхронные TTS endpoint'ы

- `POST /v1/audio/speech`
- `POST /api/v1/tts/custom`
- `POST /api/v1/tts/design`
- `POST /api/v1/tts/clone`

### Асинхронные job endpoint'ы

- `POST /v1/audio/speech/jobs`
- `POST /api/v1/tts/custom/jobs`
- `POST /api/v1/tts/design/jobs`
- `POST /api/v1/tts/clone/jobs`
- `GET /api/v1/tts/jobs/{job_id}`
- `GET /api/v1/tts/jobs/{job_id}/result`
- `POST /api/v1/tts/jobs/{job_id}/cancel`

Async submit endpoint'ы принимают заголовок `Idempotency-Key`.

## Важное поведение runtime

- Адаптер возвращает готовое аудио как обычный HTTP response body.
- `QWEN_TTS_ENABLE_STREAMING` остаётся конфигурационным флагом, но сам по себе не превращает готовый аудио-ответ в streaming transport.
- Clone uploads временно размещаются в `QWEN_TTS_UPLOAD_STAGING_DIR` и удаляются после обработки запроса.
- Слишком длинные текстовые запросы завершаются стандартной validation error.
- При превышении таймаута inference возвращается `request_timeout` с HTTP `504`.
- Unsupported model/family combinations теперь возвращают controlled error `model_capability_not_supported` с явными capability details.
- Runtime capability bindings ожидаются из `QWEN_TTS_ACTIVE_FAMILY`, `QWEN_TTS_DEFAULT_CUSTOM_MODEL`, `QWEN_TTS_DEFAULT_DESIGN_MODEL` и `QWEN_TTS_DEFAULT_CLONE_MODEL`, а не из неявной инспекции `.models/`.
- Если для режима нет активной runtime binding, ожидаемое API behavior — controlled unsupported-mode response, а не internal failure.
- `GET /health/ready` теперь включает host snapshot и mixed-backend routing summary, чтобы оператор видел, почему Piper может маршрутизироваться в ONNX даже при глобально выбранном MLX.
- Ускоренный `qwen_fast` lane остаётся additive и optional; readiness/model payload'ы могут показывать его как отклонённого route candidate с явной причиной отклонения, а не как автоматическое переключение на другой backend.

## Конфигурация

Server-specific настройки расширяют [`CoreSettings`](../core/config.py:27) через [`ServerSettings`](bootstrap.py:17).

Ключевые переменные:

- `QWEN_TTS_HOST`
- `QWEN_TTS_PORT`
- `QWEN_TTS_LOG_LEVEL`
- `QWEN_TTS_ACTIVE_FAMILY`
- `QWEN_TTS_DEFAULT_CUSTOM_MODEL`
- `QWEN_TTS_DEFAULT_DESIGN_MODEL`
- `QWEN_TTS_DEFAULT_CLONE_MODEL`
- `QWEN_TTS_DEFAULT_SAVE_OUTPUT`
- `QWEN_TTS_ENABLE_STREAMING`
- `QWEN_TTS_MAX_UPLOAD_SIZE_BYTES`
- `QWEN_TTS_MAX_INPUT_TEXT_CHARS`
- `QWEN_TTS_REQUEST_TIMEOUT_SECONDS`
- `QWEN_TTS_INFERENCE_BUSY_STATUS_CODE`

Также применяются общие переменные из [../core/README.ru.md](../core/README.ru.md).

Если нужно полностью отключить ускоренный Qwen lane, установите `QWEN_TTS_QWEN_FAST_ENABLED=false`. Это влияет только на optional fast lane и не отключает стандартный Torch Qwen path.

Если lane нужно включить на поддерживаемом Linux/Windows CUDA-хосте, установите optional accelerated runtime отдельно через `pip install faster-qwen3-tts` и соблюдайте upstream prerequisites (Python 3.10+, PyTorch 2.5.1+, NVIDIA GPU с CUDA).

## Связанные исходники

- [api/routes_health.py](api/routes_health.py)
- [api/routes_models.py](api/routes_models.py)
- [api/routes_tts.py](api/routes_tts.py)
- [api/errors.py](api/errors.py)
- [api/auth.py](api/auth.py)
- [api/policies.py](api/policies.py)
- [schemas/audio.py](schemas/audio.py)

## Связанные документы

- [../README.ru.md](../README.ru.md) — обзор репозитория
- [../core/README.ru.md](../core/README.ru.md) — общий runtime
- [../telegram_bot/README.ru.md](../telegram_bot/README.ru.md) — Telegram-адаптер
- [../cli/README.ru.md](../cli/README.ru.md) — CLI-адаптер
