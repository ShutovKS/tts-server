# Модульный локальный TTS runtime

English version: [README.md](README.md)

## Обзор

Репозиторий содержит локальный стек синтеза речи с общим модульным runtime и тремя транспортными адаптерами:

- [server/](server/README.ru.md) — HTTP API на FastAPI
- [telegram_bot/](telegram_bot/README.ru.md) — Telegram-бот на long polling
- [cli/](cli/README.ru.md) — интерактивный локальный CLI
- [core/](core/README.ru.md) — общий runtime, реестр моделей, бэкенды, jobs и observability

После реорганизации Docker-структуры артефакты сборки находятся рядом с соответствующими компонентами:

- образ сервера: [server/Dockerfile](server/Dockerfile)
- образ Telegram-бота: [telegram_bot/Dockerfile](telegram_bot/Dockerfile)
- compose-сценарий сервера: [docker-compose.server.yaml](docker-compose.server.yaml)
- compose-сценарий Telegram-бота: [docker-compose.telegram-bot.yaml](docker-compose.telegram-bot.yaml)

Старые корневые Docker-артефакты, такие как удалённые `Dockerfile` и `compose.yaml`, больше не используются.

## Возможности

- Локальный Qwen3 TTS inference на общей платформе из [core/](core/README.ru.md)
- Локальный Piper inference через ONNX runtime с использованием поддерживаемого Python API `piper-tts`
- OpenAI-совместимый endpoint `POST /v1/audio/speech`
- Расширенные HTTP endpoints для custom voice, voice design и voice cloning
- Команды Telegram-бота `/start`, `/help`, `/tts`, `/design`, `/clone`
- Интерактивный CLI для локальных сценариев синтеза
- Необязательный async job flow в HTTP-сервере
- Структурированные логи, request correlation и операционные метрики
- Изолированная staging-директория для clone-загрузок в [`.uploads/`](.uploads)
- Необязательное сохранение результатов в [`.outputs/`](.outputs)

## Требования

- Python 3.11+
- `ffmpeg`, доступный в `PATH`
- Локальные директории моделей в [`.models/`](.models)
- Для macOS Apple Silicon: MLX-совместимое окружение и MLX-подготовленные артефакты Qwen или ONNX-совместимое окружение для Piper
- Для Linux или Windows: окружение, совместимое с PyTorch/Transformers для Qwen, либо ONNX runtime для Piper

## Установка

### Наборы зависимостей

- `requirements.txt` — полный набор локальных operator-зависимостей, включая optional backend runtime-пакеты для поддерживаемых deployment lanes
- `requirements-ci.txt` — облегчённый набор для CI/проверок репозитория без тяжёлых optional runtime-зависимостей

### macOS Apple Silicon

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
brew install ffmpeg
```

### Linux или Windows

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Быстрый старт в Windows PowerShell

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
choco install ffmpeg -y
```

Если PowerShell блокирует запуск activation script, сначала выполните `Set-ExecutionPolicy -Scope Process Bypass` в текущей сессии.

### Важная оговорка по Qwen Torch lane

Немаковый Qwen lane зависит от официального Python-пакета `qwen-tts`, который используется в [`TorchBackend`](core/backends/torch_backend.py). Upstream-репозиторий Qwen3-TTS документирует `pip install -U qwen-tts` как стандартный путь установки и после этого использует `from qwen_tts import Qwen3TTSModel`. Поэтому для Linux/Windows у Qwen уже есть authoritative install path, но поддержку всё равно нужно считать частично подтверждённой или best effort, пока полный Torch lane не будет эмпирически прогнан на этих хостах.

### Важная оговорка по ускоренному Qwen lane

В репозитории также появился дополнительный backend `qwen_fast` для **custom-only** Qwen synthesis. Этот lane optional, не подменяет стандартный `torch` backend key и при отсутствии нужных runtime prerequisites автоматически уходит в безопасный fallback на стандартный Torch Qwen path.

Pinned README проекта faster-qwen3-tts документирует путь установки ускоренного runtime так:

```bash
pip install faster-qwen3-tts
```

Тот же upstream README указывает prerequisites для fast-lane: Python 3.10+, PyTorch 2.5.1+ и NVIDIA GPU с CUDA. В рамках этого репозитория такой install path нужно считать operator-managed optional dependency для поддерживаемых Linux/Windows CUDA-хостов, а не универсальной зависимостью для любого окружения.

### Примечание по Piper

В репозитории теперь есть поддерживаемый Piper lane через `piper-tts` + `onnxruntime`. На macOS wheel `piper-tts` уже включает `espeakbridge` и `espeak-ng-data`; на других платформах всё равно нужно отдельно проверить phonemization/runtime-совместимость в целевом окружении.

## Модели

Поместите загруженные директории моделей в [`.models/`](.models). Поддерживаемые локальные model ID регистрируются в [`ModelRegistry`](core/services/model_registry.py:20) и описаны в [core/models/manifest.v1.json](core/models/manifest.v1.json).

Обычно используются каталоги:

- `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`
- `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit`
- `Qwen3-TTS-12Hz-1.7B-Base-8bit`
- `Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit`
- `Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit`
- `Qwen3-TTS-12Hz-0.6B-Base-8bit`
- `Piper-en_US-lessac-medium`

### Формат Piper voice directory

Для встроенного Piper lane поместите каждый Piper voice в отдельную директорию внутри `.models/`, например:

- `.models/Piper-en_US-lessac-medium/model.onnx`
- `.models/Piper-en_US-lessac-medium/model.onnx.json`

Скачать Piper voice можно встроенным downloader, например:

```bash
source .venv311/bin/activate
mkdir -p .models/Piper-en_US-lessac-medium
.venv311/bin/python -m piper.download_voices en_US-lessac-medium --download-dir .models/Piper-en_US-lessac-medium
mv .models/Piper-en_US-lessac-medium/en_US-lessac-medium.onnx .models/Piper-en_US-lessac-medium/model.onnx
mv .models/Piper-en_US-lessac-medium/en_US-lessac-medium.onnx.json .models/Piper-en_US-lessac-medium/model.onnx.json
```

Эквивалент для Windows PowerShell:

```powershell
.\.venv311\Scripts\Activate.ps1
New-Item -ItemType Directory -Force -Path ".models/Piper-en_US-lessac-medium" | Out-Null
python -m piper.download_voices en_US-lessac-medium --download-dir .models/Piper-en_US-lessac-medium
Move-Item ".models/Piper-en_US-lessac-medium/en_US-lessac-medium.onnx" ".models/Piper-en_US-lessac-medium/model.onnx" -Force
Move-Item ".models/Piper-en_US-lessac-medium/en_US-lessac-medium.onnx.json" ".models/Piper-en_US-lessac-medium/model.onnx.json" -Force
```

## Runtime self-check

Для проверки selected backend, per-model execution backend, missing artifacts и host/runtime-сигналов используйте встроенную утилиту self-check:

```bash
source .venv311/bin/activate
python scripts/runtime_self_check.py
```

В automation можно включать строгий режим, чтобы получать non-zero exit code при degraded runtime или missing assets:

```bash
python scripts/runtime_self_check.py --strict
```

Если `qwen_fast` включён или рассматривается для маршрутизации, self-check теперь дополнительно показывает `backend_support`, route candidates и явные причины fallback/rejection, чтобы оператор видел, когда ускоренный custom-only lane был выбран, а когда нет.

Для воспроизводимых validation-сценариев используйте automation entry point, а не собирайте команды вручную:

```bash
python scripts/validate_runtime.py host-matrix
python scripts/validate_runtime.py smoke-server
python scripts/validate_runtime.py smoke-server --smoke-model-id Piper-en_US-lessac-medium --expected-backend onnx
python scripts/validate_runtime.py telegram-live --bot-token "$QWEN_TTS_TELEGRAM_BOT_TOKEN"
python scripts/validate_runtime.py telegram-live --bot-token "$QWEN_TTS_TELEGRAM_BOT_TOKEN" --chat-id "$QWEN_TTS_TELEGRAM_VALIDATION_CHAT_ID" --expect-update-chat-id "$QWEN_TTS_TELEGRAM_VALIDATION_CHAT_ID" --expect-update-text "Qwen3-TTS validation ping."
```

- `host-matrix` проверяет текущий host snapshot и симулированные сценарии optional-lane для `qwen_fast`.
- `smoke-server` поднимает локальный HTTP-сервер, ждёт health probes, запускает smoke-suite и затем корректно останавливает runtime.
- `smoke-server --smoke-model-id Piper-en_US-lessac-medium --expected-backend onnx` явно валидирует Piper HTTP path через `POST /v1/audio/speech` и одновременно проверяет, что для этой модели используется ONNX routing.
- `telegram-live` проверяет доступность реального Telegram Bot API и при необходимости может отправить validation message, если передан `--chat-id`.
- Добавьте `--expect-update-chat-id` и при необходимости `--expect-update-text`, если нужен opt-in сценарий с выделенным чатом, который дополнительно подтверждает, что новый подходящий inbound update виден через `getUpdates` без запуска long-polling runtime.

## Optional GRACE CLI install

Для локального lint этого репозитория можно использовать optional `grace` CLI. Публичный GRACE packaging-репозиторий документирует Bun-based install path:

```bash
bun add -g @osovv/grace-cli
grace lint --path /path/to/grace-project
```

В CI этого репозитория `grace` всё равно остаётся optional, потому что мы не требуем Bun на каждом validation host.

## Запуск CLI

```bash
source .venv311/bin/activate
python -m cli
```

```powershell
.\.venv311\Scripts\Activate.ps1
python -m cli
```

Подробности по адаптеру — в [cli/README.ru.md](cli/README.ru.md).

## Запуск HTTP-сервера

### Локально

```bash
source .venv311/bin/activate
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

```powershell
.\.venv311\Scripts\Activate.ps1
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### Через Docker Compose

```bash
docker compose -f docker-compose.server.yaml up --build
```

Этот сценарий собирает образ из [server/Dockerfile](server/Dockerfile) с корневым build context репозитория, монтирует общие рабочие директории и по умолчанию публикует порт `8000`.

Подробности по endpoint'ам, async jobs и конфигурации — в [server/README.ru.md](server/README.ru.md).

## Запуск Telegram-бота

### Локально

```bash
source .venv311/bin/activate
export QWEN_TTS_TELEGRAM_BOT_TOKEN="ваш_токен_бота"
export QWEN_TTS_TELEGRAM_ALLOWED_USER_IDS="123456789,987654321"
export QWEN_TTS_TELEGRAM_ADMIN_USER_IDS="123456789"
export QWEN_TTS_TELEGRAM_RATE_LIMIT_ENABLED=true
export QWEN_TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE=20
export QWEN_TTS_TELEGRAM_DELIVERY_STORE_PATH=.state/telegram_delivery_store.json
python -m telegram_bot
```

```powershell
.\.venv311\Scripts\Activate.ps1
$env:QWEN_TTS_TELEGRAM_BOT_TOKEN="ваш_токен_бота"
$env:QWEN_TTS_TELEGRAM_ALLOWED_USER_IDS="123456789,987654321"
$env:QWEN_TTS_TELEGRAM_ADMIN_USER_IDS="123456789"
$env:QWEN_TTS_TELEGRAM_RATE_LIMIT_ENABLED="true"
$env:QWEN_TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE="20"
$env:QWEN_TTS_TELEGRAM_DELIVERY_STORE_PATH=".state/telegram_delivery_store.json"
python -m telegram_bot
```

### Через Docker Compose

```bash
docker compose -f docker-compose.telegram-bot.yaml up --build
```

Этот сценарий собирает образ из [telegram_bot/Dockerfile](telegram_bot/Dockerfile), монтирует общие директории моделей и результатов и сохраняет delivery metadata в именованном volume, описанном в [docker-compose.telegram-bot.yaml](docker-compose.telegram-bot.yaml).

### Важная оговорка про Telegram-токен

На текущем Windows-хосте с Docker Desktop Linux containers compose-развёртывание было проверено через реальный старт бота, проверку доступности Telegram API и переход в healthy polling loop. При этом полноценная end-to-end интеграция всё равно требует реального и корректного Telegram-токена и подходящего chat/user context.

Подробности по командам, эксплуатации и деплою — в [telegram_bot/README.ru.md](telegram_bot/README.ru.md).

## Ключевые переменные окружения

Общие настройки читаются через [`CoreSettings.from_env()`](core/config.py:112). Основные переменные:

- `QWEN_TTS_MODELS_DIR`
- `QWEN_TTS_OUTPUTS_DIR`
- `QWEN_TTS_VOICES_DIR`
- `QWEN_TTS_UPLOAD_STAGING_DIR`
- `QWEN_TTS_BACKEND`
- `QWEN_TTS_BACKEND_AUTOSELECT`
- `QWEN_TTS_SAMPLE_RATE`
- `QWEN_TTS_MAX_INPUT_TEXT_CHARS`

Поддерживаемые backend keys теперь включают:

- `mlx` — Qwen3 на Apple Silicon
- `qwen_fast` — optional ускоренный Qwen custom-only lane с безопасным fallback на `torch`
- `torch` — Qwen3 на Torch CPU/CUDA-совместимых runtime
- `onnx` — Piper local voice inference через ONNX runtime

Для release-facing platform claims используйте [docs/support-matrix.md](docs/support-matrix.md) как канонический источник.

Настройки HTTP-сервера описаны в [server/README.ru.md](server/README.ru.md), а настройки Telegram-бота — в [telegram_bot/README.ru.md](telegram_bot/README.ru.md).

## Карта документации

- [README.md](README.md) / [README.ru.md](README.ru.md) — корневой quick start
- [core/README.md](core/README.md) / [core/README.ru.md](core/README.ru.md) — общий runtime и архитектура
- [server/README.md](server/README.md) / [server/README.ru.md](server/README.ru.md) — HTTP API
- [telegram_bot/README.md](telegram_bot/README.md) / [telegram_bot/README.ru.md](telegram_bot/README.ru.md) — Telegram-адаптер
- [cli/README.md](cli/README.md) / [cli/README.ru.md](cli/README.ru.md) — интерактивный CLI
