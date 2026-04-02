# Qwen3 TTS на Apple Silicon

English version: [README.md](README.md)

## Обзор

Репозиторий содержит локальный стек синтеза речи на базе Qwen3 TTS и разделён на три транспортных адаптера:

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
- Для macOS Apple Silicon: MLX-совместимое окружение и MLX-подготовленные артефакты моделей
- Для Linux или Windows: окружение, совместимое с PyTorch/Transformers

## Установка

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

## Модели

Поместите загруженные директории моделей в [`.models/`](.models). Поддерживаемые локальные model ID регистрируются в [`ModelRegistry`](core/services/model_registry.py:20) и описаны в [core/models/manifest.v1.json](core/models/manifest.v1.json).

Обычно используются каталоги:

- `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`
- `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit`
- `Qwen3-TTS-12Hz-1.7B-Base-8bit`
- `Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit`
- `Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit`
- `Qwen3-TTS-12Hz-0.6B-Base-8bit`

## Запуск CLI

```bash
source .venv311/bin/activate
python -m cli
```

Подробности по адаптеру — в [cli/README.ru.md](cli/README.ru.md).

## Запуск HTTP-сервера

### Локально

```bash
source .venv311/bin/activate
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

### Через Docker Compose

```bash
docker compose -f docker-compose.telegram-bot.yaml up --build
```

Этот сценарий собирает образ из [telegram_bot/Dockerfile](telegram_bot/Dockerfile), монтирует общие директории моделей и результатов и сохраняет delivery metadata в именованном volume, описанном в [docker-compose.telegram-bot.yaml](docker-compose.telegram-bot.yaml).

### Важная оговорка про Telegram-токен

Сборка контейнера и базовый запуск процесса бота подтверждены, но полноценная внешняя интеграция возможна только при наличии реального и корректного Telegram-токена. Без него бот не сможет пройти live-проверки Telegram API и обрабатывать реальные обновления.

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

Настройки HTTP-сервера описаны в [server/README.ru.md](server/README.ru.md), а настройки Telegram-бота — в [telegram_bot/README.ru.md](telegram_bot/README.ru.md).

## Карта документации

- [README.md](README.md) / [README.ru.md](README.ru.md) — корневой quick start
- [core/README.md](core/README.md) / [core/README.ru.md](core/README.ru.md) — общий runtime и архитектура
- [server/README.md](server/README.md) / [server/README.ru.md](server/README.ru.md) — HTTP API
- [telegram_bot/README.md](telegram_bot/README.md) / [telegram_bot/README.ru.md](telegram_bot/README.ru.md) — Telegram-адаптер
- [cli/README.md](cli/README.md) / [cli/README.ru.md](cli/README.ru.md) — интерактивный CLI
