# Модуль Telegram Bot — long-polling адаптер

English version: [README.md](README.md)

## Назначение

[telegram_bot/](./) — отдельный транспортный адаптер поверх общего runtime из [../core/README.ru.md](../core/README.ru.md). Он предоставляет локальный TTS через Telegram-команды и запускается как самостоятельный polling-процесс.

## Текущее состояние

Бот поддерживается как отдельная deployment unit со своим контейнерным образом и compose-сценарием.

Реализовано:

- только private-чаты
- `/start`, `/help`, `/tts`, `/design`, `/clone`
- async submit / poll / deliver workflow
- сохранение delivery metadata и восстановление после рестарта
- startup self-checks
- retry с backoff для polling и delivery
- structured observability и degraded-state tracking
- отдельный Docker-образ в [Dockerfile](Dockerfile)
- отдельный compose-сценарий в [../docker-compose.telegram-bot.yaml](../docker-compose.telegram-bot.yaml)

Не реализовано:

- webhook mode
- поддержка групповых чатов
- inline queries
- callback buttons
- conversational memory

## Локальный запуск

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

## Запуск через Docker Compose

```bash
docker compose -f docker-compose.telegram-bot.yaml up --build
```

Файл [../docker-compose.telegram-bot.yaml](../docker-compose.telegram-bot.yaml) собирает образ из [Dockerfile](Dockerfile) с корневым build context `.` и монтирует:

- [../.models](../.models)
- [../.outputs](../.outputs)
- [../.uploads](../.uploads)
- [../.voices](../.voices)
- именованный volume `telegram_bot_state` в `/app/.state`

Образ включает `ffmpeg` и хранит delivery metadata по пути `/app/.state/telegram_delivery_store.json`.

## Ограничение по Telegram-токену

Сборка контейнера и старт процесса бота подтверждены, но полноценная внешняя интеграция требует реального Telegram-токена. Без валидного токена бот не сможет пройти проверку доступности Telegram API и обрабатывать live updates.

## Команды

| Команда | Описание |
|---|---|
| `/start` | Показать приветственное сообщение |
| `/help` | Показать справку и список доступных спикеров |
| `/tts` | Синтезировать текст с опциональными speaker и speed |
| `/design` | Синтезировать речь по текстовому описанию голоса |
| `/clone` | Клонировать голос по reference audio из replied сообщения |

### Синтаксис `/tts`

```text
/tts [-- speaker=<speaker>] [-- speed=<speed>] -- <text>
```

### Синтаксис `/design`

```text
/design <voice_description> -- <text>
```

### Синтаксис `/clone`

```text
/clone [-- ref=<transcript>] -- <text>
```

Для `/clone` команда должна быть отправлена reply на сообщение с reference audio.

## Поддерживаемые clone media

- `voice`
- `audio`
- `document` с поддерживаемым аудио-форматом

Поддерживаемые форматы: WAV, MP3, FLAC, OGG, WebM, M4A, MP4.

## Конфигурация

Telegram-specific настройки определяются через [`TelegramSettings`](config.py:42).

Ключевые переменные окружения:

- `QWEN_TTS_TELEGRAM_BOT_TOKEN`
- `QWEN_TTS_TELEGRAM_ALLOWED_USER_IDS`
- `QWEN_TTS_TELEGRAM_ADMIN_USER_IDS`
- `QWEN_TTS_TELEGRAM_DEV_MODE`
- `QWEN_TTS_TELEGRAM_RATE_LIMIT_ENABLED`
- `QWEN_TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE`
- `QWEN_TTS_TELEGRAM_LOG_LEVEL`
- `QWEN_TTS_TELEGRAM_DEFAULT_SPEAKER`
- `QWEN_TTS_TELEGRAM_MAX_TEXT_LENGTH`
- `QWEN_TTS_TELEGRAM_DELIVERY_STORE_PATH`
- `QWEN_TTS_TELEGRAM_POLL_INTERVAL_SECONDS`
- `QWEN_TTS_TELEGRAM_MAX_RETRIES`

Также применяются общие core-переменные из [../core/README.ru.md](../core/README.ru.md).

## Эксплуатационные замечания

- Бот работает только в private chat.
- Для аудиоконвертации и clone flow необходим `ffmpeg`.
- Delivery metadata сохраняется, чтобы завершённые jobs могли быть доставлены после рестарта.
- Startup self-checks валидируют наличие токена, wiring runtime, доступность backend и соединение с Telegram API.

## Документация по деплою

Дополнительные deployment-артефакты:

- [../docs/telegram-bot-deployment.md](../docs/telegram-bot-deployment.md)
- [../docs/telegram-bot.service](../docs/telegram-bot.service)

## Связанные исходники

- [__main__.py](__main__.py)
- [bootstrap.py](bootstrap.py)
- [config.py](config.py)
- [polling.py](polling.py)
- [job_orchestrator.py](job_orchestrator.py)
- [media.py](media.py)
- [sender.py](sender.py)
- [handlers/commands.py](handlers/commands.py)
- [handlers/dispatcher.py](handlers/dispatcher.py)
- [handlers/tts_handler.py](handlers/tts_handler.py)

## Связанные документы

- [../README.ru.md](../README.ru.md) — обзор репозитория
- [../core/README.ru.md](../core/README.ru.md) — общий runtime
- [../server/README.ru.md](../server/README.ru.md) — HTTP-адаптер
- [../cli/README.ru.md](../cli/README.ru.md) — CLI-адаптер
