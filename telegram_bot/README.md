# Telegram bot module — long-polling adapter

Русская версия: [README.ru.md](README.ru.md)

## Purpose

[telegram_bot/](./) is a separate transport adapter built on top of the shared runtime from [../core/README.md](../core/README.md). It exposes local TTS through Telegram commands and runs as its own polling process.

## Current status

The bot is maintained as a standalone deployment unit with its own container image and compose scenario.

Implemented:

- private chats only
- `/start`, `/help`, `/tts`, `/design`, `/clone`
- async submit / poll / deliver workflow
- delivery metadata persistence and recovery after restart
- startup self-checks
- retry with backoff for polling and delivery
- structured observability and degraded-state tracking
- separate Docker image in [Dockerfile](Dockerfile)
- separate compose scenario in [../docker-compose.telegram-bot.yaml](../docker-compose.telegram-bot.yaml)

Not implemented:

- webhook mode
- group chat support
- inline queries
- callback buttons
- conversational memory

## Running locally

```bash
source .venv311/bin/activate
export QWEN_TTS_TELEGRAM_BOT_TOKEN="your_bot_token_here"
export QWEN_TTS_TELEGRAM_ALLOWED_USER_IDS="123456789,987654321"
export QWEN_TTS_TELEGRAM_ADMIN_USER_IDS="123456789"
export QWEN_TTS_TELEGRAM_RATE_LIMIT_ENABLED=true
export QWEN_TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE=20
export QWEN_TTS_TELEGRAM_DELIVERY_STORE_PATH=.state/telegram_delivery_store.json
python -m telegram_bot
```

## Running with Docker Compose

```bash
docker compose -f docker-compose.telegram-bot.yaml up --build
```

The compose file [../docker-compose.telegram-bot.yaml](../docker-compose.telegram-bot.yaml) builds from [Dockerfile](Dockerfile) with repository-root context `.` and mounts:

- [../.models](../.models)
- [../.outputs](../.outputs)
- [../.uploads](../.uploads)
- [../.voices](../.voices)
- named volume `telegram_bot_state` at `/app/.state`

The image includes `ffmpeg` and persists delivery metadata at `/app/.state/telegram_delivery_store.json`.

## Telegram token limitation

Container image build and bot process startup were verified, but full external integration requires a real Telegram bot token. Without a valid token, the bot cannot complete Telegram API connectivity checks or process live updates.

## Commands

| Command | Description |
|---|---|
| `/start` | Show welcome message |
| `/help` | Show help and available speakers |
| `/tts` | Synthesize text with optional speaker and speed |
| `/design` | Synthesize using a natural-language voice description |
| `/clone` | Clone a voice from replied reference audio |

### `/tts` syntax

```text
/tts [-- speaker=<speaker>] [-- speed=<speed>] -- <text>
```

### `/design` syntax

```text
/design <voice_description> -- <text>
```

### `/clone` syntax

```text
/clone [-- ref=<transcript>] -- <text>
```

For `/clone`, the command must be sent as a reply to a message containing reference audio.

## Supported clone media

- `voice`
- `audio`
- `document` containing a supported audio format

Supported formats: WAV, MP3, FLAC, OGG, WebM, M4A, MP4.

## Configuration

Telegram-specific settings are defined by [`TelegramSettings`](config.py:42).

Important environment variables:

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

Shared core variables from [../core/README.md](../core/README.md) also apply.

## Operational notes

- The bot only handles private chats.
- `ffmpeg` must be available for audio conversion and clone flows.
- Delivery metadata is persisted so completed jobs can still be delivered after a restart.
- Startup self-checks validate token presence, runtime wiring, backend availability, and Telegram API connectivity.

## Deployment docs

Additional deployment artifacts:

- [../docs/telegram-bot-deployment.md](../docs/telegram-bot-deployment.md)
- [../docs/telegram-bot.service](../docs/telegram-bot.service)

## Related source files

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

## Related docs

- [../README.md](../README.md) — repository overview
- [../core/README.md](../core/README.md) — shared runtime
- [../server/README.md](../server/README.md) — HTTP adapter
- [../cli/README.md](../cli/README.md) — CLI adapter
