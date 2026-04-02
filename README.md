# Qwen3 TTS on Apple Silicon

Русская версия: [README.ru.md](README.ru.md)

## Overview

This repository provides a local text-to-speech stack built around Qwen3 TTS models and split into three transport adapters:

- [server/](server/README.md) — FastAPI HTTP API
- [telegram_bot/](telegram_bot/README.md) — Telegram bot based on long polling
- [cli/](cli/README.md) — interactive local CLI
- [core/](core/README.md) — shared runtime, model registry, backends, jobs, and observability

The repository layout was updated so Docker assets now live next to the components they build:

- server image: [server/Dockerfile](server/Dockerfile)
- Telegram bot image: [telegram_bot/Dockerfile](telegram_bot/Dockerfile)
- server compose scenario: [docker-compose.server.yaml](docker-compose.server.yaml)
- Telegram bot compose scenario: [docker-compose.telegram-bot.yaml](docker-compose.telegram-bot.yaml)

Legacy root-level Docker assets such as the removed `Dockerfile` and `compose.yaml` are no longer part of the project.

## Features

- Local Qwen3 TTS inference with shared runtime from [core/](core/README.md)
- OpenAI-style speech endpoint `POST /v1/audio/speech`
- Extended HTTP endpoints for custom voice, voice design, and voice cloning
- Telegram bot commands `/start`, `/help`, `/tts`, `/design`, `/clone`
- Interactive CLI for local synthesis workflows
- Optional async job flow in the HTTP server
- Structured logging, request correlation, and operational metrics
- Isolated staging directory for uploaded clone references in [`.uploads/`](.uploads)
- Optional output persistence in [`.outputs/`](.outputs)

## Requirements

- Python 3.11+
- `ffmpeg` available in `PATH`
- Local model directories available in [`.models/`](.models)
- On macOS Apple Silicon: MLX-compatible environment and MLX-ready model artifacts
- On Linux or Windows: environment compatible with PyTorch/Transformers

## Installation

### macOS Apple Silicon

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
brew install ffmpeg
```

### Linux or Windows

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Models

Place downloaded model directories in [`.models/`](.models). The supported local model IDs are registered by [`ModelRegistry`](core/services/model_registry.py:20) and described in [core/models/manifest.v1.json](core/models/manifest.v1.json).

Typical directories include:

- `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`
- `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit`
- `Qwen3-TTS-12Hz-1.7B-Base-8bit`
- `Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit`
- `Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit`
- `Qwen3-TTS-12Hz-0.6B-Base-8bit`

## Running the CLI

```bash
source .venv311/bin/activate
python -m cli
```

See [cli/README.md](cli/README.md) for adapter-specific details.

## Running the HTTP server

### Local environment

```bash
source .venv311/bin/activate
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### Docker Compose

```bash
docker compose -f docker-compose.server.yaml up --build
```

The compose scenario builds from [server/Dockerfile](server/Dockerfile) with repository-root build context, mounts shared working directories, and exposes port `8000` by default.

See [server/README.md](server/README.md) for endpoints, async jobs, and configuration details.

## Running the Telegram bot

### Local environment

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

### Docker Compose

```bash
docker compose -f docker-compose.telegram-bot.yaml up --build
```

The compose scenario builds from [telegram_bot/Dockerfile](telegram_bot/Dockerfile), mounts shared model/output directories, and persists delivery metadata in the named volume declared by [docker-compose.telegram-bot.yaml](docker-compose.telegram-bot.yaml).

### Telegram token note

Container startup and basic bot process startup were validated, but full end-to-end Telegram interaction still depends on a real and valid bot token. Without it, the bot cannot complete external API checks or process live updates.

See [telegram_bot/README.md](telegram_bot/README.md) for command syntax, operational notes, and deployment details.

## Key environment variables

Shared settings are parsed by [`CoreSettings.from_env()`](core/config.py:112). Common variables include:

- `QWEN_TTS_MODELS_DIR`
- `QWEN_TTS_OUTPUTS_DIR`
- `QWEN_TTS_VOICES_DIR`
- `QWEN_TTS_UPLOAD_STAGING_DIR`
- `QWEN_TTS_BACKEND`
- `QWEN_TTS_BACKEND_AUTOSELECT`
- `QWEN_TTS_SAMPLE_RATE`
- `QWEN_TTS_MAX_INPUT_TEXT_CHARS`

Server-specific settings are documented in [server/README.md](server/README.md), and Telegram-specific settings are documented in [telegram_bot/README.md](telegram_bot/README.md).

## Repository map

- [README.md](README.md) / [README.ru.md](README.ru.md) — repository-level quick start
- [core/README.md](core/README.md) / [core/README.ru.md](core/README.ru.md) — shared runtime and architecture
- [server/README.md](server/README.md) / [server/README.ru.md](server/README.ru.md) — HTTP API adapter
- [telegram_bot/README.md](telegram_bot/README.md) / [telegram_bot/README.ru.md](telegram_bot/README.ru.md) — Telegram adapter
- [cli/README.md](cli/README.md) / [cli/README.ru.md](cli/README.ru.md) — interactive CLI adapter
