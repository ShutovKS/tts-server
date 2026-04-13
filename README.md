# Modular Local TTS Runtime

Русская версия: [README.ru.md](README.ru.md)

## Overview

This repository provides a local text-to-speech stack with a shared modular runtime and split into three transport adapters:

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
- Local Piper voice inference through ONNX runtime using the supported `piper-tts` Python API
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
- On macOS Apple Silicon: MLX-compatible environment and MLX-ready Qwen artifacts, or ONNX-compatible environment for Piper voices
- On Linux or Windows: environment compatible with PyTorch/Transformers for Qwen, or ONNX runtime for Piper voices

## Installation

### Dependency sets

- `requirements.txt` — full local operator dependencies, including optional backend runtimes used by supported deployment lanes
- `requirements-ci.txt` — lighter CI/test dependency set for repository verification without heavyweight optional runtimes

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

### Windows PowerShell quick start

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
choco install ffmpeg -y
```

If PowerShell blocks activation scripts, run `Set-ExecutionPolicy -Scope Process Bypass` in the current shell first.

### Optional Qwen Torch lane note

The non-macOS Qwen lane depends on the official `qwen-tts` Python package used by [`TorchBackend`](core/backends/torch_backend.py). The upstream Qwen3-TTS repository documents `pip install -U qwen-tts` as the standard installation path and exposes `from qwen_tts import Qwen3TTSModel` after installation. Linux/Windows Qwen support therefore has an authoritative package path, but should still be treated as partially proven or best effort here until the full Torch lane is empirically validated on those hosts.

### Optional accelerated Qwen lane note

The repository also includes an additive `qwen_fast` backend for **custom-only** Qwen synthesis. This lane is optional, remains separate from the standard `torch` backend key, and falls back to the standard Torch Qwen path when its runtime prerequisites are not satisfied.

The pinned faster-qwen3-tts README documents the accelerated install path as:

```bash
pip install faster-qwen3-tts
```

That same upstream README documents the fast-lane prerequisites as Python 3.10+, PyTorch 2.5.1+, and an NVIDIA GPU with CUDA. In this repository, treat that install path as an operator-managed optional dependency for supported Linux/Windows CUDA hosts rather than as a universal dependency for every environment.

### Piper-specific note

The repository now includes a supported Piper lane through `piper-tts` + `onnxruntime`. On macOS the `piper-tts` wheel bundles the required `espeakbridge` runtime and `espeak-ng-data`; on other platforms you should still verify local phonemization/runtime compatibility in your deployment environment.

## Models

Place downloaded model directories in [`.models/`](.models). The supported local model IDs are registered by [`ModelRegistry`](core/services/model_registry.py:20) and described in [core/models/manifest.v1.json](core/models/manifest.v1.json).

Typical directories include:

- `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`
- `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit`
- `Qwen3-TTS-12Hz-1.7B-Base-8bit`
- `Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit`
- `Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit`
- `Qwen3-TTS-12Hz-0.6B-Base-8bit`
- `Piper-en_US-lessac-medium`

### Piper voice layout

For the built-in Piper lane, place each Piper voice in its own directory under `.models/`, for example:

- `.models/Piper-en_US-lessac-medium/model.onnx`
- `.models/Piper-en_US-lessac-medium/model.onnx.json`

You can download a Piper voice with the bundled downloader, for example:

```bash
source .venv311/bin/activate
mkdir -p .models/Piper-en_US-lessac-medium
.venv311/bin/python -m piper.download_voices en_US-lessac-medium --download-dir .models/Piper-en_US-lessac-medium
mv .models/Piper-en_US-lessac-medium/en_US-lessac-medium.onnx .models/Piper-en_US-lessac-medium/model.onnx
mv .models/Piper-en_US-lessac-medium/en_US-lessac-medium.onnx.json .models/Piper-en_US-lessac-medium/model.onnx.json
```

Windows PowerShell equivalent:

```powershell
.\.venv311\Scripts\Activate.ps1
New-Item -ItemType Directory -Force -Path ".models/Piper-en_US-lessac-medium" | Out-Null
python -m piper.download_voices en_US-lessac-medium --download-dir .models/Piper-en_US-lessac-medium
Move-Item ".models/Piper-en_US-lessac-medium/en_US-lessac-medium.onnx" ".models/Piper-en_US-lessac-medium/model.onnx" -Force
Move-Item ".models/Piper-en_US-lessac-medium/en_US-lessac-medium.onnx.json" ".models/Piper-en_US-lessac-medium/model.onnx.json" -Force
```

## Runtime self-check

Use the built-in self-check utility to inspect selected backend, per-model execution backend, missing artifacts, and host/runtime signals:

```bash
source .venv311/bin/activate
python scripts/runtime_self_check.py
```

Use strict mode in automation when you want a non-zero exit code for degraded runtime or missing assets:

```bash
python scripts/runtime_self_check.py --strict
```

When `qwen_fast` is enabled or considered, the self-check output also exposes `backend_support`, route candidates, and explicit fallback reasons so operators can see when the accelerated custom-only lane was selected, bypassed, or rejected.

For repeatable validation flows, use the automation entry point instead of assembling commands manually:

```bash
python scripts/validate_runtime.py host-matrix
python scripts/validate_runtime.py smoke-server
python scripts/validate_runtime.py smoke-server --smoke-model-id Piper-en_US-lessac-medium --expected-backend onnx
python scripts/validate_runtime.py telegram-live --bot-token "$QWEN_TTS_TELEGRAM_BOT_TOKEN"
python scripts/validate_runtime.py telegram-live --bot-token "$QWEN_TTS_TELEGRAM_BOT_TOKEN" --chat-id "$QWEN_TTS_TELEGRAM_VALIDATION_CHAT_ID" --expect-update-chat-id "$QWEN_TTS_TELEGRAM_VALIDATION_CHAT_ID" --expect-update-text "Qwen3-TTS validation ping."
```

- `host-matrix` validates the current host snapshot plus simulated `qwen_fast` optional-lane evidence.
- `smoke-server` starts a local HTTP server, waits for health probes, runs the smoke suite, and stops the server automatically.
- `smoke-server --smoke-model-id Piper-en_US-lessac-medium --expected-backend onnx` validates the Piper HTTP path explicitly through `POST /v1/audio/speech` while asserting ONNX routing for that model.
- `telegram-live` verifies real Telegram Bot API reachability and can optionally send a validation message when you also pass `--chat-id`.
- Add `--expect-update-chat-id` and optionally `--expect-update-text` when you want an opt-in dedicated-chat check that also confirms a newer matching inbound update is visible through `getUpdates` without launching the long-polling bot runtime.

## Optional GRACE CLI install

This repository can be linted locally with the optional `grace` CLI. The public GRACE packaging repository documents a Bun-based install path:

```bash
bun add -g @osovv/grace-cli
grace lint --path /path/to/grace-project
```

CI in this repository still treats `grace` as optional because we do not require Bun on every validation host.

## Running the CLI

```bash
source .venv311/bin/activate
python -m cli
```

```powershell
.\.venv311\Scripts\Activate.ps1
python -m cli
```

See [cli/README.md](cli/README.md) for adapter-specific details.

## Running the HTTP server

### Local environment

```bash
source .venv311/bin/activate
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

```powershell
.\.venv311\Scripts\Activate.ps1
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

```powershell
.\.venv311\Scripts\Activate.ps1
$env:QWEN_TTS_TELEGRAM_BOT_TOKEN="your_bot_token_here"
$env:QWEN_TTS_TELEGRAM_ALLOWED_USER_IDS="123456789,987654321"
$env:QWEN_TTS_TELEGRAM_ADMIN_USER_IDS="123456789"
$env:QWEN_TTS_TELEGRAM_RATE_LIMIT_ENABLED="true"
$env:QWEN_TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE="20"
$env:QWEN_TTS_TELEGRAM_DELIVERY_STORE_PATH=".state/telegram_delivery_store.json"
python -m telegram_bot
```

### Docker Compose

```bash
docker compose -f docker-compose.telegram-bot.yaml up --build
```

The compose scenario builds from [telegram_bot/Dockerfile](telegram_bot/Dockerfile), mounts shared model/output directories, and persists delivery metadata in the named volume declared by [docker-compose.telegram-bot.yaml](docker-compose.telegram-bot.yaml).

### Telegram token note

On the current Windows host with Docker Desktop Linux containers, the compose deployment was verified through real bot startup, Telegram API connectivity, and entry into the healthy polling loop. Full end-to-end Telegram interaction still depends on a real and valid bot token and the intended chat/user context.

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

Supported backend keys now include:

- `mlx` — Qwen3 on Apple Silicon
- `qwen_fast` — optional accelerated Qwen custom-only lane with safe fallback to `torch`
- `torch` — Qwen3 on Torch CPU/CUDA-compatible runtimes
- `onnx` — Piper local voice inference through ONNX runtime

Read the release-facing support matrix in [docs/support-matrix.md](docs/support-matrix.md) before making platform support claims.

Server-specific settings are documented in [server/README.md](server/README.md), and Telegram-specific settings are documented in [telegram_bot/README.md](telegram_bot/README.md).

## Repository map

- [README.md](README.md) / [README.ru.md](README.ru.md) — repository-level quick start
- [core/README.md](core/README.md) / [core/README.ru.md](core/README.ru.md) — shared runtime and architecture
- [server/README.md](server/README.md) / [server/README.ru.md](server/README.ru.md) — HTTP API adapter
- [telegram_bot/README.md](telegram_bot/README.md) / [telegram_bot/README.ru.md](telegram_bot/README.ru.md) — Telegram adapter
- [cli/README.md](cli/README.md) / [cli/README.ru.md](cli/README.ru.md) — interactive CLI adapter
