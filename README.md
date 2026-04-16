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
- Local OmniVoice inference through the shared Torch family-adapter lane
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
- On Linux or Windows: environment compatible with PyTorch/Transformers for Qwen and OmniVoice, or ONNX runtime for Piper voices

## Installation

### Dependency sets

- `requirements.txt` — default local operator install for the stable shared environment composed from `profiles/packs/`
- `requirements-ci.txt` — lighter CI/test dependency set for repository verification without heavyweight optional runtimes
- `profiles/packs/base/common.txt` — shared foundation for family/module/platform compilation
- `profiles/packs/module/server.txt` — server adapter dependencies
- `profiles/packs/module/telegram.txt` — Telegram adapter dependencies
- `profiles/packs/family/qwen.txt` — standard Qwen Torch lane for the shared default environment
- `profiles/packs/family/piper.txt` — Piper ONNX lane for the shared default environment
- `profiles/packs/family/qwen-fast-addon.txt` — optional accelerated Qwen lane for supported CUDA hosts
- `profiles/packs/family/omnivoice.txt` — dedicated OmniVoice family pack for a separate environment

The important operational change is that OmniVoice is not part of the default shared install anymore. It remains supported by the runtime, but should be installed in a dedicated environment when you want live execution because its upstream dependency stack may diverge from the stable Qwen environment.

The repository is now also moving toward a profile-driven launch model where each model family is treated as its own isolated runtime contour. In practice that means Qwen, Piper, and OmniVoice are expected to converge on distinct family environments rather than one implicit shared Python environment with optional package drift.

### macOS Apple Silicon

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
brew install ffmpeg
```

### Linux

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

### Interactive Windows launcher

For a guided Windows-only launch flow, use the interactive PowerShell orchestrator:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\launch-windows.ps1
```

The script reuses the profile-aware `launcher` package to resolve the selected service and family, creates or reuses `.envs/<family>`, checks whether the chosen model exists under `.models/`, optionally downloads missing assets, and then starts the selected adapter.

Important notes:

- Qwen and OmniVoice downloads use a Hugging Face snapshot flow and prompt for a repo ID only when the local model folder is missing or incomplete.
- HF tokens are requested only when needed and are kept in the current process only; the launcher does not persist them.
- Piper downloads use the documented `piper.download_voices` flow and normalize the resulting filenames into `model.onnx` and `model.onnx.json`.
- Telegram launch still requires a real bot token; the script can prompt for it transiently during the current run.

### Recommended environment layouts

#### Default shared runtime environment

Use this when you want the stable operator lane documented by `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

That environment is intended for:

- standard Qwen Torch inference
- optional `qwen_fast` route diagnostics exposed by self-checks
- Piper ONNX inference

#### Dedicated OmniVoice environment

Use a separate environment when you want to run OmniVoice locally:

```bash
python -m pip install --upgrade pip
pip install -r profiles/packs/family/omnivoice.txt
```

On the current Windows host, OmniVoice currently imports against a newer `transformers` surface than the shared Qwen environment, so isolating it in its own venv is the safe operator path.

#### Optional accelerated Qwen environment

For a CUDA-only accelerated Qwen lane, layer the fast pack onto a supported host:

```bash
python -m pip install --upgrade pip
pip install -r profiles/packs/family/qwen-fast-addon.txt
```

### Windows host prerequisites

- `ffmpeg` must be available in `PATH`
- `sox` is strongly recommended for host-side audio tooling and some upstream runtime workflows

On this host, `sox` was installed as a standalone binary in:

```text
C:\Users\shutov.k.s\AppData\Local\Programs\sox
```

If `sox --version` still fails, add that directory to your user `PATH` and open a fresh shell.

### Optional Qwen Torch lane note

The non-macOS Qwen lane depends on the official `qwen-tts` Python package used by [`TorchBackend`](core/backends/torch_backend.py). The upstream Qwen3-TTS repository documents `pip install -U qwen-tts` as the standard installation path and exposes `from qwen_tts import Qwen3TTSModel` after installation. Linux and Windows therefore both have an authoritative package path for the standard Torch lane, but the current support claim is platform-specific: Linux remains partially proven until the full Torch lane is empirically validated there, while Windows Torch Qwen support is now treated as proven by the native host validation recorded in [docs/support-matrix.md](docs/support-matrix.md).

### Optional accelerated Qwen lane note

The repository also includes an additive `qwen_fast` backend for **custom-only** Qwen synthesis. This lane is optional, remains separate from the standard `torch` backend key, and surfaces an unresolved route when its runtime prerequisites are not satisfied.

The pinned faster-qwen3-tts README documents the accelerated install path as:

```bash
pip install faster-qwen3-tts
```

That same upstream README documents the fast-lane prerequisites as Python 3.10+, PyTorch 2.5.1+, and an NVIDIA GPU with CUDA. In this repository, treat that install path as an operator-managed optional dependency for supported Linux/Windows CUDA hosts rather than as a universal dependency for every environment.

### Piper-specific note

The repository now includes a supported Piper lane through `piper-tts` + `onnxruntime`. On macOS the `piper-tts` wheel bundles the required `espeakbridge` runtime and `espeak-ng-data`; on other platforms you should still verify local phonemization/runtime compatibility in your deployment environment.

### OmniVoice-specific note

OmniVoice is integrated as a **Torch-backed model family** rather than as a separate backend key. The upstream project documents a Python package path built around `pip install omnivoice` and `from omnivoice import OmniVoice`. In this repository, treat OmniVoice as an operator-managed optional dependency that shares the existing `torch` backend selection lane, but install it in a **dedicated environment** when you want live execution on Linux/Windows hosts.

Under the new profile architecture direction, the intended operator contour is effectively:

```text
family=omnivoice
pack_refs.base=[common]
pack_refs.family=[omnivoice]
isolated_env_name=omnivoice
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
- `OmniVoice`
- `Piper-en_US-lessac-medium`

### OmniVoice model layout

For the integrated OmniVoice family, place the downloaded Hugging Face snapshot or exported local model inside `.models/OmniVoice`. The current manifest validates a minimal root config / weights / tokenizer surface, and the published upstream model repository also includes a required `audio_tokenizer/` subtree. A practical local mirror should therefore contain at least:

- `.models/OmniVoice/config.json`
- `.models/OmniVoice/model.safetensors` or `.models/OmniVoice/model.safetensors.index.json`
- `.models/OmniVoice/tokenizer_config.json` or `.models/OmniVoice/tokenizer.json`
- `.models/OmniVoice/chat_template.jinja`
- `.models/OmniVoice/audio_tokenizer/config.json`
- `.models/OmniVoice/audio_tokenizer/model.safetensors`
- `.models/OmniVoice/audio_tokenizer/preprocessor_config.json`

If the download lands in a Hugging Face cache snapshot layout, keep the nested `snapshots/<revision>/...` structure intact — [`TorchBackend`](core/backends/torch_backend.py) already resolves that layout automatically.

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

When `qwen_fast` is enabled or considered, the self-check output also exposes `backend_support`, route candidates, and explicit rejection reasons so operators can see when the accelerated custom-only lane was selected, left unresolved, or rejected.

For OmniVoice, the same self-check output now shows it as a **Torch-routed family entry**. It does not introduce a new backend key; instead, it appears as a model-family item whose `execution_backend` is expected to resolve to `torch` when its local artifacts and optional Python packages are present.

For repeatable validation flows, use the automation entry point instead of assembling commands manually:

```bash
python scripts/validate_runtime.py host-matrix
python scripts/validate_runtime.py smoke-server
python scripts/validate_runtime.py smoke-server --smoke-model-id Piper-en_US-lessac-medium --expected-backend onnx
python scripts/validate_runtime.py smoke-server --smoke-model-id OmniVoice-Custom --expected-backend torch
python scripts/validate_runtime.py telegram-live --bot-token "$QWEN_TTS_TELEGRAM_BOT_TOKEN"
python scripts/validate_runtime.py telegram-live --bot-token "$QWEN_TTS_TELEGRAM_BOT_TOKEN" --chat-id "$QWEN_TTS_TELEGRAM_VALIDATION_CHAT_ID" --expect-update-chat-id "$QWEN_TTS_TELEGRAM_VALIDATION_CHAT_ID" --expect-update-text "Qwen3-TTS validation ping."
```

- `host-matrix` validates the current host snapshot plus simulated `qwen_fast` optional-lane evidence.
- `smoke-server` starts a local HTTP server, waits for health probes, runs the smoke suite, and stops the server automatically. This remains the baseline host-mode HTTP orchestrator for live validation.
- `smoke-server --smoke-model-id Piper-en_US-lessac-medium --expected-backend onnx` validates the Piper HTTP path explicitly through `POST /v1/audio/speech` while asserting ONNX routing for that model.
- `smoke-server --smoke-model-id OmniVoice-Custom --expected-backend torch` validates the OmniVoice HTTP path through the shared Torch family lane.
- `telegram-live` verifies real Telegram Bot API reachability and can optionally send a validation message when you also pass `--chat-id`.
- Add `--expect-update-chat-id` and optionally `--expect-update-text` when you want an opt-in dedicated-chat check that also confirms a newer matching inbound update is visible through `getUpdates` without launching the long-polling bot runtime.

For the V1 HTTP adapter, read the host lane as layered evidence rather than as a smoke-only shortcut: `python -m pytest -m "unit or integration"` and named integration cases provide deterministic API contract coverage, then `python scripts/validate_runtime.py smoke-server` adds live startup/readiness, sync audio, async job verification, scenario-aware backend/model-route assertions, documented skip behavior, and retained `server_log_path` evidence.

`python -m pytest -m "unit or integration"` remains the canonical first step. If a deterministic-baseline failure appears during that check, record it separately from runtime evidence so the adapter, smoke, Docker, Telegram, and CLI lanes stay easy to interpret.

There is also an **optional local-only deep-validation lane** for operators who want stronger real-model evidence on a specific machine. In V1, that lane starts with `python scripts/validate_runtime.py host-matrix` and/or `python scripts/runtime_self_check.py`, then reuses the existing representative `smoke-server` targets for Qwen, OmniVoice, and Piper only when the corresponding local assets and runtime packs are installed. Treat outcomes explicitly as ready/passed, missing assets, corrupt or incomplete assets, unsupported hardware/backend, missing optional dependency packs, or intentionally skipped optional features. This lane is advisory and local-only; it is not a CI or release requirement.

There is also an **optional advisory-only LLM-assisted evaluation lane** for teams that want semantic review of completed evidence packs. In V1, that lane must read only persisted repository-local artifacts already produced by the existing validation flows, such as retained `server_log_path` logs, `.sisyphus/evidence/server-docker-log.txt`, Docker health/model JSON artifacts, retained `telegram-live` summaries, `.sisyphus/evidence/telegram-docker-log.txt`, and `.sisyphus/evidence/cli-*.txt` transcripts. It must not read ephemeral terminal state or live process internals, it must not create a second evidence framework, and it remains strictly advisory: deterministic failures and structured-artifact failures stay failures even if an LLM summary sounds sympathetic or inconclusive.

V1 Docker validation is intentionally narrower than host validation. Only the HTTP server and Telegram bot have Docker parity lanes, both using the checked-in compose files: `docker compose -f docker-compose.server.yaml up --build -d server` with explicit `/health/live`, `/health/ready`, and `/api/v1/models` probes retained under `.sisyphus/evidence/server-docker-health-live.json`, `.sisyphus/evidence/server-docker-health-ready.json`, and `.sisyphus/evidence/server-docker-models.json` plus `.sisyphus/evidence/server-docker-log.txt`; and `docker compose -f docker-compose.telegram-bot.yaml up --build -d telegram-bot` paired with `python scripts/validate_runtime.py telegram-live ...`, `.sisyphus/evidence/telegram-docker-log.txt`, and explicit `docker compose ... down --remove-orphans` teardown. CLI support claims remain host/transcript based and do not require Docker in V1.

For the V1 CLI lane, keep automation intentionally narrow: `python -m cli` with scripted stdin `q\n` proves launchability and runtime wiring, while the complex interactive custom/design/clone/playback paths remain transcript-captured evidence stored under `.sisyphus/evidence/cli-launchability-transcript.txt`, `.sisyphus/evidence/cli-custom-voice-transcript.txt`, `.sisyphus/evidence/cli-design-session-transcript.txt`, `.sisyphus/evidence/cli-clone-manager-transcript.txt`, and `.sisyphus/evidence/cli-playback-transcript.txt`.

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

For V1 Docker validation, use the detached compose lane rather than treating this section as launch-only documentation: `docker compose -f docker-compose.server.yaml up --build -d server`, probe `/health/live`, `/health/ready`, and `/api/v1/models` into `.sisyphus/evidence/server-docker-health-live.json`, `.sisyphus/evidence/server-docker-health-ready.json`, and `.sisyphus/evidence/server-docker-models.json`, retain `docker compose -f docker-compose.server.yaml logs --no-color server > .sisyphus/evidence/server-docker-log.txt`, then stop the lane with `docker compose -f docker-compose.server.yaml down --remove-orphans`.

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

This is a documented Telegram Docker lane, but this README does not claim retained host proof for startup, Bot API reachability, or polling success on this machine.

For V1 Docker validation, use the detached compose lane and pair it with the existing live API checker instead of inventing a second Bot API probe: `docker compose -f docker-compose.telegram-bot.yaml up --build -d telegram-bot`, run `python scripts/validate_runtime.py telegram-live --bot-token "$QWEN_TTS_TELEGRAM_BOT_TOKEN"` (plus the optional `--chat-id` / `--expect-update-*` flags when you have a dedicated validation chat), retain `docker compose -f docker-compose.telegram-bot.yaml logs --no-color telegram-bot > .sisyphus/evidence/telegram-docker-log.txt`, and finish with `docker compose -f docker-compose.telegram-bot.yaml down --remove-orphans`.

### Telegram token note

On the current Windows host with Docker Desktop Linux containers, the Telegram compose lane remains documented and runnable, but the retained evidence set here does not prove startup, Telegram API connectivity, or the healthy polling loop on this machine. Full end-to-end Telegram interaction still depends on a real and valid bot token and the intended chat/user context.

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
- `qwen_fast` — optional accelerated Qwen custom-only lane with explicit readiness and route diagnostics
- `torch` — Qwen3 and OmniVoice on Torch CPU/CUDA-compatible runtimes
- `onnx` — Piper local voice inference through ONNX runtime

Read the release-facing support matrix in [docs/support-matrix.md](docs/support-matrix.md) before making platform support claims.

Treat [docs/support-matrix.md](docs/support-matrix.md) and [docs/verification-plan.xml](docs/verification-plan.xml) as paired release evidence docs. If a validation command, critical flow, critical scenario, or evidence level changes, update both files together so the support claim and its executable proof stay in sync.

Future validation scenarios should extend that same evidence model. Keep new scenario onboarding tied to the existing support matrix and verification plan, and avoid adding a second registry, benchmarking lane, latency-certification lane, or subjective audio-quality certification path.

Server-specific settings are documented in [server/README.md](server/README.md), and Telegram-specific settings are documented in [telegram_bot/README.md](telegram_bot/README.md).

## Repository map

- [README.md](README.md) / [README.ru.md](README.ru.md) — repository-level quick start
- [core/README.md](core/README.md) / [core/README.ru.md](core/README.ru.md) — shared runtime and architecture
- [server/README.md](server/README.md) / [server/README.ru.md](server/README.ru.md) — HTTP API adapter
- [telegram_bot/README.md](telegram_bot/README.md) / [telegram_bot/README.ru.md](telegram_bot/README.ru.md) — Telegram adapter
- [cli/README.md](cli/README.md) / [cli/README.ru.md](cli/README.ru.md) — interactive CLI adapter
