# Server module — FastAPI HTTP API

Русская версия: [README.ru.md](README.ru.md)

## Purpose

[server/](./) is the HTTP transport adapter for the shared TTS runtime from [../core/README.md](../core/README.md). It exposes synchronous and asynchronous API endpoints, health probes, model discovery, and unified error responses.

## Entry points

- [__main__.py](__main__.py) — package entry point
- [app.py](app.py) — FastAPI composition root
- [`build_server_runtime()`](bootstrap.py:41) in [bootstrap.py](bootstrap.py)

## Running locally

```bash
source .venv311/bin/activate
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

## Running with Docker Compose

```bash
docker compose -f docker-compose.server.yaml up --build
```

The compose file [../docker-compose.server.yaml](../docker-compose.server.yaml) builds the image from [Dockerfile](Dockerfile) using repository-root context `.` and mounts:

- [../.models](../.models)
- [../.outputs](../.outputs)
- [../.uploads](../.uploads)
- [../.voices](../.voices)

This replaces the old root-level compose layout.

## Container image layout

[Dockerfile](Dockerfile) installs Python dependencies, `ffmpeg`, and copies the shared sources needed by the adapter:

- [../core/](../core/)
- [./](./)
- [../cli/](../cli/)
- [../requirements.txt](../requirements.txt)
- [../.env.example](../.env.example)

The image exposes port `8000` and uses `python -m uvicorn server:app --host 0.0.0.0 --port 8000` as the default command.

## API surface

### Health endpoints

- `GET /health/live`
- `GET /health/ready`

### Model discovery

- `GET /api/v1/models`

### Synchronous TTS endpoints

- `POST /v1/audio/speech`
- `POST /api/v1/tts/custom`
- `POST /api/v1/tts/design`
- `POST /api/v1/tts/clone`

### Asynchronous job endpoints

- `POST /v1/audio/speech/jobs`
- `POST /api/v1/tts/custom/jobs`
- `POST /api/v1/tts/design/jobs`
- `POST /api/v1/tts/clone/jobs`
- `GET /api/v1/tts/jobs/{job_id}`
- `GET /api/v1/tts/jobs/{job_id}/result`
- `POST /api/v1/tts/jobs/{job_id}/cancel`

Async submission endpoints accept the `Idempotency-Key` header.

## Important runtime behavior

- The adapter returns completed audio as regular HTTP response bodies.
- `QWEN_TTS_ENABLE_STREAMING` remains a configuration flag, but it does not turn completed audio responses into a streaming transport automatically.
- Clone uploads are staged in `QWEN_TTS_UPLOAD_STAGING_DIR` and cleaned up after request processing.
- Oversized text requests fail with the standard validation error.
- Inference timeouts return `request_timeout` with HTTP `504`.

## Configuration

Server-specific settings extend [`CoreSettings`](../core/config.py:27) through [`ServerSettings`](bootstrap.py:17).

Important variables:

- `QWEN_TTS_HOST`
- `QWEN_TTS_PORT`
- `QWEN_TTS_LOG_LEVEL`
- `QWEN_TTS_DEFAULT_SAVE_OUTPUT`
- `QWEN_TTS_ENABLE_STREAMING`
- `QWEN_TTS_MAX_UPLOAD_SIZE_BYTES`
- `QWEN_TTS_MAX_INPUT_TEXT_CHARS`
- `QWEN_TTS_REQUEST_TIMEOUT_SECONDS`
- `QWEN_TTS_INFERENCE_BUSY_STATUS_CODE`

Shared variables from [../core/README.md](../core/README.md) also apply.

## Related source files

- [api/routes_health.py](api/routes_health.py)
- [api/routes_models.py](api/routes_models.py)
- [api/routes_tts.py](api/routes_tts.py)
- [api/errors.py](api/errors.py)
- [api/auth.py](api/auth.py)
- [api/policies.py](api/policies.py)
- [schemas/audio.py](schemas/audio.py)

## Related docs

- [../README.md](../README.md) — repository overview
- [../core/README.md](../core/README.md) — shared runtime
- [../telegram_bot/README.md](../telegram_bot/README.md) — Telegram adapter
- [../cli/README.md](../cli/README.md) — CLI adapter
