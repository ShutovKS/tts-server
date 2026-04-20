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

```powershell
.\.venv311\Scripts\Activate.ps1
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

### CORS note for separate frontend modules

The HTTP server no longer owns the demo UI or any other frontend assets. A standalone local demo frontend now lives in the repository root under `frontend_demo/` and calls the API over HTTP.

To support that local split, the server exposes development CORS for:

- `http://127.0.0.1:8030`
- `http://localhost:8030`

For a fully working clone demo on this host, run the server in the qwen isolated contour rather than the generic base runtime. The launcher-approved path is:

```powershell
python -m launcher inspect --family qwen --module server
python -m launcher check-env --family qwen --module server
$env:QWEN_TTS_HOST='127.0.0.1'
$env:QWEN_TTS_PORT='8020'
& .\.envs\qwen\Scripts\python.exe -m server
```

If readiness reports `selected_backend=onnx` and `supports_clone=false`, the standalone demo will intentionally refuse to submit clone requests because that contour cannot complete the advertised flow.

### Health endpoints

- `GET /health/live`
- `GET /health/ready`

### Model discovery

- `GET /api/v1/models`

Model discovery now exposes family metadata, supported capabilities, selected backend, per-model execution backend, missing artifacts, and route explanations for mixed-family deployments. For Qwen models this may include the optional `qwen_fast` route candidate, which remains unresolved when its fast-path prerequisites are not satisfied.

### Synchronous TTS endpoints

- `POST /v1/audio/speech`
- `POST /api/v1/tts/custom`
- `POST /api/v1/tts/design`
- `POST /api/v1/tts/clone`

All TTS request payloads accept `language` with default `auto`. Clone form endpoints also accept `language=auto` when omitted.

### Asynchronous job endpoints

- `POST /v1/audio/speech/jobs`
- `POST /api/v1/tts/custom/jobs`
- `POST /api/v1/tts/design/jobs`
- `POST /api/v1/tts/clone/jobs`
- `GET /api/v1/tts/jobs/{job_id}`
- `GET /api/v1/tts/jobs/{job_id}/result`
- `POST /api/v1/tts/jobs/{job_id}/cancel`

Async submission endpoints accept the `Idempotency-Key` header.
`language` participates in async idempotency fingerprints, so different language selections create different jobs.

## Important runtime behavior

- The adapter returns completed audio as regular HTTP response bodies.
- `QWEN_TTS_ENABLE_STREAMING` remains a configuration flag, but it does not turn completed audio responses into a streaming transport automatically.
- Clone uploads are staged in `QWEN_TTS_UPLOAD_STAGING_DIR` and cleaned up after request processing.
- Oversized text requests fail with the standard validation error.
- Inference timeouts return `request_timeout` with HTTP `504`.
- Unsupported model/family combinations now return controlled `model_capability_not_supported` errors with explicit capability metadata.
- Runtime capability bindings are now expected to come from `QWEN_TTS_ACTIVE_FAMILY`, `QWEN_TTS_DEFAULT_CUSTOM_MODEL`, `QWEN_TTS_DEFAULT_DESIGN_MODEL`, and `QWEN_TTS_DEFAULT_CLONE_MODEL` rather than from implicit `.models/` inspection.
- When a requested mode has no active runtime binding, the expected API behavior is a controlled unsupported-mode response rather than an internal failure.
- `GET /health/ready` now includes host snapshot, mixed-backend routing summaries, `runtime_capability_map`, and per-mode `capability_status` so operators and thin adapters can distinguish artifact availability from runtime-bound capability availability.
- The accelerated `qwen_fast` lane is additive and optional; readiness and model payloads may show it as a rejected route candidate with an explicit rejection reason when the selected fast lane is unavailable.
- Qwen clone requests are sensitive to reference-audio quality and transcript alignment. Provide `ref_text` only when it exactly matches the spoken content of the uploaded reference audio. The server rejects implausibly short clone outputs instead of returning a misleading near-empty WAV.

## Host-mode validation lane

For V1 host validation, keep `python scripts/validate_runtime.py smoke-server` as the baseline HTTP orchestrator. That command is meant to sit on top of deterministic adapter checks rather than replace them: run `python -m pytest -m "unit or integration"` and the named HTTP integration cases for contract coverage, then use `smoke-server` for live `/health/live` and `/health/ready` startup proof, synchronous audio generation, async job submit/status/result verification, and scenario-aware backend assertions for the selected smoke target.

`python -m pytest -m "unit or integration"` remains the canonical first step for this lane. If that check reports a deterministic-baseline failure, record it separately from host runtime evidence so the pytest surface and the smoke-server lane stay easy to distinguish.

For the standalone frontend demo specifically, pair the smoke-server lane with a qwen-contour browser or HTTP proof that the `frontend_demo/` page loads, reads `/health/ready`, and submits `POST /api/v1/tts/clone` successfully through a clone-capable backend.

The host lane is target-aware by design. The existing smoke suite reuses one path for default Qwen custom validation plus `OmniVoice-Custom --expected-backend torch` and `Piper-en_US-lessac-medium --expected-backend onnx`; those backend expectations should stay tied to the chosen smoke target instead of being treated as one global backend rule.

When you capture host evidence, retain the `server_log_path` returned by `python scripts/validate_runtime.py smoke-server` and pair it with stable response/readiness fields such as `x-model-id`, `x-backend-id`, request ids, job ids, and readiness routing metadata. Skip host smoke only for documented prerequisite failures such as missing `ffmpeg`, unreachable local server, absent runtime-ready models, missing model folders, or missing optional runtime endpoints/dependencies for the selected target.

## Docker-mode validation lane

Docker is an additive V1 parity lane for the HTTP adapter, not a second copy of the host smoke flow. Keep parity at the scenario/assertion level: the compose deployment must prove healthy startup and HTTP discovery behavior, but it does not need to run the exact same command sequence as `smoke-server`.

1. Start the checked-in compose scenario in detached mode: `docker compose -f docker-compose.server.yaml up --build -d server`.
2. Probe the containerized adapter surface explicitly from inside the service and retain those JSON artifacts under `.sisyphus/evidence/`:
   - `docker compose -f docker-compose.server.yaml exec -T server python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health/live').read().decode())" > .sisyphus/evidence/server-docker-health-live.json`
   - `docker compose -f docker-compose.server.yaml exec -T server python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health/ready').read().decode())" > .sisyphus/evidence/server-docker-health-ready.json`
   - `docker compose -f docker-compose.server.yaml exec -T server python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/models').read().decode())" > .sisyphus/evidence/server-docker-models.json`
3. Retain raw Docker logs as the canonical container artifact: `docker compose -f docker-compose.server.yaml logs --no-color server > .sisyphus/evidence/server-docker-log.txt`.
4. Tear down the lane explicitly with `docker compose -f docker-compose.server.yaml down --remove-orphans`.

Skip the Docker HTTP lane only when Docker/Compose is unavailable, the image cannot build or start, the service never reaches a probeable HTTP state, or the compose-mounted local runtime assets are intentionally absent. Model-dependent sync/async speech smoke remains a host-lane concern in V1, so Docker parity should not be rejected just because it does not replay every host-only smoke branch.

## Configuration

Server-specific settings extend [`CoreSettings`](../core/config.py:27) through [`ServerSettings`](bootstrap.py:17).

Important variables:

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

Shared variables from [../core/README.md](../core/README.md) also apply.

If you want to disable the accelerated Qwen lane entirely, set `QWEN_TTS_QWEN_FAST_ENABLED=false`. This affects only the optional fast lane and does not disable the standard Torch Qwen path.

When enabling this lane on a supported Linux/Windows CUDA host, install the optional accelerated runtime separately with `pip install faster-qwen3-tts` and follow the upstream prerequisites (Python 3.10+, PyTorch 2.5.1+, NVIDIA GPU with CUDA).

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
