# Server module — FastAPI HTTP API

Русская версия: [README.ru.md](README.ru.md)

## Purpose

[server/](./) is the HTTP transport adapter for the shared TTS runtime from [../core/README.md](../core/README.md). It exposes synchronous and asynchronous API endpoints, health probes, model discovery, and unified error responses.

## Entry points

- [__main__.py](__main__.py) — package entry point
- [app.py](app.py) — FastAPI composition root
- [`build_server_runtime()`](bootstrap.py:41) in [bootstrap.py](bootstrap.py)

## Running locally

### Phase 1 deployment contract

Phase 1 is a single internal HTTP server deployment. It is the sole runtime and model host for the phase, and it is expected to sit behind a network perimeter rather than be exposed as a public-internet service.

Keep these concerns separate:

- `TTS_HOST` and `TTS_PORT` define the listener bind address.
- `TTS_CORS_ALLOWED_ORIGINS` defines which browser origins may call the server.
- client code must be configured with the server base URL and must not guess capability from local model folders.

Phase 1 uses No Auth. The security boundary is the network placement of the server, not a user-facing authentication layer.

The authoritative server-side capability bindings remain `TTS_ACTIVE_FAMILY`, `TTS_DEFAULT_CUSTOM_MODEL`, `TTS_DEFAULT_DESIGN_MODEL`, and `TTS_DEFAULT_CLONE_MODEL`. Those bindings tell the running server which modes are available. A `.models/` directory on some other machine does not make that machine a valid client runtime.

For any downstream integrator, [../docs/server-http-contract.md](../docs/server-http-contract.md) is the canonical HTTP server contract and the source of truth for Phase 1. If a client, deployment note, or operator runbook disagrees with that document, the contract document wins.

Cutover stays server first. Bring up the central HTTP server, verify `/health/live`, `/health/ready`, and `/api/v1/models`, then point Telegram and any other remote client at that server base URL. If a client cutover fails, move the clients back to a known good server first, recover the server, then repoint clients after readiness returns.

Clients must not infer runtime ownership from local `.models/` folders or silently fall back to local inference when the server is unavailable. They should fail with a clear remote-contract error instead.

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

For Phase 1 operators, the compose path is the documented central-server deployment shape. Remote clients should point at the published server base URL and should not depend on local model folders.

## Container image layout

[Dockerfile](Dockerfile) installs Python dependencies, `ffmpeg`, and copies the shared sources needed by the adapter:

- [../core/](../core/)
- [./](./)
- [../cli/](../cli/)
- [../requirements.txt](../requirements.txt)
- [../.env.example](../.env.example)

The image exposes port `8000` and uses `python -m uvicorn server:app --host 0.0.0.0 --port 8000` as the default command.

## API surface

### Phase 1 canonical remote contract

Phase 1 clients should treat the HTTP server as the canonical remote contract for the repository.

- Canonical sync TTS endpoint: `POST /v1/audio/speech`
- Canonical async submit endpoint for the same sync contract: `POST /v1/audio/speech/jobs`
- Canonical async job resources: `GET /api/v1/tts/jobs/{job_id}`, `GET /api/v1/tts/jobs/{job_id}/result`, `POST /api/v1/tts/jobs/{job_id}/cancel`
- Canonical control-plane endpoints: `GET /health/live`, `GET /health/ready`, `GET /api/v1/models`

The extended `/api/v1/tts/custom`, `/api/v1/tts/design`, and `/api/v1/tts/clone` families remain supported because they expose server-native synthesis modes that are not represented by the OpenAI-compatible request shape. They are part of the same HTTP product boundary, but `/v1/audio/speech` is the canonical framing clients should prefer for preset-speaker synchronous and asynchronous speech generation.

Legacy-compatible routes may remain in the codebase, but this README and [../docs/server-http-contract.md](../docs/server-http-contract.md) define the authoritative Phase 1 client contract.

### CORS note for separate frontend modules

The HTTP server no longer owns the demo UI or any other frontend assets. A standalone local demo frontend now lives in the repository root under `frontend_demo/` and calls the API over HTTP.

That frontend is a client of the server. It needs a configured API base URL and server-approved CORS origins, but it does not host or infer runtime capability on its own.

To support that local split, the server exposes development CORS for these default origins when `TTS_CORS_ALLOWED_ORIGINS` is unset:

- `http://127.0.0.1:8030`
- `http://localhost:8030`
- `http://0.0.0.0:8030`

For forwarded or public-host browser access, set `TTS_CORS_ALLOWED_ORIGINS` explicitly as a comma-separated list, for example:

```text
TTS_CORS_ALLOWED_ORIGINS=http://127.0.0.1:8030,http://localhost:8030,http://0.0.0.0:8030,http://185.186.142.205:8030,https://split-tts.drive-vr.ru
```

That setting controls browser CORS only; it does not change the server bind address. Keep using `TTS_HOST` and `TTS_PORT` for the actual listener.

Remote clients still need the server base URL. CORS decides whether the browser may talk to that URL, but it does not choose the URL for the client.

If the frontend is served over HTTPS, the browser also requires the API URL to be HTTPS. CORS alone does not permit `https://...` pages to call `http://...` APIs; use an HTTPS reverse proxy or publish the API over HTTPS directly.

For a fully working clone demo on this host, run the server in the qwen isolated contour rather than the generic base runtime. The launcher-approved path is:

```powershell
python -m launcher inspect --family qwen --module server
python -m launcher check-env --family qwen --module server
$env:TTS_HOST='127.0.0.1'
$env:TTS_PORT='8020'
& .\.envs\qwen\Scripts\python.exe -m server
```

If readiness reports `selected_backend=onnx` and `supports_clone=false`, the standalone demo will intentionally refuse to submit clone requests because that contour cannot complete the advertised flow.

### Health endpoints

- `GET /health/live`
- `GET /health/ready`

`GET /health/live` is a process liveness probe and returns a minimal `status=ok` payload when the server process is running.

`GET /health/ready` is the authoritative readiness probe. It returns `status` plus machine-readable `checks` covering model availability, ffmpeg availability, configured runtime capability bindings, selected backend metadata, and per-capability readiness diagnostics.

### Model discovery

- `GET /api/v1/models`

Model discovery now exposes family metadata, supported capabilities, selected backend, per-model execution backend, missing artifacts, and route explanations for mixed-family deployments. For Qwen models this may include the optional `qwen_fast` route candidate, which remains unresolved when its fast-path prerequisites are not satisfied.

For Phase 1 clients, `GET /api/v1/models` is the authoritative machine-readable discovery surface for:

- stable public model IDs
- per-model supported capabilities
- whether a model is locally available and runtime-ready
- selected backend versus per-model execution backend
- artifact gaps and routing diagnostics that explain why a model is not ready

### Synchronous TTS endpoints

- `POST /v1/audio/speech`
- `POST /api/v1/tts/custom`
- `POST /api/v1/tts/design`
- `POST /api/v1/tts/clone`

All TTS request payloads accept `language` with default `auto`. Clone form endpoints also accept `language=auto` when omitted.

Successful synchronous audio responses are materialized HTTP bodies, not server-side job handles. Public response headers are:

- `x-request-id` - per-request correlation identifier
- `x-model-id` - public model ID used for synthesis
- `x-tts-mode` - normalized synthesis mode such as `custom`, `design`, or `clone`
- `x-backend-id` - backend that produced the audio
- `x-saved-output-file` - optional public-safe artifact filename when output persistence was enabled

`POST /v1/audio/speech` is the canonical sync speech contract. It accepts the OpenAI-style request body and supports `response_format=wav|pcm`.

Central-host sync semantics are intentionally explicit for Phase 1 multi-client operation:

- sync requests run as bounded synchronous server work through `run_inference_with_timeout`
- when the sync timeout is exceeded, the server returns `504 request_timeout`
- the server does not silently convert a timed-out sync request into an async job
- clients that want retryable background work should call the canonical async submit endpoint directly

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

Official async job states are:

- `queued`
- `running`
- `succeeded`
- `failed`
- `cancelled`

Clients must treat those five values as the entire public async lifecycle model for Phase 1.

Async submission responses return HTTP `202` and a job snapshot payload with:

- `job_id` - stable async job identifier
- `status` - one of the five official async states
- `submit_request_id` - original submit request correlation identifier
- `status_url`, `result_url`, `cancel_url` - canonical follow-up endpoints
- `terminal_error` - populated only for terminal failed or cancelled outcomes

Async lifecycle responses also expose correlation headers so multi-client consumers can stitch together submit, poll, result, and cancel flows without parsing logs:

- `x-request-id` - correlation id for the current HTTP interaction
- `x-job-id` - stable async job identifier
- `x-submit-request-id` - original submission request correlation id carried across later status, result, and cancel reads

Internal execution timeouts remain implementation detail. When an async job exceeds its bounded execution timeout, the public job snapshot still reports `status=failed` with a terminal error code such as `job_execution_timeout` rather than introducing a sixth public lifecycle state.

Async result responses return audio bytes plus the standard audio headers above and `x-job-id`.

Result retrieval semantics are explicit:

- `GET /api/v1/tts/jobs/{job_id}/result` returns `409 job_not_ready` while the job is `queued` or `running`
- `GET /api/v1/tts/jobs/{job_id}/result` returns `409 job_not_succeeded` when the terminal state is `failed` or `cancelled`
- `POST /api/v1/tts/jobs/{job_id}/cancel` is only accepted while the job is still cancellable; otherwise it returns `409 job_not_cancellable`

Async state snapshots, result payloads, and cancellation responses all use the same machine-readable error envelope described below when the request cannot be completed.

### Public error envelope

Every controlled API failure uses one JSON envelope:

```json
{
  "code": "machine_readable_error_code",
  "message": "Human-readable summary",
  "details": {},
  "request_id": "req_..."
}
```

Contract rules:

- `code` is the stable machine-readable discriminator clients should branch on
- `message` is human-readable but less stable than `code`
- `details` is structured JSON with safe public diagnostics only
- `request_id` is always present for correlation
- local filesystem paths are sanitized before reaching clients

Representative error codes include `validation_error`, `model_not_available`, `model_capability_not_supported`, `runtime_capability_not_configured`, `rate_limit_exceeded`, `quota_exceeded`, `request_timeout`, `job_not_found`, `job_not_ready`, `job_not_succeeded`, `job_not_cancellable`, and `job_idempotency_conflict`.

When the server has a retry hint, it also returns standard HTTP headers such as `Retry-After`.

For the complete Phase 1 contract surface, including canonical semantics and response examples, read [../docs/server-http-contract.md](../docs/server-http-contract.md).

## Important runtime behavior

- The adapter returns completed audio as regular HTTP response bodies.
- Clone uploads are staged in `TTS_UPLOAD_STAGING_DIR` and cleaned up after request processing.
- Oversized text requests fail with the standard validation error.
- Inference timeouts return `request_timeout` with HTTP `504`.
- Sync timeout handling is explicit and non-fallback: a bounded sync request either returns audio within the configured timeout or fails with `request_timeout`; background execution is only entered through the async submit routes.
- Unsupported model/family combinations now return controlled `model_capability_not_supported` errors with explicit capability metadata.
- Runtime capability bindings are now expected to come from `TTS_ACTIVE_FAMILY`, `TTS_DEFAULT_CUSTOM_MODEL`, `TTS_DEFAULT_DESIGN_MODEL`, and `TTS_DEFAULT_CLONE_MODEL` rather than from implicit `.models/` inspection.
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

- `TTS_HOST`
- `TTS_PORT`
- `TTS_LOG_LEVEL`
- `TTS_ACTIVE_FAMILY`
- `TTS_DEFAULT_CUSTOM_MODEL`
- `TTS_DEFAULT_DESIGN_MODEL`
- `TTS_DEFAULT_CLONE_MODEL`
- `TTS_DEFAULT_SAVE_OUTPUT`
- `TTS_MAX_UPLOAD_SIZE_BYTES`
- `TTS_MAX_INPUT_TEXT_CHARS`
- `TTS_REQUEST_TIMEOUT_SECONDS`
- `TTS_INFERENCE_BUSY_STATUS_CODE`

Shared variables from [../core/README.md](../core/README.md) also apply.

If you want to disable the accelerated Qwen lane entirely, set `TTS_QWEN_FAST_ENABLED=false`. This affects only the optional fast lane and does not disable the standard Torch Qwen path.

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
