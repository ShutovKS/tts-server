# Phase 1 server HTTP contract

## Scope

This document freezes the repository's Phase 1 canonical remote contract for the HTTP server only.

It covers:

- synchronous TTS
- asynchronous job submission and retrieval
- model discovery
- health and readiness probes
- public response headers
- the shared machine-readable error envelope
- the official async state model

It does **not** define Telegram- or CLI-specific payloads. Those transports must adapt to this HTTP contract rather than introducing alternate per-client contracts.

## Canonical endpoint map

### Control plane

- `GET /health/live`
- `GET /health/ready`
- `GET /api/v1/models`

### Canonical sync speech

- `POST /v1/audio/speech`

### Canonical async speech submit

- `POST /v1/audio/speech/jobs`

### Canonical async job resources

- `GET /api/v1/tts/jobs/{job_id}`
- `GET /api/v1/tts/jobs/{job_id}/result`
- `POST /api/v1/tts/jobs/{job_id}/cancel`

### Server-native synthesis extensions within the same product boundary

- `POST /api/v1/tts/custom`
- `POST /api/v1/tts/design`
- `POST /api/v1/tts/clone`
- `POST /api/v1/tts/custom/jobs`
- `POST /api/v1/tts/design/jobs`
- `POST /api/v1/tts/clone/jobs`

These `/api/v1/tts/*` routes remain official server endpoints, but they are extensions of the same single HTTP contract. Clients should not expect alternative envelopes, alternate async states, or transport-specific variants.

## Canonical `/v1` semantics

`POST /v1/audio/speech` is the canonical Phase 1 synchronous speech contract.

Request shape:

```json
{
  "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
  "input": "Hello world",
  "voice": "Vivian",
  "language": "auto",
  "response_format": "wav",
  "speed": 1.0
}
```

Behavioral rules:

- returns audio as the response body, not a JSON wrapper
- supports `response_format` of `wav` or `pcm`
- may omit `model` when the runtime has an active binding for custom synthesis
- uses the same shared error envelope as every other controlled failure path

`POST /v1/audio/speech/jobs` is the canonical asynchronous version of the same speech contract.

Behavioral rules:

- returns `202 Accepted`
- accepts the same request body as `POST /v1/audio/speech`
- accepts optional `Idempotency-Key`
- returns a job snapshot rather than audio bytes

## Health and readiness semantics

### `GET /health/live`

Purpose: process liveness only.

Response shape:

```json
{
  "status": "ok",
  "checks": {
    "process": "alive"
  }
}
```

### `GET /health/ready`

Purpose: operational readiness for clients and operators.

Public contract:

- `status` is `ok` or `degraded`
- `checks` is machine-readable structured data
- readiness exposes runtime capability bindings, backend selection, model readiness, and ffmpeg/config diagnostics

`/health/ready` is the canonical machine-readable readiness surface. Clients that need to understand whether a runtime can satisfy custom/design/clone flows should prefer this endpoint over guessing from local UI assumptions.

## Model discovery semantics

`GET /api/v1/models` is the canonical machine-readable discovery endpoint.

Each item describes:

- stable public model identity (`id`)
- synthesis mode and family metadata
- supported capabilities
- availability and runtime readiness
- selected backend and effective execution backend
- route diagnostics and missing artifacts

This endpoint is the authoritative discovery source for whether a model can be used through the current running server.

## Public success headers

### Sync audio responses

- `x-request-id`
- `x-model-id`
- `x-tts-mode`
- `x-backend-id`
- `x-saved-output-file` when output persistence produced a public artifact

### Async result responses

Async result responses return the same audio headers as sync responses plus:

- `x-job-id`

Header meanings are stable across sync and async flows:

- `x-request-id`: correlation identifier for the current HTTP request
- `x-model-id`: public model ID used for the generation
- `x-tts-mode`: normalized mode such as `custom`, `design`, or `clone`
- `x-backend-id`: backend that actually generated the audio
- `x-job-id`: async job identifier for result retrieval calls

## Shared error envelope

Every controlled HTTP failure returns:

```json
{
  "code": "machine_readable_error_code",
  "message": "Human-readable summary",
  "details": {},
  "request_id": "req_..."
}
```

Semantics:

- `code` is the primary field clients should use for branching
- `message` is for operators and humans
- `details` carries structured diagnostics and may include nested error metadata
- `request_id` is always present for correlation

The server may additionally emit HTTP headers such as `Retry-After` when retry hints exist.

Representative public error codes:

- `validation_error`
- `model_not_available`
- `model_capability_not_supported`
- `runtime_capability_not_configured`
- `rate_limit_exceeded`
- `quota_exceeded`
- `request_timeout`
- `generation_failed`
- `job_queue_full`
- `job_not_found`
- `job_not_ready`
- `job_not_succeeded`
- `job_not_cancellable`
- `job_idempotency_conflict`
- `unauthorized`
- `forbidden`

The contract guarantees that public error details are sanitized before leaving the server, including local filesystem path redaction.

## Official async state model

The only official Phase 1 async job states are:

- `queued`
- `running`
- `succeeded`
- `failed`
- `cancelled`

Clients must treat this set as authoritative.

State semantics:

- `queued`: accepted but not yet executing
- `running`: currently executing
- `succeeded`: terminal success; result retrieval must return audio
- `failed`: terminal failure; result retrieval must return `job_not_succeeded`
- `cancelled`: terminal cancellation; result retrieval must return `job_not_succeeded`

## Async lifecycle contract

### Submit

`POST .../jobs`

- returns `202`
- returns a job snapshot payload
- optional `Idempotency-Key` enables principal-scoped replay for identical payloads
- conflicting reuse of the same idempotency key with a different payload returns `409 job_idempotency_conflict`

### Status

`GET /api/v1/tts/jobs/{job_id}`

- returns the current job snapshot for an owned job
- returns `404 job_not_found` when the job does not exist
- returns `403 forbidden` when the caller does not own the job

### Result

`GET /api/v1/tts/jobs/{job_id}/result`

- returns audio on `succeeded`
- returns `409 job_not_ready` for `queued` or `running`
- returns `409 job_not_succeeded` for `failed` or `cancelled`

### Cancel

`POST /api/v1/tts/jobs/{job_id}/cancel`

- accepts cancellation while the job is still cancellable
- returns a snapshot with `status=cancelled` when cancellation succeeds
- returns `409 job_not_cancellable` when the job is already running to completion or already terminal in a non-cancellable state

## Compatibility note

The repository may keep legacy-compatible server-native routes, but this document defines the canonical client contract for Phase 1. New clients should use this contract and should not depend on alternate envelopes, alternate lifecycle labels, or Telegram-shaped payloads.
