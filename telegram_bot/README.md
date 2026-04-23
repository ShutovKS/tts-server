# Telegram bot module — long-polling adapter

Русская версия: [README.ru.md](README.ru.md)

## Purpose

[telegram_bot/](./) is a separate transport adapter built on top of the shared runtime from [../core/README.md](../core/README.md). In Phase 1 it runs as a long-polling Telegram client that submits work to the central HTTP server over the canonical remote contract rather than hosting local inference inside the bot process.

The canonical HTTP server contract lives in [../docs/server-http-contract.md](../docs/server-http-contract.md). Telegram should treat that document as the source of truth for request shapes, async behavior, readiness semantics, and error handling. It should not infer local model ownership or local runtime fallback from its own environment.

## Current status

The bot is maintained as a standalone deployment unit with its own container image and compose scenario.

Phase 1 cutover is server first, then Telegram. Bring the central HTTP server up and verify it before pointing Telegram at it. If the bot cutover has problems, stop or repoint Telegram first, recover the server, and only then restart or repoint Telegram after server readiness is healthy again.

`telegram-live` is boundary proof and Bot API proof, not server-execution proof by itself. Pair it with companion server evidence from the same deployment, usually `GET /health/ready` and `GET /api/v1/models`, before you call the migration complete.

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
export TTS_TELEGRAM_BOT_TOKEN="your_bot_token_here"
export TTS_TELEGRAM_SERVER_BASE_URL="http://server.internal:8000"
export TTS_TELEGRAM_ALLOWED_USER_IDS="123456789,987654321"
export TTS_TELEGRAM_ADMIN_USER_IDS="123456789"
export TTS_TELEGRAM_RATE_LIMIT_ENABLED=true
export TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE=20
export TTS_TELEGRAM_DELIVERY_STORE_PATH=.state/telegram_delivery_store.json
python -m telegram_bot
```

```powershell
.\.venv311\Scripts\Activate.ps1
$env:TTS_TELEGRAM_BOT_TOKEN="your_bot_token_here"
$env:TTS_TELEGRAM_SERVER_BASE_URL="http://server.internal:8000"
$env:TTS_TELEGRAM_ALLOWED_USER_IDS="123456789,987654321"
$env:TTS_TELEGRAM_ADMIN_USER_IDS="123456789"
$env:TTS_TELEGRAM_RATE_LIMIT_ENABLED="true"
$env:TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE="20"
$env:TTS_TELEGRAM_DELIVERY_STORE_PATH=".state/telegram_delivery_store.json"
python -m telegram_bot
```

## Running with Docker Compose

```bash
docker compose -f docker-compose.telegram-bot.yaml up --build
```

The compose file [../docker-compose.telegram-bot.yaml](../docker-compose.telegram-bot.yaml) builds from [Dockerfile](Dockerfile) with repository-root context `.` and configures the bot as a remote client through `TTS_TELEGRAM_SERVER_BASE_URL`. It retains only adapter-local working state mounts:

- [../.outputs](../.outputs)
- named volume `telegram_bot_state` at `/app/.state`

The image includes `ffmpeg`, stages delivery metadata at `/app/.state/telegram_delivery_store.json`, and does not claim local model-host ownership inside the bot container.

This is the documented Telegram Docker lane for the module, but this README does not claim retained host proof for remote server execution, Bot API reachability, or polling success on the current machine unless the corresponding evidence artifacts are captured from a specific run.

## Telegram token limitation

On the current Windows host with Docker Desktop Linux containers, the Telegram compose lane remains documented and runnable, but the retained evidence set here does not prove startup self-checks, Telegram API connectivity, entry into the healthy polling loop, or server-side inference on this machine. Full external integration still requires a real Telegram bot token, a configured `TTS_TELEGRAM_SERVER_BASE_URL`, and the intended chat/user context for end-to-end command handling.

## Docker-mode validation lane

The Telegram Docker lane is intentionally split between Telegram-client proof and server-host proof. Use the checked-in compose file to exercise remote-server readiness wiring, state-volume wiring, and the polling loop, and pair that with `python scripts/validate_runtime.py telegram-live ...` for host-side Bot API behavior. Treat the lane as documented operator procedure unless you also retain run-specific logs and companion server evidence from the same execution.

1. Export a real `TTS_TELEGRAM_BOT_TOKEN` and the canonical `TTS_TELEGRAM_SERVER_BASE_URL` before launching the compose lane. Add `TTS_TELEGRAM_VALIDATION_CHAT_ID` only when you want the advisory `sendMessage` / `getUpdates` subchecks.
2. Start the checked-in compose scenario in detached mode: `docker compose -f docker-compose.telegram-bot.yaml up --build -d telegram-bot`.
3. Run the host-side Bot API boundary check with `python scripts/validate_runtime.py telegram-live --bot-token "$TTS_TELEGRAM_BOT_TOKEN"`; add `--chat-id "$TTS_TELEGRAM_VALIDATION_CHAT_ID" --expect-update-chat-id "$TTS_TELEGRAM_VALIDATION_CHAT_ID" --expect-update-text "Qwen3-TTS validation ping."` only when you have a dedicated validation chat. This proves Telegram-side Bot API reachability and records that any missing server-side execution proof must come from companion server evidence.
4. Retain raw compose logs as the Telegram-client runtime artifact: `docker compose -f docker-compose.telegram-bot.yaml logs --no-color telegram-bot > .sisyphus/evidence/telegram-docker-log.txt`. Prefer stable markers such as `Remote server readiness verified`, `Telegram API connectivity verified`, and `[Poller][start][BLOCK_DISPATCH_UPDATES]` over prose-only snippets.
5. Pair the Telegram artifacts with companion server evidence from the configured base URL, ideally `python scripts/validate_runtime.py docker-server` or an equivalent retained `/health/ready` proof for that same server deployment, so Telegram-client failures stay distinguishable from central-server failures.
6. Tear down explicitly with `docker compose -f docker-compose.telegram-bot.yaml down --remove-orphans`. Reserve `down -v` for intentional state resets because the named `/app/.state` volume is part of the deployment contract.

Skip the Docker Telegram lane only when Docker is unavailable, the bot token is missing, the remote server base URL is missing, or Telegram is unreachable. Missing validation-chat credentials should downgrade only the advisory `sendMessage` / `getUpdates` subchecks rather than failing the whole compose startup/polling procedure.

## Commands

| Command | Description |
|---|---|
| `/start` | Show welcome message |
| `/help` | Show help and available speakers |
| `/tts` | Synthesize text with optional speaker, speed, and language |
| `/design` | Synthesize using a natural-language voice description and optional language |
| `/clone` | Clone a voice from replied reference audio with optional language |

The bot command surface still targets Qwen-oriented `/tts`, `/design`, and `/clone` workflows. Families that do not support design or clone operations should be surfaced through controlled capability errors rather than ambiguous runtime failures.

The running bot now also treats `TTS_ACTIVE_FAMILY` and `TTS_DEFAULT_*_MODEL` bindings as the source of truth for whether `custom`, `design`, or `clone` are operational in this process. Command syntax remains visible, but unbound modes are rejected explicitly as runtime capability configuration errors instead of failing later with implicit transport-level behavior.

The bot must never silently fall back to local inference or local model ownership when the remote server is missing or not ready. Clear failure is the correct behavior.

### `/tts` syntax

```text
/tts [speaker=<speaker>] [speed=<speed>] [lang=<language>] -- <text>
```

Omitted `lang` defaults to `auto`.

For backward compatibility, the bot still accepts legacy `/tts <text>` without the `--` separator when no structured options are needed. The `--` form remains the documented preferred syntax because it stays unambiguous once speaker, speed, or language options are present.

### `/design` syntax

```text
/design [lang=<language>] <voice_description> -- <text>
```

Omitted `lang` defaults to `auto`.

### `/clone` syntax

```text
/clone [lang=<language>] [ref=<transcript>] -- <text>
```

For `/clone`, the command must be sent as a reply to a message containing reference audio.
Omitted `lang` defaults to `auto`.

## Supported clone media

- `voice`
- `audio`
- `document` containing a supported audio format

Supported formats: WAV, MP3, FLAC, OGG, WebM, M4A, MP4.

## Configuration

Telegram-specific settings are defined by [`TelegramSettings`](config.py:42).

Important environment variables:

- `TTS_TELEGRAM_BOT_TOKEN`
- `TTS_TELEGRAM_SERVER_BASE_URL`
- `TTS_TELEGRAM_ALLOWED_USER_IDS`
- `TTS_TELEGRAM_ADMIN_USER_IDS`
- `TTS_TELEGRAM_DEV_MODE`
- `TTS_TELEGRAM_RATE_LIMIT_ENABLED`
- `TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE`
- `TTS_TELEGRAM_LOG_LEVEL`
- `TTS_TELEGRAM_DEFAULT_SPEAKER`
- `TTS_TELEGRAM_MAX_TEXT_LENGTH`
- `TTS_TELEGRAM_DELIVERY_STORE_PATH`
- `TTS_TELEGRAM_POLL_INTERVAL_SECONDS`
- `TTS_TELEGRAM_MAX_RETRIES`

Shared core variables from [../core/README.md](../core/README.md) also apply.

Important runtime binding variables from the shared contract:

- `TTS_ACTIVE_FAMILY`
- `TTS_DEFAULT_CUSTOM_MODEL`
- `TTS_DEFAULT_DESIGN_MODEL`
- `TTS_DEFAULT_CLONE_MODEL`


## Operational notes

- The bot only handles private chats.
- `ffmpeg` must be available for audio conversion and clone flows.
- Delivery metadata is persisted so completed jobs can still be delivered after a restart.
- Startup self-checks validate token presence, remote server configuration/readiness, Telegram API connectivity, and adapter-local warnings.
- In Phase 1, startup self-checks validate remote server configuration and readiness before the polling loop; they are not evidence that the bot container owns inference locally.

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
