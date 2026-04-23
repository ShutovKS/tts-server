# Telegram Bot Deployment Guide

This document describes deployment options for the Telegram bot service.

## Prerequisites

1. **ffmpeg** must be available in PATH:
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # Verify
   ffmpeg -version
   ```

2. **Python dependencies**: install the Telegram adapter dependencies needed to run the remote client process. In Phase 1 the bot does not own model hosting or local inference; the central HTTP server does.
   ```bash
   # Stable shared lane for the Telegram adapter process
   pip install -r requirements.txt

   # Optional dedicated lane only when repository packaging requires it
   # for the Telegram process on a specific host; this does not move
   # runtime/model ownership into the bot process.
   pip install -r profiles/packs/family/omnivoice.txt
   ```

   For CI-style repository verification without heavyweight optional runtime packages, use:
   ```bash
   pip install -r requirements-ci.txt
   ```

   If you are using the profile-aware launcher flow, prefer checking or creating the resolved Telegram adapter environment instead of guessing the pack manually:
   ```bash
   python -m launcher doctor --family qwen --module telegram
   python -m launcher create-env --family qwen --module telegram --apply
   python -m launcher check-env --family qwen --module telegram
   ```

## Environment Variables

Create a `.env` file with the following variables:

```bash
# === Core Settings ===
TTS_MODELS_DIR=.models
TTS_OUTPUTS_DIR=.outputs
TTS_VOICES_DIR=.voices
TTS_BACKEND_AUTOSELECT=true

# === Telegram Bot Settings ===
TTS_TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
TTS_TELEGRAM_SERVER_BASE_URL=http://server.internal:8000
# Optional: Comma-separated user IDs for allowlist (empty = all users allowed)
TTS_TELEGRAM_ALLOWED_USER_IDS=
# Optional: Admin user IDs with elevated access
TTS_TELEGRAM_ADMIN_USER_IDS=
# Optional: Enable dev mode (relaxed security checks)
TTS_TELEGRAM_DEV_MODE=false
# Optional: Rate limiting (default: true)
TTS_TELEGRAM_RATE_LIMIT_ENABLED=true
# Optional: Rate limit per user per minute (default: 20)
TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE=20
# Optional: Default speaker (default: Vivian)
TTS_TELEGRAM_DEFAULT_SPEAKER=Vivian
# Optional: Max text length (default: 1000, max: 5000)
TTS_TELEGRAM_MAX_TEXT_LENGTH=1000
# Optional: Job poller interval in seconds (default: 1.0)
TTS_TELEGRAM_POLL_INTERVAL_SECONDS=1.0
# Optional: Max retry attempts for API calls (default: 3)
TTS_TELEGRAM_MAX_RETRIES=3
```

## Phase 1 cutover and rollback

Phase 1 rollout order is fixed, bring up the central HTTP server first, verify its readiness, then point Telegram at that server, then consider any other remote client later. The server is the runtime and model host for the phase, not the bot container.

Before you call the migration good, collect both server-side and Telegram-side evidence. On the server side, keep the output from `GET /health/live`, `GET /health/ready`, and `GET /api/v1/models`, or the corresponding `python scripts/validate_runtime.py smoke-server` or `python scripts/validate_runtime.py docker-server` output for the chosen deployment. On the Telegram side, retain the `telegram-live` result plus the compose logs or service logs from the same cutover window. `telegram-live` proves Bot API reachability and client boundary behavior only, so it must be paired with the server proof.

If Telegram or another remote client starts failing after cutover, roll back in the opposite order. Stop or repoint the clients first, recover the central server until readiness is healthy again, then restart or repoint the clients after the server is ready. Do not keep client traffic pointed at a broken server while you try to diagnose it from the client side.

At no point should the bot silently infer local runtime ownership, local model hosting, or local inference fallback. If the remote server is unavailable or the contract is not satisfied, fail clearly and surface the remote boundary problem.

## Deployment Options

### Option 1: Docker Compose (Recommended)

```bash
# Start telegram bot scenario only
docker compose -f docker-compose.telegram-bot.yaml up -d --build

# View logs
docker compose -f docker-compose.telegram-bot.yaml logs -f telegram-bot

# Stop
docker compose -f docker-compose.telegram-bot.yaml down
```

### Option 2: Systemd Service

For Linux servers, use the provided systemd unit file:

```bash
# Copy unit file
sudo cp docs/telegram-bot.service /etc/systemd/system/

# Edit to adjust paths and the resolved Telegram adapter interpreter
sudo nano /etc/systemd/system/telegram-bot.service

# Reload systemd
sudo systemctl daemon-reload

# Enable and start
sudo systemctl enable telegram-bot
sudo systemctl start telegram-bot

# Check status
sudo systemctl status telegram-bot

# View logs
journalctl -u telegram-bot -f
```

### Option 3: Direct Python Execution

```bash
# Run the Telegram remote client process
python -m telegram_bot

# Or with custom settings
TTS_TELEGRAM_BOT_TOKEN=your_token python -m telegram_bot
```

Under the profile-aware environment layout, the practical operator flow is still about resolving the interpreter for the Telegram adapter process, not assigning runtime/model ownership to Telegram:

```bash
# Inspect which interpreter the Telegram adapter should use on this host
python -m launcher plan-run --family qwen --module telegram

# Execute through the resolved interpreter in dry-run mode first
python -m launcher exec --family qwen --module telegram --dry-run
```

## Startup Sequence

The Telegram bot follows this startup sequence:

1. **Configuration validation** - Validate required settings
2. **ffmpeg check** - Verify ffmpeg is available
3. **Remote server configuration and readiness check** - Verify `TTS_TELEGRAM_SERVER_BASE_URL` is configured and the central server reports a usable remote contract
4. **Token validation** - Test bot token via Telegram API
5. **Polling and delivery startup** - Begin receiving updates from Telegram and polling remote async jobs for delivery

## Health Checks

### Startup Self-Checks

The bot performs self-checks at startup:
- Bot token validation
- ffmpeg availability
- remote server base URL configuration
- remote server readiness and contract reachability
- Telegram API connectivity

These checks validate the Telegram client boundary only. In Phase 1 they are not evidence that the bot container owns local inference, local job execution, or local model loading.

### Runtime Health

Monitor the bot's operational state:
- Consecutive errors counter
- Degraded mode threshold (5 consecutive errors)
- Auto-recovery after degraded state

## Restart Policy

| Scenario | Behavior |
|----------|----------|
| Normal shutdown | Clean stop, no restart |
| Crash | Auto-restart via systemd |
| System reboot | Auto-restart via `restart: unless-stopped` |

## Troubleshooting

### Bot not responding

1. Check bot token: `TTS_TELEGRAM_BOT_TOKEN` is set correctly
2. Check logs: `journalctl -u telegram-bot -n 50`
3. Verify ffmpeg: `ffmpeg -version`

### Rate limiting

If users are hitting rate limits, increase the limit:
```bash
TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE=30
```

### Empty allowlist warning

In production, set an allowlist to restrict access:
```bash
TTS_TELEGRAM_ALLOWED_USER_IDS=123456789,987654321
```

## Security Recommendations

1. **Always use allowlist in production** - Set `TTS_TELEGRAM_ALLOWED_USER_IDS`
2. **Keep dev mode off** - Set `TTS_TELEGRAM_DEV_MODE=false`
3. **Enable rate limiting** - Keep `TTS_TELEGRAM_RATE_LIMIT_ENABLED=true`
4. **Set admin users** - Define `TTS_TELEGRAM_ADMIN_USER_IDS` for elevated access
5. **Protect bot token** - Never commit `.env` to version control
