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

2. **Python dependencies**: Ensure requirements are installed:
   ```bash
   pip install -r requirements.txt
   ```

## Environment Variables

Create a `.env` file with the following variables:

```bash
# === Core Settings ===
QWEN_TTS_MODELS_DIR=.models
QWEN_TTS_OUTPUTS_DIR=.outputs
QWEN_TTS_VOICES_DIR=.voices
QWEN_TTS_BACKEND_AUTOSELECT=true

# === Telegram Bot Settings ===
QWEN_TTS_TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
# Optional: Comma-separated user IDs for allowlist (empty = all users allowed)
QWEN_TTS_TELEGRAM_ALLOWED_USER_IDS=
# Optional: Admin user IDs with elevated access
QWEN_TTS_TELEGRAM_ADMIN_USER_IDS=
# Optional: Enable dev mode (relaxed security checks)
QWEN_TTS_TELEGRAM_DEV_MODE=false
# Optional: Rate limiting (default: true)
QWEN_TTS_TELEGRAM_RATE_LIMIT_ENABLED=true
# Optional: Rate limit per user per minute (default: 20)
QWEN_TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE=20
# Optional: Default speaker (default: Vivian)
QWEN_TTS_TELEGRAM_DEFAULT_SPEAKER=Vivian
# Optional: Max text length (default: 1000, max: 5000)
QWEN_TTS_TELEGRAM_MAX_TEXT_LENGTH=1000
# Optional: Job poller interval in seconds (default: 1.0)
QWEN_TTS_TELEGRAM_POLL_INTERVAL_SECONDS=1.0
# Optional: Max retry attempts for API calls (default: 3)
QWEN_TTS_TELEGRAM_MAX_RETRIES=3
```

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

# Edit to adjust paths if needed
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
# Run directly
python -m telegram_bot

# Or with custom settings
QWEN_TTS_TELEGRAM_BOT_TOKEN=your_token python -m telegram_bot
```

## Startup Sequence

The Telegram bot follows this startup sequence:

1. **Configuration validation** - Validate required settings
2. **ffmpeg check** - Verify ffmpeg is available
3. **Token validation** - Test bot token via Telegram API
4. **Core runtime initialization** - Load MLX/Torch backend
5. **Job execution setup** - Initialize job execution infrastructure
6. **Polling start** - Begin receiving updates from Telegram

## Health Checks

### Startup Self-Checks

The bot performs self-checks at startup:
- Bot token validation
- ffmpeg availability
- Core runtime initialization
- Model loading (if preloaded)

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

1. Check bot token: `QWEN_TTS_TELEGRAM_BOT_TOKEN` is set correctly
2. Check logs: `journalctl -u telegram-bot -n 50`
3. Verify ffmpeg: `ffmpeg -version`

### Rate limiting

If users are hitting rate limits, increase the limit:
```bash
QWEN_TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE=30
```

### Empty allowlist warning

In production, set an allowlist to restrict access:
```bash
QWEN_TTS_TELEGRAM_ALLOWED_USER_IDS=123456789,987654321
```

## Security Recommendations

1. **Always use allowlist in production** - Set `QWEN_TTS_TELEGRAM_ALLOWED_USER_IDS`
2. **Keep dev mode off** - Set `QWEN_TTS_TELEGRAM_DEV_MODE=false`
3. **Enable rate limiting** - Keep `QWEN_TTS_TELEGRAM_RATE_LIMIT_ENABLED=true`
4. **Set admin users** - Define `QWEN_TTS_TELEGRAM_ADMIN_USER_IDS` for elevated access
5. **Protect bot token** - Never commit `.env` to version control
