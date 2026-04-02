"""
Telegram Bot Transport Adapter for Qwen3-TTS.

This package provides a Telegram bot as a separate transport layer
on top of the existing core TTS infrastructure.

MVP Scope:
- Private chat only
- Command-based interface (/start, /help, /tts)
- Custom voice synthesis only
- Async UX with fast acknowledgment
"""

from telegram_bot.config import TelegramSettings
from telegram_bot.bootstrap import TelegramRuntime, build_telegram_runtime

__all__ = [
    "TelegramSettings",
    "TelegramRuntime",
    "build_telegram_runtime",
]
