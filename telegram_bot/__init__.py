# FILE: telegram_bot/__init__.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Telegram bot package marker.
#   SCOPE: package init
#   DEPENDS: none
#   LINKS: M-TELEGRAM
#   ROLE: BARREL
#   MAP_MODE: NONE
# END_MODULE_CONTRACT
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, function contracts, semantic blocks, and migrated log events to block-reference format]
# END_CHANGE_SUMMARY

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
