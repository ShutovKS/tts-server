# FILE: telegram_bot/handlers/__init__.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Handlers sub-package barrel.
#   SCOPE: barrel
#   DEPENDS: none
#   LINKS: M-TELEGRAM
#   ROLE: BARREL
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   Command parsing surface - Re-export command parsing, validation, and private-chat helpers
#   Dispatcher wiring - Re-export the command dispatcher used by polling updates
#   TTS handler surface - Re-export the Telegram TTS synthesizer implementation
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, function contracts, semantic blocks, and migrated log events to block-reference format]
# END_CHANGE_SUMMARY

"""
Telegram bot handlers package.

This package contains command handlers and message processing logic
for the Telegram bot transport layer.
"""

from telegram_bot.handlers.commands import (
    CommandType,
    ParsedCommand,
    CommandValidationResult,
    parse_command,
    validate_tts_command,
    is_private_chat,
)
from telegram_bot.handlers.dispatcher import CommandDispatcher
from telegram_bot.handlers.tts_handler import TTSSynthesizer

__all__ = [
    "CommandType",
    "ParsedCommand",
    "CommandValidationResult",
    "parse_command",
    "validate_tts_command",
    "is_private_chat",
    "CommandDispatcher",
    "TTSSynthesizer",
]
