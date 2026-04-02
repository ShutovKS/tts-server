"""
Unit tests for command parsing and validation.

Tests command parsing, validation, and policy enforcement.
"""

from __future__ import annotations

import pytest

from telegram_bot.handlers.commands import (
    CommandType,
    ParsedCommand,
    ParsedTTSArgs,
    ParsedDesignArgs,
    ParsedCloneArgs,
    CommandValidationResult,
    MIN_SPEED,
    MAX_SPEED,
    VALID_SPEAKERS,
    MIN_VOICE_DESCRIPTION_LENGTH,
    MAX_VOICE_DESCRIPTION_LENGTH,
    MAX_TEXT_LENGTH,
    parse_command,
    parse_tts_args,
    parse_design_args,
    parse_clone_args,
    validate_tts_args,
    validate_design_args,
    validate_clone_args,
    validate_tts_command,
    validate_design_command,
    validate_clone_command,
    is_private_chat,
    get_valid_speakers,
)


class TestCommandParsing:
    """Tests for command parsing logic."""
    
    def test_parse_start_command(self):
        """Test parsing /start command."""
        result = parse_command("/start", 12345, 67890, 100)
        
        assert result is not None
        assert result.command == CommandType.START
        assert result.raw_text == "/start"
        assert result.args == ""
        assert result.user_id == 12345
        assert result.chat_id == 67890
        assert result.message_id == 100

    def test_parse_command_preserves_reply_message(self):
        """Test parse_command keeps reply_to_message payload."""
        reply_message = {"message_id": 222, "voice": {"file_id": "voice123"}}

        result = parse_command(
            "/clone -- Hello",
            12345,
            67890,
            100,
            reply_to_message=reply_message,
        )

        assert result is not None
        assert result.reply_to_message == reply_message
    
    def test_parse_help_command(self):
        """Test parsing /help command."""
        result = parse_command("/help", 12345, 67890, 100)
        
        assert result is not None
        assert result.command == CommandType.HELP
        assert result.args == ""
    
    def test_parse_tts_command_with_text(self):
        """Test parsing /tts with text argument."""
        text = "Hello world"
        result = parse_command(f"/tts {text}", 12345, 67890, 100)
        
        assert result is not None
        assert result.command == CommandType.TTS
        assert result.args == text
        assert result.tts_text == text
    
    def test_parse_tts_command_empty_args(self):
        """Test parsing /tts without arguments."""
        result = parse_command("/tts", 12345, 67890, 100)
        
        assert result is not None
        assert result.command == CommandType.TTS
        assert result.args == ""
        assert result.tts_text == ""
    
    def test_parse_unknown_command(self):
        """Test parsing unknown command."""
        result = parse_command("/unknown", 12345, 67890, 100)
        
        assert result is not None
        assert result.command == CommandType.UNKNOWN
    
    def test_parse_command_case_insensitive(self):
        """Test command parsing is case insensitive."""
        result = parse_command("/START", 12345, 67890, 100)
        
        assert result is not None
        assert result.command == CommandType.START
    
    def test_parse_command_with_bot_name(self):
        """Test parsing command with @botname suffix."""
        result = parse_command("/tts@QwenTTSBot hello", 12345, 67890, 100)
        
        assert result is not None
        assert result.command == CommandType.TTS
        assert result.args == "hello"
    
    def test_parse_non_command_text(self):
        """Test that non-command text returns None."""
        result = parse_command("Just a regular message", 12345, 67890, 100)
        
        assert result is None
    
    def test_parse_empty_text(self):
        """Test that empty text returns None."""
        result = parse_command("", 12345, 67890, 100)
        
        assert result is None
    
    def test_parse_none_text(self):
        """Test that None text returns None."""
        result = parse_command(None, 12345, 67890, 100)
        
        assert result is None


class TestTTSArgsParsing:
    """Tests for /tts extended argument parsing."""
    
    def test_parse_basic_syntax(self):
        """Test basic /tts -- <text> syntax."""
        result = parse_tts_args("-- Hello world")
        
        assert result is not None
        assert result.text == "Hello world"
        assert result.speaker is None
        assert result.speed is None
    
    def test_parse_with_speaker(self):
        """Test /tts speaker=<name> -- <text> syntax."""
        result = parse_tts_args("speaker=Vivian -- Hello world")
        
        assert result is not None
        assert result.text == "Hello world"
        assert result.speaker == "Vivian"
        assert result.speed is None
    
    def test_parse_with_speed(self):
        """Test /tts speed=<val> -- <text> syntax."""
        result = parse_tts_args("speed=1.5 -- Hello world")
        
        assert result is not None
        assert result.text == "Hello world"
        assert result.speaker is None
        assert result.speed == 1.5
    
    def test_parse_with_both_params(self):
        """Test /tts speaker=<name> speed=<val> -- <text> syntax."""
        result = parse_tts_args("speaker=Ryan speed=0.8 -- Hello world")
        
        assert result is not None
        assert result.text == "Hello world"
        assert result.speaker == "Ryan"
        assert result.speed == 0.8
    
    def test_parse_with_both_params_order(self):
        """Test /tts speaker=<name> speed=<val> -- <text> syntax."""
        result = parse_tts_args("speaker=Vivian speed=1.5 -- Hello world")
        
        assert result is not None
        assert result.text == "Hello world"
        assert result.speaker == "Vivian"
        assert result.speed == 1.5
    
    def test_parse_legacy_syntax(self):
        """Test legacy syntax without -- separator (backward compatibility)."""
        result = parse_tts_args("Hello world without separator")
        
        assert result is not None
        assert result.text == "Hello world without separator"
        assert result.speaker is None
        assert result.speed is None
    
    def test_parse_empty_args_returns_none(self):
        """Test that empty args returns None."""
        result = parse_tts_args("")
        
        assert result is None
    
    def test_parse_falls_back_to_legacy_on_invalid_extended_syntax(self):
        """Test that invalid extended syntax falls back to legacy (text only)."""
        # When there's no -- separator but text looks like legacy,
        # it should be treated as legacy syntax
        result = parse_tts_args("speaker=Vivian Hello world")
        
        # Legacy fallback: entire string becomes text
        assert result is not None
        assert result.text == "speaker=Vivian Hello world"
    
    def test_parse_speed_decimal(self):
        """Test speed parsing with decimal value."""
        result = parse_tts_args("speed=0.75 -- Test")
        
        assert result is not None
        assert result.speed == 0.75
    
    def test_parse_text_with_extra_spaces(self):
        """Test parsing with extra spaces around delimiter."""
        result = parse_tts_args("speaker=Ryan   --   Hello")
        
        assert result is not None
        assert result.text == "Hello"
        assert result.speaker == "Ryan"
    
    def test_parse_text_with_leading_spaces(self):
        """Test parsing with leading spaces in text."""
        result = parse_tts_args("--   Hello world")
        
        assert result is not None
        assert result.text == "Hello world"
    
    def test_parse_case_insensitive_speaker_key(self):
        """Test that speaker key is case insensitive."""
        result = parse_tts_args("SPEAKER=Vivian -- Test")
        
        assert result is not None
        assert result.speaker == "Vivian"
    
    def test_parse_case_insensitive_speed_key(self):
        """Test that speed key is case insensitive."""
        result = parse_tts_args("SPEED=1.5 -- Test")
        
        assert result is not None
        assert result.speed == 1.5
    
    def test_parse_invalid_speed_returns_none(self):
        """Test that non-numeric speed returns None."""
        result = parse_tts_args("speed=fast -- Test")
        
        assert result is None
    
    def test_parse_empty_text_after_delimiter(self):
        """Test parsing with empty text after delimiter."""
        result = parse_tts_args("--")
        
        assert result is not None
        assert result.text == ""


class TestTTSArgsValidation:
    """Tests for /tts argument validation."""
    
    def test_validate_valid_args(self):
        """Test validation of valid arguments."""
        parsed_args = ParsedTTSArgs(text="Hello", speaker="Vivian", speed=1.0)
        result = validate_tts_args(parsed_args)
        
        assert result.is_valid is True
        assert result.error_message is None
    
    def test_validate_empty_text(self):
        """Test validation rejects empty text."""
        parsed_args = ParsedTTSArgs(text="", speaker=None, speed=None)
        result = validate_tts_args(parsed_args)
        
        assert result.is_valid is False
        assert "provide text" in result.error_message.lower()
    
    def test_validate_invalid_speaker(self):
        """Test validation rejects invalid speaker."""
        parsed_args = ParsedTTSArgs(text="Hello", speaker="InvalidSpeaker", speed=None)
        result = validate_tts_args(parsed_args)
        
        assert result.is_valid is False
        assert "Unknown speaker" in result.error_message
        assert "InvalidSpeaker" in result.error_message
    
    def test_validate_valid_speaker(self):
        """Test validation accepts valid speakers."""
        for speaker in VALID_SPEAKERS:
            parsed_args = ParsedTTSArgs(text="Hello", speaker=speaker, speed=None)
            result = validate_tts_args(parsed_args)
            assert result.is_valid is True, f"Speaker {speaker} should be valid"
    
    def test_validate_speed_too_low(self):
        """Test validation rejects speed below minimum."""
        parsed_args = ParsedTTSArgs(text="Hello", speaker=None, speed=0.1)
        result = validate_tts_args(parsed_args)
        
        assert result.is_valid is False
        assert "Speed must be between" in result.error_message
        assert "0.1" in result.error_message
    
    def test_validate_speed_too_high(self):
        """Test validation rejects speed above maximum."""
        parsed_args = ParsedTTSArgs(text="Hello", speaker=None, speed=3.0)
        result = validate_tts_args(parsed_args)
        
        assert result.is_valid is False
        assert "Speed must be between" in result.error_message
        assert "3.0" in result.error_message
    
    def test_validate_speed_at_minimum(self):
        """Test validation accepts speed at minimum boundary."""
        parsed_args = ParsedTTSArgs(text="Hello", speaker=None, speed=MIN_SPEED)
        result = validate_tts_args(parsed_args)
        
        assert result.is_valid is True
    
    def test_validate_speed_at_maximum(self):
        """Test validation accepts speed at maximum boundary."""
        parsed_args = ParsedTTSArgs(text="Hello", speaker=None, speed=MAX_SPEED)
        result = validate_tts_args(parsed_args)
        
        assert result.is_valid is True
    
    def test_validate_speed_zero(self):
        """Test validation rejects zero speed."""
        parsed_args = ParsedTTSArgs(text="Hello", speaker=None, speed=0.0)
        result = validate_tts_args(parsed_args)
        
        assert result.is_valid is False
    
    def test_validate_both_invalid(self):
        """Test validation with both speaker and speed invalid."""
        parsed_args = ParsedTTSArgs(text="Hello", speaker="Bad", speed=5.0)
        result = validate_tts_args(parsed_args)
        
        # Should fail on speaker first (since it's checked first)
        assert result.is_valid is False
        assert "Unknown speaker" in result.error_message


class TestTTSValidation:
    """Tests for TTS command validation."""
    
    def test_validate_valid_tts_command(self):
        """Test validation of valid TTS command."""
        parsed = ParsedCommand(
            command=CommandType.TTS,
            raw_text="/tts -- Hello",
            args="-- Hello",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_tts_command(parsed, max_length=1000)
        
        assert result.is_valid is True
        assert result.error_message is None
    
    def test_validate_empty_tts_text(self):
        """Test validation rejects empty TTS text."""
        parsed = ParsedCommand(
            command=CommandType.TTS,
            raw_text="/tts -- ",
            args="-- ",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_tts_command(parsed, max_length=1000)
        
        assert result.is_valid is False
        assert "provide text" in result.error_message.lower()
    
    def test_validate_tts_text_too_long(self):
        """Test validation rejects too long text."""
        long_text = "a" * 2000
        parsed = ParsedCommand(
            command=CommandType.TTS,
            raw_text=f"/tts -- {long_text}",
            args=f"-- {long_text}",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_tts_command(parsed, max_length=1000)
        
        assert result.is_valid is False
        assert "too long" in result.error_message.lower()
    
    def test_validate_tts_invalid_speaker(self):
        """Test validation rejects invalid speaker in command."""
        parsed = ParsedCommand(
            command=CommandType.TTS,
            raw_text="/tts speaker=Bad -- Hello",
            args="speaker=Bad -- Hello",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_tts_command(parsed, max_length=1000)
        
        assert result.is_valid is False
        assert "Unknown speaker" in result.error_message
    
    def test_validate_tts_invalid_speed(self):
        """Test validation rejects invalid speed in command."""
        parsed = ParsedCommand(
            command=CommandType.TTS,
            raw_text="/tts speed=5.0 -- Hello",
            args="speed=5.0 -- Hello",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_tts_command(parsed, max_length=1000)
        
        assert result.is_valid is False
        assert "Speed must be between" in result.error_message
    
    def test_validate_tts_valid_speaker_and_speed(self):
        """Test validation accepts valid speaker and speed."""
        parsed = ParsedCommand(
            command=CommandType.TTS,
            raw_text="/tts speaker=Ryan speed=0.8 -- Hello",
            args="speaker=Ryan speed=0.8 -- Hello",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_tts_command(parsed, max_length=1000)
        
        assert result.is_valid is True
    
    def test_validate_non_tts_command_passes(self):
        """Test that non-TTS commands always pass validation."""
        parsed = ParsedCommand(
            command=CommandType.START,
            raw_text="/start",
            args="",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_tts_command(parsed, max_length=1000)
        
        assert result.is_valid is True
    
    def test_validate_empty_args(self):
        """Test validation rejects empty args."""
        parsed = ParsedCommand(
            command=CommandType.TTS,
            raw_text="/tts",
            args="",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_tts_command(parsed, max_length=1000)
        
        assert result.is_valid is False


class TestChatTypePolicy:
    """Tests for chat type policy enforcement."""
    
    def test_is_private_chat_true(self):
        """Test that private chat type is correctly identified."""
        assert is_private_chat("private") is True
    
    def test_is_private_chat_false_for_group(self):
        """Test that group chat is not private."""
        assert is_private_chat("group") is False
    
    def test_is_private_chat_false_for_supergroup(self):
        """Test that supergroup is not private."""
        assert is_private_chat("supergroup") is False
    
    def test_is_private_chat_false_for_channel(self):
        """Test that channel is not private."""
        assert is_private_chat("channel") is False


class TestGetValidSpeakers:
    """Tests for get_valid_speakers function."""
    
    def test_returns_list(self):
        """Test that function returns a list."""
        result = get_valid_speakers()
        
        assert isinstance(result, list)
    
    def test_returns_sorted_list(self):
        """Test that returned list is sorted alphabetically."""
        result = get_valid_speakers()
        
        assert result == sorted(result)
    
    def test_contains_expected_speakers(self):
        """Test that list contains expected speakers."""
        result = get_valid_speakers()
        
        # These are from SPEAKER_MAP in core/models/catalog.py
        assert "Vivian" in result
        assert "Ryan" in result
    
    def test_matches_valid_speakers_set(self):
        """Test that returned list matches VALID_SPEAKERS frozenset."""
        result = get_valid_speakers()
        
        assert set(result) == VALID_SPEAKERS


class TestTTSCommandEdgeCases:
    """Edge case tests for TTS command processing."""
    
    def test_tts_text_with_multiple_spaces(self):
        """Test TTS text with multiple spaces."""
        result = parse_command("/tts Hello    World", 12345, 67890, 100)
        
        assert result is not None
        assert result.tts_text == "Hello    World"
    
    def test_tts_text_with_newlines(self):
        """Test TTS text with newlines."""
        result = parse_command("/tts Line 1\nLine 2", 12345, 67890, 100)
        
        assert result is not None
        assert "Line 1" in result.tts_text
        assert "Line 2" in result.tts_text
    
    def test_tts_text_unicode(self):
        """Test TTS text with unicode characters."""
        text = "Привет мир! Hello 世界 🌍"
        result = parse_command(f"/tts {text}", 12345, 67890, 100)
        
        assert result is not None
        assert result.tts_text == text
    
    def test_tts_text_emoji(self):
        """Test TTS text with emoji."""
        text = "Hello! 🎙️ How are you? 😊"
        result = parse_command(f"/tts {text}", 12345, 67890, 100)
        
        assert result is not None
        assert result.tts_text == text
    
    def test_tts_text_exactly_max_length(self):
        """Test TTS text at exactly max length."""
        text = "a" * 1000
        parsed = ParsedCommand(
            command=CommandType.TTS,
            raw_text=f"/tts -- {text}",
            args=f"-- {text}",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_tts_command(parsed, max_length=1000)
        
        assert result.is_valid is True
    
    def test_tts_text_one_over_max_length(self):
        """Test TTS text one character over max length."""
        text = "a" * 1001
        parsed = ParsedCommand(
            command=CommandType.TTS,
            raw_text=f"/tts -- {text}",
            args=f"-- {text}",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_tts_command(parsed, max_length=1000)
        
        assert result.is_valid is False


class TestDesignCommandParsing:
    """Tests for /design command parsing."""
    
    def test_parse_design_command(self):
        """Test parsing /design command with voice description and text."""
        result = parse_command("/design calm narrator -- Hello world", 12345, 67890, 100)
        
        assert result is not None
        assert result.command == CommandType.DESIGN
        assert result.raw_text == "/design calm narrator -- Hello world"
        assert result.args == "calm narrator -- Hello world"
    
    def test_parse_design_command_case_insensitive(self):
        """Test that /DESIGN is also parsed correctly."""
        result = parse_command("/DESIGN calm narrator -- Hello", 12345, 67890, 100)
        
        assert result is not None
        assert result.command == CommandType.DESIGN
    
    def test_parse_design_command_with_bot_name(self):
        """Test parsing /design with bot name."""
        result = parse_command("/design@mybot calm narrator -- Hello", 12345, 67890, 100)
        
        assert result is not None
        assert result.command == CommandType.DESIGN
    
    def test_parse_design_no_args(self):
        """Test parsing /design without arguments returns None."""
        result = parse_command("/design", 12345, 67890, 100)
        
        assert result is not None
        assert result.command == CommandType.DESIGN
        assert result.args == ""
    
    def test_parse_non_design_command(self):
        """Test that non-design commands are not misidentified."""
        result = parse_command("/start", 12345, 67890, 100)
        
        assert result is not None
        assert result.command == CommandType.START


class TestDesignArgsParsing:
    """Tests for /design argument parsing."""
    
    def test_parse_basic_syntax(self):
        """Test basic voice_description -- text parsing."""
        result = parse_design_args("calm narrator -- Hello world")
        
        assert result is not None
        assert result.voice_description == "calm narrator"
        assert result.text == "Hello world"
    
    def test_parse_simple_voice_description(self):
        """Test parsing with simple voice description."""
        result = parse_design_args("young female -- Привет")
        
        assert result is not None
        assert result.voice_description == "young female"
        assert result.text == "Привет"
    
    def test_parse_complex_voice_description(self):
        """Test parsing with complex voice description."""
        desc = "calm professional documentary narrator with slight british accent"
        text = "Расскажи о космосе"
        result = parse_design_args(f"{desc} -- {text}")
        
        assert result is not None
        assert result.voice_description == desc
        assert result.text == text
    
    def test_parse_voice_description_with_numbers(self):
        """Test parsing voice description with numbers."""
        result = parse_design_args("25 year old male -- Test")
        
        assert result is not None
        assert result.voice_description == "25 year old male"
        assert result.text == "Test"
    
    def test_parse_empty_args_returns_none(self):
        """Test that empty args return None."""
        result = parse_design_args("")
        
        assert result is None
    
    def test_parse_only_separator_returns_none(self):
        """Test that only separator returns None."""
        result = parse_design_args("--")
        
        assert result is None
    
    def test_parse_missing_text_returns_none(self):
        """Test that missing text after separator returns None."""
        result = parse_design_args("calm narrator --")
        
        assert result is None
    
    def test_parse_missing_voice_description_returns_none(self):
        """Test that missing voice description before separator returns None."""
        result = parse_design_args("-- Hello")
        
        assert result is None
    
    def test_parse_text_with_extra_spaces(self):
        """Test parsing text with extra spaces."""
        result = parse_design_args("  calm narrator  --  Hello world  ")
        
        assert result is not None
        assert result.voice_description == "calm narrator"
        assert result.text == "Hello world"
    
    def test_parse_multiline_text(self):
        """Test parsing multiline text."""
        result = parse_design_args("calm narrator -- Line one\nLine two\nLine three")
        
        assert result is not None
        assert result.voice_description == "calm narrator"
        assert result.text == "Line one\nLine two\nLine three"
    
    def test_parse_text_with_leading_spaces(self):
        """Test parsing text with leading spaces."""
        result = parse_design_args("calm narrator --   Hello")
        
        assert result is not None
        assert result.text == "Hello"


class TestDesignArgsValidation:
    """Tests for /design argument validation."""
    
    def test_validate_valid_args(self):
        """Test validation accepts valid arguments."""
        args = ParsedDesignArgs(
            voice_description="calm narrator",
            text="Hello world"
        )
        
        result = validate_design_args(args)
        
        assert result.is_valid is True
    
    def test_validate_empty_voice_description(self):
        """Test validation rejects empty voice description."""
        args = ParsedDesignArgs(
            voice_description="",
            text="Hello"
        )
        
        result = validate_design_args(args)
        
        assert result.is_valid is False
        assert "voice description" in result.error_message.lower()
    
    def test_validate_empty_text(self):
        """Test validation rejects empty text."""
        args = ParsedDesignArgs(
            voice_description="calm narrator",
            text=""
        )
        
        result = validate_design_args(args)
        
        assert result.is_valid is False
        assert "text" in result.error_message.lower()
    
    def test_validate_voice_description_too_short(self):
        """Test validation rejects voice description that's too short."""
        args = ParsedDesignArgs(
            voice_description="ab",
            text="Hello"
        )
        
        result = validate_design_args(args)
        
        assert result.is_valid is False
        assert "voice description" in result.error_message.lower()
    
    def test_validate_voice_description_at_minimum(self):
        """Test validation accepts voice description at minimum length."""
        args = ParsedDesignArgs(
            voice_description="abc",
            text="Hello"
        )
        
        result = validate_design_args(args)
        
        assert result.is_valid is True
    
    def test_validate_voice_description_too_long(self):
        """Test validation rejects voice description that's too long."""
        args = ParsedDesignArgs(
            voice_description="a" * 501,
            text="Hello"
        )
        
        result = validate_design_args(args)
        
        assert result.is_valid is False
        assert "voice description" in result.error_message.lower()
    
    def test_validate_voice_description_at_maximum(self):
        """Test validation accepts voice description at maximum length."""
        args = ParsedDesignArgs(
            voice_description="a" * 500,
            text="Hello"
        )
        
        result = validate_design_args(args)
        
        assert result.is_valid is True
    
    def test_validate_text_at_maximum(self):
        """Test validation accepts text at maximum length."""
        args = ParsedDesignArgs(
            voice_description="calm narrator",
            text="a" * 1000
        )
        
        result = validate_design_args(args)
        
        assert result.is_valid is True
    
    def test_validate_text_over_maximum(self):
        """Test validation rejects text over maximum length."""
        args = ParsedDesignArgs(
            voice_description="calm narrator",
            text="a" * 1001
        )
        
        result = validate_design_args(args)
        
        assert result.is_valid is False


class TestDesignCommandValidation:
    """Tests for /design command-level validation."""
    
    def test_validate_design_valid_command(self):
        """Test validation accepts valid design command."""
        parsed = ParsedCommand(
            command=CommandType.DESIGN,
            raw_text="/design calm narrator -- Hello",
            args="calm narrator -- Hello",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_design_command(parsed)
        
        assert result.is_valid is True
    
    def test_validate_design_empty_args(self):
        """Test validation rejects empty args."""
        parsed = ParsedCommand(
            command=CommandType.DESIGN,
            raw_text="/design",
            args="",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_design_command(parsed)
        
        assert result.is_valid is False
    
    def test_validate_design_missing_separator(self):
        """Test validation rejects missing separator."""
        parsed = ParsedCommand(
            command=CommandType.DESIGN,
            raw_text="/design calm narrator Hello",
            args="calm narrator Hello",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_design_command(parsed)
        
        assert result.is_valid is False
    
    def test_validate_design_empty_voice_description(self):
        """Test validation rejects empty voice description."""
        parsed = ParsedCommand(
            command=CommandType.DESIGN,
            raw_text="/design -- Hello",
            args="-- Hello",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_design_command(parsed)
        
        assert result.is_valid is False
        assert "/design" in result.error_message.lower()
        assert "voice description" in result.error_message.lower() or "syntax" in result.error_message.lower()
    
    def test_validate_design_empty_text(self):
        """Test validation rejects empty text."""
        parsed = ParsedCommand(
            command=CommandType.DESIGN,
            raw_text="/design calm narrator --",
            args="calm narrator --",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_design_command(parsed)
        
        assert result.is_valid is False
        assert "text" in result.error_message.lower()
    
    def test_validate_design_non_design_command_passes(self):
        """Test that non-DESIGN commands always pass validation."""
        parsed = ParsedCommand(
            command=CommandType.START,
            raw_text="/start",
            args="",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_design_command(parsed)
        
        assert result.is_valid is True


class TestCloneCommandParsing:
    """Tests for /clone command parsing."""
    
    def test_parse_clone_command(self):
        """Test parsing /clone command with text."""
        result = parse_command("/clone -- Hello world", 12345, 67890, 100)
        
        assert result is not None
        assert result.command == CommandType.CLONE
        assert result.raw_text == "/clone -- Hello world"
        assert result.args == "-- Hello world"
    
    def test_parse_clone_command_case_insensitive(self):
        """Test that /CLONE is also parsed correctly."""
        result = parse_command("/CLONE -- Hello", 12345, 67890, 100)
        
        assert result is not None
        assert result.command == CommandType.CLONE
    
    def test_parse_clone_command_with_bot_name(self):
        """Test parsing /clone with bot name."""
        result = parse_command("/clone@mybot -- Hello", 12345, 67890, 100)
        
        assert result is not None
        assert result.command == CommandType.CLONE
    
    def test_parse_clone_no_args(self):
        """Test parsing /clone without arguments returns command with empty args."""
        result = parse_command("/clone", 12345, 67890, 100)
        
        assert result is not None
        assert result.command == CommandType.CLONE
        assert result.args == ""
    
    def test_parse_clone_with_ref(self):
        """Test parsing /clone with ref= parameter."""
        result = parse_command("/clone ref=This is my sample -- Say this in my voice", 12345, 67890, 100)
        
        assert result is not None
        assert result.command == CommandType.CLONE
        assert result.args == "ref=This is my sample -- Say this in my voice"
    
    def test_parse_non_clone_command(self):
        """Test that non-clone commands are not misidentified."""
        result = parse_command("/start", 12345, 67890, 100)
        
        assert result is not None
        assert result.command == CommandType.START


class TestCloneArgsParsing:
    """Tests for /clone argument parsing."""
    
    def test_parse_basic_syntax(self):
        """Test basic -- text parsing."""
        result = parse_clone_args("-- Hello world")
        
        assert result is not None
        assert result.ref_text is None
        assert result.text == "Hello world"
    
    def test_parse_with_ref_text(self):
        """Test parsing with ref= parameter."""
        result = parse_clone_args("ref=This is my sample -- Hello world")
        
        assert result is not None
        assert result.ref_text == "This is my sample"
        assert result.text == "Hello world"
    
    def test_parse_empty_ref_text(self):
        """Test parsing with empty ref text - treated as no ref_text."""
        result = parse_clone_args("ref= -- Hello")
        
        assert result is not None
        # Empty ref= is treated as None (no transcript)
        assert result.ref_text is None
        assert result.text == "Hello"
    
    def test_parse_multiline_ref_text(self):
        """Test parsing multiline reference text."""
        result = parse_clone_args("ref=Line one\nLine two -- Hello world")
        
        assert result is not None
        assert result.ref_text == "Line one\nLine two"
        assert result.text == "Hello world"
    
    def test_parse_cyrillic_ref_text(self):
        """Test parsing Cyrillic reference text."""
        result = parse_clone_args("ref=Это мой образец -- Скажи это моим голосом")
        
        assert result is not None
        assert result.ref_text == "Это мой образец"
        assert result.text == "Скажи это моим голосом"
    
    def test_parse_cyrillic_text(self):
        """Test parsing Cyrillic synthesis text."""
        result = parse_clone_args("-- Привет мир")
        
        assert result is not None
        assert result.text == "Привет мир"
    
    def test_parse_unicode_text(self):
        """Test parsing text with unicode characters."""
        result = parse_clone_args("-- Hello 世界 🌍")
        
        assert result is not None
        assert result.text == "Hello 世界 🌍"
    
    def test_parse_text_with_extra_spaces(self):
        """Test parsing text with extra spaces."""
        result = parse_clone_args("  --  Hello world  ")
        
        assert result is not None
        assert result.text == "Hello world"
    
    def test_parse_ref_text_with_leading_trailing_spaces(self):
        """Test parsing ref text with leading/trailing spaces."""
        result = parse_clone_args("ref=  sample audio  -- Hello")
        
        assert result is not None
        assert result.ref_text == "sample audio"
        assert result.text == "Hello"
    
    def test_parse_empty_args_returns_none(self):
        """Test that empty args return None."""
        result = parse_clone_args("")
        
        assert result is None
    
    def test_parse_only_separator_returns_args(self):
        """Test that only separator returns parsed args with empty text."""
        result = parse_clone_args("--")
        
        # Parsing succeeds but validation will reject empty text
        assert result is not None
        assert result.ref_text is None
        assert result.text == ""
    
    def test_parse_missing_text_returns_args(self):
        """Test that missing text after separator returns parsed args with empty text."""
        result = parse_clone_args("--")
        
        # Parsing succeeds but validation will reject empty text
        assert result is not None
        assert result.ref_text is None
        assert result.text == ""
    
    def test_parse_missing_separator_returns_none(self):
        """Test that missing separator returns None."""
        result = parse_clone_args("Hello world")
        
        assert result is None
    
    def test_parse_ref_before_separator(self):
        """Test that ref= is only valid before -- separator."""
        result = parse_clone_args("Hello ref=text -- world")
        
        # The pattern expects ref= before --, so this should not match ref pattern
        # The text will be everything after --
        assert result is not None


class TestCloneArgsValidation:
    """Tests for /clone argument validation."""
    
    def test_validate_valid_args(self):
        """Test validation accepts valid arguments."""
        args = ParsedCloneArgs(
            ref_text=None,
            text="Hello world"
        )
        
        result = validate_clone_args(args)
        
        assert result.is_valid is True
    
    def test_validate_valid_args_with_ref(self):
        """Test validation accepts valid arguments with ref_text."""
        args = ParsedCloneArgs(
            ref_text="This is my sample",
            text="Hello world"
        )
        
        result = validate_clone_args(args)
        
        assert result.is_valid is True
    
    def test_validate_empty_text(self):
        """Test validation rejects empty text."""
        args = ParsedCloneArgs(
            ref_text=None,
            text=""
        )
        
        result = validate_clone_args(args)
        
        assert result.is_valid is False
        assert "text" in result.error_message.lower()
    
    def test_validate_text_at_maximum(self):
        """Test validation accepts text at maximum length."""
        args = ParsedCloneArgs(
            ref_text=None,
            text="a" * 1000
        )
        
        result = validate_clone_args(args)
        
        assert result.is_valid is True
    
    def test_validate_text_over_maximum(self):
        """Test validation rejects text over maximum length."""
        args = ParsedCloneArgs(
            ref_text=None,
            text="a" * 1001
        )
        
        result = validate_clone_args(args)
        
        assert result.is_valid is False
        assert "text" in result.error_message.lower()
    
    def test_validate_ref_text_at_maximum(self):
        """Test validation accepts ref_text at maximum length."""
        args = ParsedCloneArgs(
            ref_text="a" * 500,
            text="Hello"
        )
        
        result = validate_clone_args(args)
        
        assert result.is_valid is True
    
    def test_validate_ref_text_over_maximum(self):
        """Test validation rejects ref_text over maximum length."""
        args = ParsedCloneArgs(
            ref_text="a" * 501,
            text="Hello"
        )
        
        result = validate_clone_args(args)
        
        assert result.is_valid is False
        # Error message contains "Reference text" not "ref_text"
        assert "reference" in result.error_message.lower()
        assert "too long" in result.error_message.lower()


class TestCloneCommandValidation:
    """Tests for /clone command-level validation."""
    
    def test_validate_clone_valid_command(self):
        """Test validation accepts valid clone command."""
        parsed = ParsedCommand(
            command=CommandType.CLONE,
            raw_text="/clone -- Hello",
            args="-- Hello",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_clone_command(parsed)
        
        assert result.is_valid is True
    
    def test_validate_clone_with_ref(self):
        """Test validation accepts clone command with ref parameter."""
        parsed = ParsedCommand(
            command=CommandType.CLONE,
            raw_text="/clone ref=Sample -- Hello",
            args="ref=Sample -- Hello",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_clone_command(parsed)
        
        assert result.is_valid is True
    
    def test_validate_clone_empty_args(self):
        """Test validation rejects empty args."""
        parsed = ParsedCommand(
            command=CommandType.CLONE,
            raw_text="/clone",
            args="",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_clone_command(parsed)
        
        assert result.is_valid is False
    
    def test_validate_clone_missing_separator(self):
        """Test validation rejects missing separator."""
        parsed = ParsedCommand(
            command=CommandType.CLONE,
            raw_text="/clone Hello",
            args="Hello",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_clone_command(parsed)
        
        assert result.is_valid is False
    
    def test_validate_clone_empty_text(self):
        """Test validation rejects empty text."""
        parsed = ParsedCommand(
            command=CommandType.CLONE,
            raw_text="/clone --",
            args="--",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_clone_command(parsed)
        
        assert result.is_valid is False
        assert "text" in result.error_message.lower()
    
    def test_validate_clone_non_clone_command_passes(self):
        """Test that non-CLONE commands always pass validation."""
        parsed = ParsedCommand(
            command=CommandType.START,
            raw_text="/start",
            args="",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_clone_command(parsed)
        
        assert result.is_valid is True
    
    def test_validate_clone_unsupported_params_rejected(self):
        """Test that unsupported parameters are rejected."""
        parsed = ParsedCommand(
            command=CommandType.CLONE,
            raw_text="/clone speaker=Ryan -- Hello",
            args="speaker=Ryan -- Hello",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_clone_command(parsed)
        
        assert result.is_valid is False
        assert "speaker" in result.error_message.lower()
    
    def test_validate_clone_speed_param_rejected(self):
        """Test that speed parameter is rejected."""
        parsed = ParsedCommand(
            command=CommandType.CLONE,
            raw_text="/clone speed=1.5 -- Hello",
            args="speed=1.5 -- Hello",
            user_id=12345,
            chat_id=67890,
            message_id=100,
        )
        
        result = validate_clone_command(parsed)
        
        assert result.is_valid is False
        assert "speed" in result.error_message.lower()
