"""
Unit tests for Telegram dispatcher.

Tests message routing, command handling, and response generation.
"""

# FILE: tests/unit/test_telegram_bot/test_dispatcher.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for Telegram command dispatch and response templates.
#   SCOPE: Message routing, authorization, rate-limit handling, help/start text
#   DEPENDS: M-TELEGRAM
#   LINKS: V-M-TELEGRAM
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   Template tests - Verify help, start, accepted, success, and error message templates
#   TestDispatcherInitialization - Verifies dispatcher dependency wiring
#   TestDispatcherGetHelpMessage - Verifies speaker list formatting in help text
#   TestMessageSenderProtocol - Verifies sender protocol methods used by dispatcher
#   TestDispatcherRouting - Verifies routing, authorization, and private-chat handling across updates
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.1 - Added duplicate delivery requeue coverage for remote /tts and /design job submits]
# END_CHANGE_SUMMARY

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from telegram_bot.handlers.dispatcher import (
    CommandDispatcher,
    MessageSender,
)


class TestHelpMessage:
    """Tests for help message content and formatting."""

    def test_help_message_contains_speaker_info(self):
        """Test that help message contains speaker information."""
        assert "speaker" in CommandDispatcher.HELP_MESSAGE.lower()
        assert "имя голоса" in CommandDispatcher.HELP_MESSAGE.lower()

    def test_help_message_contains_syntax_examples(self):
        """Test that help message contains syntax examples."""
        assert "--" in CommandDispatcher.HELP_MESSAGE
        assert "speaker=" in CommandDispatcher.HELP_MESSAGE
        assert "speed=" in CommandDispatcher.HELP_MESSAGE

    def test_help_message_contains_speaker_list_placeholder(self):
        """Test that help message has speakers placeholder."""
        assert "{speakers}" in CommandDispatcher.HELP_MESSAGE

    def test_help_message_contains_clone_info(self):
        """Test that help message documents clone flow."""
        assert "/clone" in CommandDispatcher.HELP_MESSAGE
        assert "reply" in CommandDispatcher.HELP_MESSAGE.lower()
        assert "audio" in CommandDispatcher.HELP_MESSAGE.lower()


class TestStartMessage:
    """Tests for start message content."""

    def test_start_message_contains_all_commands(self):
        """Test start message lists all supported commands."""
        assert "/start" in CommandDispatcher.START_MESSAGE
        assert "/help" in CommandDispatcher.START_MESSAGE
        assert "/tts" in CommandDispatcher.START_MESSAGE
        assert "/design" in CommandDispatcher.START_MESSAGE
        assert "/clone" in CommandDispatcher.START_MESSAGE

    def test_start_message_mentions_private_chat_only(self):
        """Test start message explains private chat limitation."""
        assert "личном чате" in CommandDispatcher.START_MESSAGE

    def test_start_message_mentions_clone_prerequisite(self):
        """Test start message explains clone requires prior audio."""
        assert "voice" in CommandDispatcher.START_MESSAGE.lower()
        assert "audio" in CommandDispatcher.START_MESSAGE.lower()
        assert "/clone" in CommandDispatcher.START_MESSAGE


class TestAcceptedMessage:
    """Tests for ACCEPTED_MESSAGE template."""

    def test_accepted_message_has_speaker_placeholder(self):
        """Test that ACCEPTED_MESSAGE has speaker placeholder."""
        assert "{speaker}" in CommandDispatcher.ACCEPTED_MESSAGE

    def test_accepted_message_has_speed_placeholder(self):
        """Test that ACCEPTED_MESSAGE has speed placeholder."""
        assert "{speed}" in CommandDispatcher.ACCEPTED_MESSAGE

    def test_accepted_message_explicit_status(self):
        """Test that ACCEPTED_MESSAGE clearly states accepted status."""
        assert "принят" in CommandDispatcher.ACCEPTED_MESSAGE.lower()

    def test_accepted_message_explains_result_will_arrive_later(self):
        """Test accepted message explains async result delivery."""
        assert "отдельное voice сообщение" in CommandDispatcher.ACCEPTED_MESSAGE.lower()


class TestSuccessTemplate:
    """Tests for SUCCESS_TEMPLATE."""

    def test_success_template_has_duration_placeholder(self):
        """Test that SUCCESS_TEMPLATE has duration placeholder."""
        assert "{duration:.1f}" in CommandDispatcher.SUCCESS_TEMPLATE

    def test_success_template_explicit_status(self):
        """Test that SUCCESS_TEMPLATE clearly states success status."""
        assert "готово" in CommandDispatcher.SUCCESS_TEMPLATE.lower()

    def test_success_template_mentions_voice_message(self):
        """Test success template says result is a voice message."""
        assert "voice сообщение" in CommandDispatcher.SUCCESS_TEMPLATE.lower()


class TestErrorTemplate:
    """Tests for error template."""

    def test_error_template_has_placeholder(self):
        """Test that ERROR_TEMPLATE has error placeholder."""
        assert "{error}" in CommandDispatcher.ERROR_TEMPLATE

    def test_error_template_points_to_help(self):
        """Test that error template points user to help."""
        assert "/help" in CommandDispatcher.ERROR_TEMPLATE
        assert (
            "синтаксис" in CommandDispatcher.ERROR_TEMPLATE.lower()
            or "примеры" in CommandDispatcher.ERROR_TEMPLATE.lower()
        )


class TestDispatcherInitialization:
    """Tests for dispatcher initialization."""

    def test_dispatcher_stores_dependencies(self):
        """Test dispatcher stores dependencies."""
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_sender = MagicMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
        )

        assert dispatcher._synthesizer is mock_synth
        assert dispatcher._settings is mock_settings
        assert dispatcher._sender is mock_sender


class TestDispatcherGetHelpMessage:
    """Tests for help message formatting."""

    def test_get_help_message_formats_speakers(self):
        """Test that _get_help_message formats speakers list."""
        mock_settings = MagicMock()
        mock_sender = MagicMock()

        dispatcher = CommandDispatcher(
            synthesizer=MagicMock(),
            settings=mock_settings,
            sender=mock_sender,
        )

        help_msg = dispatcher._get_help_message()

        # Should have Vivian (from SPEAKER_MAP)
        assert "Vivian" in help_msg
        # Should have placeholder replaced
        assert "{speakers}" not in help_msg


class TestMessageSenderProtocol:
    """Tests for MessageSender protocol."""

    def test_message_sender_has_send_text_method(self):
        """Test MessageSender has send_text method."""
        assert hasattr(MessageSender, "send_text")

    def test_message_sender_has_send_voice_method(self):
        """Test MessageSender has send_voice method."""
        assert hasattr(MessageSender, "send_voice")


class TestDispatcherRouting:
    """Tests for dispatcher command routing via handle_update."""

    @pytest.mark.anyio
    async def test_dispatcher_ignores_non_private_chat(self):
        """Test that dispatcher ignores non-private chats."""
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_sender = MagicMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
        )

        # Call with group chat type
        await dispatcher.handle_update(
            text="/help",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="group",
        )

        # Should not send any message
        mock_sender.send_text.assert_not_called()

    @pytest.mark.anyio
    async def test_dispatcher_denies_unauthorized_user(self):
        """Test that dispatcher denies unauthorized users."""
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = False
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
        )

        await dispatcher.handle_update(
            text="/help",
            user_id=99999,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        # Should send access denied message
        mock_sender.send_text.assert_called_once()
        args = mock_sender.send_text.call_args[0]
        assert "denied" in args[1].lower() or "not authorized" in args[1].lower()

    @pytest.mark.anyio
    async def test_dispatcher_allows_authorized_user(self):
        """Test that dispatcher allows authorized users."""
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
        )

        # Use /help command which should respond
        await dispatcher.handle_update(
            text="/help",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        # Should send help message
        mock_sender.send_text.assert_called()

    @pytest.mark.anyio
    async def test_dispatcher_rejects_rate_limited_user(self):
        """Test dispatcher rejects commands when rate limit is exceeded."""
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()
        mock_rate_limiter = MagicMock()
        mock_rate_limiter.is_enabled = True
        mock_rate_limiter.check_and_consume.return_value = MagicMock(
            allowed=False,
            retry_after_seconds=3.2,
            limit=20,
        )

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
            rate_limiter=mock_rate_limiter,
        )

        await dispatcher.handle_update(
            text="/help",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        mock_rate_limiter.check_and_consume.assert_called_once_with(67890)
        mock_sender.send_text.assert_called_once()
        args = mock_sender.send_text.call_args[0]
        assert "слишком много запросов" in args[1].lower()
        assert "4" in args[1]

    @pytest.mark.anyio
    async def test_dispatcher_ignores_non_command_text(self):
        """Test dispatcher ignores non-command text."""
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_sender = MagicMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
        )

        await dispatcher.handle_update(
            text="Just a regular message",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        # Should not send any response
        mock_sender.send_text.assert_not_called()

    @pytest.mark.anyio
    async def test_dispatcher_ignores_empty_text(self):
        """Test dispatcher ignores empty text."""
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_sender = MagicMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
        )

        await dispatcher.handle_update(
            text="",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        # Should not send any response
        mock_sender.send_text.assert_not_called()


class TestDispatcherTTSHandling:
    """Tests for dispatcher /tts handling with new syntax."""

    @pytest.mark.anyio
    async def test_dispatcher_processes_basic_tts(self):
        """Test dispatcher processes basic /tts command."""
        mock_synth = MagicMock()
        mock_synth.synthesize = AsyncMock(
            return_value=MagicMock(
                audio_bytes=b"fake_audio",
                duration_seconds=1.5,
            )
        )
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_settings.telegram_max_text_length = 1000
        mock_settings.resolve_runtime_model_binding.side_effect = (
            lambda mode: f"model-{mode}"
        )
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()
        mock_sender.send_voice = AsyncMock()
        mock_orchestrator = _AsyncJobOrchestratorDouble()
        mock_delivery_store = MagicMock()
        mock_delivery_store.create = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
            job_orchestrator=mock_orchestrator,
            delivery_store=mock_delivery_store,
        )

        await dispatcher.handle_update(
            text="/tts -- Hello world",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        # Should send accepted message first
        assert mock_sender.send_text.called
        # Should enqueue remote job instead of local synthesis fallback
        mock_orchestrator.submit_tts_job.assert_awaited_once()
        mock_synth.synthesize.assert_not_called()

    @pytest.mark.anyio
    async def test_dispatcher_with_speaker_param(self):
        """Test dispatcher handles /tts with speaker parameter."""
        mock_synth = MagicMock()
        mock_synth.synthesize = AsyncMock(
            return_value=MagicMock(
                audio_bytes=b"fake_audio",
                duration_seconds=1.0,
            )
        )
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_settings.telegram_max_text_length = 1000
        mock_settings.resolve_runtime_model_binding.side_effect = (
            lambda mode: f"model-{mode}"
        )
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()
        mock_sender.send_voice = AsyncMock()
        mock_orchestrator = _AsyncJobOrchestratorDouble()
        mock_delivery_store = MagicMock()
        mock_delivery_store.create = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
            job_orchestrator=mock_orchestrator,
            delivery_store=mock_delivery_store,
        )

        await dispatcher.handle_update(
            text="/tts speaker=Vivian -- Hello",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        # Should process the command
        mock_orchestrator.submit_tts_job.assert_awaited_once()

    @pytest.mark.anyio
    async def test_dispatcher_with_speed_param(self):
        """Test dispatcher handles /tts with speed parameter."""
        mock_synth = MagicMock()
        mock_synth.synthesize = AsyncMock(
            return_value=MagicMock(
                audio_bytes=b"fake_audio",
                duration_seconds=1.5,
            )
        )
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_settings.telegram_max_text_length = 1000
        mock_settings.resolve_runtime_model_binding.side_effect = (
            lambda mode: f"model-{mode}"
        )
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()
        mock_sender.send_voice = AsyncMock()
        mock_orchestrator = _AsyncJobOrchestratorDouble()
        mock_delivery_store = MagicMock()
        mock_delivery_store.create = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
            job_orchestrator=mock_orchestrator,
            delivery_store=mock_delivery_store,
        )

        await dispatcher.handle_update(
            text="/tts speed=1.5 -- Hello",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        mock_orchestrator.submit_tts_job.assert_awaited_once()

    @pytest.mark.anyio
    async def test_dispatcher_with_both_params(self):
        """Test dispatcher handles /tts with both speaker and speed."""
        mock_synth = MagicMock()
        mock_synth.synthesize = AsyncMock(
            return_value=MagicMock(
                audio_bytes=b"fake_audio",
                duration_seconds=1.0,
            )
        )
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_settings.telegram_max_text_length = 1000
        mock_settings.resolve_runtime_model_binding.side_effect = (
            lambda mode: f"model-{mode}"
        )
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()
        mock_sender.send_voice = AsyncMock()
        mock_orchestrator = _AsyncJobOrchestratorDouble()
        mock_delivery_store = MagicMock()
        mock_delivery_store.create = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
            job_orchestrator=mock_orchestrator,
            delivery_store=mock_delivery_store,
        )

        await dispatcher.handle_update(
            text="/tts speaker=Ryan speed=0.8 -- Hello",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        mock_orchestrator.submit_tts_job.assert_awaited_once()

    @pytest.mark.anyio
    async def test_tts_duplicate_requeues_delivery_metadata(self):
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_settings.telegram_max_text_length = 1000
        mock_settings.resolve_runtime_model_binding.side_effect = (
            lambda mode: f"model-{mode}"
        )
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()
        mock_sender.send_voice = AsyncMock()
        mock_orchestrator = _AsyncJobOrchestratorDouble()
        mock_orchestrator.submit_tts_job = AsyncMock(
            return_value=MagicMock(
                success=True,
                is_duplicate=True,
                job_id="job-tts-dup-123",
                submit_request_id="submit-tts-dup-123",
            )
        )
        mock_delivery_store = MagicMock()
        mock_delivery_store.get = AsyncMock(return_value=None)
        mock_delivery_store.create = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
            job_orchestrator=mock_orchestrator,
            delivery_store=mock_delivery_store,
        )

        await dispatcher.handle_update(
            text="/tts -- Hello world",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        mock_delivery_store.create.assert_awaited_once_with(
            chat_id=12345,
            message_id=1,
            job_id="job-tts-dup-123",
            submit_request_id="submit-tts-dup-123",
        )

    @pytest.mark.anyio
    async def test_tts_duplicate_preserves_existing_delivered_metadata(self):
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_settings.telegram_max_text_length = 1000
        mock_settings.resolve_runtime_model_binding.side_effect = (
            lambda mode: f"model-{mode}"
        )
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()
        mock_sender.send_voice = AsyncMock()
        mock_orchestrator = _AsyncJobOrchestratorDouble()
        mock_orchestrator.submit_tts_job = AsyncMock(
            return_value=MagicMock(
                success=True,
                is_duplicate=True,
                job_id="job-tts-dup-123",
                submit_request_id="submit-tts-dup-123",
            )
        )
        mock_delivery_store = MagicMock()
        mock_delivery_store.get = AsyncMock(
            return_value={
                "chat_id": 12345,
                "message_id": 1,
                "job_id": "job-tts-dup-123",
                "status": "delivered",
                "delivered_at": "2026-04-23T10:00:00+00:00",
            }
        )
        mock_delivery_store.create = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
            job_orchestrator=mock_orchestrator,
            delivery_store=mock_delivery_store,
        )

        await dispatcher.handle_update(
            text="/tts -- Hello world",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        mock_delivery_store.create.assert_not_awaited()



class TestDispatcherErrorHandling:
    """Tests for dispatcher error handling."""

    @pytest.mark.anyio
    async def test_dispatcher_handles_invalid_speaker(self):
        """Test dispatcher handles invalid speaker gracefully."""
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
        )

        await dispatcher.handle_update(
            text="/tts speaker=Invalid -- Hello",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        # Should send error message
        mock_sender.send_text.assert_called()
        args = mock_sender.send_text.call_args[0]
        assert "speaker" in args[1].lower() or "unknown" in args[1].lower()

    @pytest.mark.anyio
    async def test_dispatcher_handles_invalid_speed(self):
        """Test dispatcher handles invalid speed gracefully."""
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
        )

        await dispatcher.handle_update(
            text="/tts speed=5.0 -- Hello",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        # Should send error message
        mock_sender.send_text.assert_called()
        args = mock_sender.send_text.call_args[0]
        assert "speed" in args[1].lower() or "between" in args[1].lower()

    @pytest.mark.anyio
    async def test_dispatcher_handles_empty_text(self):
        """Test dispatcher handles empty TTS text."""
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
        )

        await dispatcher.handle_update(
            text="/tts",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        # Should send error message
        mock_sender.send_text.assert_called()
        args = mock_sender.send_text.call_args[0]
        assert "provide" in args[1].lower() or "empty" in args[1].lower()


class TestDispatcherDesignHandling:
    """Tests for dispatcher /design command handling."""

    def test_help_message_contains_design_info(self):
        """Test that help message contains /design information."""
        assert "/design" in CommandDispatcher.HELP_MESSAGE
        assert "voice design" in CommandDispatcher.HELP_MESSAGE.lower()

    def test_help_message_contains_clone_info(self):
        """Test that help message contains /clone information."""
        assert "/clone" in CommandDispatcher.HELP_MESSAGE
        assert "reply" in CommandDispatcher.HELP_MESSAGE.lower()
        assert "audio" in CommandDispatcher.HELP_MESSAGE.lower()

    def test_design_accepted_message_exists(self):
        """Test that DESIGN_ACCEPTED_MESSAGE exists."""
        assert hasattr(CommandDispatcher, "DESIGN_ACCEPTED_MESSAGE")
        assert (
            "голос" in CommandDispatcher.DESIGN_ACCEPTED_MESSAGE.lower()
            or "описанию" in CommandDispatcher.DESIGN_ACCEPTED_MESSAGE.lower()
        )
        assert "запрос принят" in CommandDispatcher.DESIGN_ACCEPTED_MESSAGE.lower()

    @pytest.mark.anyio
    async def test_dispatcher_processes_basic_design(self):
        """Test dispatcher processes basic /design command."""
        mock_synth = MagicMock()
        mock_synth.synthesize_design = AsyncMock(
            return_value=MagicMock(
                audio_bytes=b"fake_audio",
                duration_seconds=1.5,
            )
        )
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_settings.telegram_max_text_length = 1000
        mock_settings.resolve_runtime_model_binding.side_effect = (
            lambda mode: f"model-{mode}"
        )
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()
        mock_sender.send_voice = AsyncMock()
        mock_orchestrator = _AsyncJobOrchestratorDouble()
        mock_delivery_store = MagicMock()
        mock_delivery_store.create = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
            job_orchestrator=mock_orchestrator,
            delivery_store=mock_delivery_store,
        )

        await dispatcher.handle_update(
            text="/design calm narrator -- Hello world",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        # Should send accepted message first
        assert mock_sender.send_text.called
        # Should enqueue remote design job instead of local synthesis fallback
        mock_orchestrator.submit_design_job.assert_awaited_once()
        mock_synth.synthesize_design.assert_not_called()

    @pytest.mark.anyio
    async def test_design_duplicate_requeues_delivery_metadata(self):
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_settings.telegram_max_text_length = 1000
        mock_settings.resolve_runtime_model_binding.side_effect = (
            lambda mode: f"model-{mode}"
        )
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()
        mock_sender.send_voice = AsyncMock()
        mock_orchestrator = _AsyncJobOrchestratorDouble()
        mock_orchestrator.submit_design_job = AsyncMock(
            return_value=MagicMock(
                success=True,
                is_duplicate=True,
                job_id="job-design-dup-123",
                submit_request_id="submit-design-dup-123",
            )
        )
        mock_delivery_store = MagicMock()
        mock_delivery_store.get = AsyncMock(return_value=None)
        mock_delivery_store.create = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
            job_orchestrator=mock_orchestrator,
            delivery_store=mock_delivery_store,
        )

        await dispatcher.handle_update(
            text="/design calm narrator -- Hello world",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        mock_delivery_store.create.assert_awaited_once_with(
            chat_id=12345,
            message_id=1,
            job_id="job-design-dup-123",
            submit_request_id="submit-design-dup-123",
        )

    @pytest.mark.anyio
    async def test_design_duplicate_preserves_existing_delivered_metadata(self):
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_settings.telegram_max_text_length = 1000
        mock_settings.resolve_runtime_model_binding.side_effect = (
            lambda mode: f"model-{mode}"
        )
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()
        mock_sender.send_voice = AsyncMock()
        mock_orchestrator = _AsyncJobOrchestratorDouble()
        mock_orchestrator.submit_design_job = AsyncMock(
            return_value=MagicMock(
                success=True,
                is_duplicate=True,
                job_id="job-design-dup-123",
                submit_request_id="submit-design-dup-123",
            )
        )
        mock_delivery_store = MagicMock()
        mock_delivery_store.get = AsyncMock(
            return_value={
                "chat_id": 12345,
                "message_id": 1,
                "job_id": "job-design-dup-123",
                "status": "delivered",
                "delivered_at": "2026-04-23T10:00:00+00:00",
            }
        )
        mock_delivery_store.create = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
            job_orchestrator=mock_orchestrator,
            delivery_store=mock_delivery_store,
        )

        await dispatcher.handle_update(
            text="/design calm narrator -- Hello world",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        mock_delivery_store.create.assert_not_awaited()


class TestCloneJobDelivery:
    @pytest.mark.anyio
    async def test_clone_duplicate_requeues_delivery_metadata(self):
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.submit_clone_job = AsyncMock(return_value=MagicMock(
            success=True,
            is_duplicate=True,
            job_id="job-existing-123",
            submit_request_id="submit-existing-123",
        ))
        mock_delivery_store = MagicMock()
        mock_delivery_store.get = AsyncMock(return_value=None)
        mock_delivery_store.create = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
            job_orchestrator=mock_orchestrator,
            delivery_store=mock_delivery_store,
        )

        await dispatcher._handle_clone_via_job(
            chat_id=12345,
            message_id=67890,
            text="Скажи это моим голосом",
            ref_text="пример референса",
            language="ru",
            ref_audio_path="/tmp/ref.wav",
            ref_audio_content_type="audio/wav",
        )

        mock_delivery_store.create.assert_awaited_once_with(
            chat_id=12345,
            message_id=67890,
            job_id="job-existing-123",
            submit_request_id="submit-existing-123",
        )

    @pytest.mark.anyio
    async def test_clone_duplicate_preserves_existing_delivered_metadata(self):
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.submit_clone_job = AsyncMock(return_value=MagicMock(
            success=True,
            is_duplicate=True,
            job_id="job-existing-123",
            submit_request_id="submit-existing-123",
        ))
        mock_delivery_store = MagicMock()
        mock_delivery_store.get = AsyncMock(
            return_value={
                "chat_id": 12345,
                "message_id": 67890,
                "job_id": "job-existing-123",
                "status": "delivered",
                "delivered_at": "2026-04-23T10:00:00+00:00",
            }
        )
        mock_delivery_store.create = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
            job_orchestrator=mock_orchestrator,
            delivery_store=mock_delivery_store,
        )

        await dispatcher._handle_clone_via_job(
            chat_id=12345,
            message_id=67890,
            text="Скажи это моим голосом",
            ref_text="пример референса",
            language="ru",
            ref_audio_path="/tmp/ref.wav",
            ref_audio_content_type="audio/wav",
        )

        mock_delivery_store.create.assert_not_awaited()


class TestCloneMediaPreparation:
    @pytest.mark.anyio
    async def test_clone_stage_failure_sends_user_error(self):
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_settings.telegram_max_text_length = 1000
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()
        mock_sender._client = MagicMock()
        mock_orchestrator = MagicMock()
        mock_delivery_store = MagicMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
            job_orchestrator=mock_orchestrator,
            delivery_store=mock_delivery_store,
        )

        reply_to_message = {
            "message_id": 159,
            "voice": {
                "file_id": "voice123",
                "mime_type": "audio/ogg",
                "file_size": 1024,
            },
        }

        with patch(
            "telegram_bot.handlers.dispatcher.stage_clone_media",
            new=AsyncMock(side_effect=RuntimeError("staging stuck")),
        ):
            await dispatcher.handle_update(
                text="/clone lang=ru ref=пример -- Скажи это моим голосом",
                user_id=67890,
                chat_id=12345,
                message_id=777,
                chat_type="private",
                reply_to_message=reply_to_message,
            )

        mock_sender.send_text.assert_called_once()
        args = mock_sender.send_text.call_args[0]
        assert "reference audio" in args[1]
        mock_orchestrator.submit_clone_job.assert_not_called()

    @pytest.mark.anyio
    async def test_clone_uses_command_message_id_for_job_submission(self):
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_settings.telegram_max_text_length = 1000
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()
        mock_sender._client = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.submit_clone_job = AsyncMock(return_value=MagicMock(
            success=True,
            is_duplicate=False,
            job_id="job-new-123",
            submit_request_id="submit-new-123",
        ))
        mock_delivery_store = MagicMock()
        mock_delivery_store.create = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
            job_orchestrator=mock_orchestrator,
            delivery_store=mock_delivery_store,
        )

        staged_media = MagicMock()
        staged_media.get_audio_path.return_value = "/tmp/ref.wav"
        staged_media.was_converted = True

        reply_to_message = {
            "message_id": 159,
            "audio": {
                "file_id": "audio123",
                "mime_type": "audio/mpeg",
                "file_size": 1024,
            },
        }

        with patch(
            "telegram_bot.handlers.dispatcher.stage_clone_media",
            new=AsyncMock(
                return_value=(
                    staged_media,
                    MagicMock(content_type="audio/mpeg", file_size=1024),
                )
            ),
        ):
            await dispatcher.handle_update(
                text="/clone ref=пример -- Новый текст",
                user_id=67890,
                chat_id=12345,
                message_id=777,
                chat_type="private",
                reply_to_message=reply_to_message,
            )

        mock_orchestrator.submit_clone_job.assert_called_once()
        assert mock_orchestrator.submit_clone_job.call_args.kwargs["message_id"] == 777

    @pytest.mark.anyio
    async def test_dispatcher_rejects_design_missing_separator(self):
        """Test dispatcher rejects /design without separator."""
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
        )

        await dispatcher.handle_update(
            text="/design calm narrator Hello",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        # Should send error message
        mock_sender.send_text.assert_called()
        args = mock_sender.send_text.call_args[0]
        assert "design" in args[1].lower()

    @pytest.mark.anyio
    async def test_dispatcher_rejects_design_empty_args(self):
        """Test dispatcher rejects /design without arguments."""
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
        )

        await dispatcher.handle_update(
            text="/design",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        # Should send error message
        mock_sender.send_text.assert_called()
        args = mock_sender.send_text.call_args[0]
        assert "design" in args[1].lower()

    @pytest.mark.anyio
    async def test_dispatcher_rejects_design_empty_text(self):
        """Test dispatcher rejects /design with empty text."""
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
        )

        await dispatcher.handle_update(
            text="/design calm narrator --",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        # Should send error message
        mock_sender.send_text.assert_called()
        args = mock_sender.send_text.call_args[0]
        assert "design" in args[1].lower() or "text" in args[1].lower()

    @pytest.mark.anyio
    async def test_dispatcher_rejects_design_short_voice_description(self):
        """Test dispatcher rejects /design with too short voice description."""
        mock_synth = MagicMock()
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
        )

        await dispatcher.handle_update(
            text="/design ab -- Hello",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        # Should send error message
        mock_sender.send_text.assert_called()
        args = mock_sender.send_text.call_args[0]
        assert "design" in args[1].lower() or "voice" in args[1].lower()

    @pytest.mark.anyio
    async def test_dispatcher_handles_design_with_cyrillic_text(self):
        """Test dispatcher handles /design with Cyrillic text."""
        mock_synth = MagicMock()
        mock_synth.synthesize_design = AsyncMock(
            return_value=MagicMock(
                audio_bytes=b"fake_audio",
                duration_seconds=1.5,
            )
        )
        mock_settings = MagicMock()
        mock_settings.is_user_allowed.return_value = True
        mock_settings.telegram_max_text_length = 1000
        mock_settings.resolve_runtime_model_binding.side_effect = (
            lambda mode: f"model-{mode}"
        )
        mock_sender = MagicMock()
        mock_sender.send_text = AsyncMock()
        mock_sender.send_voice = AsyncMock()
        mock_orchestrator = _AsyncJobOrchestratorDouble()
        mock_delivery_store = MagicMock()
        mock_delivery_store.create = AsyncMock()

        dispatcher = CommandDispatcher(
            synthesizer=mock_synth,
            settings=mock_settings,
            sender=mock_sender,
            job_orchestrator=mock_orchestrator,
            delivery_store=mock_delivery_store,
        )

        await dispatcher.handle_update(
            text="/design молодой ведущий подкаста -- Привет мир",
            user_id=67890,
            chat_id=12345,
            message_id=1,
            chat_type="private",
        )

        mock_orchestrator.submit_design_job.assert_awaited_once()
        mock_synth.synthesize_design.assert_not_called()
class _AsyncJobOrchestratorDouble:
    def __init__(self):
        self.submit_tts_job = AsyncMock(
            return_value=MagicMock(
                success=True,
                is_duplicate=False,
                job_id="job-tts-123",
                submit_request_id="submit-tts-123",
            )
        )
        self.submit_design_job = AsyncMock(
            return_value=MagicMock(
                success=True,
                is_duplicate=False,
                job_id="job-design-123",
                submit_request_id="submit-design-123",
            )
        )
        self.submit_clone_job = AsyncMock(
            return_value=MagicMock(
                success=True,
                is_duplicate=False,
                job_id="job-clone-123",
                submit_request_id="submit-clone-123",
            )
        )
