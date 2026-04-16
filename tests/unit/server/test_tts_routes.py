# FILE: tests/unit/server/test_tts_routes.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for HTTP TTS route business-rule helpers.
#   SCOPE: Model capability gating, async idempotency fingerprints, and async submission shaping helpers
#   DEPENDS: M-SERVER
#   LINKS: V-M-SERVER
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _make_request - Build a minimal request stub with registry, settings, request id, and principal state
#   test_ensure_requested_model_capability_allows_supported_model - Verifies capability validation is a no-op for supported model/mode combinations
#   test_ensure_requested_model_capability_rejects_unsupported_model - Verifies explicit model capability mismatches raise typed model capability errors
#   test_build_idempotency_fingerprint_is_stable_for_equivalent_payloads - Verifies payload ordering does not affect idempotency fingerprints
#   test_build_idempotency_fingerprint_changes_when_language_changes - Verifies language participates in async idempotency fingerprints
#   test_create_custom_job_submission_from_openai_builds_idempotent_submission - Verifies OpenAI payload shaping preserves defaults and idempotency metadata
#   test_create_custom_job_submission_from_custom_uses_instruction_fallback_and_save_override - Verifies custom payload shaping honors save-output overrides and instruction fallback rules
#   test_create_design_job_submission_builds_design_operation_submission - Verifies design payload shaping preserves normalized text, language, and idempotency metadata
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added focused unit coverage for routes_tts capability checks and async submission shaping helpers]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.contracts.jobs import JobOperation
from core.errors import ModelCapabilityError
from server.api.routes_tts import (
    build_idempotency_fingerprint,
    create_custom_job_submission_from_custom,
    create_custom_job_submission_from_openai,
    create_design_job_submission,
    ensure_requested_model_capability,
)
from server.bootstrap import ServerSettings
from server.schemas.audio import CustomTTSRequest, DesignTTSRequest, OpenAISpeechRequest
from tests.support.api_fakes import DummyRegistry


pytestmark = pytest.mark.unit


def _make_request(tmp_path: Path) -> SimpleNamespace:
    settings = ServerSettings(
        models_dir=tmp_path / ".models",
        outputs_dir=tmp_path / ".outputs",
        voices_dir=tmp_path / ".voices",
        upload_staging_dir=tmp_path / ".uploads",
        default_save_output=False,
        max_input_text_chars=64,
        request_timeout_seconds=123.0,
    )
    settings.ensure_directories()
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                registry=DummyRegistry(settings),
                settings=settings,
            )
        ),
        state=SimpleNamespace(
            request_id="req-123",
            principal=SimpleNamespace(principal_id="principal-1"),
        ),
    )


def test_ensure_requested_model_capability_allows_supported_model(tmp_path: Path):
    request = _make_request(tmp_path)

    ensure_requested_model_capability(
        request,
        "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        execution_mode="custom",
    )


def test_ensure_requested_model_capability_rejects_unsupported_model(tmp_path: Path):
    request = _make_request(tmp_path)

    with pytest.raises(ModelCapabilityError) as exc_info:
        ensure_requested_model_capability(
            request,
            "Piper-en_US-lessac-medium",
            execution_mode="design",
        )

    details = exc_info.value.context.to_dict()
    assert details["model"] == "Piper-en_US-lessac-medium"
    assert details["capability"] == "voice_description_tts"
    assert details["family"] == "Piper"
    assert details["supported_capabilities"] == ["preset_speaker_tts"]


def test_build_idempotency_fingerprint_is_stable_for_equivalent_payloads():
    first = build_idempotency_fingerprint(
        operation=JobOperation.SYNTHESIZE_CUSTOM,
        payload={"language": "auto", "text": "hello", "voice": "Vivian"},
    )
    second = build_idempotency_fingerprint(
        operation=JobOperation.SYNTHESIZE_CUSTOM,
        payload={"voice": "Vivian", "text": "hello", "language": "auto"},
    )

    assert first == second


def test_build_idempotency_fingerprint_changes_when_language_changes():
    auto_fingerprint = build_idempotency_fingerprint(
        operation=JobOperation.SYNTHESIZE_CUSTOM,
        payload={"text": "hello", "voice": "Vivian", "language": "auto"},
    )
    ru_fingerprint = build_idempotency_fingerprint(
        operation=JobOperation.SYNTHESIZE_CUSTOM,
        payload={"text": "hello", "voice": "Vivian", "language": "ru"},
    )

    assert auto_fingerprint != ru_fingerprint


def test_create_custom_job_submission_from_openai_builds_idempotent_submission(
    tmp_path: Path,
):
    request = _make_request(tmp_path)
    payload = OpenAISpeechRequest(
        model="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        input="  Hello async world  ",
        voice="Vivian",
        language="RU",
        response_format="pcm",
        speed=1.25,
    )

    submission = create_custom_job_submission_from_openai(
        request,
        payload,
        idempotency_key="idem-openai",
    )

    assert submission.operation is JobOperation.SYNTHESIZE_CUSTOM
    assert submission.submit_request_id == "req-123"
    assert submission.owner_principal_id == "principal-1"
    assert submission.response_format == "pcm"
    assert submission.save_output is False
    assert submission.execution_timeout_seconds == 123.0
    assert submission.idempotency_key == "idem-openai"
    assert submission.idempotency_scope == "principal-1"
    assert submission.idempotency_fingerprint is not None
    assert submission.command.text == "Hello async world"
    assert submission.command.language == "ru"
    assert submission.command.speaker == "Vivian"
    assert submission.command.instruct == "Normal tone"
    assert submission.command.speed == 1.25


def test_create_custom_job_submission_from_custom_uses_instruction_fallback_and_save_override(
    tmp_path: Path,
):
    request = _make_request(tmp_path)
    payload = CustomTTSRequest(
        model="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        text="  Make it energetic  ",
        speaker="Alice",
        emotion="Excited",
        instruct=None,
        language="EN",
        speed=0.9,
        save_output=True,
    )

    submission = create_custom_job_submission_from_custom(
        request,
        payload,
        idempotency_key="idem-custom",
    )

    assert submission.operation is JobOperation.SYNTHESIZE_CUSTOM
    assert submission.response_format == "wav"
    assert submission.save_output is True
    assert submission.idempotency_key == "idem-custom"
    assert submission.idempotency_scope == "principal-1"
    assert submission.idempotency_fingerprint is not None
    assert submission.command.text == "Make it energetic"
    assert submission.command.speaker == "Alice"
    assert submission.command.instruct == "Excited"
    assert submission.command.language == "en"
    assert submission.command.save_output is True


def test_create_design_job_submission_builds_design_operation_submission(tmp_path: Path):
    request = _make_request(tmp_path)
    payload = DesignTTSRequest(
        model="Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        text="  Design this voice  ",
        voice_description=" calm narrator ",
        language="DE",
        save_output=None,
    )

    submission = create_design_job_submission(
        request,
        payload,
        idempotency_key="idem-design",
    )

    assert submission.operation is JobOperation.SYNTHESIZE_DESIGN
    assert submission.response_format == "wav"
    assert submission.save_output is False
    assert submission.idempotency_key == "idem-design"
    assert submission.idempotency_scope == "principal-1"
    assert submission.idempotency_fingerprint is not None
    assert submission.command.text == "Design this voice"
    assert submission.command.voice_description == "calm narrator"
    assert submission.command.language == "de"
    assert submission.command.save_output is False
