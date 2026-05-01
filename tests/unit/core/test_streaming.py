# FILE: tests/unit/core/test_streaming.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Verify the audio streaming chunker that backs the /api/v1/tts/custom/stream endpoint.
#   SCOPE: iter_audio_chunks chunk boundaries, empty payload handling, chunk_size validation, and stream_generation_result metadata propagation.
#   DEPENDS: M-STREAMING
#   LINKS: V-M-STREAMING
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _make_result - Build a GenerationResult with a configurable audio payload.
#   test_iter_audio_chunks_yields_fixed_size_chunks - Verifies a payload is split into chunks of the requested size with a possibly smaller tail chunk.
#   test_iter_audio_chunks_emits_single_chunk_for_short_payload - Verifies short payloads stay in a single chunk.
#   test_iter_audio_chunks_emits_one_empty_chunk_for_empty_payload - Verifies empty payloads still yield exactly one boundary chunk.
#   test_iter_audio_chunks_rejects_invalid_chunk_size - Verifies non-positive chunk_size raises ValueError.
#   test_stream_generation_result_propagates_metadata - Verifies AudioStreamChunk values carry model/mode/backend/media_type and final flag.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Phase 4.12: introduced unit coverage for iter_audio_chunks and stream_generation_result, exercising both edge cases and the metadata propagation contract]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path

import pytest

from core.contracts.results import AudioResult, GenerationResult
from core.services.streaming import (
    DEFAULT_AUDIO_STREAM_CHUNK_SIZE,
    AudioStreamChunk,
    iter_audio_chunks,
    stream_generation_result,
)

pytestmark = pytest.mark.unit


def _make_result(payload: bytes, *, media_type: str = "audio/wav") -> GenerationResult:
    return GenerationResult(
        audio=AudioResult(path=Path("/tmp/audio.wav"), bytes_data=payload, media_type=media_type),
        saved_path=None,
        model="qwen3-custom-1.7b",
        mode="custom",
        backend="torch",
    )


def test_iter_audio_chunks_yields_fixed_size_chunks() -> None:
    payload = b"x" * 250
    chunks = list(iter_audio_chunks(payload, chunk_size=100))
    assert chunks == [b"x" * 100, b"x" * 100, b"x" * 50]


def test_iter_audio_chunks_emits_single_chunk_for_short_payload() -> None:
    payload = b"hello"
    chunks = list(iter_audio_chunks(payload, chunk_size=DEFAULT_AUDIO_STREAM_CHUNK_SIZE))
    assert chunks == [payload]


def test_iter_audio_chunks_emits_one_empty_chunk_for_empty_payload() -> None:
    chunks = list(iter_audio_chunks(b"", chunk_size=64))
    assert chunks == [b""]


def test_iter_audio_chunks_rejects_invalid_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        list(iter_audio_chunks(b"abc", chunk_size=0))


def test_stream_generation_result_propagates_metadata() -> None:
    result = _make_result(b"abcdefghij")
    chunks = list(stream_generation_result(result, chunk_size=4))

    assert len(chunks) == 3
    assert all(isinstance(chunk, AudioStreamChunk) for chunk in chunks)
    assert b"".join(chunk.data for chunk in chunks) == b"abcdefghij"

    indices = [chunk.index for chunk in chunks]
    assert indices == [0, 1, 2]
    assert [chunk.total_chunks for chunk in chunks] == [3, 3, 3]
    assert [chunk.final for chunk in chunks] == [False, False, True]
    assert {chunk.model for chunk in chunks} == {"qwen3-custom-1.7b"}
    assert {chunk.mode for chunk in chunks} == {"custom"}
    assert {chunk.backend for chunk in chunks} == {"torch"}
    assert {chunk.media_type for chunk in chunks} == {"audio/wav"}
