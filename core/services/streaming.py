# FILE: core/services/streaming.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide a transport-agnostic seam that turns a completed GenerationResult into an iterator of fixed-size audio chunks so HTTP streaming endpoints (and future native-streaming backends) share one chunking contract.
#   SCOPE: AudioStreamChunk dataclass, iter_audio_chunks(bytes) generic byte chunker, and stream_generation_result(GenerationResult) helper that wraps a result's audio payload with structured metadata.
#   DEPENDS: M-CONTRACTS
#   LINKS: M-TTS-SERVICE, M-STREAMING
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   DEFAULT_AUDIO_STREAM_CHUNK_SIZE - Default chunk size in bytes (64 KiB) used when callers do not override it.
#   AudioStreamChunk - Frozen container describing a single chunk of streamed audio plus its position metadata.
#   iter_audio_chunks - Generic byte chunker that yields fixed-size byte slices from a bytes blob.
#   stream_generation_result - Wrap a GenerationResult's audio payload as an iterator of AudioStreamChunk values with model/mode/backend metadata.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Phase 4.12: introduced the audio streaming chunker seam used by the new /api/v1/tts/custom/stream endpoint and reusable by future native-streaming backends]
# END_CHANGE_SUMMARY

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from core.contracts.results import GenerationResult

DEFAULT_AUDIO_STREAM_CHUNK_SIZE: int = 64 * 1024


# START_CONTRACT: AudioStreamChunk
#   PURPOSE: Describe a single chunk of streamed audio plus its position metadata so transports can render either chunked HTTP responses, SSE frames, or future binary websocket frames.
#   INPUTS: { index: int - Zero-based chunk index, total_chunks: int - Total chunk count for the result, data: bytes - Raw chunk payload, final: bool - True for the last chunk in the stream, media_type: str - MIME type of the underlying audio payload, model: str - Model identifier from the originating GenerationResult, mode: str - Synthesis mode from the originating GenerationResult, backend: str - Backend key that produced the result }
#   OUTPUTS: { instance - Immutable chunk descriptor }
#   SIDE_EFFECTS: none
#   LINKS: M-STREAMING
# END_CONTRACT: AudioStreamChunk
@dataclass(frozen=True)
class AudioStreamChunk:
    index: int
    total_chunks: int
    data: bytes
    final: bool
    media_type: str
    model: str
    mode: str
    backend: str


# START_CONTRACT: iter_audio_chunks
#   PURPOSE: Yield fixed-size byte slices from a raw bytes payload so any blob (regardless of its origin) can be streamed.
#   INPUTS: { payload: bytes - Source bytes to chunk, chunk_size: int - Maximum chunk size in bytes (must be > 0) }
#   OUTPUTS: { Iterator[bytes] - Iterator that yields one or more byte slices whose concatenation equals the payload }
#   SIDE_EFFECTS: none
#   LINKS: M-STREAMING
# END_CONTRACT: iter_audio_chunks
def iter_audio_chunks(
    payload: bytes,
    *,
    chunk_size: int = DEFAULT_AUDIO_STREAM_CHUNK_SIZE,
) -> Iterator[bytes]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not payload:
        # Always yield at least one chunk so callers get a deterministic boundary even for empty payloads.
        yield b""
        return
    for offset in range(0, len(payload), chunk_size):
        yield payload[offset : offset + chunk_size]


# START_CONTRACT: stream_generation_result
#   PURPOSE: Wrap a completed GenerationResult as an iterator of AudioStreamChunk values that include model/mode/backend metadata so transports can render structured streaming responses.
#   INPUTS: { result: GenerationResult - Completed generation result whose audio payload should be streamed, chunk_size: int - Maximum chunk size in bytes (must be > 0) }
#   OUTPUTS: { Iterator[AudioStreamChunk] - Iterator that yields one AudioStreamChunk per data chunk; the last chunk has final=True }
#   SIDE_EFFECTS: none
#   LINKS: M-STREAMING, M-TTS-SERVICE
# END_CONTRACT: stream_generation_result
def stream_generation_result(
    result: GenerationResult,
    *,
    chunk_size: int = DEFAULT_AUDIO_STREAM_CHUNK_SIZE,
) -> Iterator[AudioStreamChunk]:
    payload = result.audio.bytes_data
    chunks: Iterable[bytes] = list(iter_audio_chunks(payload, chunk_size=chunk_size))
    total = len(chunks)
    for index, data in enumerate(chunks):
        yield AudioStreamChunk(
            index=index,
            total_chunks=total,
            data=data,
            final=index == total - 1,
            media_type=result.audio.media_type,
            model=result.model,
            mode=result.mode,
            backend=result.backend,
        )


__all__ = [
    "DEFAULT_AUDIO_STREAM_CHUNK_SIZE",
    "AudioStreamChunk",
    "iter_audio_chunks",
    "stream_generation_result",
]
