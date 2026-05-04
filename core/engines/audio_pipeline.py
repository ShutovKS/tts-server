# FILE: core/engines/audio_pipeline.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide a standard post-synthesis audio pipeline seam for TTSEngine outputs before persistence.
#   SCOPE: AudioPipeline process hook over AudioBuffer values with centralized WAV conversion, sample-rate normalization, loudness normalization, and peak limiting
#   DEPENDS: M-ENGINE-CONTRACTS
#   LINKS: M-ENGINE-CONTRACTS, M-TTS-COORDINATOR
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   AudioPipeline - Post-synthesis engine audio processing seam
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.2.0 - Added loudness-style RMS normalization with peak limiting for engine outputs]
# END_CHANGE_SUMMARY

from __future__ import annotations

import audioop
import io
import wave

from core.engines.contracts import AudioBuffer


# START_CONTRACT: AudioPipeline
#   PURPOSE: Centralize post-synthesis normalization for TTSEngine outputs before persistence.
#   INPUTS: { target_sample_rate: int - Desired output sample rate, target_rms_dbfs: float - Target loudness in dBFS, limit_peak: bool - Whether peak limiting is applied after loudness normalization }
#   OUTPUTS: { instance - Audio normalization pipeline }
#   SIDE_EFFECTS: none on construction
#   LINKS: M-ENGINE-AUDIO-PIPELINE, M-TTS-COORDINATOR
# END_CONTRACT: AudioPipeline
class AudioPipeline:
    def __init__(
        self,
        *,
        target_sample_rate: int = 24000,
        target_rms_dbfs: float = -18.0,
        limit_peak: bool = True,
    ) -> None:
        self._target_sample_rate = target_sample_rate
        self._target_rms_amplitude = 32767.0 * (10.0 ** (target_rms_dbfs / 20.0))
        self._limit_peak = limit_peak

    # START_CONTRACT: process
    #   PURPOSE: Normalize one engine-produced audio buffer into the shared WAV output contract.
    #   INPUTS: { audio: AudioBuffer - Engine output buffer in any supported format }
    #   OUTPUTS: { AudioBuffer - WAV-normalized buffer with target sample rate and loudness shaping applied }
    #   SIDE_EFFECTS: none
    #   LINKS: M-ENGINE-AUDIO-PIPELINE, M-ENGINE-CONTRACTS
    # END_CONTRACT: process
    def process(self, audio: AudioBuffer) -> AudioBuffer:
        wav_bytes = self._to_wav_bytes(audio)
        normalized_bytes, normalized_rate = self._normalize_wav_bytes(wav_bytes, audio.sample_rate)
        return AudioBuffer(
            waveform=normalized_bytes,
            sample_rate=normalized_rate,
            audio_format="wav",
        )

    def _to_wav_bytes(self, audio: AudioBuffer) -> bytes:
        if isinstance(audio.waveform, (bytes, bytearray)) and audio.audio_format.casefold() == "wav":
            return bytes(audio.waveform)

        import soundfile

        buffer = io.BytesIO()
        soundfile.write(buffer, audio.waveform, audio.sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    def _normalize_wav_bytes(self, wav_bytes: bytes, fallback_sample_rate: int) -> tuple[bytes, int]:
        with wave.open(io.BytesIO(wav_bytes), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            source_rate = source.getframerate() or fallback_sample_rate
            frames = source.readframes(source.getnframes())

        target_rate = self._target_sample_rate or source_rate
        if source_rate != target_rate:
            frames, _ = audioop.ratecv(frames, sample_width, channels, source_rate, target_rate, None)

        if sample_width == 2:
            rms = audioop.rms(frames, sample_width)
            if rms > 0:
                loudness_multiplier = self._target_rms_amplitude / rms
                if loudness_multiplier > 0:
                    frames = audioop.mul(frames, sample_width, loudness_multiplier)
            if self._limit_peak:
                peak = audioop.max(frames, sample_width)
                if peak > 0:
                    target_peak = int(32767 * 0.95)
                    limiter_multiplier = min(1.0, target_peak / peak)
                    if limiter_multiplier > 0:
                        frames = audioop.mul(frames, sample_width, limiter_multiplier)

        output_buffer = io.BytesIO()
        with wave.open(output_buffer, "wb") as target:
            target.setnchannels(channels)
            target.setsampwidth(sample_width)
            target.setframerate(target_rate)
            target.writeframes(frames)
        return output_buffer.getvalue(), target_rate


__all__ = ["AudioPipeline"]
