# FILE: tests/unit/core/test_torch_backend_clone_guard.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Verify the Torch clone guard rejects implausibly short clone outputs instead of persisting misleading near-empty WAVs.
#   SCOPE: clone-duration guard behavior for short and sufficient clone audio payloads
#   DEPENDS: M-BACKENDS, M-ERRORS
#   LINKS: V-M-BACKENDS
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_torch_clone_guard_rejects_implausibly_short_audio - Verifies clone outputs shorter than the guard threshold raise a generation error.
#   test_torch_clone_guard_allows_normal_duration_audio - Verifies clone outputs above the threshold pass the guard.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added unit coverage for the Torch clone-duration guard]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.backends.torch_backend import TorchBackend
from core.errors import TTSGenerationError


pytestmark = pytest.mark.unit


def test_torch_clone_guard_rejects_implausibly_short_audio(tmp_path: Path):
    backend = TorchBackend(tmp_path)

    with pytest.raises(TTSGenerationError) as exc_info:
        backend._assert_clone_audio_duration(
            [np.zeros(3840, dtype=np.float32)],
            24000,
            text="hello world",
            family="qwen3_tts",
            ref_audio_path=tmp_path / "ref.wav",
            ref_text="hello world",
        )

    assert "implausibly short audio" in str(exc_info.value)
    assert exc_info.value.context.to_dict()["failure_kind"] == "clone_audio_too_short"


def test_torch_clone_guard_allows_normal_duration_audio(tmp_path: Path):
    backend = TorchBackend(tmp_path)

    backend._assert_clone_audio_duration(
        [np.zeros(48000, dtype=np.float32)],
        24000,
        text="hello world",
        family="qwen3_tts",
        ref_audio_path=tmp_path / "ref.wav",
        ref_text=None,
    )
