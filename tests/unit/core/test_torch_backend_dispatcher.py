# FILE: tests/unit/core/test_torch_backend_dispatcher.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Verify Torch backend strategy registry wiring remains deterministic and compatibility-safe.
#   SCOPE: built-in strategy registration, additive injected strategy wiring, duplicate-key rejection, availability behavior with unavailable optional injected families
#   DEPENDS: M-BACKENDS
#   LINKS: V-M-BACKENDS
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _ExtraStrategy - In-test Torch strategy with a unique family key used to verify additive injection.
#   _UnavailableExtraStrategy - In-test Torch strategy with a unique family key and unavailable runtime used to verify optional injected failures do not hide valid built-ins.
#   _DuplicateQwenStrategy - In-test Torch strategy that collides with the built-in qwen3_tts family key.
#   test_built_in_torch_family_strategies_are_deterministic - Verifies the built-in registry order remains qwen3_tts then omnivoice.
#   test_build_torch_strategy_map_rejects_duplicate_keys - Verifies duplicate family registration raises with the duplicate key in the message.
#   test_torch_backend_adds_injected_strategies_without_replacing_built_ins - Verifies explicit injection extends the default registry instead of replacing it.
#   test_torch_backend_is_available_ignores_unavailable_optional_injected_strategy - Verifies an unavailable injected optional family does not hide an available registered family.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Task 4 compatibility wiring: added focused dispatcher/strategy-registry coverage for built-ins, duplicate keys, and unavailable optional injected families]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.backends.torch_backend import (
    TorchBackend,
    TorchFamilyStrategy,
    build_torch_strategy_map,
    built_in_torch_family_strategies,
)

pytestmark = pytest.mark.unit


# START_BLOCK_TEST_STUBS
class _ExtraStrategy(TorchFamilyStrategy):
    family_key = "extra_family"
    runtime_dependency = "tests.extra"

    def load_model_class(self) -> Any | None:
        return object

    def import_error(self) -> Exception | None:
        return None

    def synthesize_custom(self, runtime_model: Any, *, text: str, language: str, speaker: str, instruct: str, speed: float) -> tuple[list[Any], int]:  # pragma: no cover
        raise NotImplementedError

    def synthesize_design(self, runtime_model: Any, *, text: str, language: str, voice_description: str) -> tuple[list[Any], int]:  # pragma: no cover
        raise NotImplementedError

    def synthesize_clone(self, runtime_model: Any, *, text: str, language: str, ref_audio: str, ref_text: str | None) -> tuple[list[Any], int]:  # pragma: no cover
        raise NotImplementedError


class _UnavailableExtraStrategy(_ExtraStrategy):
    family_key = "extra_unavailable"
    runtime_dependency = "tests.unavailable"

    def load_model_class(self) -> Any | None:
        return None

    def import_error(self) -> Exception | None:
        return ImportError("optional extra runtime unavailable")


class _DuplicateQwenStrategy(_ExtraStrategy):
    family_key = "qwen3_tts"


# END_BLOCK_TEST_STUBS


def test_built_in_torch_family_strategies_are_deterministic() -> None:
    strategies = built_in_torch_family_strategies()

    assert tuple(strategy.family_key for strategy in strategies) == ("qwen3_tts", "omnivoice")


def test_build_torch_strategy_map_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="qwen3_tts"):
        build_torch_strategy_map((_DuplicateQwenStrategy(),))


def test_torch_backend_adds_injected_strategies_without_replacing_built_ins(tmp_path: Path) -> None:
    backend = TorchBackend(tmp_path, strategies=(_ExtraStrategy(),))

    assert tuple(sorted(backend._strategies)) == ("extra_family", "omnivoice", "qwen3_tts")


def test_torch_backend_is_available_ignores_unavailable_optional_injected_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = TorchBackend(tmp_path, strategies=(_UnavailableExtraStrategy(),))

    monkeypatch.setattr(backend._strategies["qwen3_tts"], "load_model_class", lambda: object)
    monkeypatch.setattr("core.backends.torch_backend.dispatcher.torch", object())

    assert backend.is_available() is True
