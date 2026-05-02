# FILE: tests/unit/core/test_engine_config.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for the typed and discriminated engine config models in core/engines/config.py.
#   SCOPE: valid config parsing, explicit disabled configs, missing field failures, unknown kinds, duplicate alias rejection
#   DEPENDS: M-ENGINE-CONFIG
#   LINKS: V-M-ENGINE-CONFIG
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_parse_engine_config_accepts_enabled_variants - Verifies known valid enabled engine configs parse into typed variants.
#   test_parse_engine_config_accepts_disabled_variant - Verifies the explicit disabled case stays deterministic and identifiable.
#   test_parse_engine_config_rejects_unknown_kind - Verifies unsupported discriminator values fail validation.
#   test_parse_engine_config_rejects_missing_required_fields - Verifies required shared fields stay typed and mandatory.
#   test_engine_settings_reject_duplicate_aliases - Verifies duplicate aliases or names fail fast.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added focused unit coverage for discriminated engine config parsing and duplicate alias guards]
# END_CHANGE_SUMMARY

from __future__ import annotations

import pytest
from pydantic import ValidationError  # pyright: ignore[reportMissingImports]

from core.engines import (
    DisabledEngineConfig,
    EngineSettings,
    MlxEngineConfig,
    OnnxEngineConfig,
    QwenFastEngineConfig,
    TorchEngineConfig,
    parse_engine_config,
    parse_engine_settings,
)

pytestmark = pytest.mark.unit


def test_parse_engine_config_accepts_enabled_variants() -> None:
    torch_config = parse_engine_config(
        {
            "kind": "torch",
            "name": "qwen-torch",
            "aliases": ["qwen_cpu", "qwen_cuda", "qwen_cpu"],
            "family": "qwen3_tts",
            "capabilities": ["preset_speaker_tts", "voice_description_tts"],
            "priority": 20,
            "params": {"device": "cuda:0"},
        }
    )
    mlx_config = parse_engine_config(
        {
            "kind": "mlx",
            "name": "qwen-mlx",
            "family": "qwen3_tts",
            "capabilities": ["preset_speaker_tts"],
        }
    )
    onnx_config = parse_engine_config(
        {
            "kind": "onnx",
            "name": "piper-onnx",
            "family": "piper",
            "capabilities": ["preset_speaker_tts"],
            "params": {"voice": "lessac"},
        }
    )
    fast_config = parse_engine_config(
        {
            "kind": "qwen_fast",
            "name": "qwen-fast",
            "family": "qwen3_tts",
            "capabilities": ["preset_speaker_tts", "reference_voice_clone"],
        }
    )

    assert isinstance(torch_config, TorchEngineConfig)
    assert torch_config.backend == "torch"
    assert torch_config.aliases == ("qwen_cpu", "qwen_cuda")
    assert torch_config.params == {"device": "cuda:0"}
    assert isinstance(mlx_config, MlxEngineConfig)
    assert mlx_config.backend == "mlx"
    assert isinstance(onnx_config, OnnxEngineConfig)
    assert onnx_config.family == "piper"
    assert isinstance(fast_config, QwenFastEngineConfig)
    assert fast_config.enabled is True


def test_parse_engine_config_accepts_disabled_variant() -> None:
    config = parse_engine_config(
        {
            "kind": "disabled",
            "name": "legacy-engine",
            "aliases": ["do-not-use"],
            "reason": "not enabled on this host",
            "params": {"ticket": "future-registry"},
        }
    )
    settings = parse_engine_settings({"engines": [config.model_dump()]})

    assert isinstance(config, DisabledEngineConfig)
    assert config.enabled is False
    assert config.reason == "not enabled on this host"
    assert settings.enabled_engines == ()
    assert settings.disabled_engines == (config,)


def test_parse_engine_config_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError, match="union_tag_invalid|expected tags"):
        parse_engine_config(
            {
                "kind": "future_engine",
                "name": "experimental",
                "family": "qwen3_tts",
                "capabilities": ["preset_speaker_tts"],
            }
        )


def test_parse_engine_config_rejects_missing_required_fields() -> None:
    with pytest.raises(ValidationError, match="family"):
        parse_engine_config(
            {
                "kind": "torch",
                "name": "missing-family",
                "capabilities": ["preset_speaker_tts"],
            }
        )

    with pytest.raises(ValidationError, match="capabilities"):
        parse_engine_config(
            {
                "kind": "mlx",
                "name": "missing-capabilities",
                "family": "qwen3_tts",
            }
        )


def test_engine_settings_reject_duplicate_aliases() -> None:
    with pytest.raises(ValidationError, match="Duplicate engine alias/name 'shared'"):
        parse_engine_settings(
            {
                "engines": [
                    {
                        "kind": "torch",
                        "name": "primary",
                        "aliases": ["shared"],
                        "family": "qwen3_tts",
                        "capabilities": ["preset_speaker_tts"],
                    },
                    {
                        "kind": "disabled",
                        "name": "secondary",
                        "aliases": ["shared"],
                        "reason": "operator disabled",
                    },
                ]
            }
        )

    settings = EngineSettings.model_validate(
        {
            "engines": [
                {
                    "kind": "torch",
                    "name": "alpha",
                    "aliases": ["alpha-primary"],
                    "family": "qwen3_tts",
                    "capabilities": ["preset_speaker_tts"],
                },
                {
                    "kind": "disabled",
                    "name": "beta",
                    "aliases": ["beta-disabled"],
                    "reason": "not deployed",
                },
            ]
        }
    )

    assert [config.name for config in settings.enabled_engines] == ["alpha"]
    assert [config.name for config in settings.disabled_engines] == ["beta"]
