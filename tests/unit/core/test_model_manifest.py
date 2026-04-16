# FILE: tests/unit/core/test_model_manifest.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for model manifest loading and artifact validation rules.
#   SCOPE: Manifest parsing, backend artifact requirements, validation reporting
#   DEPENDS: M-CORE
#   LINKS: V-M-CORE
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_default_model_manifest_preserves_existing_public_identifiers - Verifies shipped manifest keeps stable public ids and metadata
#   test_load_model_manifest_rejects_unknown_version - Verifies manifest loader rejects unsupported schema versions
#   test_load_model_manifest_requires_backend_artifact_validation_for_affinity - Verifies backend affinity requires matching validation rules
#   test_manifest_artifact_validation_rules_match_existing_backend_requirements - Verifies manifest validation rules match expected runtime artifacts
#   test_manifest_artifact_validation_reports_missing_torch_preprocessor - Verifies missing torch artifacts are surfaced in validation output
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT and MODULE_MAP]
# END_CHANGE_SUMMARY

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.models.catalog import MODEL_SPECS, get_model_manifest
from core.models.manifest import (
    ModelDescriptor,
    ModelManifestValidationError,
    load_model_manifest,
)


pytestmark = pytest.mark.unit


REQUIRED_TORCH_ARTIFACTS = [
    "config.json",
    "model.safetensors|model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer_config.json|vocab.json",
]


REQUIRED_QWEN_FAST_ARTIFACTS = [
    "config.json",
    "model.safetensors|model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer_config.json|vocab.json",
]


REQUIRED_MLX_ARTIFACTS = [
    "config.json",
    "model.safetensors|model.safetensors.index.json",
    "tokenizer_config.json|vocab.json",
]


def test_default_model_manifest_preserves_existing_public_identifiers():
    manifest = get_model_manifest()

    assert manifest.version == 1
    assert list(manifest.models)[:6] == ["1", "2", "3", "4", "5", "6"]
    assert "piper-1" in manifest.models
    assert MODEL_SPECS["1"].api_name == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    assert MODEL_SPECS["2"].mode == "design"
    assert MODEL_SPECS["3"].mode_metadata.id == "clone"
    assert MODEL_SPECS["4"].rollout.default_preference == 50
    assert MODEL_SPECS["1"].backend_affinity == ("mlx", "qwen_fast", "torch")
    assert "omnivoice-custom-1" in manifest.models


def test_load_model_manifest_rejects_unknown_version(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"version": 99, "metadata": {}, "modes": [], "models": []}),
        encoding="utf-8",
    )

    with pytest.raises(
        ModelManifestValidationError, match="Unsupported manifest version: 99"
    ):
        load_model_manifest(path)


def test_load_model_manifest_requires_backend_artifact_validation_for_affinity(
    tmp_path: Path,
):
    path = tmp_path / "manifest.json"
    payload = {
        "version": 1,
        "metadata": {},
        "modes": [
            {
                "id": "custom",
                "label": "Custom Voice",
                "semantics": "Instruction-guided synthesis",
            },
        ],
        "models": [
            {
                "key": "1",
                "public_name": "Custom Voice",
                "folder": "folder",
                "mode": "custom",
                "output_subfolder": "CustomVoice",
                "metadata": {"variant": "1.7B"},
                "mode_metadata": {
                    "id": "custom",
                    "label": "Custom Voice",
                    "semantics": "Instruction-guided synthesis",
                },
                "backend_affinity": ["mlx", "torch"],
                "rollout": {
                    "enabled": True,
                    "stage": "general",
                    "default_preference": 1,
                },
                "artifact_validation": {
                    "mlx": {
                        "required_rules": [
                            {"name": "config", "any_of": ["config.json"]},
                        ]
                    }
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ModelManifestValidationError, match="missing artifact_validation for backends"
    ):
        load_model_manifest(path)


def test_manifest_artifact_validation_rules_match_existing_backend_requirements(
    tmp_path: Path,
):
    model_path = tmp_path / MODEL_SPECS["1"].folder
    model_path.mkdir(parents=True)
    for filename in [
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "preprocessor_config.json",
    ]:
        (model_path / filename).write_text("{}", encoding="utf-8")

    mlx_check = (
        MODEL_SPECS["1"].artifact_validation_for_backend("mlx").validate(model_path)
    )
    torch_check = (
        MODEL_SPECS["1"].artifact_validation_for_backend("torch").validate(model_path)
    )
    fast_check = (
        MODEL_SPECS["1"]
        .artifact_validation_for_backend("qwen_fast")
        .validate(model_path)
    )

    assert mlx_check == {
        "loadable": True,
        "required_artifacts": REQUIRED_MLX_ARTIFACTS,
        "missing_artifacts": [],
    }
    assert torch_check == {
        "loadable": True,
        "required_artifacts": REQUIRED_TORCH_ARTIFACTS,
        "missing_artifacts": [],
    }
    assert fast_check == {
        "loadable": True,
        "required_artifacts": REQUIRED_QWEN_FAST_ARTIFACTS,
        "missing_artifacts": [],
    }


def test_manifest_artifact_validation_reports_missing_torch_preprocessor(
    tmp_path: Path,
):
    model_path = tmp_path / MODEL_SPECS["1"].folder
    model_path.mkdir(parents=True)
    for filename in ["config.json", "model.safetensors", "tokenizer_config.json"]:
        (model_path / filename).write_text("{}", encoding="utf-8")

    torch_check = (
        MODEL_SPECS["1"].artifact_validation_for_backend("torch").validate(model_path)
    )

    assert torch_check == {
        "loadable": False,
        "required_artifacts": REQUIRED_TORCH_ARTIFACTS,
        "missing_artifacts": ["preprocessor_config.json"],
    }


def test_model_descriptor_exposes_family_aware_compatibility_fields():
    descriptor = ModelDescriptor.from_model_spec(MODEL_SPECS["1"])

    assert descriptor.model_id == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    assert descriptor.family == "Qwen3-TTS"
    assert descriptor.family_key == "qwen3_tts"
    assert descriptor.execution_mode == "custom"
    assert descriptor.supported_capabilities == ("preset_speaker_tts",)
    assert descriptor.backend_support == ("mlx", "qwen_fast", "torch")
    assert descriptor.artifact_format == "local_model_dir"


def test_non_custom_qwen_models_do_not_advertise_fast_backend():
    assert MODEL_SPECS["2"].backend_affinity == ("mlx", "torch")
    assert MODEL_SPECS["3"].backend_affinity == ("mlx", "torch")
    assert MODEL_SPECS["5"].backend_affinity == ("mlx", "torch")
    assert MODEL_SPECS["6"].backend_affinity == ("mlx", "torch")


def test_manifest_descriptors_preserve_enabled_model_count():
    manifest = get_model_manifest()

    descriptors = manifest.descriptors()

    assert len(descriptors) == len(manifest.models)
    assert descriptors[0].family_key == "qwen3_tts"


def test_omnivoice_manifest_entries_expose_expected_family_metadata():
    custom = MODEL_SPECS["omnivoice-custom-1"]
    design = MODEL_SPECS["omnivoice-design-1"]
    clone = MODEL_SPECS["omnivoice-clone-1"]

    assert custom.family == "OmniVoice"
    assert custom.family_key == "omnivoice"
    assert custom.backend_affinity == ("torch",)
    assert design.supported_capabilities == ("voice_description_tts",)
    assert clone.supported_capabilities == ("reference_voice_clone",)
