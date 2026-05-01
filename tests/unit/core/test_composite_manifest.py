# FILE: tests/unit/core/test_composite_manifest.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for the composite manifest loader and .models/ auto-discovery introduced in Phase 2.7.
#   SCOPE: load_composite_manifest behavior with fragments_dir + models_dir, fragment shape acceptance, duplicate-key rejection, malformed-fragment skipping
#   DEPENDS: M-MODELS
#   LINKS: V-M-MODELS
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _BASE_MODEL_PAYLOAD - In-test base manifest payload reused by each scenario.
#   _make_fragment_model - Helper that produces a fragment-shaped model dict for a given key/folder.
#   test_base_manifest_loads_without_fragments - Verifies behavior matches load_model_manifest when no fragments are provided.
#   test_fragments_dir_appends_models - Verifies fragments under fragments_dir/*.json are merged.
#   test_models_dir_auto_discovery_appends_models - Verifies per-folder model_manifest.json files in models_dir are merged.
#   test_fragment_with_duplicate_key_raises - Verifies duplicate `key` across fragments fails via ModelManifest validation.
#   test_malformed_fragment_is_skipped_with_warning - Verifies non-JSON / non-object fragments are skipped (logged) instead of raising.
#   test_fragment_accepts_three_shapes - Verifies models[]/model/single-object fragment shapes are all accepted.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Phase 2.7 composite manifest coverage]
# END_CHANGE_SUMMARY

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from core.models.composite import (
    CompositeManifestError,
    load_composite_manifest,
)
from core.models.manifest import ModelManifestValidationError

pytestmark = pytest.mark.unit


def _base_manifest_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "metadata": {"catalog": "test_catalog", "default_selection_policy": "mode_first_available"},
        "modes": [
            {
                "id": "custom",
                "label": "Custom Voice",
                "semantics": "Instruction-guided synthesis with predefined speakers",
            }
        ],
        "models": [
            {
                "key": "base-1",
                "public_name": "Base Custom",
                "folder": "base-folder-1",
                "mode": "custom",
                "output_subfolder": "Base",
                "metadata": {"family": "Qwen3-TTS"},
                "mode_metadata": {
                    "id": "custom",
                    "label": "Custom Voice",
                    "semantics": "Instruction-guided synthesis with predefined speakers",
                },
                "backend_affinity": ["torch"],
                "rollout": {
                    "enabled": True,
                    "stage": "general",
                    "default_preference": 100,
                },
                "artifact_validation": {
                    "torch": {
                        "required_rules": [
                            {"name": "config", "any_of": ["config.json"]},
                        ]
                    }
                },
            }
        ],
    }


def _make_fragment_model(key: str, folder: str, *, public_name: str) -> dict[str, Any]:
    return {
        "key": key,
        "public_name": public_name,
        "folder": folder,
        "mode": "custom",
        "output_subfolder": "Frag",
        "metadata": {"family": "Qwen3-TTS"},
        "mode_metadata": {
            "id": "custom",
            "label": "Custom Voice",
            "semantics": "Instruction-guided synthesis with predefined speakers",
        },
        "backend_affinity": ["torch"],
        "rollout": {"enabled": True, "stage": "general", "default_preference": 50},
        "artifact_validation": {
            "torch": {
                "required_rules": [
                    {"name": "config", "any_of": ["config.json"]},
                ]
            }
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_base(tmp_path: Path) -> Path:
    base_path = tmp_path / "base.json"
    _write_json(base_path, _base_manifest_payload())
    return base_path


def test_base_manifest_loads_without_fragments(tmp_path: Path) -> None:
    base_path = _write_base(tmp_path)

    manifest = load_composite_manifest(base_path=base_path, fragments_dir=None, models_dir=None)

    assert set(manifest.models.keys()) == {"base-1"}


def test_fragments_dir_appends_models(tmp_path: Path) -> None:
    base_path = _write_base(tmp_path)
    fragments_dir = tmp_path / "fragments"
    fragments_dir.mkdir()
    _write_json(
        fragments_dir / "extra.json",
        {"models": [_make_fragment_model("frag-1", "frag-folder-1", public_name="Frag One")]},
    )

    manifest = load_composite_manifest(
        base_path=base_path, fragments_dir=fragments_dir, models_dir=None
    )

    assert set(manifest.models.keys()) == {"base-1", "frag-1"}


def test_models_dir_auto_discovery_appends_models(tmp_path: Path) -> None:
    base_path = _write_base(tmp_path)
    models_dir = tmp_path / "models_dir"
    bark_dir = models_dir / "Bark"
    bark_dir.mkdir(parents=True)
    _write_json(
        bark_dir / "model_manifest.json",
        {"model": _make_fragment_model("bark-1", "Bark", public_name="Bark Custom")},
    )

    manifest = load_composite_manifest(
        base_path=base_path, fragments_dir=None, models_dir=models_dir
    )

    assert set(manifest.models.keys()) == {"base-1", "bark-1"}


def test_fragment_with_duplicate_key_raises(tmp_path: Path) -> None:
    base_path = _write_base(tmp_path)
    fragments_dir = tmp_path / "fragments"
    fragments_dir.mkdir()
    _write_json(
        fragments_dir / "dup.json",
        {"models": [_make_fragment_model("base-1", "different-folder", public_name="Dup")]},
    )

    with pytest.raises(ModelManifestValidationError):
        load_composite_manifest(base_path=base_path, fragments_dir=fragments_dir, models_dir=None)


def test_malformed_fragment_is_skipped_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    base_path = _write_base(tmp_path)
    fragments_dir = tmp_path / "fragments"
    fragments_dir.mkdir()
    (fragments_dir / "broken.json").write_text("{not json", encoding="utf-8")
    (fragments_dir / "non_object.json").write_text("[1, 2, 3]", encoding="utf-8")

    with caplog.at_level("WARNING", logger="core.models.composite"):
        manifest = load_composite_manifest(
            base_path=base_path, fragments_dir=fragments_dir, models_dir=None
        )

    assert set(manifest.models.keys()) == {"base-1"}
    warnings = [rec.message for rec in caplog.records if rec.name == "core.models.composite"]
    assert any("broken.json" in msg for msg in warnings)
    assert any("non_object.json" in msg for msg in warnings)


def test_fragment_accepts_three_shapes(tmp_path: Path) -> None:
    base_path = _write_base(tmp_path)
    fragments_dir = tmp_path / "fragments"
    fragments_dir.mkdir()

    _write_json(
        fragments_dir / "list.json",
        {"models": [_make_fragment_model("shape-list", "shape-list", public_name="A")]},
    )
    _write_json(
        fragments_dir / "single_keyed.json",
        {"model": _make_fragment_model("shape-keyed", "shape-keyed", public_name="B")},
    )
    _write_json(
        fragments_dir / "single_root.json",
        _make_fragment_model("shape-root", "shape-root", public_name="C"),
    )

    manifest = load_composite_manifest(
        base_path=base_path, fragments_dir=fragments_dir, models_dir=None
    )

    assert set(manifest.models.keys()) == {
        "base-1",
        "shape-list",
        "shape-keyed",
        "shape-root",
    }


def test_fragment_with_invalid_root_shape_raises(tmp_path: Path) -> None:
    base_path = _write_base(tmp_path)
    fragments_dir = tmp_path / "fragments"
    fragments_dir.mkdir()
    # Object with no recognized fragment key
    _write_json(fragments_dir / "weird.json", {"unrelated": 1})

    with pytest.raises(CompositeManifestError):
        load_composite_manifest(base_path=base_path, fragments_dir=fragments_dir, models_dir=None)
