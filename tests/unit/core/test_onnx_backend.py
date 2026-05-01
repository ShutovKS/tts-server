# FILE: tests/unit/core/test_onnx_backend.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for the ONNX/Piper backend integration.
#   SCOPE: Artifact inspection, runtime loading, and basic synthesis path
#   DEPENDS: M-CORE
#   LINKS: V-M-BACKENDS-V2
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _write_piper_artifacts - Helper to create a minimal Piper voice directory
#   test_onnx_backend_inspects_missing_artifacts_for_piper_voice - Verifies artifact validation for Piper voice directories
#   test_onnx_backend_loads_piper_voice_from_local_directory - Verifies Piper voice loading through the supported API
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added ONNX/Piper backend unit coverage]
# END_CHANGE_SUMMARY

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.backends.onnx_backend import ONNXBackend
from core.models.catalog import MODEL_SPECS

pytestmark = pytest.mark.unit


def _write_piper_artifacts(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.onnx").write_bytes(b"onnx")
    (model_dir / "model.onnx.json").write_text(
        json.dumps(
            {
                "audio": {"sample_rate": 22050},
                "espeak": {"voice": "en-us"},
                "phoneme_type": "espeak",
                "num_symbols": 10,
                "num_speakers": 1,
                "phoneme_id_map": {"_": [0], "^": [1], "$": [2], "a": [3]},
            }
        ),
        encoding="utf-8",
    )


def test_onnx_backend_inspects_missing_artifacts_for_piper_voice(tmp_path: Path):
    backend = ONNXBackend(models_dir=tmp_path)
    spec = MODEL_SPECS["piper-1"]

    inspection = backend.inspect_model(spec)

    assert inspection["backend"] == "onnx"
    assert inspection["available"] is False
    assert inspection["loadable"] is False
    assert inspection["missing_artifacts"] == ["model.onnx", "model.onnx.json"]


def test_onnx_backend_loads_piper_voice_from_local_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    backend = ONNXBackend(models_dir=tmp_path)
    spec = MODEL_SPECS["piper-1"]
    model_dir = tmp_path / spec.folder
    _write_piper_artifacts(model_dir)

    monkeypatch.setattr(
        "core.backends.onnx_backend.PiperVoice",
        SimpleNamespace(
            load=lambda model_path, config_path, use_cuda=False: {
                "model_path": str(model_path),
                "config_path": str(config_path),
                "use_cuda": use_cuda,
            }
        ),
    )

    handle = backend.load_model(spec)
    inspection = backend.inspect_model(spec)

    assert handle.backend_key == "onnx"
    assert handle.runtime_model["model_path"].endswith("model.onnx")
    assert handle.runtime_model["config_path"].endswith("model.onnx.json")
    assert inspection["available"] is True
    assert inspection["loadable"] is True
    assert inspection["runtime_ready"] is True
