# FILE: tests/unit/core/test_qwen_fast_backend.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for the additive accelerated Qwen custom-only backend.
#   SCOPE: Readiness diagnostics, artifact inspection, model loading, and custom-only execution boundaries
#   DEPENDS: M-CORE
#   LINKS: V-M-BACKENDS-V2
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _write_fast_qwen_artifacts - Helper to create minimal local model artifacts for fast-backend inspection
#   test_qwen_fast_backend_reports_disabled_by_config - Verifies explicit config disablement is surfaced
#   test_qwen_fast_backend_inspects_missing_artifacts - Verifies artifact validation for fast-capable custom models
#   test_qwen_fast_backend_loads_model_with_stub_runtime - Verifies dependency-gated loading through a stub runtime
#   test_qwen_fast_backend_rejects_design_and_clone_execution - Verifies MVP custom-only scope is enforced at execution time
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added accelerated Qwen backend unit coverage]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path

import pytest

from core.backends.base import ExecutionRequest
from core.backends.qwen_fast_backend import QwenFastBackend
from core.errors import TTSGenerationError
from core.models.catalog import MODEL_SPECS


pytestmark = pytest.mark.unit


def _write_fast_qwen_artifacts(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer_config.json",
    ]:
        (model_dir / filename).write_text("{}", encoding="utf-8")


def test_qwen_fast_backend_reports_disabled_by_config(tmp_path: Path):
    backend = QwenFastBackend(models_dir=tmp_path, enabled=False)

    diagnostics = backend.readiness_diagnostics()

    assert diagnostics.available is False
    assert diagnostics.ready is False
    assert diagnostics.reason == "disabled_by_config"
    assert diagnostics.details["enabled"] is False


def test_qwen_fast_backend_inspects_missing_artifacts(tmp_path: Path):
    backend = QwenFastBackend(models_dir=tmp_path)
    spec = MODEL_SPECS["1"]

    inspection = backend.inspect_model(spec)

    assert inspection["backend"] == "qwen_fast"
    assert inspection["available"] is False
    assert inspection["loadable"] is False
    assert inspection["missing_artifacts"] == ["model_directory"]


def test_qwen_fast_backend_loads_model_with_stub_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    backend = QwenFastBackend(models_dir=tmp_path)
    spec = MODEL_SPECS["1"]
    model_dir = tmp_path / spec.folder
    _write_fast_qwen_artifacts(model_dir)

    monkeypatch.setattr(
        backend,
        "readiness_diagnostics",
        lambda: type(
            "DiagnosticsStub",
            (),
            {
                "ready": True,
                "reason": None,
                "details": {},
            },
        )(),
    )
    monkeypatch.setattr(
        backend,
        "_load_runtime_model",
        lambda model_path: type(
            "RuntimeStub",
            (),
            {
                "model_path": str(model_path),
                "generate_custom_voice": staticmethod(
                    lambda **kwargs: ([[0.0, 0.0]], 24000)
                ),
            },
        )(),
    )

    handle = backend.load_model(spec)
    inspection = backend.inspect_model(spec)

    assert handle.backend_key == "qwen_fast"
    assert str(handle.resolved_path).endswith(spec.folder)
    assert inspection["available"] is True
    assert inspection["loadable"] is True
    assert inspection["cached"] is True


def test_qwen_fast_backend_rejects_design_and_clone_execution(tmp_path: Path):
    backend = QwenFastBackend(models_dir=tmp_path)
    handle = type(
        "HandleStub",
        (),
        {"spec": MODEL_SPECS["1"], "runtime_model": object()},
    )()

    with pytest.raises(TTSGenerationError, match="custom synthesis only"):
        backend.execute(
            ExecutionRequest(
                handle=handle,
                text="hello",
                output_dir=tmp_path,
                language="auto",
                execution_mode="design",
                generation_kwargs={"instruct": "Warm narrator"},
            )
        )

    with pytest.raises(TTSGenerationError, match="custom synthesis only"):
        backend.execute(
            ExecutionRequest(
                handle=handle,
                text="hello",
                output_dir=tmp_path,
                language="auto",
                execution_mode="clone",
                generation_kwargs={"ref_audio": tmp_path / "reference.wav", "ref_text": "hello"},
            )
        )
