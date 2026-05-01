# FILE: tests/unit/core/test_metrics_observability.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for core operational metrics and observability tracking.
#   SCOPE: Execution metrics, model cache metrics, load failure metrics
#   DEPENDS: M-CORE
#   LINKS: V-M-CORE
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _write_model_artifacts - Helper that writes minimal backend artifacts for metrics tests
#   _make_nested_qwen3_config - Helper that builds a nested Qwen3 config fixture
#   test_execution_metrics_collect_lifecycle_and_queue_depth - Verifies execution metrics track lifecycle counts and queue depth
#   test_mlx_metrics_collect_cache_hits_misses_and_load_failures - Verifies MLX metrics record cache and load outcomes
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT and MODULE_MAP]
# END_CHANGE_SUMMARY

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.application.job_execution import InMemoryJobExecutor
from core.backends.mlx_backend import MLXBackend
from core.metrics import InMemoryMetricsCollector, OperationalMetricsRegistry
from core.models.catalog import MODEL_SPECS
from tests.unit.core.test_job_execution import (
    StubApplicationService,
    _make_submission,
    _wait_for_status,
)

pytestmark = pytest.mark.unit


def _write_model_artifacts(model_dir: Path, config: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (model_dir / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "vocab.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    speech_tokenizer_dir = model_dir / "speech_tokenizer"
    speech_tokenizer_dir.mkdir(exist_ok=True)
    (speech_tokenizer_dir / "config.json").write_text("{}\n", encoding="utf-8")


def _make_nested_qwen3_config() -> dict:
    return {
        "architectures": ["Qwen3TTSForConditionalGeneration"],
        "model_type": "qwen3_tts",
        "tts_model_type": "custom_voice",
        "quantization": {"group_size": 64, "bits": 8, "mode": "affine"},
        "talker_config": {
            "model_type": "qwen3_tts_talker",
            "hidden_size": 2048,
            "num_hidden_layers": 28,
            "intermediate_size": 6144,
            "num_attention_heads": 16,
            "rms_norm_eps": 1e-6,
            "vocab_size": 3072,
            "num_key_value_heads": 8,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000,
            "head_dim": 128,
            "tie_word_embeddings": False,
            "rope_scaling": {"type": "default"},
        },
    }


def test_execution_metrics_collect_lifecycle_and_queue_depth():
    metrics = OperationalMetricsRegistry(InMemoryMetricsCollector())
    from core.infrastructure.job_execution_local import (
        LocalBoundedExecutionManager,
        LocalInMemoryJobStore,
    )

    store = LocalInMemoryJobStore()
    manager = LocalBoundedExecutionManager(
        store=store,
        executor=InMemoryJobExecutor(application_service=StubApplicationService()),
        worker_count=1,
        queue_capacity=2,
        metrics=metrics,
    )

    try:
        created = manager.submit(_make_submission())
        _wait_for_status(
            store,
            created.job_id,
            status=store.get_snapshot(created.job_id).status.SUCCEEDED,
        )  # type: ignore[union-attr]
    finally:
        manager.stop()

    summary = metrics.execution_summary()
    assert summary["submitted"] == 1
    assert summary["started"] == 1
    assert summary["completed"] == 1
    assert summary["failed"] == 0
    assert summary["timeout"] == 0
    assert summary["cancelled"] == 0
    assert summary["queue_depth"]["peak"] >= 1
    assert summary["queue_depth"]["current"] == 0


def test_mlx_metrics_collect_cache_hits_misses_and_load_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    metrics = OperationalMetricsRegistry(InMemoryMetricsCollector())
    backend = MLXBackend(models_dir=tmp_path, metrics=metrics)
    spec = MODEL_SPECS["1"]
    model_dir = tmp_path / spec.folder
    _write_model_artifacts(model_dir, _make_nested_qwen3_config())

    monkeypatch.setattr(
        "core.backends.mlx_backend.load_model",
        lambda path: type("RuntimeModel", (), {"tokenizer": object()})(),
    )

    backend.load_model(spec)
    backend.load_model(spec)

    summary = metrics.model_summary()
    assert summary["cache"]["miss"]["mlx"] == 1
    assert summary["cache"]["hit"]["mlx"] == 1
    assert summary["load"]["duration_ms"]["mlx"]["count"] == 1

    failing_metrics = OperationalMetricsRegistry(InMemoryMetricsCollector())
    failing_backend = MLXBackend(models_dir=tmp_path, metrics=failing_metrics)
    monkeypatch.setattr(
        "core.backends.mlx_backend.load_model",
        lambda path: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(Exception):
        failing_backend.load_model(spec)

    failing_summary = failing_metrics.model_summary()
    assert failing_summary["cache"]["miss"]["mlx"] == 1
    assert failing_summary["load"]["failures"]["mlx"] == 1
