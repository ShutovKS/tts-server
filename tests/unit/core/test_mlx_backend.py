from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.backends.mlx_backend import MLXBackend
from core.errors import ModelLoadError
from core.models.catalog import MODEL_SPECS


pytestmark = pytest.mark.unit


def _write_model_artifacts(model_dir: Path, config: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (model_dir / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
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


def test_qwen3_nested_config_is_normalized_into_temp_runtime_dir(tmp_path: Path):
    backend = MLXBackend(models_dir=tmp_path)
    spec = MODEL_SPECS["1"]
    model_dir = tmp_path / spec.folder
    original_config = _make_nested_qwen3_config()
    _write_model_artifacts(model_dir, original_config)

    runtime_dir = backend._prepare_runtime_model_path(spec=spec, model_path=model_dir)

    assert runtime_dir != model_dir
    normalized_config = json.loads(
        (runtime_dir / "config.json").read_text(encoding="utf-8")
    )
    assert normalized_config["model_type"] == "qwen3_tts"
    assert (
        normalized_config["hidden_size"]
        == original_config["talker_config"]["hidden_size"]
    )
    assert (
        normalized_config["num_hidden_layers"]
        == original_config["talker_config"]["num_hidden_layers"]
    )
    assert normalized_config["talker_config"] == original_config["talker_config"]
    assert (runtime_dir / "tokenizer_config.json").exists()
    assert (runtime_dir / "speech_tokenizer").exists()


def test_qwen3_nested_config_validation_rejects_incomplete_talker_config(
    tmp_path: Path,
):
    backend = MLXBackend(models_dir=tmp_path)
    spec = MODEL_SPECS["1"]
    model_dir = tmp_path / spec.folder
    broken_config = _make_nested_qwen3_config()
    broken_config["talker_config"].pop("hidden_size")
    _write_model_artifacts(model_dir, broken_config)

    with pytest.raises(ModelLoadError) as exc_info:
        backend._prepare_runtime_model_path(spec=spec, model_path=model_dir)

    details = exc_info.value.context.to_dict()
    assert details["reason"] == "Qwen3-TTS MLX talker_config is incomplete"
    assert "hidden_size" in details["missing_fields"]
    assert details["backend"] == "mlx"


def test_non_qwen3_config_is_not_rewritten(tmp_path: Path):
    backend = MLXBackend(models_dir=tmp_path)
    spec = MODEL_SPECS["1"]
    model_dir = tmp_path / spec.folder
    plain_config = {"model_type": "other_model", "hidden_size": 1}
    _write_model_artifacts(model_dir, plain_config)

    runtime_dir = backend._prepare_runtime_model_path(spec=spec, model_path=model_dir)

    assert runtime_dir == model_dir


def test_cache_diagnostics_reports_normalized_runtime_dirs(tmp_path: Path):
    backend = MLXBackend(models_dir=tmp_path)
    spec = MODEL_SPECS["1"]
    model_dir = tmp_path / spec.folder
    _write_model_artifacts(model_dir, _make_nested_qwen3_config())

    runtime_dir = backend._prepare_runtime_model_path(spec=spec, model_path=model_dir)
    backend._cache[spec.folder] = object()

    diagnostics = backend.cache_diagnostics()
    inspection = backend.inspect_model(spec)

    assert diagnostics["cached_model_count"] == 1
    assert diagnostics["cache_policy"]["normalized_runtime_dirs"] == 1
    assert diagnostics["loaded_models"][0]["normalized_runtime"] is True
    assert diagnostics["loaded_models"][0]["runtime_path"] == str(runtime_dir)
    assert inspection["cache"]["loaded"] is True
    assert inspection["cache"]["normalized_runtime"] is True
    assert inspection["runtime_path"] == str(runtime_dir)


def test_normalized_runtime_rebinds_resources_to_original_model_path(tmp_path: Path):
    backend = MLXBackend(models_dir=tmp_path)
    spec = MODEL_SPECS["1"]
    model_dir = tmp_path / spec.folder
    _write_model_artifacts(model_dir, _make_nested_qwen3_config())
    runtime_dir = backend._prepare_runtime_model_path(spec=spec, model_path=model_dir)
    calls: list[Path] = []

    class RuntimeModel:
        @classmethod
        def post_load_hook(cls, model, model_path: Path):
            calls.append(model_path)
            return model

    backend._rebind_runtime_resources(
        runtime_model=RuntimeModel(),
        model_path=model_dir,
        runtime_path=runtime_dir,
        spec=spec,
    )

    assert calls == [model_dir]


def test_normalized_runtime_loader_rebinds_resources_after_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    backend = MLXBackend(models_dir=tmp_path)
    spec = MODEL_SPECS["1"]
    model_dir = tmp_path / spec.folder
    _write_model_artifacts(model_dir, _make_nested_qwen3_config())
    runtime_dir = backend._prepare_runtime_model_path(spec=spec, model_path=model_dir)
    loaded_paths: list[str] = []
    rebound_paths: list[Path] = []

    monkeypatch.setattr(
        "core.backends.mlx_backend.load_model",
        lambda path: loaded_paths.append(path) or SimpleNamespace(),
    )

    def fake_rebind(**kwargs):
        rebound_paths.append(kwargs["model_path"])
        return kwargs["runtime_model"]

    monkeypatch.setattr(backend, "_rebind_runtime_resources", fake_rebind)

    backend._invoke_runtime_loader(
        spec=spec,
        model_path=model_dir,
        runtime_path=runtime_dir,
        normalized_runtime=True,
    )

    assert loaded_paths == [str(runtime_dir)]
    assert rebound_paths == [model_dir]
