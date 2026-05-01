# FILE: tests/unit/core/test_config_and_bootstrap.py
# VERSION: 1.0.1
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for core settings parsing and runtime bootstrap.
#   SCOPE: Environment parsing, backend wiring, runtime construction
#   DEPENDS: M-CORE
#   LINKS: V-M-CORE
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_parse_core_settings_from_env_uses_local_job_backend_defaults - Verifies local runtime defaults are parsed from environment
#   test_parse_core_settings_from_env_reads_explicit_job_backends - Verifies explicit backend and auth settings are parsed correctly
#   test_job_wiring_factories_reject_unknown_backend_ids - Verifies bootstrap factories reject unsupported backend ids
#   test_job_wiring_factories_keep_local_runtime_defaults - Verifies local bootstrap factories preserve default runtime wiring
#   test_build_runtime_passes_manifest_path_to_backend_registry - Verifies runtime bootstrap passes explicit manifest configuration to registry construction
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.2.1 - Added coverage for canonical TTS_CORS_ALLOWED_ORIGINS parsing so transport adapters can read explicit browser origin allowlists from shared config]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path

import pytest

from core.application.job_execution import InMemoryJobExecutor
from core.bootstrap import (
    build_job_artifact_store,
    build_job_execution_backend,
    build_job_metadata_store,
    build_runtime,
)
from core.config import (
    DEFAULT_MODELS_DIR,
    DEFAULT_OUTPUTS_DIR,
    DEFAULT_UPLOAD_STAGING_DIR,
    DEFAULT_VOICES_DIR,
    LOCAL_JOB_ARTIFACT_BACKEND,
    LOCAL_JOB_EXECUTION_BACKEND,
    LOCAL_JOB_METADATA_BACKEND,
    LOCAL_QUOTA_BACKEND,
    LOCAL_RATE_LIMIT_BACKEND,
    PROJECT_ROOT,
    CoreSettings,
    parse_core_settings_from_env,
)
from core.infrastructure.job_execution_local import (
    LocalBoundedExecutionManager,
    LocalInMemoryJobStore,
    LocalJobArtifactStore,
)
from core.metrics import OperationalMetricsRegistry
from tests.unit.core.test_job_execution import StubApplicationService

pytestmark = pytest.mark.unit


def test_parse_core_settings_from_env_uses_local_job_backend_defaults():
    values = parse_core_settings_from_env({})

    assert values["job_execution_backend"] == LOCAL_JOB_EXECUTION_BACKEND
    assert values["job_metadata_backend"] == LOCAL_JOB_METADATA_BACKEND
    assert values["job_artifact_backend"] == LOCAL_JOB_ARTIFACT_BACKEND
    assert values["auth_mode"] == "off"
    assert values["auth_static_bearer_token"] is None
    assert values["cors_allowed_origins"] == ()
    assert values["mlx_models_dir"] == (DEFAULT_MODELS_DIR / "mlx").resolve()
    assert values["active_family"] is None
    assert values["default_custom_model"] is None
    assert values["default_design_model"] is None
    assert values["default_clone_model"] is None
    assert (
        values["model_manifest_path"]
        == (PROJECT_ROOT / "core" / "models" / "manifest.v1.json").resolve()
    )
    assert values["qwen_fast_enabled"] is True
    assert values["model_preload_policy"] == "none"
    assert values["model_preload_ids"] == ()
    assert values["rate_limit_enabled"] is False
    assert values["rate_limit_backend"] == LOCAL_RATE_LIMIT_BACKEND
    assert values["quota_enabled"] is False
    assert values["quota_backend"] == LOCAL_QUOTA_BACKEND


def test_parse_core_settings_from_env_reads_explicit_job_backends(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    values = parse_core_settings_from_env(
        {
            "TTS_MODELS_DIR": str(tmp_path / "models"),
            "TTS_MLX_MODELS_DIR": str(tmp_path / "mlx-models"),
            "TTS_OUTPUTS_DIR": str(tmp_path / "outputs"),
            "TTS_VOICES_DIR": str(tmp_path / "voices"),
            "TTS_UPLOAD_STAGING_DIR": str(tmp_path / "uploads"),
            "TTS_MODEL_MANIFEST_PATH": str(manifest_path),
            "TTS_ACTIVE_FAMILY": "qwen",
            "TTS_DEFAULT_CUSTOM_MODEL": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "TTS_DEFAULT_DESIGN_MODEL": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
            "TTS_DEFAULT_CLONE_MODEL": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
            "TTS_QWEN_FAST_ENABLED": "false",
            "TTS_MODEL_PRELOAD_POLICY": "listed",
            "TTS_MODEL_PRELOAD_IDS": "model-a, model-b,model-a",
            "TTS_JOB_EXECUTION_BACKEND": "future-executor",
            "TTS_JOB_METADATA_BACKEND": "future-metadata",
            "TTS_JOB_ARTIFACT_BACKEND": "future-artifacts",
            "TTS_AUTH_MODE": "static_bearer",
            "TTS_AUTH_STATIC_BEARER_TOKEN": "secret-token",
            "TTS_AUTH_STATIC_BEARER_PRINCIPAL_ID": "principal-configured",
            "TTS_AUTH_STATIC_BEARER_CREDENTIAL_ID": "cred-configured",
            "TTS_CORS_ALLOWED_ORIGINS": "http://127.0.0.1:8030, http://185.186.142.205:8030,http://127.0.0.1:8030",
            "TTS_RATE_LIMIT_ENABLED": "true",
            "TTS_RATE_LIMIT_BACKEND": "future-rate-limit",
            "TTS_RATE_LIMIT_SYNC_TTS_PER_MINUTE": "11",
            "TTS_RATE_LIMIT_ASYNC_SUBMIT_PER_MINUTE": "12",
            "TTS_RATE_LIMIT_JOB_READ_PER_MINUTE": "13",
            "TTS_RATE_LIMIT_JOB_CANCEL_PER_MINUTE": "14",
            "TTS_RATE_LIMIT_CONTROL_PLANE_PER_MINUTE": "15",
            "TTS_QUOTA_ENABLED": "true",
            "TTS_QUOTA_BACKEND": "future-quota",
            "TTS_QUOTA_COMPUTE_REQUESTS_PER_WINDOW": "21",
            "TTS_QUOTA_COMPUTE_WINDOW_SECONDS": "120",
            "TTS_QUOTA_MAX_ACTIVE_JOBS_PER_PRINCIPAL": "3",
        }
    )

    assert values["model_manifest_path"] == manifest_path.resolve()
    assert values["mlx_models_dir"] == (tmp_path / "mlx-models").resolve()
    assert values["active_family"] == "qwen"
    assert values["default_custom_model"] == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    assert values["default_design_model"] == "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"
    assert values["default_clone_model"] == "Qwen3-TTS-12Hz-1.7B-Base-8bit"
    assert values["qwen_fast_enabled"] is False
    assert values["model_preload_policy"] == "listed"
    assert values["model_preload_ids"] == ("model-a", "model-b")
    assert values["job_execution_backend"] == "future-executor"
    assert values["job_metadata_backend"] == "future-metadata"
    assert values["job_artifact_backend"] == "future-artifacts"
    assert values["auth_mode"] == "static_bearer"
    assert values["auth_static_bearer_token"] == "secret-token"
    assert values["auth_static_bearer_principal_id"] == "principal-configured"
    assert values["auth_static_bearer_credential_id"] == "cred-configured"
    assert values["cors_allowed_origins"] == (
        "http://127.0.0.1:8030",
        "http://185.186.142.205:8030",
    )
    assert values["rate_limit_enabled"] is True
    assert values["rate_limit_backend"] == "future-rate-limit"
    assert values["rate_limit_sync_tts_per_minute"] == 11
    assert values["rate_limit_async_submit_per_minute"] == 12
    assert values["rate_limit_job_read_per_minute"] == 13
    assert values["rate_limit_job_cancel_per_minute"] == 14
    assert values["rate_limit_control_plane_per_minute"] == 15
    assert values["quota_enabled"] is True
    assert values["quota_backend"] == "future-quota"
    assert values["quota_compute_requests_per_window"] == 21
    assert values["quota_compute_window_seconds"] == 120
    assert values["quota_max_active_jobs_per_principal"] == 3


def test_parse_core_settings_from_env_ignores_legacy_qwen_names(tmp_path: Path):
    values = parse_core_settings_from_env(
        {
            "LEGACY_MODELS_DIR": str(tmp_path / "legacy-models"),
            "LEGACY_BACKEND": "torch",
            "LEGACY_SAMPLE_RATE": "22050",
        }
    )

    assert values["models_dir"] == DEFAULT_MODELS_DIR.resolve()
    assert values["backend"] is None
    assert values["sample_rate"] == 24000


def test_parse_core_settings_from_env_uses_only_tts_names_when_legacy_names_are_also_set(
    tmp_path: Path,
):
    values = parse_core_settings_from_env(
        {
            "TTS_MODELS_DIR": str(tmp_path / "canonical-models"),
            "LEGACY_MODELS_DIR": str(tmp_path / "legacy-models"),
            "TTS_BACKEND": "onnx",
            "LEGACY_BACKEND": "torch",
            "TTS_SAMPLE_RATE": "16000",
            "LEGACY_SAMPLE_RATE": "24000",
        }
    )

    assert values["models_dir"] == (tmp_path / "canonical-models").resolve()
    assert values["backend"] == "onnx"
    assert values["sample_rate"] == 16000


@pytest.mark.parametrize(
    ("factory_name", "settings_overrides", "expected_message"),
    [
        (
            "artifact",
            {"job_artifact_backend": "s3"},
            "Unsupported job artifact backend: s3",
        ),
        (
            "metadata",
            {"job_metadata_backend": "postgres"},
            "Unsupported job metadata backend: postgres",
        ),
        (
            "execution",
            {"job_execution_backend": "redis"},
            "Unsupported job execution backend: redis",
        ),
    ],
)
def test_job_wiring_factories_reject_unknown_backend_ids(
    factory_name: str,
    settings_overrides: dict[str, str],
    expected_message: str,
):
    settings = CoreSettings(
        models_dir=DEFAULT_MODELS_DIR,
        mlx_models_dir=DEFAULT_MODELS_DIR / "mlx",
        outputs_dir=DEFAULT_OUTPUTS_DIR,
        voices_dir=DEFAULT_VOICES_DIR,
        upload_staging_dir=DEFAULT_UPLOAD_STAGING_DIR,
        **settings_overrides,
    )

    with pytest.raises(ValueError, match=expected_message):
        if factory_name == "artifact":
            build_job_artifact_store(settings)
        elif factory_name == "metadata":
            build_job_metadata_store(settings, artifact_store=LocalJobArtifactStore())
        else:
            build_job_execution_backend(
                settings,
                store=LocalInMemoryJobStore(),
                executor=InMemoryJobExecutor(application_service=StubApplicationService()),
                metrics=OperationalMetricsRegistry(),
            )


def test_job_wiring_factories_keep_local_runtime_defaults():
    settings = CoreSettings(
        models_dir=DEFAULT_MODELS_DIR,
        mlx_models_dir=DEFAULT_MODELS_DIR / "mlx",
        outputs_dir=DEFAULT_OUTPUTS_DIR,
        voices_dir=DEFAULT_VOICES_DIR,
        upload_staging_dir=DEFAULT_UPLOAD_STAGING_DIR,
    )

    artifact_store = build_job_artifact_store(settings)
    metadata_store = build_job_metadata_store(settings, artifact_store=artifact_store)
    execution_backend = build_job_execution_backend(
        settings,
        store=metadata_store,
        executor=InMemoryJobExecutor(application_service=StubApplicationService()),
        metrics=OperationalMetricsRegistry(),
    )

    assert isinstance(artifact_store, LocalJobArtifactStore)
    assert isinstance(metadata_store, LocalInMemoryJobStore)
    assert isinstance(execution_backend, LocalBoundedExecutionManager)
    assert metadata_store.artifact_store is artifact_store
    assert execution_backend.store is metadata_store

    execution_backend.stop()


def test_build_runtime_passes_manifest_path_to_backend_registry(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """
        {
          "version": 1,
          "metadata": {"catalog": "test"},
          "modes": [
            {"id": "custom", "label": "Custom Voice", "semantics": "Instruction-guided synthesis with predefined speakers"}
          ],
          "models": [
            {
              "key": "1",
              "public_name": "Custom Voice",
              "folder": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
              "mode": "custom",
              "output_subfolder": "CustomVoice",
              "metadata": {"variant": "1.7B"},
              "mode_metadata": {"id": "custom", "label": "Custom Voice", "semantics": "Instruction-guided synthesis with predefined speakers"},
               "backend_affinity": ["mlx", "qwen_fast", "torch"],
               "rollout": {"enabled": true, "stage": "general", "default_preference": 1},
               "artifact_validation": {
                 "mlx": {"required_rules": [{"name": "config", "any_of": ["config.json"]}]},
                 "torch": {"required_rules": [{"name": "config", "any_of": ["config.json"]}]},
                 "qwen_fast": {"required_rules": [{"name": "config", "any_of": ["config.json"]}]}
               }
             }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    settings = CoreSettings(
        models_dir=tmp_path / "models",
        mlx_models_dir=tmp_path / "mlx-models",
        outputs_dir=tmp_path / "outputs",
        voices_dir=tmp_path / "voices",
        upload_staging_dir=tmp_path / "uploads",
        model_manifest_path=manifest_path,
        model_preload_policy="listed",
        model_preload_ids=("Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",),
    )

    runtime = build_runtime(settings)

    assert (
        runtime.backend_registry.model_specs[0].api_name == "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    )
    assert runtime.backend_registry._model_manifest.metadata["catalog"] == "test"
    assert runtime.backend_registry._backends["mlx"].models_dir == (tmp_path / "mlx-models")
    assert runtime.backend_registry._backends["torch"].models_dir == (tmp_path / "models")
    assert runtime.backend_registry._backends["qwen_fast"].enabled is True
    assert runtime.settings.runtime_capability_map() == {
        "family": None,
        "custom_model": None,
        "design_model": None,
        "clone_model": None,
    }
    assert runtime.registry.readiness_report()["preload"]["policy"] == "listed"
    assert runtime.registry.readiness_report()["preload"]["requested_model_ids"] == [
        "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    ]
    assert runtime.metrics.readiness_summary()["execution"]["submitted"] == 0
    assert runtime.rate_limiter is not None
    assert runtime.quota_guard is not None

    runtime.job_manager.stop()


def test_build_backends_auto_discovers_concrete_subclasses(tmp_path: Path):
    """Phase 2.8: build_backends() must instantiate every importable concrete
    TTSBackend subclass via cls.from_settings(...) instead of relying on a
    hardcoded class list in core/bootstrap.py.
    """

    from core.backends.base import TTSBackend
    from core.bootstrap import build_backends
    from core.metrics import OperationalMetricsRegistry

    settings = CoreSettings(
        models_dir=tmp_path / "models",
        mlx_models_dir=tmp_path / "mlx-models",
        outputs_dir=tmp_path / "outputs",
        voices_dir=tmp_path / "voices",
        upload_staging_dir=tmp_path / "uploads",
    )
    metrics = OperationalMetricsRegistry()

    backends = build_backends(settings, metrics=metrics)

    keys = {backend.key for backend in backends}
    # The four built-in backends imported by core.bootstrap must all show up.
    assert {"mlx", "torch", "onnx", "qwen_fast"}.issubset(keys)
    # Every produced instance is a concrete TTSBackend.
    for backend in backends:
        assert isinstance(backend, TTSBackend)
    # MLXBackend overrides from_settings to use mlx_models_dir, not models_dir.
    mlx_backend = next(b for b in backends if b.key == "mlx")
    assert mlx_backend.models_dir == (tmp_path / "mlx-models")
    # QwenFastBackend overrides from_settings to thread qwen_fast_enabled.
    qwen_fast_backend = next(b for b in backends if b.key == "qwen_fast")
    assert qwen_fast_backend.enabled is True
    # Torch + ONNX use the default from_settings; both bind to settings.models_dir.
    torch_backend = next(b for b in backends if b.key == "torch")
    assert torch_backend.models_dir == (tmp_path / "models")
    onnx_backend = next(b for b in backends if b.key == "onnx")
    assert onnx_backend.models_dir == (tmp_path / "models")


def test_core_settings_resolve_runtime_model_binding_by_mode():
    settings = CoreSettings(
        models_dir=DEFAULT_MODELS_DIR,
        mlx_models_dir=DEFAULT_MODELS_DIR / "mlx",
        outputs_dir=DEFAULT_OUTPUTS_DIR,
        voices_dir=DEFAULT_VOICES_DIR,
        upload_staging_dir=DEFAULT_UPLOAD_STAGING_DIR,
        active_family="qwen",
        default_custom_model="custom-model",
        default_design_model="design-model",
        default_clone_model="clone-model",
    )

    assert settings.resolve_runtime_model_binding("custom") == "custom-model"
    assert settings.resolve_runtime_model_binding("design") == "design-model"
    assert settings.resolve_runtime_model_binding("clone") == "clone-model"
    assert settings.resolve_runtime_model_binding("unknown") is None
