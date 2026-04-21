# FILE: tests/unit/scripts/test_family_env_guidance.py
# VERSION: 1.1.1
# START_MODULE_CONTRACT
#   PURPOSE: Validate dedicated family-environment guidance exposure in the runtime self-check payload without bootstrapping the full runtime.
#   SCOPE: dedicated-family-env metadata for optional OmniVoice family environments and its machine-readable payload shape
#   DEPENDS: M-RUNTIME-SELF-CHECK, M-PROFILE-SCHEMA
#   LINKS: V-M-BOOTSTRAP, V-M-PROFILES
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_runtime_self_check_exposes_dedicated_family_env_guidance - Verifies the self-check payload keeps dedicated-family guidance bounded to OmniVoice and converts pack refs into JSON-friendly lists
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.1 - Aligned module verification links with verification ownership while preserving bounded dedicated-family guidance coverage]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from profiles import FamilyProfile, HostProfile, ModuleProfile, ResolvedLaunchProfile
from scripts import runtime_self_check


pytestmark = pytest.mark.unit


def test_runtime_self_check_exposes_dedicated_family_env_guidance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    settings_payload = {
        "models_dir": tmp_path / "models",
        "mlx_models_dir": tmp_path / "models" / "mlx",
        "outputs_dir": tmp_path / "outputs",
        "voices_dir": tmp_path / "voices",
        "upload_staging_dir": tmp_path / "uploads",
        "model_manifest_path": tmp_path / "manifest.json",
    }

    class StubProfileResolver:
        def __init__(self, project_root: Path):
            self.project_root = project_root
            self._families = (
                FamilyProfile(
                    key="qwen",
                    label="Qwen3-TTS",
                    pack_refs={"family": ("qwen",)},
                    isolated_env_name="qwen",
                    supported_capabilities=("preset_speaker_tts",),
                    allowed_backends=("torch",),
                    required_artifacts=("config.json",),
                    benchmark_command="benchmark qwen",
                    self_check_command="self-check qwen",
                ),
                FamilyProfile(
                    key="omnivoice",
                    label="OmniVoice",
                    pack_refs={"base": ("common",), "family": ("omnivoice",)},
                    isolated_env_name="omnivoice",
                    supported_capabilities=("preset_speaker_tts",),
                    allowed_backends=("torch",),
                    required_artifacts=("config.json",),
                    benchmark_command="benchmark omnivoice",
                    self_check_command="self-check omnivoice",
                    optional=True,
                ),
                FamilyProfile(
                    key="piper",
                    label="Piper",
                    pack_refs={"family": ("piper",)},
                    isolated_env_name="piper",
                    supported_capabilities=("preset_speaker_tts",),
                    allowed_backends=("onnx",),
                    required_artifacts=("model.onnx",),
                    benchmark_command="benchmark piper",
                    self_check_command="self-check piper",
                ),
            )
            self._modules = (
                ModuleProfile(
                    key="server",
                    label="HTTP Server",
                    entrypoint="python -m server",
                    transport="http",
                    docker_supported=True,
                    pack_refs={"module": ("server",)},
                    env_prefixes=("TTS_",),
                ),
            )

        def list_family_profiles(self) -> tuple[FamilyProfile, ...]:
            return self._families

        def list_module_profiles(self) -> tuple[ModuleProfile, ...]:
            return self._modules

        def resolve(self, *, family: str, module: str) -> ResolvedLaunchProfile:
            family_profile = next(item for item in self._families if item.key == family)
            module_profile = self._modules[0]
            return ResolvedLaunchProfile(
                host=HostProfile(
                    key="windows-amd64",
                    platform_system="windows",
                    architecture="amd64",
                    python_version="3.11.9",
                    ffmpeg_available=True,
                    docker_available=True,
                    torch_runtime_available=True,
                    cuda_available=False,
                    onnx_providers=("CPUExecutionProvider",),
                ),
                family=family_profile,
                module=module_profile,
                compatible=True,
                reasons=(),
                selected_backend=family_profile.allowed_backends[0],
                required_env_name=family_profile.isolated_env_name,
                expected_python_path=str(tmp_path / ".envs" / family_profile.isolated_env_name / "python"),
                backend_candidates=family_profile.allowed_backends,
                metadata={"pack_refs": {"family": [family_profile.key]}},
            )

    monkeypatch.setattr(runtime_self_check, "parse_core_settings_from_env", lambda environ=None: settings_payload)
    monkeypatch.setattr(
        runtime_self_check,
        "build_runtime",
        lambda settings: SimpleNamespace(
            registry=SimpleNamespace(readiness_report=lambda: {"registry_ready": True, "items": []})
        ),
    )
    monkeypatch.setattr(runtime_self_check, "ProfileResolver", StubProfileResolver)

    # START_BLOCK_ASSERT_DEDICATED_ENV_GUIDANCE
    payload = runtime_self_check.build_self_check_payload()

    dedicated = payload["profiles"]["dedicated_family_envs"]
    indexed = {item["family"]: item for item in dedicated}

    assert set(indexed) == {"omnivoice"}
    assert indexed["omnivoice"]["isolated_env_name"] == "omnivoice"
    assert indexed["omnivoice"]["pack_refs"] == {
        "base": ["common"],
        "family": ["omnivoice"],
    }
    assert all(isinstance(values, list) for values in indexed["omnivoice"]["pack_refs"].values())
    # END_BLOCK_ASSERT_DEDICATED_ENV_GUIDANCE
