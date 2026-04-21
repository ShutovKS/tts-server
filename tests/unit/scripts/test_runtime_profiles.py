# FILE: tests/unit/scripts/test_runtime_profiles.py
# VERSION: 1.1.1
# START_MODULE_CONTRACT
#   PURPOSE: Validate the public runtime profile schema, resolver, package exports, and bounded runtime self-check profile payload exposure.
#   SCOPE: profiles package barrel exports, profile DTO serialization, resolver catalog and resolution behavior, runtime self-check profile payload assembly
#   DEPENDS: M-PROFILES, M-PROFILE-SCHEMA, M-PROFILE-RESOLVER, M-RUNTIME-SELF-CHECK
#   LINKS: V-M-BOOTSTRAP, V-M-PROFILE-SCHEMA, V-M-PROFILE-RESOLVER, V-M-PROFILES, V-M-LAUNCHER
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   PROJECT_ROOT - Repository root used to instantiate the real profile resolver
#   _expected_python_path_for_env - Build the platform-aware isolated-environment interpreter path expected from resolver payloads
#   test_profiles_package_exports_public_schema_and_resolver_symbols - Verifies the profiles package barrel re-exports the public schema and resolver surface intentionally
#   test_profile_schema_dataclasses_serialize_nested_payloads_without_aliasing_metadata - Verifies profile DTO serialization remains deterministic and metadata copies are isolated from callers
#   test_profile_resolver_lists_known_family_and_module_profiles - Verifies the resolver exposes known family and module catalogs through its public listing surface
#   test_profile_resolver_resolves_qwen_server_with_platform_aware_metadata - Verifies resolver output carries explicit compatibility, pack metadata, and interpreter-path evidence for qwen/server
#   test_profile_resolver_uses_family_env_when_qwen_host_runtime_is_missing - Verifies qwen resolution falls back to dedicated-family runtime readiness when host torch is absent
#   test_profile_resolver_prefers_onnx_for_piper_cli_when_provider_is_available - Verifies Piper resolution stays explicitly ONNX-backed when an ONNX provider is available
#   test_profile_resolver_reports_incompatibility_reasons_when_qwen_runtime_prerequisites_are_missing - Verifies resolver failure payloads carry explicit missing-runtime reasons for unsupported qwen host contours
#   test_runtime_self_check_builds_profile_payload_from_resolver_surface - Verifies runtime self-check assembles the profile payload surface from the imported resolver boundary without bootstrapping a real runtime
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.1 - Aligned module verification links with verification ownership and added direct negative resolver coverage for explicit qwen incompatibility reasons]
# END_CHANGE_SUMMARY

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import profiles
from profiles import FamilyProfile, HostProfile, ModuleProfile, ProfileResolver, ResolvedLaunchProfile
from scripts import runtime_self_check


PROJECT_ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.unit


# START_CONTRACT: _expected_python_path_for_env
#   PURPOSE: Compute the platform-aware isolated-environment interpreter path expected from resolver payloads.
#   INPUTS: { project_root: Path - repository root, env_name: str - isolated environment name }
#   OUTPUTS: { str - canonical expected interpreter path for the active platform }
#   SIDE_EFFECTS: none
#   LINKS: V-M-PROFILE-RESOLVER
# END_CONTRACT: _expected_python_path_for_env
def _expected_python_path_for_env(project_root: Path, env_name: str) -> str:
    env_root = project_root / ".envs" / env_name
    if sys.platform.startswith("win"):
        return str(env_root / "Scripts" / "python.exe")
    return str(env_root / "bin" / "python")


def test_profiles_package_exports_public_schema_and_resolver_symbols():
    # START_BLOCK_ASSERT_PUBLIC_BARREL_EXPORTS
    assert profiles.ProfileResolver is ProfileResolver
    assert profiles.HostProfile is HostProfile
    assert profiles.FamilyProfile is FamilyProfile
    assert profiles.ModuleProfile is ModuleProfile
    assert profiles.ResolvedLaunchProfile is ResolvedLaunchProfile
    assert set(profiles.__all__) >= {
        "FamilyProfile",
        "HostProfile",
        "ModuleProfile",
        "ProfileResolver",
        "ResolvedLaunchProfile",
    }
    # END_BLOCK_ASSERT_PUBLIC_BARREL_EXPORTS


def test_profile_schema_dataclasses_serialize_nested_payloads_without_aliasing_metadata():
    # START_BLOCK_BUILD_PROFILE_DTOS
    host = HostProfile(
        key="windows-amd64",
        platform_system="windows",
        architecture="amd64",
        python_version="3.11.9",
        ffmpeg_available=True,
        docker_available=True,
        torch_runtime_available=False,
        cuda_available=False,
        onnx_providers=("CPUExecutionProvider",),
    )
    family = FamilyProfile(
        key="qwen",
        label="Qwen3-TTS",
        pack_refs={"base": ("common",), "family": ("qwen",)},
        isolated_env_name="qwen",
        supported_capabilities=(
            "preset_speaker_tts",
            "voice_description_tts",
            "reference_voice_clone",
        ),
        allowed_backends=("mlx", "qwen_fast", "torch"),
        required_artifacts=("config.json",),
        benchmark_command="python scripts/validate_runtime.py representative-models --target qwen",
        self_check_command="python scripts/runtime_self_check.py",
    )
    module = ModuleProfile(
        key="server",
        label="HTTP Server",
        entrypoint="python -m server",
        transport="http",
        docker_supported=True,
        pack_refs={"module": ("server",)},
        supported_families=("qwen", "omnivoice"),
        env_prefixes=("TTS_",),
    )
    resolved = ResolvedLaunchProfile(
        host=host,
        family=family,
        module=module,
        compatible=True,
        selected_backend="torch",
        required_env_name="qwen",
        expected_python_path=_expected_python_path_for_env(PROJECT_ROOT, "qwen"),
        backend_candidates=("mlx", "qwen_fast", "torch"),
        metadata={"pack_refs": {"family": ["qwen"]}, "project_root": str(PROJECT_ROOT)},
    )
    # END_BLOCK_BUILD_PROFILE_DTOS

    # START_BLOCK_ASSERT_PROFILE_DTO_SERIALIZATION
    host_payload = host.to_dict()
    family_payload = family.to_dict()
    module_payload = module.to_dict()
    resolved_payload = resolved.to_dict()

    assert host_payload["onnx_providers"] == ("CPUExecutionProvider",)
    assert family_payload["pack_refs"]["family"] == ("qwen",)
    assert module_payload["supported_families"] == ("qwen", "omnivoice")
    assert resolved_payload["family"]["key"] == "qwen"
    assert resolved_payload["module"]["entrypoint"] == "python -m server"
    assert resolved_payload["backend_candidates"] == ("mlx", "qwen_fast", "torch")
    assert resolved_payload["metadata"] == {
        "pack_refs": {"family": ["qwen"]},
        "project_root": str(PROJECT_ROOT),
    }

    resolved_payload["metadata"]["mutated"] = True

    assert resolved.metadata == {
        "pack_refs": {"family": ["qwen"]},
        "project_root": str(PROJECT_ROOT),
    }
    # END_BLOCK_ASSERT_PROFILE_DTO_SERIALIZATION


def test_profile_resolver_lists_known_family_and_module_profiles():
    resolver = ProfileResolver(PROJECT_ROOT)

    # START_BLOCK_ASSERT_CATALOG_SURFACE
    family_keys = {profile.key for profile in resolver.list_family_profiles()}
    module_keys = {profile.key for profile in resolver.list_module_profiles()}
    qwen = resolver.get_family_profile("qwen")
    server = resolver.get_module_profile("server")

    assert {"qwen", "piper", "omnivoice"}.issubset(family_keys)
    assert {"server", "cli", "telegram"}.issubset(module_keys)
    assert qwen.pack_refs["family"] == ("qwen",)
    assert qwen.allowed_backends == ("mlx", "qwen_fast", "torch")
    assert server.entrypoint == "python -m server"
    assert server.transport == "http"
    assert server.pack_refs["module"] == ("server",)
    # END_BLOCK_ASSERT_CATALOG_SURFACE


def test_profile_resolver_resolves_qwen_server_with_platform_aware_metadata(monkeypatch: pytest.MonkeyPatch):
    resolver = ProfileResolver(PROJECT_ROOT)
    host = HostProfile(
        key="linux-amd64",
        platform_system="linux",
        architecture="amd64",
        python_version="3.11.9",
        ffmpeg_available=True,
        docker_available=True,
        torch_runtime_available=True,
        cuda_available=False,
        onnx_providers=(),
    )
    monkeypatch.setattr(resolver, "resolve_host", lambda: host)
    monkeypatch.setattr(resolver, "_family_env_runtime_ready", lambda family: False)

    # START_BLOCK_ASSERT_QWEN_SERVER_RESOLUTION
    resolved = resolver.resolve(family="qwen", module="server").to_dict()

    assert resolved["compatible"] is True
    assert list(resolved["reasons"]) == []
    assert resolved["family"]["key"] == "qwen"
    assert resolved["module"]["key"] == "server"
    assert resolved["selected_backend"] == "torch"
    assert resolved["required_env_name"] == "qwen"
    assert resolved["expected_python_path"] == _expected_python_path_for_env(PROJECT_ROOT, "qwen")
    assert resolved["backend_candidates"] == ("mlx", "qwen_fast", "torch")
    assert resolved["metadata"]["project_root"] == str(PROJECT_ROOT)
    assert resolved["metadata"]["pack_refs"] == {
        "base": ["common"],
        "platform": ["linux", "cpu"],
        "module": ["server"],
        "family": ["qwen"],
    }
    pack_files = [path.replace("\\", "/") for path in resolved["metadata"]["pack_files"]]

    assert any(path.endswith("profiles/packs/base/common.txt") for path in pack_files)
    assert any(path.endswith("profiles/packs/platform/linux.txt") for path in pack_files)
    assert any(path.endswith("profiles/packs/platform/cpu.txt") for path in pack_files)
    assert any(path.endswith("profiles/packs/module/server.txt") for path in pack_files)
    assert any(path.endswith("profiles/packs/family/qwen.txt") for path in pack_files)
    # END_BLOCK_ASSERT_QWEN_SERVER_RESOLUTION


def test_profile_resolver_uses_family_env_when_qwen_host_runtime_is_missing(monkeypatch: pytest.MonkeyPatch):
    resolver = ProfileResolver(PROJECT_ROOT)

    monkeypatch.setattr(
        resolver,
        "resolve_host",
        lambda: HostProfile(
            key="windows-amd64",
            platform_system="windows",
            architecture="amd64",
            python_version="3.11.9",
            ffmpeg_available=True,
            docker_available=True,
            torch_runtime_available=False,
            cuda_available=False,
            onnx_providers=("CPUExecutionProvider",),
        ),
    )
    monkeypatch.setattr(resolver, "_family_env_runtime_ready", lambda family: family.key == "qwen")

    resolved = resolver.resolve(family="qwen", module="server").to_dict()

    assert resolved["compatible"] is True
    assert list(resolved["reasons"]) == []
    assert resolved["selected_backend"] == "torch"


def test_profile_resolver_prefers_onnx_for_piper_cli_when_provider_is_available(monkeypatch: pytest.MonkeyPatch):
    resolver = ProfileResolver(PROJECT_ROOT)

    monkeypatch.setattr(
        resolver,
        "resolve_host",
        lambda: HostProfile(
            key="windows-amd64",
            platform_system="windows",
            architecture="amd64",
            python_version="3.11.9",
            ffmpeg_available=True,
            docker_available=True,
            torch_runtime_available=False,
            cuda_available=False,
            onnx_providers=("CPUExecutionProvider",),
        ),
    )

    resolved = resolver.resolve(family="piper", module="cli").to_dict()

    assert resolved["family"]["key"] == "piper"
    assert resolved["module"]["key"] == "cli"
    assert resolved["compatible"] is True
    assert list(resolved["reasons"]) == []
    assert resolved["selected_backend"] == "onnx"
    assert resolved["metadata"]["pack_refs"]["family"] == ["piper"]
    assert resolved["metadata"]["pack_refs"]["module"] == ["cli"]


def test_profile_resolver_reports_incompatibility_reasons_when_qwen_runtime_prerequisites_are_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    resolver = ProfileResolver(PROJECT_ROOT)
    host = HostProfile(
        key="linux-amd64",
        platform_system="linux",
        architecture="amd64",
        python_version="3.11.9",
        ffmpeg_available=False,
        docker_available=True,
        torch_runtime_available=False,
        cuda_available=False,
        onnx_providers=(),
    )
    monkeypatch.setattr(resolver, "resolve_host", lambda: host)
    monkeypatch.setattr(resolver, "_family_env_runtime_ready", lambda family: False)

    resolved = resolver.resolve(family="qwen", module="server").to_dict()

    assert resolved["compatible"] is False
    assert tuple(resolved["reasons"]) == ("ffmpeg_missing", "torch_runtime_missing")
    assert resolved["selected_backend"] == "torch"
    assert resolved["required_env_name"] == "qwen"


def test_runtime_self_check_builds_profile_payload_from_resolver_surface(
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
    readiness_report = {
        "registry_ready": True,
        "items": [],
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
                    supported_capabilities=(
                        "preset_speaker_tts",
                        "voice_description_tts",
                        "reference_voice_clone",
                    ),
                    allowed_backends=("torch",),
                    required_artifacts=("config.json",),
                    benchmark_command="benchmark qwen",
                    self_check_command="self-check qwen",
                ),
                FamilyProfile(
                    key="omnivoice",
                    label="OmniVoice",
                    pack_refs={"family": ("omnivoice",), "base": ("common",)},
                    isolated_env_name="omnivoice",
                    supported_capabilities=("preset_speaker_tts",),
                    allowed_backends=("torch",),
                    required_artifacts=("config.json",),
                    benchmark_command="benchmark omnivoice",
                    self_check_command="self-check omnivoice",
                    optional=True,
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
                ModuleProfile(
                    key="cli",
                    label="CLI",
                    entrypoint="python -m cli",
                    transport="local_cli",
                    docker_supported=False,
                    pack_refs={"module": ("cli",)},
                    supported_families=("qwen", "omnivoice"),
                    env_prefixes=("TTS_",),
                ),
            )

        def list_family_profiles(self) -> tuple[FamilyProfile, ...]:
            return self._families

        def list_module_profiles(self) -> tuple[ModuleProfile, ...]:
            return self._modules

        def resolve(self, *, family: str, module: str) -> ResolvedLaunchProfile:
            family_profile = next(item for item in self._families if item.key == family)
            module_profile = next(item for item in self._modules if item.key == module)
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
                    onnx_providers=(),
                ),
                family=family_profile,
                module=module_profile,
                compatible=True,
                reasons=(),
                selected_backend="torch",
                required_env_name=family_profile.isolated_env_name,
                expected_python_path=str(tmp_path / ".envs" / family_profile.isolated_env_name / "python"),
                backend_candidates=family_profile.allowed_backends,
                metadata={"pack_refs": {"family": [family_profile.key], "module": [module_profile.key]}},
            )

    monkeypatch.setattr(runtime_self_check, "parse_core_settings_from_env", lambda environ=None: settings_payload)
    monkeypatch.setattr(
        runtime_self_check,
        "build_runtime",
        lambda settings: SimpleNamespace(
            registry=SimpleNamespace(readiness_report=lambda: readiness_report)
        ),
    )
    monkeypatch.setattr(runtime_self_check, "ProfileResolver", StubProfileResolver)

    # START_BLOCK_ASSERT_SELF_CHECK_PROFILE_PAYLOAD
    payload = runtime_self_check.build_self_check_payload(
        {
            "TTS_QWEN_FAST_TEST_MODE": "simulated",
        }
    )
    profiles_payload = payload["profiles"]

    assert payload["status"] == "ok"
    assert payload["settings"] == {
        "models_dir": str(settings_payload["models_dir"]),
        "outputs_dir": str(settings_payload["outputs_dir"]),
        "voices_dir": str(settings_payload["voices_dir"]),
        "upload_staging_dir": str(settings_payload["upload_staging_dir"]),
        "model_manifest_path": str(settings_payload["model_manifest_path"]),
        "configured_backend": None,
        "backend_autoselect": True,
        "qwen_fast_enabled": True,
        "qwen_fast_test_mode": "simulated",
        "model_preload_policy": "none",
        "model_preload_ids": [],
    }
    assert [item["key"] for item in profiles_payload["families"]] == ["qwen", "omnivoice"]
    assert [item["key"] for item in profiles_payload["modules"]] == ["server", "cli"]
    assert {
        (item["family"]["key"], item["module"]["key"])
        for item in profiles_payload["resolved_launch_profiles"]
    } == {
        ("qwen", "server"),
        ("qwen", "cli"),
        ("omnivoice", "server"),
        ("omnivoice", "cli"),
    }
    assert profiles_payload["dedicated_family_envs"] == [
        {
            "family": "omnivoice",
            "isolated_env_name": "omnivoice",
            "pack_refs": {
                "family": ["omnivoice"],
                "base": ["common"],
            },
        }
    ]
    # END_BLOCK_ASSERT_SELF_CHECK_PROFILE_PAYLOAD
