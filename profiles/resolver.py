# FILE: profiles/resolver.py
# VERSION: 1.0.2
# START_MODULE_CONTRACT
#   PURPOSE: Resolve host, family, and module profiles into a launchable runtime profile.
#   SCOPE: profile lookup, host synthesis, compatibility evaluation, dependency pack resolution, expected interpreter calculation, dedicated-env runtime probing, and resolved-profile assembly
#   DEPENDS: M-HOST-PROBE, M-PROFILE-SCHEMA
#   LINKS: M-PROFILE-RESOLVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _load_json_profile_map - Load and validate profile JSON object maps from disk
#   ProfileResolver - Resolve host, family, and module inputs into concrete launch-profile payloads
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.2 - Added an exported-surface contract for ProfileResolver and refined resolver scope wording without changing runtime behavior]
# END_CHANGE_SUMMARY

from __future__ import annotations

import shutil
from pathlib import Path
import json
import platform
from collections import OrderedDict
import subprocess

from core.planning.host_probe import HostProbe
from profiles.schema import FamilyProfile, HostProfile, ModuleProfile, ResolvedLaunchProfile


# START_CONTRACT: _load_json_profile_map
#   PURPOSE: Load a JSON-backed profile map and verify that it is object-keyed.
#   INPUTS: { file_path: Path - profile JSON file path }
#   OUTPUTS: { dict[str, dict[str, object]] - validated raw profile mapping }
#   SIDE_EFFECTS: Reads profile configuration from disk
#   LINKS: M-PROFILE-RESOLVER
# END_CONTRACT: _load_json_profile_map
def _load_json_profile_map(file_path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Profile file must contain an object map: {file_path}")
    return payload


# START_CONTRACT: ProfileResolver
#   PURPOSE: Expose the public runtime-profile resolution surface that loads catalogs, evaluates compatibility, resolves dependency packs, computes expected interpreters, and probes dedicated environments.
#   INPUTS: { project_root: Path | None - optional repository root override used to load profile catalogs and environment paths }
#   OUTPUTS: { ProfileResolver - initialized resolver exposing host, catalog, and launch-profile resolution operations }
#   SIDE_EFFECTS: Loads profile catalogs from disk and instantiates host probing utilities for later compatibility and environment checks
#   LINKS: M-PROFILE-RESOLVER, M-HOST-PROBE, M-PROFILE-SCHEMA
# END_CONTRACT: ProfileResolver
class ProfileResolver:
    # START_CONTRACT: ProfileResolver.__init__
    #   PURPOSE: Initialize profile resolver state rooted at the repository profiles directory.
    #   INPUTS: { project_root: Path | None - optional repository root override }
    #   OUTPUTS: { None - resolver is initialized in place }
    #   SIDE_EFFECTS: Instantiates host probing utilities and loads family and module profile catalogs from disk
    #   LINKS: M-PROFILE-RESOLVER, M-HOST-PROBE
    # END_CONTRACT: ProfileResolver.__init__
    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = (project_root or Path(__file__).resolve().parent.parent).resolve()
        self._host_probe = HostProbe()
        self._profiles_dir = self.project_root / "profiles"
        self._family_profiles = self._load_family_profiles()
        self._module_profiles = self._load_module_profiles()

    # START_CONTRACT: ProfileResolver._load_family_profiles
    #   PURPOSE: Load and normalize family profile definitions from disk.
    #   INPUTS: none
    #   OUTPUTS: { dict[str, FamilyProfile] - keyed family profile catalog }
    #   SIDE_EFFECTS: Reads the family profile catalog from disk
    #   LINKS: M-PROFILE-RESOLVER, M-PROFILE-SCHEMA
    # END_CONTRACT: ProfileResolver._load_family_profiles
    def _load_family_profiles(self) -> dict[str, FamilyProfile]:
        raw_profiles = _load_json_profile_map(self._profiles_dir / "families.json")
        return {
            key: FamilyProfile(
                key=key,
                label=str(payload["label"]),
                pack_refs={
                    category: tuple(values)
                    for category, values in dict(payload.get("pack_refs", {})).items()
                },
                isolated_env_name=str(payload["isolated_env_name"]),
                supported_capabilities=tuple(payload.get("supported_capabilities", [])),
                allowed_backends=tuple(payload.get("allowed_backends", [])),
                required_artifacts=tuple(payload.get("required_artifacts", [])),
                benchmark_command=str(payload["benchmark_command"]),
                self_check_command=str(payload["self_check_command"]),
                optional=bool(payload.get("optional", False)),
            )
            for key, payload in raw_profiles.items()
        }

    # START_CONTRACT: ProfileResolver._load_module_profiles
    #   PURPOSE: Load and normalize module profile definitions from disk.
    #   INPUTS: none
    #   OUTPUTS: { dict[str, ModuleProfile] - keyed module profile catalog }
    #   SIDE_EFFECTS: Reads the module profile catalog from disk
    #   LINKS: M-PROFILE-RESOLVER, M-PROFILE-SCHEMA
    # END_CONTRACT: ProfileResolver._load_module_profiles
    def _load_module_profiles(self) -> dict[str, ModuleProfile]:
        raw_profiles = _load_json_profile_map(self._profiles_dir / "modules.json")
        return {
            key: ModuleProfile(
                key=key,
                label=str(payload["label"]),
                entrypoint=str(payload["entrypoint"]),
                transport=str(payload["transport"]),
                docker_supported=bool(payload["docker_supported"]),
                pack_refs={
                    category: tuple(values)
                    for category, values in dict(payload.get("pack_refs", {})).items()
                },
                supported_families=tuple(payload.get("supported_families", [])),
                env_prefixes=tuple(payload.get("env_prefixes", [])),
            )
            for key, payload in raw_profiles.items()
        }

    # START_CONTRACT: ProfileResolver.resolve_host
    #   PURPOSE: Synthesize the current host capability profile used during launch resolution.
    #   INPUTS: none
    #   OUTPUTS: { HostProfile - normalized host runtime profile }
    #   SIDE_EFFECTS: Probes local host capabilities and may import optional runtime packages
    #   LINKS: M-PROFILE-RESOLVER, M-HOST-PROBE, M-PROFILE-SCHEMA
    # END_CONTRACT: ProfileResolver.resolve_host
    def resolve_host(self) -> HostProfile:
        snapshot = self._host_probe.probe()
        onnx_providers: tuple[str, ...] = ()
        try:
            import onnxruntime as ort  # type: ignore

            onnx_providers = tuple(ort.get_available_providers())
        except Exception:
            onnx_providers = ()
        docker_available = shutil.which("docker") is not None
        return HostProfile(
            key=f"{snapshot.platform_system}-{snapshot.architecture}",
            platform_system=snapshot.platform_system,
            architecture=snapshot.architecture,
            python_version=snapshot.python_version,
            ffmpeg_available=snapshot.ffmpeg_available,
            docker_available=docker_available,
            torch_runtime_available=snapshot.torch_runtime_available,
            cuda_available=snapshot.cuda_available,
            onnx_providers=onnx_providers,
        )

    # START_CONTRACT: ProfileResolver.resolve
    #   PURPOSE: Resolve a requested family and module into a concrete launch profile for the current host.
    #   INPUTS: { family: str - requested runtime family key, module: str - requested transport/module key }
    #   OUTPUTS: { ResolvedLaunchProfile - assembled launch contour with compatibility and dependency metadata }
    #   SIDE_EFFECTS: Probes the host and reads profile metadata already loaded by the resolver
    #   LINKS: M-PROFILE-RESOLVER, M-PROFILE-SCHEMA
    # END_CONTRACT: ProfileResolver.resolve
    def resolve(self, *, family: str, module: str) -> ResolvedLaunchProfile:
        # START_BLOCK_ASSEMBLE_RESOLUTION_INPUTS
        family_profile = self.get_family_profile(family)
        module_profile = self.get_module_profile(module)
        host_profile = self.resolve_host()
        compatible, reasons, backend = self._evaluate_compatibility(
            host=host_profile,
            family=family_profile,
            module=module_profile,
        )
        pack_refs = self._resolve_pack_refs(
            host=host_profile,
            family=family_profile,
            module=module_profile,
        )
        pack_files = self._resolve_pack_files(pack_refs)
        # END_BLOCK_ASSEMBLE_RESOLUTION_INPUTS
        return ResolvedLaunchProfile(
            host=host_profile,
            family=family_profile,
            module=module_profile,
            compatible=compatible,
            reasons=tuple(reasons),
            selected_backend=backend,
            required_env_name=family_profile.isolated_env_name,
            expected_python_path=self._expected_python_path(family_profile.isolated_env_name),
            backend_candidates=family_profile.allowed_backends,
            metadata={
                "pack_refs": {key: list(values) for key, values in pack_refs.items()},
                "pack_files": [str(path) for path in pack_files],
                "project_root": str(self.project_root),
            },
        )

    # START_CONTRACT: ProfileResolver.get_family_profile
    #   PURPOSE: Return a normalized family profile by key or raise a user-facing error.
    #   INPUTS: { key: str - family profile key }
    #   OUTPUTS: { FamilyProfile - resolved family profile }
    #   SIDE_EFFECTS: none
    #   LINKS: M-PROFILE-RESOLVER, M-PROFILE-SCHEMA
    # END_CONTRACT: ProfileResolver.get_family_profile
    def get_family_profile(self, key: str) -> FamilyProfile:
        try:
            return self._family_profiles[key]
        except KeyError as exc:
            raise ValueError(f"Unknown family profile: {key}") from exc

    # START_CONTRACT: ProfileResolver.get_module_profile
    #   PURPOSE: Return a normalized module profile by key or raise a user-facing error.
    #   INPUTS: { key: str - module profile key }
    #   OUTPUTS: { ModuleProfile - resolved module profile }
    #   SIDE_EFFECTS: none
    #   LINKS: M-PROFILE-RESOLVER, M-PROFILE-SCHEMA
    # END_CONTRACT: ProfileResolver.get_module_profile
    def get_module_profile(self, key: str) -> ModuleProfile:
        try:
            return self._module_profiles[key]
        except KeyError as exc:
            raise ValueError(f"Unknown module profile: {key}") from exc

    # START_CONTRACT: ProfileResolver.list_family_profiles
    #   PURPOSE: List all loaded family profiles in resolver order.
    #   INPUTS: none
    #   OUTPUTS: { tuple[FamilyProfile, ...] - loaded family profile catalog }
    #   SIDE_EFFECTS: none
    #   LINKS: M-PROFILE-RESOLVER, M-PROFILE-SCHEMA
    # END_CONTRACT: ProfileResolver.list_family_profiles
    def list_family_profiles(self) -> tuple[FamilyProfile, ...]:
        return tuple(self._family_profiles.values())

    # START_CONTRACT: ProfileResolver.list_module_profiles
    #   PURPOSE: List all loaded module profiles in resolver order.
    #   INPUTS: none
    #   OUTPUTS: { tuple[ModuleProfile, ...] - loaded module profile catalog }
    #   SIDE_EFFECTS: none
    #   LINKS: M-PROFILE-RESOLVER, M-PROFILE-SCHEMA
    # END_CONTRACT: ProfileResolver.list_module_profiles
    def list_module_profiles(self) -> tuple[ModuleProfile, ...]:
        return tuple(self._module_profiles.values())

    # START_CONTRACT: ProfileResolver._evaluate_compatibility
    #   PURPOSE: Evaluate host, family, and module compatibility and select the preferred backend.
    #   INPUTS: { host: HostProfile - probed host profile, family: FamilyProfile - requested family profile, module: ModuleProfile - requested module profile }
    #   OUTPUTS: { tuple[bool, list[str], str | None] - compatibility flag, reasons, and selected backend }
    #   SIDE_EFFECTS: May probe the dedicated family environment for installed runtime dependencies
    #   LINKS: M-PROFILE-RESOLVER, M-PROFILE-SCHEMA
    # END_CONTRACT: ProfileResolver._evaluate_compatibility
    def _evaluate_compatibility(
        self,
        *,
        host: HostProfile,
        family: FamilyProfile,
        module: ModuleProfile,
    ) -> tuple[bool, list[str], str | None]:
        reasons: list[str] = []
        selected_backend: str | None = None
        env_runtime_ready = self._family_env_runtime_ready(family)

        if not host.ffmpeg_available:
            reasons.append("ffmpeg_missing")

        if family.key == "piper":
            if "CPUExecutionProvider" not in host.onnx_providers and "CUDAExecutionProvider" not in host.onnx_providers:
                reasons.append("onnx_provider_missing")
            selected_backend = "onnx"
        elif family.key == "qwen":
            if not host.torch_runtime_available and not env_runtime_ready:
                reasons.append("torch_runtime_missing")
            if host.platform_system == "darwin":
                selected_backend = "mlx"
            elif host.cuda_available:
                selected_backend = "qwen_fast"
            else:
                selected_backend = "torch"
        elif family.key == "omnivoice":
            if not host.torch_runtime_available and not env_runtime_ready:
                reasons.append("torch_runtime_missing")
            selected_backend = "torch"

        if module.supported_families and family.key not in module.supported_families:
            reasons.append("module_family_unsupported")

        if module.key == "telegram" and not module.docker_supported and host.docker_available:
            pass

        compatible = not reasons
        return compatible, reasons, selected_backend

    # START_CONTRACT: ProfileResolver._expected_python_path
    #   PURPOSE: Compute the canonical interpreter path for a dedicated family environment.
    #   INPUTS: { env_name: str - isolated environment name }
    #   OUTPUTS: { str - platform-specific interpreter path }
    #   SIDE_EFFECTS: none
    #   LINKS: M-PROFILE-RESOLVER
    # END_CONTRACT: ProfileResolver._expected_python_path
    def _expected_python_path(self, env_name: str) -> str:
        env_root = self.project_root / ".envs" / env_name
        if platform.system().lower() == "windows":
            return str(env_root / "Scripts" / "python.exe")
        return str(env_root / "bin" / "python")

    # START_CONTRACT: ProfileResolver._resolve_pack_refs
    #   PURPOSE: Merge family, platform, and module dependency pack references for a resolution request.
    #   INPUTS: { host: HostProfile - probed host profile, family: FamilyProfile - requested family profile, module: ModuleProfile - requested module profile }
    #   OUTPUTS: { dict[str, tuple[str, ...]] - ordered dependency pack names grouped by category }
    #   SIDE_EFFECTS: none
    #   LINKS: M-PROFILE-RESOLVER, M-PROFILE-SCHEMA
    # END_CONTRACT: ProfileResolver._resolve_pack_refs
    def _resolve_pack_refs(
        self,
        *,
        host: HostProfile,
        family: FamilyProfile,
        module: ModuleProfile,
    ) -> dict[str, tuple[str, ...]]:
        refs: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        refs["base"] = self._merge_pack_names(family.pack_refs.get("base", ()))
        refs["platform"] = self._merge_pack_names(self._host_pack_refs(host))
        refs["module"] = self._merge_pack_names(module.pack_refs.get("module", ()))
        refs["family"] = self._merge_pack_names(family.pack_refs.get("family", ()))
        return dict(refs)

    # START_CONTRACT: ProfileResolver._merge_pack_names
    #   PURPOSE: Merge dependency pack name groups while preserving first-seen order and uniqueness.
    #   INPUTS: { groups: tuple[str, ...] | list[str] - dependency pack name groups }
    #   OUTPUTS: { tuple[str, ...] - ordered unique pack names }
    #   SIDE_EFFECTS: none
    #   LINKS: M-PROFILE-RESOLVER
    # END_CONTRACT: ProfileResolver._merge_pack_names
    @staticmethod
    def _merge_pack_names(*groups: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        merged: list[str] = []
        for group in groups:
            for value in group:
                if value not in merged:
                    merged.append(value)
        return tuple(merged)

    # START_CONTRACT: ProfileResolver._host_pack_refs
    #   PURPOSE: Derive host-specific dependency pack references for the current platform contour.
    #   INPUTS: { host: HostProfile - probed host profile }
    #   OUTPUTS: { tuple[str, ...] - host-derived pack reference names }
    #   SIDE_EFFECTS: none
    #   LINKS: M-PROFILE-RESOLVER, M-PROFILE-SCHEMA
    # END_CONTRACT: ProfileResolver._host_pack_refs
    def _host_pack_refs(self, host: HostProfile) -> tuple[str, ...]:
        refs: list[str] = [host.platform_system.lower()]
        if host.platform_system.lower() == "darwin" and host.architecture.lower() in {"arm64", "aarch64"}:
            refs.append("apple-silicon")
        elif host.cuda_available:
            refs.append("cuda")
        else:
            refs.append("cpu")
        return tuple(refs)

    # START_CONTRACT: ProfileResolver._resolve_pack_files
    #   PURPOSE: Translate resolved dependency pack references into concrete pack file paths.
    #   INPUTS: { pack_refs: dict[str, tuple[str, ...]] - grouped dependency pack names }
    #   OUTPUTS: { tuple[Path, ...] - ordered dependency pack file paths }
    #   SIDE_EFFECTS: none
    #   LINKS: M-PROFILE-RESOLVER
    # END_CONTRACT: ProfileResolver._resolve_pack_files
    def _resolve_pack_files(self, pack_refs: dict[str, tuple[str, ...]]) -> tuple[Path, ...]:
        pack_files: list[Path] = []
        for category in ("base", "platform", "module", "family"):
            for name in pack_refs.get(category, ()): 
                pack_path = self._profiles_dir / "packs" / category / f"{name}.txt"
                pack_files.append(pack_path)
        return tuple(pack_files)

    # START_CONTRACT: ProfileResolver._family_env_runtime_ready
    #   PURPOSE: Check whether the dedicated family environment already contains its required runtime packages.
    #   INPUTS: { family: FamilyProfile - requested family profile }
    #   OUTPUTS: { bool - true when the environment probe reports all required imports present }
    #   SIDE_EFFECTS: Executes the family interpreter to run a small import probe
    #   LINKS: M-PROFILE-RESOLVER, M-PROFILE-SCHEMA
    # END_CONTRACT: ProfileResolver._family_env_runtime_ready
    def _family_env_runtime_ready(self, family: FamilyProfile) -> bool:
        expected_python = Path(self._expected_python_path(family.isolated_env_name))
        if not expected_python.exists():
            return False

        probe = self._family_env_probe_snippet(family.key)
        if probe is None:
            return False

        completed = subprocess.run(
            [str(expected_python), "-c", probe],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return False
        try:
            payload = json.loads(completed.stdout.strip())
        except Exception:
            return False
        return all(bool(value) for value in payload.values())

    # START_CONTRACT: ProfileResolver._family_env_probe_snippet
    #   PURPOSE: Return the interpreter probe snippet used to verify family runtime packages inside an isolated environment.
    #   INPUTS: { family_key: str - runtime family key }
    #   OUTPUTS: { str | None - inline Python snippet or None when no probe is defined }
    #   SIDE_EFFECTS: none
    #   LINKS: M-PROFILE-RESOLVER
    # END_CONTRACT: ProfileResolver._family_env_probe_snippet
    @staticmethod
    def _family_env_probe_snippet(family_key: str) -> str | None:
        probes = {
            "qwen": "import importlib.util, json; print(json.dumps({'torch': importlib.util.find_spec('torch') is not None, 'qwen_tts': importlib.util.find_spec('qwen_tts') is not None}))",
            "piper": "import importlib.util, json; print(json.dumps({'onnxruntime': importlib.util.find_spec('onnxruntime') is not None, 'piper': importlib.util.find_spec('piper') is not None or importlib.util.find_spec('piper_tts') is not None}))",
            "omnivoice": "import importlib.util, json; print(json.dumps({'torch': importlib.util.find_spec('torch') is not None, 'omnivoice': importlib.util.find_spec('omnivoice') is not None}))",
        }
        return probes.get(family_key)
