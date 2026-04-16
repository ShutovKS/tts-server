# FILE: profiles/schema.py
# VERSION: 1.0.2
# START_MODULE_CONTRACT
#   PURPOSE: Define first-class schema objects for host, family, module, and resolved runtime profiles.
#   SCOPE: immutable dataclasses and simple serialization helpers
#   DEPENDS: none
#   LINKS: M-PROFILE-SCHEMA
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   HostProfile - Immutable host capability snapshot used during launch-profile resolution
#   FamilyProfile - Immutable family definition carrying runtime pack and capability metadata
#   ModuleProfile - Immutable module definition carrying entrypoint and transport metadata
#   ResolvedLaunchProfile - Immutable resolved launch contour with compatibility, backend, and metadata payloads
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.2 - Added exported-surface contracts for profile DTO dataclasses while preserving serialization behavior]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# START_CONTRACT: HostProfile
#   PURPOSE: Represent an immutable snapshot of host identity and runtime capabilities used during launch-profile resolution.
#   INPUTS: { key: str - stable host identifier, platform_system: str - detected operating system, architecture: str - detected CPU architecture, python_version: str - active Python version, ffmpeg_available: bool - ffmpeg availability flag, docker_available: bool - docker availability flag, torch_runtime_available: bool - host torch availability flag, cuda_available: bool - CUDA availability flag, onnx_providers: tuple[str, ...] - detected ONNX runtime providers }
#   OUTPUTS: { HostProfile - immutable host capability DTO shared across resolver and launcher payloads }
#   SIDE_EFFECTS: none
#   LINKS: M-PROFILE-SCHEMA, M-PROFILE-RESOLVER
# END_CONTRACT: HostProfile
@dataclass(frozen=True)
class HostProfile:
    key: str
    platform_system: str
    architecture: str
    python_version: str
    ffmpeg_available: bool
    docker_available: bool
    torch_runtime_available: bool
    cuda_available: bool
    onnx_providers: tuple[str, ...] = ()

    # START_CONTRACT: HostProfile.to_dict
    #   PURPOSE: Serialize the immutable host profile into a plain dictionary payload.
    #   INPUTS: none
    #   OUTPUTS: { dict[str, Any] - JSON-serializable host profile fields }
    #   SIDE_EFFECTS: none
    #   LINKS: M-PROFILE-SCHEMA
    # END_CONTRACT: HostProfile.to_dict
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# START_CONTRACT: FamilyProfile
#   PURPOSE: Represent an immutable runtime family definition with dependency pack references, backend policy, and lifecycle commands.
#   INPUTS: { key: str - stable family identifier, label: str - operator-facing family label, pack_refs: dict[str, tuple[str, ...]] - grouped dependency pack references, isolated_env_name: str - dedicated environment name, supported_capabilities: tuple[str, ...] - supported capability identifiers, allowed_backends: tuple[str, ...] - permitted backend candidates, required_artifacts: tuple[str, ...] - required runtime artifacts, benchmark_command: str - benchmark entry command, self_check_command: str - family self-check command, optional: bool - optional family availability flag }
#   OUTPUTS: { FamilyProfile - immutable family DTO used by profile resolution and launch planning }
#   SIDE_EFFECTS: none
#   LINKS: M-PROFILE-SCHEMA, M-PROFILE-RESOLVER
# END_CONTRACT: FamilyProfile
@dataclass(frozen=True)
class FamilyProfile:
    key: str
    label: str
    pack_refs: dict[str, tuple[str, ...]]
    isolated_env_name: str
    supported_capabilities: tuple[str, ...]
    allowed_backends: tuple[str, ...]
    required_artifacts: tuple[str, ...]
    benchmark_command: str
    self_check_command: str
    optional: bool = False

    # START_CONTRACT: FamilyProfile.to_dict
    #   PURPOSE: Serialize the immutable family profile into a plain dictionary payload.
    #   INPUTS: none
    #   OUTPUTS: { dict[str, Any] - JSON-serializable family profile fields }
    #   SIDE_EFFECTS: none
    #   LINKS: M-PROFILE-SCHEMA
    # END_CONTRACT: FamilyProfile.to_dict
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# START_CONTRACT: ModuleProfile
#   PURPOSE: Represent an immutable transport or launcher module definition with entrypoint and pack-overlay metadata.
#   INPUTS: { key: str - stable module identifier, label: str - operator-facing module label, entrypoint: str - configured module entrypoint, transport: str - transport kind, docker_supported: bool - docker support flag, pack_refs: dict[str, tuple[str, ...]] - module-specific dependency pack references, supported_families: tuple[str, ...] - compatible family keys, env_prefixes: tuple[str, ...] - environment variable prefixes for the module }
#   OUTPUTS: { ModuleProfile - immutable module DTO used by resolver selection and launcher execution planning }
#   SIDE_EFFECTS: none
#   LINKS: M-PROFILE-SCHEMA, M-PROFILE-RESOLVER
# END_CONTRACT: ModuleProfile
@dataclass(frozen=True)
class ModuleProfile:
    key: str
    label: str
    entrypoint: str
    transport: str
    docker_supported: bool
    pack_refs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    supported_families: tuple[str, ...] = ()
    env_prefixes: tuple[str, ...] = ()

    # START_CONTRACT: ModuleProfile.to_dict
    #   PURPOSE: Serialize the immutable module profile into a plain dictionary payload.
    #   INPUTS: none
    #   OUTPUTS: { dict[str, Any] - JSON-serializable module profile fields }
    #   SIDE_EFFECTS: none
    #   LINKS: M-PROFILE-SCHEMA
    # END_CONTRACT: ModuleProfile.to_dict
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# START_CONTRACT: ResolvedLaunchProfile
#   PURPOSE: Represent the fully resolved launch contour produced from host, family, and module inputs plus compatibility and environment metadata.
#   INPUTS: { host: HostProfile - resolved host capabilities, family: FamilyProfile - resolved family definition, module: ModuleProfile - resolved module definition, compatible: bool - overall compatibility flag, reasons: tuple[str, ...] - incompatibility reasons, selected_backend: str | None - chosen backend, required_env_name: str | None - dedicated environment name, expected_python_path: str | None - computed interpreter path, backend_candidates: tuple[str, ...] - allowed backend candidates, metadata: dict[str, Any] - dependency-pack and repository metadata }
#   OUTPUTS: { ResolvedLaunchProfile - immutable resolved launch DTO consumed by launcher commands }
#   SIDE_EFFECTS: none
#   LINKS: M-PROFILE-SCHEMA, M-PROFILE-RESOLVER, M-LAUNCHER
# END_CONTRACT: ResolvedLaunchProfile
@dataclass(frozen=True)
class ResolvedLaunchProfile:
    host: HostProfile
    family: FamilyProfile
    module: ModuleProfile
    compatible: bool
    reasons: tuple[str, ...] = ()
    selected_backend: str | None = None
    required_env_name: str | None = None
    expected_python_path: str | None = None
    backend_candidates: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    # START_CONTRACT: ResolvedLaunchProfile.to_dict
    #   PURPOSE: Serialize the resolved launch profile while copying mutable metadata for safe callers.
    #   INPUTS: none
    #   OUTPUTS: { dict[str, Any] - JSON-serializable resolved launch profile payload }
    #   SIDE_EFFECTS: none
    #   LINKS: M-PROFILE-SCHEMA, M-PROFILE-RESOLVER
    # END_CONTRACT: ResolvedLaunchProfile.to_dict
    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata)
        return payload
