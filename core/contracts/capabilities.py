# FILE: core/contracts/capabilities.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide a process-local CapabilityRegistry so synthesis capabilities are declared as runtime data instead of as a closed Literal type, allowing new families to introduce new capabilities (and matching execution modes) without editing the shared contract module.
#   SCOPE: CapabilitySpec dataclass, CapabilityRegistry class with register/get/is_supported/execution_mode_for/capability_for_mode helpers, the default DEFAULT_CAPABILITY_REGISTRY pre-populated with the three built-in capabilities (preset_speaker_tts, voice_description_tts, reference_voice_clone) used by the Qwen3 family today.
#   DEPENDS: M-CONTRACTS
#   LINKS: M-EXECUTION-PLAN, M-CAPABILITIES
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   CapabilitySpec - Frozen description of a single synthesis capability (name, execution_mode, description).
#   CapabilityRegistry - Process-local registry that owns the capability <-> execution_mode mapping.
#   DEFAULT_CAPABILITY_REGISTRY - Module-level registry pre-populated with the three built-in capabilities.
#   register_default_capability - Convenience function to register an additional capability into the default registry.
#   is_supported_capability - Predicate that defers to the default registry.
#   default_capability_for_mode - Convenience reverse lookup against the default registry.
#   default_execution_mode_for - Convenience forward lookup against the default registry.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Phase 3.10: introduced CapabilityRegistry and DEFAULT_CAPABILITY_REGISTRY so capabilities are declared as runtime data and synthesis.SynthesisCapability is now an open str alias instead of a closed Literal]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import dataclass, field


# START_CONTRACT: CapabilitySpec
#   PURPOSE: Describe a single synthesis capability — its public name, the execution mode it maps to, and an optional human-readable description for diagnostics and docs.
#   INPUTS: { name: str - Stable capability identifier (e.g., "preset_speaker_tts"), execution_mode: str - Execution mode handled by the planner/registry (e.g., "custom"), description: str - Optional human-readable description, aliases: tuple[str, ...] - Optional alternative identifiers that should resolve to this capability via lookup }
#   OUTPUTS: { instance - Frozen capability descriptor }
#   SIDE_EFFECTS: none
#   LINKS: M-CAPABILITIES
# END_CONTRACT: CapabilitySpec
@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    execution_mode: str
    description: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)


# START_CONTRACT: CapabilityRegistry
#   PURPOSE: Hold a process-local set of CapabilitySpec instances and answer forward (name -> execution_mode) and reverse (execution_mode -> name) lookups so callers do not have to hardcode capability identifiers.
#   INPUTS: { specs: tuple[CapabilitySpec, ...] | None - Optional initial specs }
#   OUTPUTS: { instance - Registry ready for register/get lookups }
#   SIDE_EFFECTS: none on construction; register(...) mutates the registry
#   LINKS: M-CAPABILITIES
# END_CONTRACT: CapabilityRegistry
class CapabilityRegistry:
    def __init__(self, specs: tuple[CapabilitySpec, ...] | None = None) -> None:
        self._by_name: dict[str, CapabilitySpec] = {}
        self._by_mode: dict[str, str] = {}
        for spec in specs or ():
            self.register(spec)

    # START_CONTRACT: register
    #   PURPOSE: Register a new capability spec; raises if the name or execution_mode collides with an existing entry.
    #   INPUTS: { spec: CapabilitySpec - Capability to register }
    #   OUTPUTS: { None }
    #   SIDE_EFFECTS: Mutates the registry's name and mode indices
    #   LINKS: M-CAPABILITIES
    # END_CONTRACT: register
    def register(self, spec: CapabilitySpec) -> None:
        if not spec.name:
            raise ValueError("CapabilitySpec must declare a non-empty 'name'")
        if not spec.execution_mode:
            raise ValueError(
                f"CapabilitySpec '{spec.name}' must declare a non-empty 'execution_mode'"
            )
        if spec.name in self._by_name:
            raise ValueError(f"Capability '{spec.name}' is already registered")
        if spec.execution_mode in self._by_mode:
            raise ValueError(
                f"Execution mode '{spec.execution_mode}' is already mapped to "
                f"'{self._by_mode[spec.execution_mode]}'"
            )
        self._by_name[spec.name] = spec
        self._by_mode[spec.execution_mode] = spec.name
        for alias in spec.aliases:
            if alias and alias not in self._by_name:
                self._by_name[alias] = spec

    # START_CONTRACT: get
    #   PURPOSE: Return the spec registered under the given name (or alias), or None when absent.
    #   INPUTS: { name: str - Capability name or alias }
    #   OUTPUTS: { CapabilitySpec | None }
    #   SIDE_EFFECTS: none
    #   LINKS: M-CAPABILITIES
    # END_CONTRACT: get
    def get(self, name: str) -> CapabilitySpec | None:
        return self._by_name.get(name)

    # START_CONTRACT: is_supported
    #   PURPOSE: Report whether a capability identifier is known to the registry (including aliases).
    #   INPUTS: { name: str - Capability name }
    #   OUTPUTS: { bool }
    #   SIDE_EFFECTS: none
    #   LINKS: M-CAPABILITIES
    # END_CONTRACT: is_supported
    def is_supported(self, name: str) -> bool:
        return name in self._by_name

    # START_CONTRACT: execution_mode_for
    #   PURPOSE: Resolve a capability name to its execution mode.
    #   INPUTS: { name: str - Capability name or alias }
    #   OUTPUTS: { str - Execution mode (e.g., "custom") }
    #   SIDE_EFFECTS: none
    #   LINKS: M-CAPABILITIES
    # END_CONTRACT: execution_mode_for
    def execution_mode_for(self, name: str) -> str:
        spec = self._by_name.get(name)
        if spec is None:
            raise KeyError(f"Unknown capability: {name!r}")
        return spec.execution_mode

    # START_CONTRACT: capability_for_mode
    #   PURPOSE: Resolve an execution mode to its primary capability name.
    #   INPUTS: { mode: str - Execution mode (e.g., "custom") }
    #   OUTPUTS: { str - Primary capability name }
    #   SIDE_EFFECTS: none
    #   LINKS: M-CAPABILITIES
    # END_CONTRACT: capability_for_mode
    def capability_for_mode(self, mode: str) -> str:
        try:
            return self._by_mode[mode]
        except KeyError as exc:
            raise ValueError(f"Unsupported execution mode: {mode}") from exc

    # START_CONTRACT: names
    #   PURPOSE: List the canonical capability names currently registered (excluding aliases) in deterministic order.
    #   INPUTS: {}
    #   OUTPUTS: { tuple[str, ...] }
    #   SIDE_EFFECTS: none
    #   LINKS: M-CAPABILITIES
    # END_CONTRACT: names
    def names(self) -> tuple[str, ...]:
        canonical = [name for name, spec in self._by_name.items() if name == spec.name]
        return tuple(sorted(canonical))

    # START_CONTRACT: execution_modes
    #   PURPOSE: List the execution modes currently registered, in deterministic order.
    #   INPUTS: {}
    #   OUTPUTS: { tuple[str, ...] }
    #   SIDE_EFFECTS: none
    #   LINKS: M-CAPABILITIES
    # END_CONTRACT: execution_modes
    def execution_modes(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_mode))

    # START_CONTRACT: specs
    #   PURPOSE: Return the registered specs in deterministic (sorted-by-name) order.
    #   INPUTS: {}
    #   OUTPUTS: { tuple[CapabilitySpec, ...] }
    #   SIDE_EFFECTS: none
    #   LINKS: M-CAPABILITIES
    # END_CONTRACT: specs
    def specs(self) -> tuple[CapabilitySpec, ...]:
        return tuple(
            self._by_name[name]
            for name in sorted(self._by_name)
            if self._by_name[name].name == name
        )


# START_CONTRACT: DEFAULT_CAPABILITY_REGISTRY
#   PURPOSE: Provide the canonical process-wide registry used by SynthesisRequest, the planner, and routes; pre-populated with the three capabilities that ship with the runtime.
#   INPUTS: { none }
#   OUTPUTS: { instance - Module-level registry callers can extend by calling register_default_capability(...) }
#   SIDE_EFFECTS: none
#   LINKS: M-CAPABILITIES
# END_CONTRACT: DEFAULT_CAPABILITY_REGISTRY
DEFAULT_CAPABILITY_REGISTRY: CapabilityRegistry = CapabilityRegistry(
    specs=(
        CapabilitySpec(
            name="preset_speaker_tts",
            execution_mode="custom",
            description="Synthesise speech using a preset speaker plus optional instruction prompt.",
        ),
        CapabilitySpec(
            name="voice_description_tts",
            execution_mode="design",
            description="Design a voice from a free-form natural language description.",
        ),
        CapabilitySpec(
            name="reference_voice_clone",
            execution_mode="clone",
            description="Clone a voice from a reference audio sample (and optional reference transcript).",
        ),
    )
)


# START_CONTRACT: register_default_capability
#   PURPOSE: Register an additional capability into the module-level default registry.
#   INPUTS: { spec: CapabilitySpec - Capability to register }
#   OUTPUTS: { None }
#   SIDE_EFFECTS: Mutates DEFAULT_CAPABILITY_REGISTRY
#   LINKS: M-CAPABILITIES
# END_CONTRACT: register_default_capability
def register_default_capability(spec: CapabilitySpec) -> None:
    DEFAULT_CAPABILITY_REGISTRY.register(spec)


# START_CONTRACT: is_supported_capability
#   PURPOSE: Predicate against the default registry; returns True when the capability is registered.
#   INPUTS: { name: str - Capability name }
#   OUTPUTS: { bool }
#   SIDE_EFFECTS: none
#   LINKS: M-CAPABILITIES
# END_CONTRACT: is_supported_capability
def is_supported_capability(name: str) -> bool:
    return DEFAULT_CAPABILITY_REGISTRY.is_supported(name)


# START_CONTRACT: default_execution_mode_for
#   PURPOSE: Forward lookup helper bound to the default registry.
#   INPUTS: { name: str - Capability name }
#   OUTPUTS: { str - Execution mode }
#   SIDE_EFFECTS: none
#   LINKS: M-CAPABILITIES
# END_CONTRACT: default_execution_mode_for
def default_execution_mode_for(name: str) -> str:
    return DEFAULT_CAPABILITY_REGISTRY.execution_mode_for(name)


# START_CONTRACT: default_capability_for_mode
#   PURPOSE: Reverse lookup helper bound to the default registry.
#   INPUTS: { mode: str - Execution mode }
#   OUTPUTS: { str - Capability name }
#   SIDE_EFFECTS: none
#   LINKS: M-CAPABILITIES
# END_CONTRACT: default_capability_for_mode
def default_capability_for_mode(mode: str) -> str:
    return DEFAULT_CAPABILITY_REGISTRY.capability_for_mode(mode)


__all__ = [
    "DEFAULT_CAPABILITY_REGISTRY",
    "CapabilityRegistry",
    "CapabilitySpec",
    "default_capability_for_mode",
    "default_execution_mode_for",
    "is_supported_capability",
    "register_default_capability",
]
