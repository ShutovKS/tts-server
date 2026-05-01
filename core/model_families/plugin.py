# FILE: core/model_families/plugin.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define the unified ModelFamilyPlugin extension contract that bundles a family's load, synthesize, and artifact-validation responsibilities behind a single seam.
#   SCOPE: FamilyExecutionRequest / FamilyExecutionResult DTOs, ModelFamilyPlugin abstract base class, FamilyPluginRegistry for in-process registration and capability lookup
#   DEPENDS: M-MODEL-FAMILY, M-MODELS
#   LINKS: M-MODEL-FAMILY
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   FamilyExecutionRequest - Normalized synthesis inputs handed to a plugin.
#   FamilyExecutionResult - Generation result returned by a plugin (waveforms + sample rate).
#   ModelFamilyPlugin - Abstract single-seam extension contract for new model families.
#   FamilyPluginRegistry - Process-local registry that holds plugins and answers capability/backend lookups.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Phase 2.5: introduced unified ModelFamilyPlugin extension contract alongside the existing ModelFamilyAdapter and TorchFamilyStrategy seams. Not yet wired into the runtime; entry-point discovery follows in Phase 2.6.]
# END_CHANGE_SUMMARY

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.models.manifest import ModelSpec


# START_CONTRACT: FamilyExecutionRequest
#   PURPOSE: Carry the inputs a plugin needs to run synthesis without leaking transport-specific or planner-specific types.
#   INPUTS: { capability: str - Synthesis capability identifier, execution_mode: str - Resolved execution mode (custom/design/clone/...), text: str - Prompt text, language: str - Resolved language code, output_dir: Path - Directory plugins may persist artifacts into, payload: dict[str, Any] - Family-specific generation inputs prepared by the planner }
#   OUTPUTS: { instance - Frozen request handed to ModelFamilyPlugin.synthesize }
#   SIDE_EFFECTS: none
#   LINKS: M-MODEL-FAMILY
# END_CONTRACT: FamilyExecutionRequest
@dataclass(frozen=True)
class FamilyExecutionRequest:
    capability: str
    execution_mode: str
    text: str
    language: str
    output_dir: Path
    payload: dict[str, Any] = field(default_factory=dict)


# START_CONTRACT: FamilyExecutionResult
#   PURPOSE: Carry generated audio back from a plugin in a backend-agnostic shape.
#   INPUTS: { waveforms: list[Any] - Sequence of generated waveforms (numpy or torch tensors); first item is treated as the primary output, sample_rate: int - Sample rate associated with the waveforms }
#   OUTPUTS: { instance - Frozen result returned by ModelFamilyPlugin.synthesize }
#   SIDE_EFFECTS: none
#   LINKS: M-MODEL-FAMILY
# END_CONTRACT: FamilyExecutionResult
@dataclass(frozen=True)
class FamilyExecutionResult:
    waveforms: list[Any]
    sample_rate: int


# START_CONTRACT: ModelFamilyPlugin
#   PURPOSE: Single seam for adding a new model family. A plugin owns its capabilities, supported backends, runtime availability, model loading, synthesis, and artifact validation.
#   INPUTS: { subclass - Concrete plugin class declaring key, label, capabilities, supported_backends }
#   OUTPUTS: { instance - Plugin ready to be registered with FamilyPluginRegistry }
#   SIDE_EFFECTS: none
#   LINKS: M-MODEL-FAMILY
# END_CONTRACT: ModelFamilyPlugin
class ModelFamilyPlugin(ABC):
    key: str = ""
    label: str = ""
    capabilities: tuple[str, ...] = ()
    supported_backends: tuple[str, ...] = ()

    # START_CONTRACT: is_available
    #   PURPOSE: Report whether the plugin's runtime dependencies are importable in the current environment.
    #   INPUTS: {}
    #   OUTPUTS: { bool - True when the plugin can load its model and run synthesis }
    #   SIDE_EFFECTS: May trigger lazy imports of the family's runtime package
    #   LINKS: M-MODEL-FAMILY
    # END_CONTRACT: is_available
    @abstractmethod
    def is_available(self) -> bool: ...

    # START_CONTRACT: import_error
    #   PURPOSE: Surface the captured ImportError from the family's runtime, if any.
    #   INPUTS: {}
    #   OUTPUTS: { Exception | None - Captured import error, or None }
    #   SIDE_EFFECTS: none
    #   LINKS: M-MODEL-FAMILY
    # END_CONTRACT: import_error
    @abstractmethod
    def import_error(self) -> Exception | None: ...

    # START_CONTRACT: load_model
    #   PURPOSE: Load (or return a cached) runtime model for the requested spec on the requested backend.
    #   INPUTS: { spec: ModelSpec - Manifest spec of the model to load, backend_key: str - Backend the model should be prepared for, model_path: Path - Resolved on-disk path to the model directory }
    #   OUTPUTS: { Any - Loaded runtime model handle (plugin-specific) }
    #   SIDE_EFFECTS: May allocate device memory and import optional runtime packages
    #   LINKS: M-MODEL-FAMILY
    # END_CONTRACT: load_model
    @abstractmethod
    def load_model(
        self,
        *,
        spec: ModelSpec,
        backend_key: str,
        model_path: Path,
    ) -> Any: ...

    # START_CONTRACT: synthesize
    #   PURPOSE: Run synthesis against a previously loaded runtime model and return audio.
    #   INPUTS: { model: Any - Runtime model returned by load_model, request: FamilyExecutionRequest - Normalized synthesis inputs, backend_key: str - Backend the runtime model is prepared for }
    #   OUTPUTS: { FamilyExecutionResult - Generated waveforms and sample rate }
    #   SIDE_EFFECTS: Performs inference; may write artifacts into request.output_dir if the plugin chooses to persist them itself
    #   LINKS: M-MODEL-FAMILY
    # END_CONTRACT: synthesize
    @abstractmethod
    def synthesize(
        self,
        model: Any,
        request: FamilyExecutionRequest,
        *,
        backend_key: str,
    ) -> FamilyExecutionResult: ...

    # START_CONTRACT: validate_artifacts
    #   PURPOSE: Validate the on-disk artifacts of the model against the family's expectations for the given backend.
    #   INPUTS: { spec: ModelSpec - Manifest spec, backend_key: str - Backend the model is being validated for, model_path: Path | None - Resolved on-disk path or None when the model is missing }
    #   OUTPUTS: { dict[str, Any] - { "loadable": bool, "required_artifacts": list, "missing_artifacts": list, ... } }
    #   SIDE_EFFECTS: Reads the filesystem to check artifact presence
    #   LINKS: M-MODEL-FAMILY
    # END_CONTRACT: validate_artifacts
    def validate_artifacts(
        self,
        *,
        spec: ModelSpec,
        backend_key: str,
        model_path: Path | None,
    ) -> dict[str, Any]:
        validation = spec.artifact_validation_for_backend(backend_key)
        if model_path is None:
            return {
                "loadable": False,
                "required_artifacts": [rule.describe() for rule in validation.required_rules],
                "missing_artifacts": ["model_directory"],
            }
        return validation.validate(model_path)


# START_CONTRACT: FamilyPluginRegistry
#   PURPOSE: Hold a process-local set of registered ModelFamilyPlugin instances and answer lookups by family key, capability, or supported backend.
#   INPUTS: { plugins: Iterable[ModelFamilyPlugin] | None - Optional initial plugins }
#   OUTPUTS: { instance - Registry ready for lookup }
#   SIDE_EFFECTS: none
#   LINKS: M-MODEL-FAMILY
# END_CONTRACT: FamilyPluginRegistry
class FamilyPluginRegistry:
    def __init__(self, plugins: tuple[ModelFamilyPlugin, ...] | None = None):
        self._plugins: dict[str, ModelFamilyPlugin] = {}
        for plugin in plugins or ():
            self.register(plugin)

    # START_CONTRACT: register
    #   PURPOSE: Add a plugin to the registry under its declared key.
    #   INPUTS: { plugin: ModelFamilyPlugin - Plugin to register }
    #   OUTPUTS: { None }
    #   SIDE_EFFECTS: Mutates the registry
    #   LINKS: M-MODEL-FAMILY
    # END_CONTRACT: register
    def register(self, plugin: ModelFamilyPlugin) -> None:
        key = plugin.key
        if not key:
            raise ValueError("ModelFamilyPlugin must declare a non-empty 'key'")
        if key in self._plugins:
            raise ValueError(f"ModelFamilyPlugin key '{key}' is already registered")
        self._plugins[key] = plugin

    # START_CONTRACT: get
    #   PURPOSE: Return the plugin registered under the given family key, or None when absent.
    #   INPUTS: { key: str - Family key }
    #   OUTPUTS: { ModelFamilyPlugin | None }
    #   SIDE_EFFECTS: none
    #   LINKS: M-MODEL-FAMILY
    # END_CONTRACT: get
    def get(self, key: str) -> ModelFamilyPlugin | None:
        return self._plugins.get(key)

    # START_CONTRACT: keys
    #   PURPOSE: List the family keys currently registered.
    #   INPUTS: {}
    #   OUTPUTS: { tuple[str, ...] - Family keys in deterministic (sorted) order }
    #   SIDE_EFFECTS: none
    #   LINKS: M-MODEL-FAMILY
    # END_CONTRACT: keys
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))

    # START_CONTRACT: plugins
    #   PURPOSE: List the registered plugins in deterministic (sorted-by-key) order.
    #   INPUTS: {}
    #   OUTPUTS: { tuple[ModelFamilyPlugin, ...] }
    #   SIDE_EFFECTS: none
    #   LINKS: M-MODEL-FAMILY
    # END_CONTRACT: plugins
    def plugins(self) -> tuple[ModelFamilyPlugin, ...]:
        return tuple(self._plugins[key] for key in sorted(self._plugins))

    # START_CONTRACT: for_capability
    #   PURPOSE: Return all plugins that declare support for the given capability.
    #   INPUTS: { capability: str - Capability identifier }
    #   OUTPUTS: { tuple[ModelFamilyPlugin, ...] - Plugins whose `capabilities` include the requested capability }
    #   SIDE_EFFECTS: none
    #   LINKS: M-MODEL-FAMILY
    # END_CONTRACT: for_capability
    def for_capability(self, capability: str) -> tuple[ModelFamilyPlugin, ...]:
        return tuple(
            self._plugins[key]
            for key in sorted(self._plugins)
            if capability in self._plugins[key].capabilities
        )

    # START_CONTRACT: for_backend
    #   PURPOSE: Return all plugins that declare support for the given backend.
    #   INPUTS: { backend_key: str - Backend identifier (e.g. "torch", "onnx", "mlx") }
    #   OUTPUTS: { tuple[ModelFamilyPlugin, ...] }
    #   SIDE_EFFECTS: none
    #   LINKS: M-MODEL-FAMILY
    # END_CONTRACT: for_backend
    def for_backend(self, backend_key: str) -> tuple[ModelFamilyPlugin, ...]:
        return tuple(
            self._plugins[key]
            for key in sorted(self._plugins)
            if backend_key in self._plugins[key].supported_backends
        )


__all__ = [
    "FamilyExecutionRequest",
    "FamilyExecutionResult",
    "FamilyPluginRegistry",
    "ModelFamilyPlugin",
]
