# FILE: core/models/manifest.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Load, validate, and expose typed model manifest metadata.
#   SCOPE: Manifest dataclasses, validation helpers, cached manifest loading
#   DEPENDS: none
#   LINKS: M-MODELS
#   ROLE: TYPES
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   ModelManifestValidationError - Validation error for malformed manifests
#   ArtifactValidationRule - Required artifact rule for a model
#   BackendArtifactValidation - Backend-specific artifact validation ruleset for a model
#   DEFAULT_MODEL_MANIFEST_PATH - Default on-disk manifest path used by loaders
#   ModeMetadata - Typed mode metadata shared across manifest and model specs
#   ModelSpec - Typed specification for a single model
#   ModelRollout - Rollout metadata controlling model enablement and preference
#   ModelDescriptor - Family-aware compatibility descriptor derived from manifest metadata
#   ModelManifest - Typed manifest container for all models
#   SUPPORTED_BACKENDS - Backend identifiers accepted by manifest validation
#   SUPPORTED_MANIFEST_VERSIONS - Manifest schema versions supported by the loader
#   load_model_manifest - Parse and validate manifest file from disk
#   iter_models_for_backend - Iterate enabled models that support a backend
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping


SUPPORTED_MANIFEST_VERSIONS = {1}
SUPPORTED_MODES = {"custom", "design", "clone"}
SUPPORTED_BACKENDS = {"mlx", "torch", "onnx", "qwen_fast"}


class ModelManifestValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactValidationRule:
    name: str
    any_of: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArtifactValidationRule":
        name = _require_non_empty_string(payload, "name")
        any_of_payload = payload.get("any_of")
        if not isinstance(any_of_payload, list) or not any_of_payload:
            raise ModelManifestValidationError(
                f"Artifact validation rule '{name}' must declare a non-empty any_of list"
            )
        any_of = tuple(
            _coerce_non_empty_string(item, f"artifact_validation.{name}.any_of[]")
            for item in any_of_payload
        )
        if len(set(any_of)) != len(any_of):
            raise ModelManifestValidationError(
                f"Artifact validation rule '{name}' contains duplicate paths"
            )
        return cls(name=name, any_of=any_of)

    def matches(self, model_path: Path) -> bool:
        return any(
            (model_path / relative_path).exists() for relative_path in self.any_of
        )

    def describe(self) -> str:
        return "|".join(self.any_of)


@dataclass(frozen=True)
class BackendArtifactValidation:
    required_rules: tuple[ArtifactValidationRule, ...]

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, backend: str
    ) -> "BackendArtifactValidation":
        rules_payload = payload.get("required_rules")
        if not isinstance(rules_payload, list) or not rules_payload:
            raise ModelManifestValidationError(
                f"Backend artifact validation for '{backend}' must declare required_rules"
            )
        rules = tuple(
            ArtifactValidationRule.from_mapping(item) for item in rules_payload
        )
        names = [rule.name for rule in rules]
        if len(set(names)) != len(names):
            raise ModelManifestValidationError(
                f"Backend artifact validation for '{backend}' contains duplicate rule names"
            )
        return cls(required_rules=rules)

    def validate(self, model_path: Path) -> dict[str, Any]:
        missing = [
            rule.describe()
            for rule in self.required_rules
            if not rule.matches(model_path)
        ]
        return {
            "loadable": not missing,
            "required_artifacts": [rule.describe() for rule in self.required_rules],
            "missing_artifacts": missing,
        }


@dataclass(frozen=True)
class ModeMetadata:
    id: str
    label: str
    semantics: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ModeMetadata":
        return cls(
            id=_require_non_empty_string(payload, "id"),
            label=_require_non_empty_string(payload, "label"),
            semantics=_require_non_empty_string(payload, "semantics"),
        )


@dataclass(frozen=True)
class ModelRollout:
    enabled: bool
    stage: str
    default_preference: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ModelRollout":
        enabled = payload.get("enabled")
        if not isinstance(enabled, bool):
            raise ModelManifestValidationError(
                "Model rollout.enabled must be a boolean"
            )
        stage = _require_non_empty_string(payload, "stage")
        default_preference = payload.get("default_preference")
        if not isinstance(default_preference, int):
            raise ModelManifestValidationError(
                "Model rollout.default_preference must be an integer"
            )
        return cls(enabled=enabled, stage=stage, default_preference=default_preference)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    public_name: str
    folder: str
    mode: str
    output_subfolder: str
    metadata: Mapping[str, Any]
    mode_metadata: ModeMetadata
    backend_affinity: tuple[str, ...]
    rollout: ModelRollout
    artifact_validation: Mapping[str, BackendArtifactValidation]

    @property
    def api_name(self) -> str:
        return self.folder

    @property
    def model_id(self) -> str:
        return self.metadata_id

    @property
    def metadata_id(self) -> str:
        metadata_id = self.metadata.get("model_id")
        if isinstance(metadata_id, str) and metadata_id.strip():
            return metadata_id.strip()
        return self.folder

    @property
    def family(self) -> str:
        family = self.metadata.get("family")
        if isinstance(family, str) and family.strip():
            return family.strip()
        return "Qwen3-TTS"

    @property
    def family_key(self) -> str:
        normalized = "".join(
            character if character.isalnum() else "_"
            for character in self.family.lower()
        )
        collapsed = "_".join(part for part in normalized.split("_") if part)
        return collapsed or "qwen3_tts"

    @property
    def supported_capabilities(self) -> tuple[str, ...]:
        explicit = self.metadata.get("supported_capabilities")
        if isinstance(explicit, list) and explicit:
            return tuple(
                _coerce_non_empty_string(
                    item, f"model.{self.key}.metadata.supported_capabilities[]"
                )
                for item in explicit
            )
        compatibility = {
            "custom": "preset_speaker_tts",
            "design": "voice_description_tts",
            "clone": "reference_voice_clone",
        }
        return (compatibility[self.mode],)

    @property
    def host_constraints(self) -> Mapping[str, Any]:
        constraints = self.metadata.get("host_constraints")
        if isinstance(constraints, Mapping):
            return dict(constraints)
        return {}

    @property
    def resource_profile(self) -> Mapping[str, Any]:
        profile = self.metadata.get("resource_profile")
        if isinstance(profile, Mapping):
            return dict(profile)
        return {}

    @property
    def artifact_format(self) -> str:
        artifact_format = self.metadata.get("artifact_format")
        if isinstance(artifact_format, str) and artifact_format.strip():
            return artifact_format.strip()
        return "local_model_dir"

    @property
    def backend_support(self) -> tuple[str, ...]:
        return self.backend_affinity

    @property
    def enabled(self) -> bool:
        return self.rollout.enabled

    def supports_backend(self, backend_key: str) -> bool:
        return backend_key in self.backend_affinity

    def artifact_validation_for_backend(
        self, backend_key: str
    ) -> BackendArtifactValidation:
        validation = self.artifact_validation.get(backend_key)
        if validation is None:
            raise ModelManifestValidationError(
                f"Model '{self.key}' does not define artifact validation for backend '{backend_key}'"
            )
        return validation

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ModelSpec":
        key = _require_non_empty_string(payload, "key")
        public_name = _require_non_empty_string(payload, "public_name")
        folder = _require_non_empty_string(payload, "folder")
        mode = _require_non_empty_string(payload, "mode")
        if mode not in SUPPORTED_MODES:
            raise ModelManifestValidationError(
                f"Model '{key}' declares unsupported mode '{mode}'"
            )
        output_subfolder = _require_non_empty_string(payload, "output_subfolder")
        metadata_payload = payload.get("metadata")
        if not isinstance(metadata_payload, Mapping):
            raise ModelManifestValidationError(f"Model '{key}' must declare metadata")
        mode_metadata_payload = payload.get("mode_metadata")
        if not isinstance(mode_metadata_payload, Mapping):
            raise ModelManifestValidationError(
                f"Model '{key}' must declare mode_metadata"
            )
        backend_affinity_payload = payload.get("backend_affinity")
        if (
            not isinstance(backend_affinity_payload, list)
            or not backend_affinity_payload
        ):
            raise ModelManifestValidationError(
                f"Model '{key}' must declare backend_affinity"
            )
        backend_affinity = tuple(
            _coerce_non_empty_string(item, f"model.{key}.backend_affinity[]")
            for item in backend_affinity_payload
        )
        unknown_backends = sorted(set(backend_affinity) - SUPPORTED_BACKENDS)
        if unknown_backends:
            raise ModelManifestValidationError(
                f"Model '{key}' declares unsupported backends: {unknown_backends}"
            )
        rollout_payload = payload.get("rollout")
        if not isinstance(rollout_payload, Mapping):
            raise ModelManifestValidationError(
                f"Model '{key}' must declare rollout metadata"
            )
        artifact_validation_payload = payload.get("artifact_validation")
        if not isinstance(artifact_validation_payload, Mapping):
            raise ModelManifestValidationError(
                f"Model '{key}' must declare artifact_validation"
            )
        validations = {
            backend: BackendArtifactValidation.from_mapping(value, backend=backend)
            for backend, value in artifact_validation_payload.items()
        }
        missing_backend_rules = sorted(set(backend_affinity) - set(validations))
        if missing_backend_rules:
            raise ModelManifestValidationError(
                f"Model '{key}' is missing artifact_validation for backends: {missing_backend_rules}"
            )
        return cls(
            key=key,
            public_name=public_name,
            folder=folder,
            mode=mode,
            output_subfolder=output_subfolder,
            metadata=dict(metadata_payload),
            mode_metadata=ModeMetadata.from_mapping(mode_metadata_payload),
            backend_affinity=backend_affinity,
            rollout=ModelRollout.from_mapping(rollout_payload),
            artifact_validation=validations,
        )


@dataclass(frozen=True)
class ModelManifest:
    version: int
    metadata: Mapping[str, Any]
    modes: Mapping[str, ModeMetadata]
    models: Mapping[str, ModelSpec]

    def enabled_models(self) -> tuple[ModelSpec, ...]:
        return tuple(spec for spec in self.models.values() if spec.enabled)

    def get(self, key: str) -> ModelSpec:
        return self.models[key]

    def descriptors(self) -> tuple["ModelDescriptor", ...]:
        return tuple(
            ModelDescriptor.from_model_spec(spec) for spec in self.models.values()
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ModelManifest":
        version = payload.get("version")
        if not isinstance(version, int):
            raise ModelManifestValidationError("Manifest version must be an integer")
        if version not in SUPPORTED_MANIFEST_VERSIONS:
            raise ModelManifestValidationError(
                f"Unsupported manifest version: {version}"
            )

        metadata_payload = payload.get("metadata")
        if not isinstance(metadata_payload, Mapping):
            raise ModelManifestValidationError("Manifest metadata must be an object")

        modes_payload = payload.get("modes")
        if not isinstance(modes_payload, list) or not modes_payload:
            raise ModelManifestValidationError("Manifest must declare modes")
        modes = {
            mode.id: mode
            for mode in (ModeMetadata.from_mapping(item) for item in modes_payload)
        }
        if len(modes) != len(modes_payload):
            raise ModelManifestValidationError("Manifest modes must have unique ids")

        models_payload = payload.get("models")
        if not isinstance(models_payload, list) or not models_payload:
            raise ModelManifestValidationError("Manifest must declare models")
        models = {
            spec.key: spec
            for spec in (ModelSpec.from_mapping(item) for item in models_payload)
        }
        if len(models) != len(models_payload):
            raise ModelManifestValidationError("Manifest models must have unique keys")

        folders = [spec.folder for spec in models.values()]
        if len(set(folders)) != len(folders):
            raise ModelManifestValidationError(
                "Manifest models must have unique folder ids"
            )

        for spec in models.values():
            declared_mode = modes.get(spec.mode)
            if declared_mode is None:
                raise ModelManifestValidationError(
                    f"Model '{spec.key}' references mode '{spec.mode}' that is not defined in manifest.modes"
                )
            if declared_mode != spec.mode_metadata:
                raise ModelManifestValidationError(
                    f"Model '{spec.key}' mode_metadata must match manifest-level definition for mode '{spec.mode}'"
                )

        return cls(
            version=version, metadata=dict(metadata_payload), modes=modes, models=models
        )


DEFAULT_MODEL_MANIFEST_PATH = Path(__file__).with_name("manifest.v1.json")


@dataclass(frozen=True)
class ModelDescriptor:
    key: str
    model_id: str
    public_name: str
    family: str
    family_key: str
    legacy_mode: str
    supported_capabilities: tuple[str, ...]
    backend_support: tuple[str, ...]
    artifact_format: str
    folder: str
    output_subfolder: str
    enabled: bool
    rollout_stage: str
    default_preference: int
    host_constraints: Mapping[str, Any]
    resource_profile: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @classmethod
    def from_model_spec(cls, spec: ModelSpec) -> "ModelDescriptor":
        return cls(
            key=spec.key,
            model_id=spec.model_id,
            public_name=spec.public_name,
            family=spec.family,
            family_key=spec.family_key,
            legacy_mode=spec.mode,
            supported_capabilities=spec.supported_capabilities,
            backend_support=spec.backend_support,
            artifact_format=spec.artifact_format,
            folder=spec.folder,
            output_subfolder=spec.output_subfolder,
            enabled=spec.enabled,
            rollout_stage=spec.rollout.stage,
            default_preference=spec.rollout.default_preference,
            host_constraints=dict(spec.host_constraints),
            resource_profile=dict(spec.resource_profile),
            metadata=dict(spec.metadata),
        )


@lru_cache(maxsize=4)
def load_model_manifest(
    path: str | Path = DEFAULT_MODEL_MANIFEST_PATH,
) -> ModelManifest:
    manifest_path = Path(path).resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelManifestValidationError(
            f"Model manifest file does not exist: {manifest_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ModelManifestValidationError(
            f"Model manifest file is not valid JSON: {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ModelManifestValidationError("Model manifest root must be an object")
    return ModelManifest.from_mapping(payload)


def _require_non_empty_string(payload: Mapping[str, Any], field: str) -> str:
    return _coerce_non_empty_string(payload.get(field), field)


def _coerce_non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelManifestValidationError(
            f"Field '{field}' must be a non-empty string"
        )
    return value.strip()


def iter_models_for_backend(
    manifest: ModelManifest, backend_key: str
) -> Iterable[ModelSpec]:
    return (
        spec for spec in manifest.enabled_models() if spec.supports_backend(backend_key)
    )


__all__ = [
    "ArtifactValidationRule",
    "BackendArtifactValidation",
    "DEFAULT_MODEL_MANIFEST_PATH",
    "ModelDescriptor",
    "ModeMetadata",
    "ModelManifest",
    "ModelManifestValidationError",
    "ModelRollout",
    "ModelSpec",
    "SUPPORTED_BACKENDS",
    "SUPPORTED_MANIFEST_VERSIONS",
    "load_model_manifest",
    "iter_models_for_backend",
]
