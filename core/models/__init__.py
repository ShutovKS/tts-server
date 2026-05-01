# FILE: core/models/__init__.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Re-export public model metadata types, catalog helpers, and the composite manifest loader.
#   SCOPE: barrel re-exports for manifest and catalog symbols plus composite manifest helpers
#   DEPENDS: M-MODELS
#   LINKS: M-MODELS
#   ROLE: BARREL
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   ModelManifest - Parsed manifest root model
#   ModelManifestValidationError - Manifest validation error type
#   ModelSpec - Model specification type
#   MODEL_SPECS - Read-only model specification mapping
#   SPEAKER_MAP - Supported speakers by language
#   EMOTION_EXAMPLES - Example voice design prompts
#   get_model_manifest - Load the model manifest
#   get_model_specs - Build mutable model spec mapping
#   load_composite_manifest - Build a manifest from base + per-model fragments + .models/ scan
#   discover_model_manifests - Yield (label, payload) pairs for fragment files
#   CompositeManifestError - Raised on composite manifest merge failures
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Phase 2.7: re-exported the composite manifest loader and discovery helpers.]
# END_CHANGE_SUMMARY

from core.models.catalog import (
    EMOTION_EXAMPLES,
    MODEL_SPECS,
    SPEAKER_MAP,
    get_model_manifest,
    get_model_specs,
)
from core.models.composite import (
    CompositeManifestError,
    discover_model_manifests,
    load_composite_manifest,
)
from core.models.manifest import ModelManifest, ModelManifestValidationError, ModelSpec

__all__ = [
    "CompositeManifestError",
    "EMOTION_EXAMPLES",
    "MODEL_SPECS",
    "ModelManifest",
    "ModelManifestValidationError",
    "ModelSpec",
    "SPEAKER_MAP",
    "discover_model_manifests",
    "get_model_manifest",
    "get_model_specs",
    "load_composite_manifest",
]
