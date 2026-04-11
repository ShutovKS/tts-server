# FILE: core/models/catalog.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Expose static model catalog helpers and current public model metadata mappings.
#   SCOPE: Speaker map, emotion examples, manifest accessors, exported model spec registry
#   DEPENDS: M-MODELS
#   LINKS: M-MODELS
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   SPEAKER_MAP - Supported speakers grouped by language
#   EMOTION_EXAMPLES - Example voice design descriptions
#   DEFAULT_MODEL_MANIFEST_PATH - Default manifest file location used by catalog helpers
#   ModelManifest - Typed manifest container re-exported for catalog consumers
#   ModelSpec - Typed model specification re-exported for registry consumers
#   get_model_manifest - Load model manifest from disk
#   get_model_specs - Return model specs as a mutable dictionary
#   MODEL_SPECS - Read-only model specs mapping
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

from types import MappingProxyType

from core.models.manifest import (
    DEFAULT_MODEL_MANIFEST_PATH,
    ModelManifest,
    ModelSpec,
    load_model_manifest,
)


SPEAKER_MAP = {
    "English": ["Ryan", "Aiden", "Ethan", "Chelsie", "Serena", "Vivian"],
    "Chinese": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"],
    "Japanese": ["Ono_Anna"],
    "Korean": ["Sohee"],
}

EMOTION_EXAMPLES = [
    "Sad and crying, speaking slowly",
    "Excited and happy, speaking very fast",
    "Angry and shouting",
    "Whispering quietly",
]


def get_model_manifest(path=DEFAULT_MODEL_MANIFEST_PATH) -> ModelManifest:
    return load_model_manifest(path)


def get_model_specs() -> dict[str, ModelSpec]:
    return dict(get_model_manifest().models)


MODEL_SPECS = MappingProxyType(get_model_specs())


__all__ = [
    "DEFAULT_MODEL_MANIFEST_PATH",
    "EMOTION_EXAMPLES",
    "MODEL_SPECS",
    "ModelManifest",
    "ModelSpec",
    "SPEAKER_MAP",
    "get_model_manifest",
    "get_model_specs",
]
