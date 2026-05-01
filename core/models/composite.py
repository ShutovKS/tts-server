# FILE: core/models/composite.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Build a ModelManifest by merging a base manifest with optional per-model JSON files under core/models/manifests/ and auto-discovered model_manifest.json files inside the on-disk .models/ directory.
#   SCOPE: load_composite_manifest, discover_model_manifests, MANIFEST_FRAGMENT_FILENAME, DEFAULT_MANIFEST_FRAGMENTS_DIR, CompositeManifestError
#   DEPENDS: M-MODELS
#   LINKS: M-MODELS
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   MANIFEST_FRAGMENT_FILENAME - Filename used by per-folder manifest fragments inside .models/.
#   DEFAULT_MANIFEST_FRAGMENTS_DIR - Default directory under core/models/ where per-model JSON fragments live.
#   CompositeManifestError - Raised when fragment merging produces an invalid combined manifest.
#   discover_model_manifests - Yield (label, payload) pairs from a fragments directory plus a .models/ scan.
#   load_composite_manifest - Build a ModelManifest from a base manifest plus optional fragments and .models/ scan.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Phase 2.7: introduced composite manifest loader with .models/ auto-discovery and per-fragment merging.]
# END_CHANGE_SUMMARY

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from core.models.manifest import (
    DEFAULT_MODEL_MANIFEST_PATH,
    ModelManifest,
    ModelManifestValidationError,
)

logger = logging.getLogger(__name__)

MANIFEST_FRAGMENT_FILENAME = "model_manifest.json"
DEFAULT_MANIFEST_FRAGMENTS_DIR = Path(__file__).with_name("manifests")


class CompositeManifestError(ModelManifestValidationError):
    """Raised when fragment merging produces an invalid combined manifest."""


# START_CONTRACT: discover_model_manifests
#   PURPOSE: Yield (label, payload) pairs for every JSON manifest fragment found under a fragments directory plus every per-folder model_manifest.json found by walking a .models/ directory tree (depth-1 children only).
#   INPUTS: { fragments_dir: Path | None - Optional directory containing per-model JSON files, models_dir: Path | None - Optional .models/-style directory to walk for model_manifest.json files }
#   OUTPUTS: { Iterator[tuple[str, Mapping[str, Any]]] - (source label, parsed JSON payload) pairs in deterministic sorted order }
#   SIDE_EFFECTS: Reads files from disk; logs a warning when a candidate file fails to parse and is skipped
#   LINKS: M-MODELS
# END_CONTRACT: discover_model_manifests
def discover_model_manifests(
    *,
    fragments_dir: Path | None,
    models_dir: Path | None,
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    candidates: list[tuple[str, Path]] = []

    if fragments_dir is not None and fragments_dir.is_dir():
        for path in sorted(fragments_dir.glob("*.json")):
            candidates.append((f"fragments::{path.name}", path))

    if models_dir is not None and models_dir.is_dir():
        for child in sorted(models_dir.iterdir()):
            if not child.is_dir():
                continue
            fragment_path = child / MANIFEST_FRAGMENT_FILENAME
            if fragment_path.is_file():
                candidates.append(
                    (f"models_dir::{child.name}/{MANIFEST_FRAGMENT_FILENAME}", fragment_path)
                )

    for label, path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("composite_manifest: skipping %s (%s)", label, exc)
            continue
        if not isinstance(payload, Mapping):
            logger.warning("composite_manifest: skipping %s (root must be a JSON object)", label)
            continue
        yield label, payload


# START_CONTRACT: load_composite_manifest
#   PURPOSE: Build a single ModelManifest from a base manifest file plus optional per-model JSON fragments and an optional .models/ auto-discovery scan, merging the `models` arrays and re-running ModelManifest validation against the combined payload.
#   INPUTS: { base_path: Path - Base manifest path, defaults to DEFAULT_MODEL_MANIFEST_PATH, fragments_dir: Path | None - Optional directory of per-model JSON files (defaults to DEFAULT_MANIFEST_FRAGMENTS_DIR if it exists), models_dir: Path | None - Optional .models/-style directory to scan for per-folder model_manifest.json fragments }
#   OUTPUTS: { ModelManifest - Combined manifest with merged models }
#   SIDE_EFFECTS: Reads the base manifest plus every discovered fragment from disk; raises CompositeManifestError on duplicate keys / model_ids across fragments
#   LINKS: M-MODELS
# END_CONTRACT: load_composite_manifest
def load_composite_manifest(
    *,
    base_path: Path = DEFAULT_MODEL_MANIFEST_PATH,
    fragments_dir: Path | None = None,
    models_dir: Path | None = None,
) -> ModelManifest:
    base_path = Path(base_path).resolve()
    try:
        base_payload = json.loads(base_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CompositeManifestError(f"Base manifest does not exist: {base_path}") from exc
    except json.JSONDecodeError as exc:
        raise CompositeManifestError(
            f"Base manifest is not valid JSON: {base_path}: {exc}"
        ) from exc
    if not isinstance(base_payload, Mapping):
        raise CompositeManifestError("Base manifest root must be a JSON object")

    if fragments_dir is None and DEFAULT_MANIFEST_FRAGMENTS_DIR.is_dir():
        fragments_dir = DEFAULT_MANIFEST_FRAGMENTS_DIR

    raw_base_models = base_payload.get("models", [])
    if not isinstance(raw_base_models, list):
        raise CompositeManifestError("Base manifest models must be a list")
    base_models: list[Mapping[str, Any]] = list(raw_base_models)

    extra_models: list[Mapping[str, Any]] = []
    for label, payload in discover_model_manifests(
        fragments_dir=fragments_dir, models_dir=models_dir
    ):
        for spec in _iter_fragment_models(payload, label=label):
            extra_models.append(spec)

    combined_payload: dict[str, Any] = dict(base_payload)
    combined_payload["models"] = [*base_models, *extra_models]
    # ModelManifest.from_mapping enforces unique `key` and unique derived
    # `model_id` across the full models list - including fragments.
    return ModelManifest.from_mapping(combined_payload)


# START_BLOCK_FRAGMENT_HELPERS
def _iter_fragment_models(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> Iterable[Mapping[str, Any]]:
    """Accept three fragment shapes:
    1. {"models": [...]}
    2. {"model": {...}}
    3. A single model object (heuristic: must contain 'key' and 'model_id').
    """

    if "models" in payload:
        models = payload["models"]
        if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
            raise CompositeManifestError(
                f"Manifest fragment {label!r} 'models' field must be a list"
            )
        for item in models:
            if not isinstance(item, Mapping):
                raise CompositeManifestError(
                    f"Manifest fragment {label!r} contains a non-object model entry"
                )
            yield item
        return

    if "model" in payload:
        single = payload["model"]
        if not isinstance(single, Mapping):
            raise CompositeManifestError(
                f"Manifest fragment {label!r} 'model' field must be an object"
            )
        yield single
        return

    # Single-model heuristic: a payload that has the two ModelSpec fields
    # `key` and `folder` is treated as a single model spec.
    if "key" in payload and "folder" in payload:
        yield payload
        return

    raise CompositeManifestError(
        f"Manifest fragment {label!r} must declare 'models', 'model', or be a single model object"
    )


# END_BLOCK_FRAGMENT_HELPERS


__all__ = [
    "CompositeManifestError",
    "DEFAULT_MANIFEST_FRAGMENTS_DIR",
    "MANIFEST_FRAGMENT_FILENAME",
    "discover_model_manifests",
    "load_composite_manifest",
]
