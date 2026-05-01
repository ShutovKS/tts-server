# FILE: core/services/result_cache.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide a deterministic cache key builder and a swappable storage facade for completed synthesis results so identical (text, model, parameters) requests can short-circuit GPU work.
#   SCOPE: build_cache_key canonicaliser, ResultCache abstract base class, NullResultCache no-op default, FileSystemResultCache local-disk implementation, and CachedResult descriptor wrapping the cached audio payload plus its metadata.
#   DEPENDS: M-CONTRACTS
#   LINKS: M-RESULT-CACHE
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LOGGER - Module logger for cache events.
#   CachedResult - Frozen container holding the cached audio bytes plus replay metadata.
#   ResultCache - Abstract base class describing the get/put/clear contract.
#   NullResultCache - Always-miss default used when caching is disabled.
#   FileSystemResultCache - Local-disk backed cache that stores binary payloads alongside JSON metadata.
#   build_cache_key - Build a stable SHA-256 cache key from a command kind and an arbitrary parameter mapping.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Phase 4.14: introduced ResultCache abstraction with NullResultCache and FileSystemResultCache implementations plus the build_cache_key canonicaliser so SynthesisRouter can short-circuit identical requests]
# END_CHANGE_SUMMARY

from __future__ import annotations

import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.observability import get_logger, log_event

LOGGER = get_logger(__name__)


# START_CONTRACT: CachedResult
#   PURPOSE: Wrap the cached audio payload alongside the metadata required to reconstruct a GenerationResult on a hit.
#   INPUTS: { payload: bytes - Audio bytes returned to the caller, media_type: str - Audio MIME type, model: str - Model identifier the cached result was produced by, mode: str - Synthesis mode (custom/design/clone), backend: str - Backend key that produced the cached result, saved_path: str | None - Optional persisted artefact path, created_at: float - Unix timestamp when the entry was written }
#   OUTPUTS: { instance - Immutable cached entry }
#   SIDE_EFFECTS: none
#   LINKS: M-RESULT-CACHE
# END_CONTRACT: CachedResult
@dataclass(frozen=True)
class CachedResult:
    payload: bytes
    media_type: str
    model: str
    mode: str
    backend: str
    saved_path: str | None
    created_at: float


# START_CONTRACT: build_cache_key
#   PURPOSE: Build a deterministic SHA-256 cache key from a command kind plus an arbitrary parameter mapping so semantically identical requests collapse onto the same key regardless of dict ordering.
#   INPUTS: { kind: str - High-level command discriminator (e.g. "custom"), params: Mapping[str, Any] - JSON-serialisable parameter mapping }
#   OUTPUTS: { str - 64-character lowercase hex digest }
#   SIDE_EFFECTS: none
#   LINKS: M-RESULT-CACHE
# END_CONTRACT: build_cache_key
def build_cache_key(kind: str, params: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"kind": kind, "params": params},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# START_CONTRACT: ResultCache
#   PURPOSE: Abstract storage facade for cached synthesis results.
#   INPUTS: { (no constructor inputs at the ABC level) }
#   OUTPUTS: { instance - Concrete subclass implementing the get/put/clear contract }
#   SIDE_EFFECTS: Subclasses may persist to disk, in-memory, or remote stores.
#   LINKS: M-RESULT-CACHE
# END_CONTRACT: ResultCache
class ResultCache(ABC):
    @abstractmethod
    def get(self, key: str) -> CachedResult | None: ...

    @abstractmethod
    def put(self, key: str, value: CachedResult) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...


# START_CONTRACT: NullResultCache
#   PURPOSE: No-op cache used when result caching is disabled so callers can always go through the same get/put seam.
#   INPUTS: {}
#   OUTPUTS: { instance - Always-miss cache }
#   SIDE_EFFECTS: none
#   LINKS: M-RESULT-CACHE
# END_CONTRACT: NullResultCache
class NullResultCache(ResultCache):
    def get(self, key: str) -> CachedResult | None:
        return None

    def put(self, key: str, value: CachedResult) -> None:
        return None

    def clear(self) -> None:
        return None


# START_CONTRACT: FileSystemResultCache
#   PURPOSE: Local-disk backed result cache that stores audio payloads as binary blobs alongside JSON metadata.
#   INPUTS: { cache_dir: Path - Directory to persist entries under (created on demand), max_entries: int - Optional in-memory soft cap; oldest entries are evicted when the limit is exceeded }
#   OUTPUTS: { instance - File-system backed cache }
#   SIDE_EFFECTS: Creates cache_dir, writes binary and JSON files when put is called, and removes files on clear/eviction.
#   LINKS: M-RESULT-CACHE
# END_CONTRACT: FileSystemResultCache
class FileSystemResultCache(ResultCache):
    def __init__(self, cache_dir: Path, *, max_entries: int = 1024) -> None:
        self._cache_dir = Path(cache_dir)
        self._max_entries = max(1, max_entries)
        self._lock = threading.RLock()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def _entry_paths(self, key: str) -> tuple[Path, Path]:
        # START_BLOCK_RESOLVE_ENTRY_PATHS
        safe_key = "".join(ch for ch in key if ch.isalnum())[:64]
        if not safe_key:
            safe_key = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return (
            self._cache_dir / f"{safe_key}.bin",
            self._cache_dir / f"{safe_key}.meta.json",
        )
        # END_BLOCK_RESOLVE_ENTRY_PATHS

    def get(self, key: str) -> CachedResult | None:
        # START_BLOCK_LOOKUP_CACHE_ENTRY
        with self._lock:
            payload_path, meta_path = self._entry_paths(key)
            if not (payload_path.exists() and meta_path.exists()):
                return None
            try:
                payload = payload_path.read_bytes()
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                log_event(
                    LOGGER,
                    level=30,
                    event="[ResultCache][get][BLOCK_DROP_CORRUPT_ENTRY]",
                    message="Dropping corrupt cache entry",
                    key=key,
                    error=str(exc),
                )
                payload_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                return None
            return CachedResult(
                payload=payload,
                media_type=str(meta.get("media_type", "audio/wav")),
                model=str(meta.get("model", "")),
                mode=str(meta.get("mode", "")),
                backend=str(meta.get("backend", "")),
                saved_path=meta.get("saved_path"),
                created_at=float(meta.get("created_at", time.time())),
            )
        # END_BLOCK_LOOKUP_CACHE_ENTRY

    def put(self, key: str, value: CachedResult) -> None:
        # START_BLOCK_STORE_CACHE_ENTRY
        with self._lock:
            payload_path, meta_path = self._entry_paths(key)
            payload_path.write_bytes(value.payload)
            meta_path.write_text(
                json.dumps(
                    {
                        "media_type": value.media_type,
                        "model": value.model,
                        "mode": value.mode,
                        "backend": value.backend,
                        "saved_path": value.saved_path,
                        "created_at": value.created_at,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            self._evict_if_needed()
        # END_BLOCK_STORE_CACHE_ENTRY

    def clear(self) -> None:
        # START_BLOCK_CLEAR_CACHE
        with self._lock:
            for path in list(self._cache_dir.iterdir()):
                if path.is_file():
                    path.unlink(missing_ok=True)
        # END_BLOCK_CLEAR_CACHE

    def _evict_if_needed(self) -> None:
        # START_BLOCK_EVICT_OVERFLOW_ENTRIES
        meta_files = sorted(
            (p for p in self._cache_dir.iterdir() if p.suffix == ".json"),
            key=lambda p: p.stat().st_mtime,
        )
        overflow = len(meta_files) - self._max_entries
        if overflow <= 0:
            return
        for meta in meta_files[:overflow]:
            payload = meta.with_name(meta.stem.removesuffix(".meta") + ".bin")
            meta.unlink(missing_ok=True)
            payload.unlink(missing_ok=True)
        log_event(
            LOGGER,
            level=20,
            event="[ResultCache][_evict_if_needed][BLOCK_EVICT_OVERFLOW_ENTRIES]",
            message="Evicted cache overflow entries",
            evicted=overflow,
        )
        # END_BLOCK_EVICT_OVERFLOW_ENTRIES


__all__ = [
    "CachedResult",
    "FileSystemResultCache",
    "NullResultCache",
    "ResultCache",
    "build_cache_key",
]
