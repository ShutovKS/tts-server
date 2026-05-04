# FILE: core/engines/model_cache.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide a thread-safe engine-layer model handle cache without exposing eviction mechanics to service orchestration.
#   SCOPE: ModelCache keying, get-or-load behavior, bounded FIFO eviction, and invalidation helpers
#   DEPENDS: M-ENGINE-CONTRACTS, M-MODELS
#   LINKS: M-ENGINE-CONTRACTS, M-ENGINE-RUNTIME-FACTORY
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   ModelCacheKey - Immutable cache key for one engine/model/backend/path tuple
#   ModelCache - Thread-safe bounded model-handle cache
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added engine-layer model handle cache seam for runtime factory composition]
# END_CHANGE_SUMMARY

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from core.engines.contracts import ModelHandle


# START_CONTRACT: ModelCacheKey
#   PURPOSE: Represent the stable identity of one cached engine/model/backend/path tuple.
#   INPUTS: { engine_key: str - Engine identifier, model_id: str - Stable model identifier, backend_key: str - Backend identifier, model_path: str | None - Resolved on-disk model path when available }
#   OUTPUTS: { instance - Immutable cache key }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-MODEL-CACHE, M-ENGINE-CONTRACTS
# END_CONTRACT: ModelCacheKey
@dataclass(frozen=True)
class ModelCacheKey:
    engine_key: str
    model_id: str
    backend_key: str
    model_path: str | None

    @classmethod
    # START_CONTRACT: from_parts
    #   PURPOSE: Build a stable cache key from runtime parts while normalizing Path input into a string payload.
    #   INPUTS: { engine_key: str - Engine identifier, model_id: str - Stable model identifier, backend_key: str - Backend identifier, model_path: Path | None - Resolved model path }
    #   OUTPUTS: { ModelCacheKey - Immutable normalized key }
    #   SIDE_EFFECTS: none
    #   LINKS: M-ENGINE-MODEL-CACHE
    # END_CONTRACT: from_parts
    def from_parts(
        cls,
        *,
        engine_key: str,
        model_id: str,
        backend_key: str,
        model_path: Path | None,
    ) -> ModelCacheKey:
        return cls(
            engine_key=engine_key,
            model_id=model_id,
            backend_key=backend_key,
            model_path=str(model_path) if model_path is not None else None,
        )


# START_CONTRACT: ModelCache
#   PURPOSE: Provide a bounded thread-safe cache for reusable ModelHandle instances in the engine runtime layer.
#   INPUTS: { max_entries: int - Maximum number of cached handles retained at once }
#   OUTPUTS: { instance - Ready model-handle cache }
#   SIDE_EFFECTS: none on construction
#   LINKS: M-ENGINE-MODEL-CACHE, M-ENGINE-RUNTIME-FACTORY
# END_CONTRACT: ModelCache
class ModelCache:
    def __init__(self, max_entries: int = 1) -> None:
        if max_entries < 0:
            raise ValueError("max_entries must be at least 0")
        self._max_entries = max_entries
        self._items: OrderedDict[ModelCacheKey, ModelHandle] = OrderedDict()
        self._lock = RLock()

    @property
    def max_entries(self) -> int:
        return self._max_entries

    # START_CONTRACT: get_or_load
    #   PURPOSE: Return a cached model handle for the key or load/store one atomically when absent.
    #   INPUTS: { key: ModelCacheKey - Cache key for the requested model handle, loader: Callable[[], ModelHandle] - Lazy loader used on cache miss }
    #   OUTPUTS: { ModelHandle - Cached or newly loaded handle }
    #   SIDE_EFFECTS: Mutates the cache and may evict the oldest entry when capacity is exceeded.
    #   LINKS: M-ENGINE-MODEL-CACHE, M-ENGINE-CONTRACTS
    # END_CONTRACT: get_or_load
    def get_or_load(self, key: ModelCacheKey, loader: Callable[[], ModelHandle]) -> ModelHandle:
        if self._max_entries == 0:
            return loader()
        with self._lock:
            cached = self._items.get(key)
            if cached is not None:
                self._items.move_to_end(key)
                return cached
            loaded = loader()
            self._items[key] = loaded
            self._items.move_to_end(key)
            while len(self._items) > self._max_entries:
                self._items.popitem(last=False)
            return loaded

    # START_CONTRACT: clear
    #   PURPOSE: Remove all cached model handles.
    #   INPUTS: {}
    #   OUTPUTS: { None }
    #   SIDE_EFFECTS: Empties the cache.
    #   LINKS: M-ENGINE-MODEL-CACHE
    # END_CONTRACT: clear
    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    # START_CONTRACT: invalidate
    #   PURPOSE: Remove one cached handle by key when present.
    #   INPUTS: { key: ModelCacheKey - Cache key to invalidate }
    #   OUTPUTS: { None }
    #   SIDE_EFFECTS: Mutates the cache by removing one entry when it exists.
    #   LINKS: M-ENGINE-MODEL-CACHE
    # END_CONTRACT: invalidate
    def invalidate(self, key: ModelCacheKey) -> None:
        with self._lock:
            self._items.pop(key, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


__all__ = ["ModelCache", "ModelCacheKey"]
