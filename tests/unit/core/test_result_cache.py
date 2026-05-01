# FILE: tests/unit/core/test_result_cache.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Verify the build_cache_key canonicalisation and FileSystemResultCache get/put/clear/eviction semantics.
#   SCOPE: build_cache_key determinism, NullResultCache no-op behaviour, FileSystemResultCache round-trip and eviction, corrupt-entry recovery.
#   DEPENDS: M-RESULT-CACHE
#   LINKS: V-M-RESULT-CACHE
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _make_entry - Build a CachedResult fixture with deterministic timestamps.
#   test_build_cache_key_is_deterministic_regardless_of_dict_order - Verifies the canonicaliser collapses dict-order variants.
#   test_build_cache_key_diverges_on_value_change - Verifies semantic differences yield distinct keys.
#   test_null_cache_is_always_miss - Verifies NullResultCache.get/put/clear behave as no-ops.
#   test_filesystem_cache_round_trip - Verifies put then get returns the original payload and metadata.
#   test_filesystem_cache_clear_removes_all_entries - Verifies clear() empties the directory.
#   test_filesystem_cache_evicts_oldest_when_overflow - Verifies oldest entries are evicted when max_entries is exceeded.
#   test_filesystem_cache_drops_corrupt_meta - Verifies corrupt JSON metadata triggers cache miss with cleanup.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Phase 4.14: introduced unit coverage for result_cache module covering build_cache_key, NullResultCache, FileSystemResultCache round-trip, eviction, and corruption recovery]
# END_CHANGE_SUMMARY

from __future__ import annotations

import time
from pathlib import Path

import pytest

from core.services.result_cache import (
    CachedResult,
    FileSystemResultCache,
    NullResultCache,
    build_cache_key,
)

pytestmark = pytest.mark.unit


def _make_entry(payload: bytes = b"data", *, model: str = "model-a") -> CachedResult:
    return CachedResult(
        payload=payload,
        media_type="audio/wav",
        model=model,
        mode="custom",
        backend="torch",
        saved_path=None,
        created_at=time.time(),
    )


def test_build_cache_key_is_deterministic_regardless_of_dict_order() -> None:
    key_a = build_cache_key("custom", {"text": "hi", "model": "m", "speaker": "Vivian"})
    key_b = build_cache_key("custom", {"speaker": "Vivian", "model": "m", "text": "hi"})
    assert key_a == key_b
    assert len(key_a) == 64


def test_build_cache_key_diverges_on_value_change() -> None:
    key_a = build_cache_key("custom", {"text": "hi", "speed": 1.0})
    key_b = build_cache_key("custom", {"text": "hi", "speed": 1.1})
    assert key_a != key_b


def test_null_cache_is_always_miss() -> None:
    cache = NullResultCache()
    cache.put("key", _make_entry())
    cache.clear()
    assert cache.get("key") is None


def test_filesystem_cache_round_trip(tmp_path: Path) -> None:
    cache = FileSystemResultCache(tmp_path)
    entry = _make_entry(b"abcdef", model="m1")
    cache.put("k1", entry)

    fetched = cache.get("k1")
    assert fetched is not None
    assert fetched.payload == b"abcdef"
    assert fetched.model == "m1"
    assert fetched.mode == "custom"
    assert fetched.backend == "torch"


def test_filesystem_cache_clear_removes_all_entries(tmp_path: Path) -> None:
    cache = FileSystemResultCache(tmp_path)
    cache.put("k1", _make_entry(b"a"))
    cache.put("k2", _make_entry(b"b"))

    cache.clear()

    assert cache.get("k1") is None
    assert cache.get("k2") is None
    assert list(tmp_path.iterdir()) == []


def test_filesystem_cache_evicts_oldest_when_overflow(tmp_path: Path) -> None:
    cache = FileSystemResultCache(tmp_path, max_entries=2)
    cache.put("k1", _make_entry(b"a"))
    time.sleep(0.01)
    cache.put("k2", _make_entry(b"b"))
    time.sleep(0.01)
    cache.put("k3", _make_entry(b"c"))

    survivors = {name for name in (k for k in ("k1", "k2", "k3") if cache.get(k) is not None)}
    assert len(survivors) == 2
    assert "k3" in survivors


def test_filesystem_cache_drops_corrupt_meta(tmp_path: Path) -> None:
    cache = FileSystemResultCache(tmp_path)
    cache.put("k1", _make_entry(b"a"))

    meta_path = next(tmp_path.glob("*.meta.json"))
    meta_path.write_text("not json")

    assert cache.get("k1") is None
    assert not meta_path.exists()
