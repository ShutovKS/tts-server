# FILE: core/discovery.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide auto-discovery helpers that enumerate concrete TTSBackend, ModelFamilyAdapter, and ModelFamilyPlugin classes from in-process __subclasses__() recursion plus optional importlib.metadata entry points.
#   SCOPE: discover_backend_classes, discover_family_adapter_classes, discover_family_plugin_classes, BACKEND_ENTRY_POINT_GROUP, FAMILY_PLUGIN_ENTRY_POINT_GROUP, FAMILY_ADAPTER_ENTRY_POINT_GROUP
#   DEPENDS: M-BACKENDS, M-MODEL-FAMILY
#   LINKS: M-DISCOVERY
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   BACKEND_ENTRY_POINT_GROUP - Importlib entry-point group for external backend classes.
#   FAMILY_PLUGIN_ENTRY_POINT_GROUP - Importlib entry-point group for external ModelFamilyPlugin classes.
#   FAMILY_ADAPTER_ENTRY_POINT_GROUP - Importlib entry-point group for external ModelFamilyAdapter classes.
#   discover_backend_classes - Enumerate concrete TTSBackend subclasses (in-process) plus entry-point-declared classes.
#   discover_family_adapter_classes - Enumerate concrete ModelFamilyAdapter subclasses plus entry-point-declared classes.
#   discover_family_plugin_classes - Enumerate concrete ModelFamilyPlugin subclasses plus entry-point-declared classes.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Phase 2.6: introduced auto-discovery helpers (subclass scan + entry-points) for backends and family plugins/adapters.]
# END_CHANGE_SUMMARY

from __future__ import annotations

import inspect
import logging
from importlib.metadata import EntryPoint, entry_points
from typing import TypeVar

from core.backends.base import TTSBackend
from core.model_families.base import ModelFamilyAdapter
from core.model_families.plugin import ModelFamilyPlugin

logger = logging.getLogger(__name__)

BACKEND_ENTRY_POINT_GROUP = "tts_server.backends"
FAMILY_PLUGIN_ENTRY_POINT_GROUP = "tts_server.model_families"
FAMILY_ADAPTER_ENTRY_POINT_GROUP = "tts_server.family_adapters"

T = TypeVar("T")


# START_CONTRACT: discover_backend_classes
#   PURPOSE: Enumerate concrete TTSBackend implementations available in the current process. Combines recursive __subclasses__() discovery (works whether or not the project is pip-installed) with importlib.metadata entry points (works for external pip packages once they declare a `tts_server.backends` entry).
#   INPUTS: { include_entry_points: bool - When True, also load entry-point-declared backend classes (default True), entry_points_loader: callable | None - Optional override for tests; called with no arguments and expected to return an iterable of EntryPoint instances }
#   OUTPUTS: { tuple[type[TTSBackend], ...] - Discovered concrete backend classes deduplicated by fully qualified name and sorted by declared `key` then qualified name }
#   SIDE_EFFECTS: May import the modules referenced by entry points; logs a warning when an entry point fails to load
#   LINKS: M-DISCOVERY
# END_CONTRACT: discover_backend_classes
def discover_backend_classes(
    *,
    include_entry_points: bool = True,
    entry_points_loader: object | None = None,
) -> tuple[type[TTSBackend], ...]:
    discovered: list[type[TTSBackend]] = list(_iter_concrete_subclasses(TTSBackend))
    if include_entry_points:
        for cls in _load_entry_point_classes(
            BACKEND_ENTRY_POINT_GROUP,
            base_class=TTSBackend,
            loader=entry_points_loader,
        ):
            discovered.append(cls)
    return _dedupe_and_sort(discovered)


# START_CONTRACT: discover_family_adapter_classes
#   PURPOSE: Enumerate concrete ModelFamilyAdapter implementations (planner-side preparation contract). Same discovery surface as backends.
#   INPUTS: { include_entry_points: bool - When True, also load entry-point-declared adapter classes (default True), entry_points_loader: callable | None - Optional override for tests }
#   OUTPUTS: { tuple[type[ModelFamilyAdapter], ...] - Discovered adapter classes deduplicated and sorted }
#   SIDE_EFFECTS: May import the modules referenced by entry points; logs a warning when an entry point fails to load
#   LINKS: M-DISCOVERY
# END_CONTRACT: discover_family_adapter_classes
def discover_family_adapter_classes(
    *,
    include_entry_points: bool = True,
    entry_points_loader: object | None = None,
) -> tuple[type[ModelFamilyAdapter], ...]:
    discovered: list[type[ModelFamilyAdapter]] = list(
        _iter_concrete_subclasses(ModelFamilyAdapter)
    )
    if include_entry_points:
        for cls in _load_entry_point_classes(
            FAMILY_ADAPTER_ENTRY_POINT_GROUP,
            base_class=ModelFamilyAdapter,
            loader=entry_points_loader,
        ):
            discovered.append(cls)
    return _dedupe_and_sort(discovered)


# START_CONTRACT: discover_family_plugin_classes
#   PURPOSE: Enumerate concrete ModelFamilyPlugin implementations (unified extension contract introduced in Phase 2.5).
#   INPUTS: { include_entry_points: bool - When True, also load entry-point-declared plugin classes (default True), entry_points_loader: callable | None - Optional override for tests }
#   OUTPUTS: { tuple[type[ModelFamilyPlugin], ...] - Discovered plugin classes deduplicated and sorted }
#   SIDE_EFFECTS: May import the modules referenced by entry points; logs a warning when an entry point fails to load
#   LINKS: M-DISCOVERY
# END_CONTRACT: discover_family_plugin_classes
def discover_family_plugin_classes(
    *,
    include_entry_points: bool = True,
    entry_points_loader: object | None = None,
) -> tuple[type[ModelFamilyPlugin], ...]:
    discovered: list[type[ModelFamilyPlugin]] = list(
        _iter_concrete_subclasses(ModelFamilyPlugin)
    )
    if include_entry_points:
        for cls in _load_entry_point_classes(
            FAMILY_PLUGIN_ENTRY_POINT_GROUP,
            base_class=ModelFamilyPlugin,
            loader=entry_points_loader,
        ):
            discovered.append(cls)
    return _dedupe_and_sort(discovered)


# START_BLOCK_DISCOVERY_HELPERS
def _iter_concrete_subclasses(base_class: type[T]) -> list[type[T]]:
    seen: set[type[T]] = set()
    out: list[type[T]] = []
    stack: list[type[T]] = list(base_class.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        if inspect.isabstract(cls):
            continue
        out.append(cls)
    return out


def _load_entry_point_classes(
    group: str,
    *,
    base_class: type[T],
    loader: object | None = None,
) -> list[type[T]]:
    out: list[type[T]] = []
    for entry in _resolve_entry_points(group, loader=loader):
        try:
            obj = entry.load()
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.warning(
                "discovery: failed to load entry point %r in group %r: %s",
                getattr(entry, "name", "<unknown>"),
                group,
                exc,
            )
            continue
        if not isinstance(obj, type):
            logger.warning(
                "discovery: entry point %r in group %r is not a class (got %r); skipping",
                getattr(entry, "name", "<unknown>"),
                group,
                type(obj).__name__,
            )
            continue
        if not issubclass(obj, base_class):
            logger.warning(
                "discovery: entry point %r in group %r is not a subclass of %s; skipping",
                getattr(entry, "name", "<unknown>"),
                group,
                base_class.__name__,
            )
            continue
        if inspect.isabstract(obj):
            logger.warning(
                "discovery: entry point %r in group %r resolves to an abstract class; skipping",
                getattr(entry, "name", "<unknown>"),
                group,
            )
            continue
        out.append(obj)
    return out


def _resolve_entry_points(
    group: str,
    *,
    loader: object | None,
) -> tuple[EntryPoint, ...]:
    if loader is not None:
        if not callable(loader):
            raise TypeError("entry_points_loader must be callable when provided")
        try:
            iterable = loader()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "discovery: entry_points_loader raised for group %r: %s", group, exc
            )
            return ()
        return tuple(iterable)
    try:
        all_entries = entry_points()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("discovery: importlib.metadata.entry_points() failed: %s", exc)
        return ()
    select = getattr(all_entries, "select", None)
    if select is not None:
        return tuple(select(group=group))
    # Fallback for older Python releases (pre-3.10 entry_points API).
    return tuple(
        getattr(all_entries, "get", lambda _key, default=(): default)(group, ())
    )


def _qualified_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _dedupe_and_sort(classes: list[type[T]]) -> tuple[type[T], ...]:
    by_name: dict[str, type[T]] = {}
    for cls in classes:
        by_name.setdefault(_qualified_name(cls), cls)

    def sort_key(cls: type[T]) -> tuple[str, str]:
        key_attr = getattr(cls, "key", "")
        key_text = key_attr if isinstance(key_attr, str) else ""
        return (key_text, _qualified_name(cls))

    return tuple(sorted(by_name.values(), key=sort_key))


# END_BLOCK_DISCOVERY_HELPERS


__all__ = [
    "BACKEND_ENTRY_POINT_GROUP",
    "FAMILY_ADAPTER_ENTRY_POINT_GROUP",
    "FAMILY_PLUGIN_ENTRY_POINT_GROUP",
    "discover_backend_classes",
    "discover_family_adapter_classes",
    "discover_family_plugin_classes",
]
