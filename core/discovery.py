# FILE: core/discovery.py
# VERSION: 1.2.1
# START_MODULE_CONTRACT
#   PURPOSE: Provide auto-discovery helpers that enumerate concrete TTSBackend, ModelFamilyAdapter, and ModelFamilyPlugin classes from in-process __subclasses__() recursion plus optional importlib.metadata entry points.
#   SCOPE: discover_backend_classes, discover_family_adapter_classes, discover_family_plugin_classes, built-in adapter seeding for subclass discovery, test-local subclass filtering for normal family-adapter discovery, BACKEND_ENTRY_POINT_GROUP, FAMILY_PLUGIN_ENTRY_POINT_GROUP, FAMILY_ADAPTER_ENTRY_POINT_GROUP
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
#   import_builtin_backend_modules - Seed built-in backend subclasses before backend discovery runs.
#   discover_family_adapter_classes - Enumerate concrete ModelFamilyAdapter subclasses plus entry-point-declared classes after seeding built-in adapter modules, excluding test-local subclass leakage by default.
#   _is_test_local_module_name - Identify test-local module names for family-adapter leakage filtering.
#   discover_family_plugin_classes - Enumerate concrete ModelFamilyPlugin subclasses plus entry-point-declared classes.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.2.1 - Task 4/5 verification cleanup: kept built-in backend import seeding in core.discovery as the single canonical seam and removed the redundant core.backends.builtins helper surface]
# END_CHANGE_SUMMARY

from __future__ import annotations

import inspect
import logging
from importlib import import_module
from importlib.metadata import EntryPoint, entry_points
from typing import Callable, Iterable, TypeVar

from core.backends.base import TTSBackend
from core.model_families.base import ModelFamilyAdapter
from core.model_families.plugin import ModelFamilyPlugin

logger = logging.getLogger(__name__)

BACKEND_ENTRY_POINT_GROUP = "tts_server.backends"
FAMILY_PLUGIN_ENTRY_POINT_GROUP = "tts_server.model_families"
FAMILY_ADAPTER_ENTRY_POINT_GROUP = "tts_server.family_adapters"
BUILTIN_BACKEND_MODULES: tuple[str, ...] = (
    "core.backends.mlx_backend",
    "core.backends.onnx_backend",
    "core.backends.qwen_fast_backend",
    "core.backends.torch_backend",
)

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
    entry_points_loader: Callable[[], Iterable[EntryPoint]] | None = None,
) -> tuple[type[TTSBackend], ...]:
    import_builtin_backend_modules()
    discovered: list[type[TTSBackend]] = list(_iter_concrete_subclasses(TTSBackend))
    if include_entry_points:
        for cls in _load_entry_point_classes(
            BACKEND_ENTRY_POINT_GROUP,
            base_class=TTSBackend,
            loader=entry_points_loader,
        ):
            discovered.append(cls)
    return _dedupe_and_sort(discovered)


# START_CONTRACT: import_builtin_backend_modules
#   PURPOSE: Import the built-in backend modules with per-module failure isolation so any importable backend still registers even if another optional backend fails to import.
#   INPUTS: {}
#   OUTPUTS: { None - Modules are imported for their TTSBackend subclass side effects }
#   SIDE_EFFECTS: Imports built-in backend modules and logs a warning for each failed optional import
#   LINKS: M-BACKENDS, M-DISCOVERY
# END_CONTRACT: import_builtin_backend_modules
def import_builtin_backend_modules() -> None:
    # START_BLOCK_IMPORT_BUILTIN_BACKEND_MODULES
    for module_name in BUILTIN_BACKEND_MODULES:
        try:
            import_module(module_name)
        except Exception as exc:  # pragma: no cover - defensive import seeding path
            logger.warning("discovery: failed to import built-in backend module %r: %s", module_name, exc)
    # END_BLOCK_IMPORT_BUILTIN_BACKEND_MODULES


# START_CONTRACT: discover_family_adapter_classes
#   PURPOSE: Enumerate concrete ModelFamilyAdapter implementations (planner-side preparation contract). Same discovery surface as backends.
#   INPUTS: { include_entry_points: bool - When True, also load entry-point-declared adapter classes (default True), entry_points_loader: callable | None - Optional override for tests, include_test_classes: bool - When True, retain subclasses declared under tests.* modules for explicit test-only assertions (default False) }
#   OUTPUTS: { tuple[type[ModelFamilyAdapter], ...] - Discovered adapter classes deduplicated and sorted }
#   SIDE_EFFECTS: May import the modules referenced by entry points; logs a warning when an entry point fails to load or resolves to a filtered tests.* module
#   LINKS: M-DISCOVERY
# END_CONTRACT: discover_family_adapter_classes
def discover_family_adapter_classes(
    *,
    include_entry_points: bool = True,
    entry_points_loader: Callable[[], Iterable[EntryPoint]] | None = None,
    include_test_classes: bool = False,
) -> tuple[type[ModelFamilyAdapter], ...]:
    _import_builtin_family_adapter_modules()
    discovered: list[type[ModelFamilyAdapter]] = list(
        _iter_concrete_subclasses(
            ModelFamilyAdapter,
            include_test_classes=include_test_classes,
        )
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
    entry_points_loader: Callable[[], Iterable[EntryPoint]] | None = None,
) -> tuple[type[ModelFamilyPlugin], ...]:
    discovered: list[type[ModelFamilyPlugin]] = list(_iter_concrete_subclasses(ModelFamilyPlugin))
    if include_entry_points:
        for cls in _load_entry_point_classes(
            FAMILY_PLUGIN_ENTRY_POINT_GROUP,
            base_class=ModelFamilyPlugin,
            loader=entry_points_loader,
        ):
            discovered.append(cls)
    return _dedupe_and_sort(discovered)


# START_BLOCK_DISCOVERY_HELPERS
def _import_builtin_family_adapter_modules() -> None:
    """Compatibility import seeding for built-in family adapters.

    Discovery relies on ``ModelFamilyAdapter.__subclasses__()``. The built-in
    family adapter modules are imported here in one controlled place so callers
    such as ``TTSService`` can discover them deterministically without keeping
    hardcoded constructor wiring in runtime services.
    """

    for module_name in (
        "core.model_families.qwen3",
        "core.model_families.omnivoice",
        "core.model_families.piper",
    ):
        import_module(module_name)


def _is_test_local_module_name(module_name: str) -> bool:
    return (
        module_name == "tests"
        or module_name.startswith("tests.")
        or module_name == "test"
        or module_name.startswith("test_")
        or ".test_" in module_name
    )


def _iter_concrete_subclasses(
    base_class: type[T],
    *,
    include_test_classes: bool = True,
) -> list[type[T]]:
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
        if not include_test_classes and _is_test_local_module_name(getattr(cls, "__module__", "")):
            continue
        out.append(cls)
    return out


def _load_entry_point_classes(
    group: str,
    *,
    base_class: type[T],
    loader: Callable[[], Iterable[EntryPoint]] | None = None,
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
        if _is_test_local_module_name(getattr(obj, "__module__", "")):
            logger.warning(
                "discovery: entry point %r in group %r resolves to test-local module %r; skipping",
                getattr(entry, "name", "<unknown>"),
                group,
                getattr(obj, "__module__", "<unknown>"),
            )
            continue
        out.append(obj)
    return out


def _resolve_entry_points(
    group: str,
    *,
    loader: Callable[[], Iterable[EntryPoint]] | None,
) -> tuple[EntryPoint, ...]:
    if loader is not None:
        if not callable(loader):
            raise TypeError("entry_points_loader must be callable when provided")
        try:
            iterable = loader()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("discovery: entry_points_loader raised for group %r: %s", group, exc)
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
    return tuple(getattr(all_entries, "get", lambda _key, default=(): default)(group, ()))


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
    "BUILTIN_BACKEND_MODULES",
    "FAMILY_ADAPTER_ENTRY_POINT_GROUP",
    "FAMILY_PLUGIN_ENTRY_POINT_GROUP",
    "discover_backend_classes",
    "discover_family_adapter_classes",
    "discover_family_plugin_classes",
    "import_builtin_backend_modules",
]
