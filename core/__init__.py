# FILE: core/__init__.py
# VERSION: 1.0.1
# START_MODULE_CONTRACT
#   PURPOSE: Re-export public API of the core runtime package.
#   SCOPE: barrel re-exports for CoreRuntime, CoreSettings, build_runtime
#   DEPENDS: M-BOOTSTRAP, M-CONFIG
#   LINKS: M-BOOTSTRAP
#   ROLE: BARREL
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   __getattr__ - Lazily resolve public core exports without importing runtime bootstrap on every package import.
#   CoreRuntime - Frozen dataclass holding all wired runtime components
#   CoreSettings - Shared runtime configuration
#   build_runtime - Factory function to assemble CoreRuntime
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.1 - Switched the core package barrel to lazy export resolution so lightweight importers do not pull runtime bootstrap dependencies eagerly]
# END_CHANGE_SUMMARY

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.bootstrap import CoreRuntime, build_runtime
    from core.config import CoreSettings


def __getattr__(name: str) -> Any:
    if name == "CoreSettings":
        from core.config import CoreSettings

        return CoreSettings
    if name in {"CoreRuntime", "build_runtime"}:
        from core.bootstrap import CoreRuntime, build_runtime

        return {
            "CoreRuntime": CoreRuntime,
            "build_runtime": build_runtime,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["CoreRuntime", "CoreSettings", "build_runtime"]
