# FILE: launcher/__init__.py
# VERSION: 1.0.1
# START_MODULE_CONTRACT
#   PURPOSE: Re-export the launcher CLI entrypoint for profile-aware runtime commands.
#   SCOPE: barrel exports for launcher consumers
#   DEPENDS: M-LAUNCHER
#   LINKS: M-LAUNCHER
#   ROLE: BARREL
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   Re-export `main` from launcher.main as the package-level launcher entrypoint.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.1 - Aligned the launcher package barrel with GRACE barrel-role semantics and summary mapping]
# END_CHANGE_SUMMARY

from launcher.main import main

__all__ = ["main"]
