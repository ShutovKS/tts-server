# FILE: profiles/__init__.py
# VERSION: 1.0.1
# START_MODULE_CONTRACT
#   PURPOSE: Re-export profile schema types and resolver entrypoints for runtime profile composition.
#   SCOPE: barrel exports for profile schema types and the profile resolver
#   DEPENDS: M-PROFILE-SCHEMA, M-PROFILE-RESOLVER
#   LINKS: M-PROFILES
#   ROLE: BARREL
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   Re-export profile schema dataclasses and `ProfileResolver` for package-level imports.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.1 - Aligned the profiles package barrel with GRACE barrel-role semantics and summary mapping]
# END_CHANGE_SUMMARY

from profiles.resolver import ProfileResolver
from profiles.schema import FamilyProfile, HostProfile, ModuleProfile, ResolvedLaunchProfile

__all__ = [
    "FamilyProfile",
    "HostProfile",
    "ModuleProfile",
    "ProfileResolver",
    "ResolvedLaunchProfile",
]
