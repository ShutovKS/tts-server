# FILE: core/planning/__init__.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Re-export public planning contracts for synthesis planning, host probing, and capability resolution.
#   SCOPE: barrel re-exports for planner surface
#   DEPENDS: M-SYNTHESIS-PLANNER, M-HOST-PROBE, M-CAPABILITY-RESOLVER
#   LINKS: M-SYNTHESIS-PLANNER, M-HOST-PROBE, M-CAPABILITY-RESOLVER
#   ROLE: BARREL
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   Planning surface - Re-export the synthesis planner compatibility bridge plus host and capability helpers
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added planning barrel exports for synthesis planner migration seams]
# END_CHANGE_SUMMARY

__all__ = [
    "SynthesisPlanner",
    "HostProbe",
    "HostSnapshot",
    "CapabilityCandidate",
    "CapabilityResolver",
]


def __getattr__(name: str):
    if name == "SynthesisPlanner":
        from core.planning.planner import SynthesisPlanner

        return SynthesisPlanner
    if name in {"HostProbe", "HostSnapshot"}:
        from core.planning.host_probe import HostProbe, HostSnapshot

        return {"HostProbe": HostProbe, "HostSnapshot": HostSnapshot}[name]
    if name in {"CapabilityCandidate", "CapabilityResolver"}:
        from core.planning.capability_resolver import (
            CapabilityCandidate,
            CapabilityResolver,
        )

        return {
            "CapabilityCandidate": CapabilityCandidate,
            "CapabilityResolver": CapabilityResolver,
        }[name]
    raise AttributeError(name)
