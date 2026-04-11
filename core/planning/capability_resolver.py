# FILE: core/planning/capability_resolver.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Rank backend candidates using host features and backend diagnostics while preserving compatibility with current runtime behavior.
#   SCOPE: CapabilityCandidate and CapabilityResolver classes
#   DEPENDS: M-HOST-PROBE, M-BACKENDS
#   LINKS: M-CAPABILITY-RESOLVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   CapabilityCandidate - Ranked backend candidate with explicit accept/reject reasons
#   CapabilityResolver - Resolver that ranks backend candidates against the active host snapshot
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added explainable capability resolver for backend selection]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from core.backends.base import TTSBackend
from core.planning.host_probe import HostSnapshot


@dataclass(frozen=True)
class CapabilityCandidate:
    backend_key: str
    score: int
    accepted: bool
    reason: str
    diagnostics: dict[str, object]


class CapabilityResolver:
    def rank_backends(
        self,
        *,
        backends: Sequence[TTSBackend],
        host: HostSnapshot,
    ) -> tuple[CapabilityCandidate, ...]:
        candidates: list[CapabilityCandidate] = []
        for backend in backends:
            platform_supported = backend.supports_platform()
            available = backend.is_available()
            diagnostics = backend.readiness_diagnostics().to_dict()
            ready = bool(diagnostics.get("ready", platform_supported and available))
            score = 0
            accepted = ready
            if platform_supported:
                score += 100
            if available:
                score += 80
            if ready:
                score += 20
            if backend.key == "mlx" and host.platform_system == "darwin":
                score += 25
            if backend.key == "qwen_fast" and host.cuda_available:
                score += 40
            if backend.key == "qwen_fast" and host.platform_system in {
                "linux",
                "windows",
            }:
                score += 20
            if backend.key == "torch" and host.cuda_available:
                score += 25
            if backend.key == "torch" and host.platform_system in {"linux", "windows"}:
                score += 15

            if accepted:
                reason = "host_and_runtime_compatible"
            elif isinstance(diagnostics.get("reason"), str) and diagnostics.get(
                "reason"
            ):
                reason = str(diagnostics["reason"])
            elif platform_supported:
                reason = "runtime_dependency_missing"
            else:
                reason = "platform_unsupported"

            candidates.append(
                CapabilityCandidate(
                    backend_key=backend.key,
                    score=score,
                    accepted=accepted,
                    reason=reason,
                    diagnostics=diagnostics,
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return tuple(candidates)


__all__ = ["CapabilityCandidate", "CapabilityResolver"]
