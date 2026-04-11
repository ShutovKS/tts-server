# FILE: tests/unit/core/test_host_probe.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for host probing and capability resolution surfaces.
#   SCOPE: Host snapshot shape and backend candidate ranking
#   DEPENDS: M-CORE
#   LINKS: V-M-HOST-PROBE, V-M-CAPABILITY-RESOLVER
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_host_probe_returns_runtime_snapshot - Verifies host probe returns a typed runtime snapshot
#   test_capability_resolver_ranks_compatible_backend_above_missing_runtime - Verifies explainable backend ranking
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added unit coverage for host probing and capability resolution]
# END_CHANGE_SUMMARY

from __future__ import annotations

import pytest

from core.backends.base import TTSBackend
from core.backends.capabilities import BackendCapabilitySet, BackendDiagnostics
from core.planning.capability_resolver import CapabilityResolver
from core.planning.host_probe import HostProbe, HostSnapshot


pytestmark = pytest.mark.unit


class ProbeBackendStub(TTSBackend):
    def __init__(self, *, key: str, available: bool, platform_supported: bool):
        self.key = key
        self.label = key.upper()
        self._available = available
        self._platform_supported = platform_supported

    def capabilities(self) -> BackendCapabilitySet:
        return BackendCapabilitySet(
            supports_custom=True,
            supports_design=True,
            supports_clone=True,
            platforms=("darwin", "linux", "windows"),
        )

    def is_available(self) -> bool:
        return self._available

    def supports_platform(self) -> bool:
        return self._platform_supported

    def resolve_model_path(self, folder_name: str):
        return None

    def load_model(self, spec):  # pragma: no cover
        raise NotImplementedError

    def inspect_model(self, spec):  # pragma: no cover
        raise NotImplementedError

    def readiness_diagnostics(self) -> BackendDiagnostics:
        return BackendDiagnostics(
            backend_key=self.key,
            backend_label=self.label,
            available=self._available,
            ready=self._available and self._platform_supported,
            reason=None,
            details={},
        )

    def synthesize_custom(self, handle, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def synthesize_design(self, handle, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def synthesize_clone(self, handle, **kwargs):  # pragma: no cover
        raise NotImplementedError


def test_host_probe_returns_runtime_snapshot():
    snapshot = HostProbe().probe()

    assert snapshot.platform_system in {"darwin", "linux", "windows"}
    assert isinstance(snapshot.architecture, str)
    assert isinstance(snapshot.ffmpeg_available, bool)
    assert isinstance(snapshot.to_dict()["python_version"], str)


def test_capability_resolver_ranks_compatible_backend_above_missing_runtime():
    resolver = CapabilityResolver()
    host = HostSnapshot(
        platform_system="darwin",
        platform_release="test",
        architecture="arm64",
        python_version="3.11.0",
        ffmpeg_available=True,
        mlx_runtime_available=True,
        torch_runtime_available=True,
        cuda_available=False,
    )

    candidates = resolver.rank_backends(
        backends=(
            ProbeBackendStub(key="mlx", available=True, platform_supported=True),
            ProbeBackendStub(key="torch", available=False, platform_supported=True),
        ),
        host=host,
    )

    assert candidates[0].backend_key == "mlx"
    assert candidates[0].accepted is True
    assert candidates[0].reason == "host_and_runtime_compatible"
    assert candidates[1].reason == "runtime_dependency_missing"
