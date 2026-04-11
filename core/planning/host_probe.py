# FILE: core/planning/host_probe.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Detect host platform, architecture, and runtime dependency availability used for explainable backend selection.
#   SCOPE: HostSnapshot dataclass and HostProbe runtime inspector
#   DEPENDS: M-CONFIG
#   LINKS: M-HOST-PROBE
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   HostSnapshot - Immutable host feature snapshot for backend planning
#   HostProbe - Host inspector that detects platform, architecture, and runtime dependency presence
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added host probing for explainable backend selection]
# END_CHANGE_SUMMARY

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class HostSnapshot:
    platform_system: str
    platform_release: str
    architecture: str
    python_version: str
    ffmpeg_available: bool
    mlx_runtime_available: bool
    torch_runtime_available: bool
    cuda_available: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "platform_system": self.platform_system,
            "platform_release": self.platform_release,
            "architecture": self.architecture,
            "python_version": self.python_version,
            "ffmpeg_available": self.ffmpeg_available,
            "mlx_runtime_available": self.mlx_runtime_available,
            "torch_runtime_available": self.torch_runtime_available,
            "cuda_available": self.cuda_available,
        }


class HostProbe:
    def probe(self) -> HostSnapshot:
        return HostSnapshot(
            platform_system=platform.system().lower(),
            platform_release=platform.release(),
            architecture=platform.machine().lower(),
            python_version=platform.python_version(),
            ffmpeg_available=shutil.which("ffmpeg") is not None,
            mlx_runtime_available=find_spec("mlx") is not None,
            torch_runtime_available=find_spec("torch") is not None,
            cuda_available=self._cuda_available(),
        )

    @staticmethod
    def _cuda_available() -> bool:
        spec = find_spec("torch")
        if spec is None:
            return False
        try:
            import torch
        except Exception:  # pragma: no cover
            return False
        return bool(torch.cuda.is_available())


__all__ = ["HostProbe", "HostSnapshot"]
