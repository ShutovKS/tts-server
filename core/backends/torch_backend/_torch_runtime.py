# FILE: core/backends/torch_backend/_torch_runtime.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide a single import surface for the optional `torch` runtime dependency used by the dispatcher and family strategies.
#   SCOPE: lazy `torch` import, device/dtype resolution helpers, shared TORCH_IMPORT_ERROR sentinel
#   DEPENDS: none
#   LINKS: M-BACKENDS
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   torch - Optional torch module reference (None when torch is not installed).
#   TORCH_IMPORT_ERROR - Optional ImportError describing why torch could not be loaded.
#   resolve_device_map - Compute the effective device_map name for Torch model loading.
#   resolve_device_map_name - Public alias used by diagnostics.
#   resolve_dtype - Return the preferred torch dtype for the available device.
#   resolve_dtype_name - Stringify the resolved dtype for diagnostics.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Extracted from monolithic torch_backend.py during Phase 1.4 strategy split]
# END_CHANGE_SUMMARY

from __future__ import annotations

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR: Exception | None = exc
else:
    TORCH_IMPORT_ERROR = None


# START_CONTRACT: resolve_device_map
#   PURPOSE: Resolve the preferred device_map name for the active Torch runtime.
#   INPUTS: {}
#   OUTPUTS: { str - "cuda:0" when CUDA is available, otherwise "cpu" }
#   SIDE_EFFECTS: none
#   LINKS: M-BACKENDS
# END_CONTRACT: resolve_device_map
def resolve_device_map() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


# START_CONTRACT: resolve_device_map_name
#   PURPOSE: Public alias for resolve_device_map used by readiness diagnostics.
#   INPUTS: {}
#   OUTPUTS: { str - Resolved device map name }
#   SIDE_EFFECTS: none
#   LINKS: M-BACKENDS
# END_CONTRACT: resolve_device_map_name
def resolve_device_map_name() -> str:
    return resolve_device_map()


# START_CONTRACT: resolve_dtype
#   PURPOSE: Pick the preferred torch dtype for the active runtime (bfloat16 on CUDA, float32 otherwise).
#   INPUTS: {}
#   OUTPUTS: { Any | None - torch.dtype handle, or None when torch is unavailable }
#   SIDE_EFFECTS: none
#   LINKS: M-BACKENDS
# END_CONTRACT: resolve_dtype
def resolve_dtype():
    if torch is None:
        return None
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


# START_CONTRACT: resolve_dtype_name
#   PURPOSE: Render the resolved dtype as a stable string for diagnostics output.
#   INPUTS: {}
#   OUTPUTS: { str | None - Stringified dtype without the "torch." prefix, or None when torch is missing }
#   SIDE_EFFECTS: none
#   LINKS: M-BACKENDS
# END_CONTRACT: resolve_dtype_name
def resolve_dtype_name() -> str | None:
    dtype = resolve_dtype()
    return None if dtype is None else str(dtype).replace("torch.", "")


__all__ = [
    "TORCH_IMPORT_ERROR",
    "resolve_device_map",
    "resolve_device_map_name",
    "resolve_dtype",
    "resolve_dtype_name",
    "torch",
]
