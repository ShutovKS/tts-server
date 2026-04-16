#!/usr/bin/env python3
# FILE: scripts/runtime_self_check.py
# VERSION: 1.2.2
# START_MODULE_CONTRACT
#   PURPOSE: Provide an operator-facing self-check utility for validating runtime dependencies, model assets, readiness metadata, and profile resolution snapshots.
#   SCOPE: CLI entry point for shared runtime self-check reporting, asset validation, representative optional-model gating, profile resolver snapshots, and support evidence capture
#   DEPENDS: M-BOOTSTRAP, M-CONFIG, M-MODEL-REGISTRY, M-PROFILES
#   LINKS: M-BOOTSTRAP
#   ROLE: SCRIPT
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   PROJECT_ROOT - Repository root inserted into sys.path for script-local imports
#   REPRESENTATIVE_MODEL_TARGETS - Representative model targets used for bounded optional real-model checks
#   _overlaid_environment - Temporarily overlay environment variables while building self-check payloads
#   parse_args - Parse CLI arguments for self-check execution
#   build_asset_report - Build filesystem and runtime validation results for declared model assets
#   _classify_representative_target - Classify a representative target into ready, skipped, or failed machine-readable status
#   build_representative_model_report - Build bounded representative-model readiness/gating details for optional real-model validation
#   build_self_check_payload_with_diagnostics - Build the self-check payload while capturing bootstrap/runtime noise as structured diagnostics
#   build_self_check_payload - Build the full runtime self-check payload for a given environment snapshot
#   main - Execute the runtime self-check and print JSON results
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.2.2 - Aligned runtime self-check dependency contract to the public profiles package boundary used by the script import surface]
# END_CHANGE_SUMMARY

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


# START_BLOCK_BOOTSTRAP_IMPORT_PATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# END_BLOCK_BOOTSTRAP_IMPORT_PATH

from core.bootstrap import build_runtime  # noqa: E402
from core.config import CoreSettings, parse_core_settings_from_env  # noqa: E402
from profiles import ProfileResolver  # noqa: E402


REPRESENTATIVE_MODEL_TARGETS: tuple[dict[str, str], ...] = (
    {
        "target": "qwen",
        "model_id": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "family_key": "qwen3_tts",
        "expected_backend": "mlx|torch|qwen_fast",
    },
    {
        "target": "omnivoice",
        "model_id": "OmniVoice-Custom",
        "family_key": "omnivoice",
        "expected_backend": "torch",
    },
    {
        "target": "piper",
        "model_id": "Piper-en_US-lessac-medium",
        "family_key": "piper",
        "expected_backend": "onnx",
    },
)

# START_CONTRACT: _overlaid_environment
#   PURPOSE: Temporarily overlay environment variables while preserving and restoring prior process state.
#   INPUTS: { environ: Mapping[str, str] | None - optional environment overrides to apply for the duration of the context }
#   OUTPUTS: { contextmanager - context that restores the original environment on exit }
#   SIDE_EFFECTS: Mutates os.environ for the lifetime of the context
#   LINKS: M-BOOTSTRAP, M-CONFIG
# END_CONTRACT: _overlaid_environment

@contextmanager
def _overlaid_environment(environ: Mapping[str, str] | None):
    if environ is None:
        yield
        return

    original: dict[str, str | None] = {key: os.environ.get(key) for key in environ}
    missing_keys = [key for key, value in original.items() if value is None]
    try:
        for key, value in environ.items():
            os.environ[key] = value
        yield
    finally:
        for key in missing_keys:
            os.environ.pop(key, None)
        for key, value in original.items():
            if value is not None:
                os.environ[key] = value


# START_CONTRACT: parse_args
#   PURPOSE: Parse CLI arguments controlling runtime self-check output and strictness.
#   INPUTS: { argv: list[str] | None - optional raw CLI arguments }
#   OUTPUTS: { argparse.Namespace - parsed argument payload }
#   SIDE_EFFECTS: none
#   LINKS: M-BOOTSTRAP
# END_CONTRACT: parse_args
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate local runtime dependencies, configured models, and readiness state.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when the shared runtime is not ready or when any configured model is missing required assets.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level for the output payload.",
    )
    return parser.parse_args(argv)


# START_CONTRACT: build_asset_report
#   PURPOSE: Build per-model asset validation details for operator-facing setup checks.
#   INPUTS: { readiness_report: dict[str, Any] - registry readiness payload }
#   OUTPUTS: { dict[str, Any] - summarized asset validation state for configured models }
#   SIDE_EFFECTS: none
#   LINKS: M-MODEL-REGISTRY
# END_CONTRACT: build_asset_report
def build_asset_report(readiness_report: dict[str, Any]) -> dict[str, Any]:
    items = readiness_report.get("items", [])
    configured = [
        {
            "id": item.get("id"),
            "family": item.get("family"),
            "backend_support": item.get("backend_support", []),
            "folder": item.get("folder"),
            "available": item.get("available"),
            "loadable": item.get("loadable"),
            "runtime_ready": item.get("runtime_ready"),
            "execution_backend": item.get("execution_backend") or item.get("backend"),
            "selected_backend": item.get("selected_backend"),
            "capabilities_supported": item.get("capabilities_supported", []),
            "route_candidates": item.get("route", {}).get("candidates", []),
            "missing_artifacts": item.get("missing_artifacts", []),
            "required_artifacts": item.get("required_artifacts", []),
            "route_reason": item.get("route", {}).get("route_reason"),
        }
        for item in items
    ]
    missing_assets = [
        item
        for item in configured
        if item["missing_artifacts"] or not item["available"]
    ]
    return {
        "configured_models": configured,
        "models_missing_assets": missing_assets,
    }


# START_CONTRACT: _classify_representative_target
#   PURPOSE: Classify a representative model target into machine-readable readiness or skip/failure reasons.
#   INPUTS: { item: dict[str, Any] | None - readiness item for the representative model target, if present }
#   OUTPUTS: { dict[str, Any] - machine-readable representative target classification }
#   SIDE_EFFECTS: none
#   LINKS: M-MODEL-REGISTRY
# END_CONTRACT: _classify_representative_target
def _classify_representative_target(item: dict[str, Any] | None) -> dict[str, Any]:
    if item is None:
        return {
            "status": "skipped",
            "reason": "representative_model_not_registered",
            "message": "Representative model is not registered in the active readiness payload.",
            "machine_readable": True,
        }

    route = item.get("route") or {}
    execution_backend = item.get("execution_backend") or item.get("backend")
    missing_artifacts = list(item.get("missing_artifacts") or [])
    required_artifacts = list(item.get("required_artifacts") or [])
    available = item.get("available") is True
    loadable = item.get("loadable") is True
    runtime_ready = item.get("runtime_ready") is True
    runtime_blockers = list(item.get("runtime_blockers") or [])

    if runtime_ready:
        return {
            "status": "ready",
            "reason": "representative_model_ready",
            "message": "Representative model is runtime-ready for opt-in validation.",
            "machine_readable": True,
        }

    if missing_artifacts or not available:
        return {
            "status": "skipped",
            "reason": "model_assets_missing",
            "message": "Representative model assets are missing.",
            "machine_readable": True,
        }

    selected_backend = route.get("selected_backend") or item.get("selected_backend")
    if route.get("selected_backend_compatible_with_model") is False:
        return {
            "status": "skipped",
            "reason": "selected_backend_incompatible_with_model",
            "message": "Representative model is installed, but the selected backend does not support it on this host.",
            "machine_readable": True,
        }

    candidate_diagnostics = []
    for candidate in route.get("candidates", []):
        diagnostics = candidate.get("diagnostics") or {}
        if diagnostics:
            candidate_diagnostics.append(diagnostics)
            reason = diagnostics.get("reason")
            if reason in {
                "runtime_dependency_missing",
                "dependency_missing",
                "python_package_missing",
                "optional_dependency_missing",
            }:
                return {
                    "status": "skipped",
                    "reason": "optional_dependency_pack_missing",
                    "message": "Representative model requires an optional runtime dependency pack that is not installed.",
                    "machine_readable": True,
                }
            if reason in {
                "platform_unsupported",
                "cuda_required",
                "backend_unavailable",
                "runtime_unavailable",
                "host_unsupported",
            }:
                return {
                    "status": "skipped",
                    "reason": "backend_or_runtime_unsupported",
                    "message": "Representative model cannot run on the current host/runtime backend combination.",
                    "machine_readable": True,
                }

    route_reason = route.get("route_reason")
    if route_reason in {
        "selected_backend_incompatible_with_model",
        "platform_unsupported",
        "cuda_required",
    }:
        return {
            "status": "skipped",
            "reason": "backend_or_runtime_unsupported",
            "message": "Representative model cannot run on the current host/runtime backend combination.",
            "machine_readable": True,
        }

    if available and not loadable:
        return {
            "status": "failed",
            "reason": "model_artifacts_corrupt_or_incomplete",
            "message": "Representative model assets exist but the runtime did not classify them as loadable.",
            "machine_readable": True,
        }

    return {
        "status": "skipped",
        "reason": "runtime_not_ready",
        "message": "Representative model is present but not runtime-ready for an unspecified readiness reason.",
        "machine_readable": True,
        "details": {
            "selected_backend": selected_backend,
            "execution_backend": execution_backend,
            "route_reason": route_reason,
            "candidate_diagnostics": candidate_diagnostics,
            "required_artifacts": required_artifacts,
        },
    }


# START_CONTRACT: build_representative_model_report
#   PURPOSE: Build bounded representative-model gating details for optional real-model validation lanes.
#   INPUTS: { readiness_report: dict[str, Any] - registry readiness payload }
#   OUTPUTS: { dict[str, Any] - representative target readiness report with explicit machine-readable reasons }
#   SIDE_EFFECTS: none
#   LINKS: M-MODEL-REGISTRY
# END_CONTRACT: build_representative_model_report
def build_representative_model_report(readiness_report: dict[str, Any]) -> dict[str, Any]:
    items = readiness_report.get("items", [])
    indexed_items = {str(item.get("id")): item for item in items}
    targets: list[dict[str, Any]] = []
    for spec in REPRESENTATIVE_MODEL_TARGETS:
        item = indexed_items.get(spec["model_id"])
        classification = _classify_representative_target(item)
        target_payload = {
            "target": spec["target"],
            "model_id": spec["model_id"],
            "family_key": spec["family_key"],
            "expected_backend": spec["expected_backend"],
            **classification,
        }
        if item is not None:
            target_payload.update(
                {
                    "selected_backend": item.get("selected_backend"),
                    "execution_backend": item.get("execution_backend") or item.get("backend"),
                    "available": item.get("available"),
                    "loadable": item.get("loadable"),
                    "runtime_ready": item.get("runtime_ready"),
                    "runtime_blockers": list(item.get("runtime_blockers") or []),
                    "missing_artifacts": list(item.get("missing_artifacts") or []),
                    "required_artifacts": list(item.get("required_artifacts") or []),
                    "route_reason": (item.get("route") or {}).get("route_reason"),
                }
            )
        targets.append(target_payload)
    return {
        "targets": targets,
        "ready_targets": [item for item in targets if item["status"] == "ready"],
        "skipped_targets": [item for item in targets if item["status"] == "skipped"],
        "failed_targets": [item for item in targets if item["status"] == "failed"],
    }


# START_CONTRACT: build_self_check_payload_with_diagnostics
#   PURPOSE: Build the self-check payload while capturing bootstrap/runtime noise into structured diagnostics.
#   INPUTS: { environ: Mapping[str, str] | None - optional environment override used to resolve runtime settings }
#   OUTPUTS: { tuple[dict[str, Any], list[dict[str, Any]]] - self-check payload plus captured runtime diagnostics }
#   SIDE_EFFECTS: Builds the shared runtime and intercepts direct stdout/stderr messages emitted by imported dependencies
#   LINKS: M-BOOTSTRAP, M-MODEL-REGISTRY
# END_CONTRACT: build_self_check_payload_with_diagnostics
def build_self_check_payload_with_diagnostics(
    environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    # START_BLOCK_CAPTURE_RUNTIME_DIAGNOSTICS
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        payload = build_self_check_payload(environ)

    diagnostics: list[dict[str, Any]] = []
    for source, raw_output in (
        ("stdout", stdout_buffer.getvalue()),
        ("stderr", stderr_buffer.getvalue()),
    ):
        for message in [line.strip() for line in raw_output.splitlines() if line.strip()]:
            diagnostics.append(
                {
                    "kind": "captured_runtime_message",
                    "source": source,
                    "message": message,
                }
            )
    # END_BLOCK_CAPTURE_RUNTIME_DIAGNOSTICS
    return payload, diagnostics


# START_CONTRACT: build_self_check_payload
#   PURPOSE: Build the full runtime self-check payload for the active or provided environment mapping.
#   INPUTS: { environ: Mapping[str, str] | None - optional environment override used to resolve runtime settings }
#   OUTPUTS: { dict[str, Any] - structured runtime self-check payload suitable for local operators, CI, and validation scripts }
#   SIDE_EFFECTS: Builds the shared runtime, inspects model readiness, and resolves configured filesystem paths
#   LINKS: M-BOOTSTRAP, M-MODEL-REGISTRY
# END_CONTRACT: build_self_check_payload
def build_self_check_payload(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    with _overlaid_environment(environ):
        # START_BLOCK_BUILD_RUNTIME_AND_PROFILE_SNAPSHOTS
        settings = CoreSettings(**parse_core_settings_from_env(environ))
        runtime = build_runtime(settings)
        readiness = runtime.registry.readiness_report()
        resolver = ProfileResolver(PROJECT_ROOT)
        family_profiles = [profile.to_dict() for profile in resolver.list_family_profiles()]
        module_profiles = [profile.to_dict() for profile in resolver.list_module_profiles()]
        resolved_profiles = [
            resolver.resolve(family=family_profile.key, module=module_profile.key).to_dict()
            for family_profile in resolver.list_family_profiles()
            for module_profile in resolver.list_module_profiles()
        ]
        # END_BLOCK_BUILD_RUNTIME_AND_PROFILE_SNAPSHOTS
    return {
        "status": "ok" if readiness["registry_ready"] else "degraded",
        "settings": {
            "models_dir": str(settings.models_dir),
            "outputs_dir": str(settings.outputs_dir),
            "voices_dir": str(settings.voices_dir),
            "upload_staging_dir": str(settings.upload_staging_dir),
            "model_manifest_path": str(settings.model_manifest_path),
            "configured_backend": settings.backend,
            "backend_autoselect": settings.backend_autoselect,
            "qwen_fast_enabled": settings.qwen_fast_enabled,
            "qwen_fast_test_mode": env.get("QWEN_TTS_QWEN_FAST_TEST_MODE") or None,
            "model_preload_policy": settings.model_preload_policy,
            "model_preload_ids": list(settings.model_preload_ids),
        },
        "readiness": readiness,
        "assets": build_asset_report(readiness),
        "representative_models": build_representative_model_report(readiness),
        "profiles": {
            "families": family_profiles,
            "modules": module_profiles,
            "resolved_launch_profiles": resolved_profiles,
            "dedicated_family_envs": [
                {
                    "family": profile["key"],
                    "isolated_env_name": profile["isolated_env_name"],
                    "pack_refs": {
                        key: list(values)
                        for key, values in dict(profile.get("pack_refs", {})).items()
                    },
                }
                for profile in family_profiles
                if profile["key"] in {"omnivoice"}
            ],
        },
    }


# START_CONTRACT: main
#   PURPOSE: Execute the runtime self-check and print a JSON readiness snapshot for operators and CI.
#   INPUTS: { argv: list[str] | None - optional raw CLI arguments }
#   OUTPUTS: { int - process exit code }
#   SIDE_EFFECTS: Builds the shared runtime, performs filesystem checks, and writes JSON to stdout
#   LINKS: M-BOOTSTRAP, M-MODEL-REGISTRY
# END_CONTRACT: main
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload, diagnostics = build_self_check_payload_with_diagnostics()
    if diagnostics:
        payload["diagnostics"] = diagnostics
    print(json.dumps(payload, indent=args.indent, sort_keys=True))

    if not args.strict:
        return 0
    if (
        payload["readiness"]["registry_ready"]
        and not payload["assets"]["models_missing_assets"]
    ):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
