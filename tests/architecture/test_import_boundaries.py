# FILE: tests/architecture/test_import_boundaries.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Verify import boundary constraints between project layers.
#   SCOPE: Architecture boundary checks ensuring adapters don't cross-depend
#   DEPENDS: none
#   LINKS: none
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _collect_import_targets - Helper that parses Python files and collects import targets
#   _matches_prefix - Helper that checks module-name prefix matches
#   test_cli_has_no_other_adapter_imports - Verifies CLI code does not import other transport adapters
#   test_core_has_no_adapter_imports - Verifies core runtime does not import transport adapters
#   test_runtime_code_has_no_legacy_server_compatibility_imports - Verifies runtime code avoids removed legacy server modules
#   test_server_adapter_depends_only_on_server_and_core_modules - Verifies server code does not import other transport adapters
#   test_telegram_adapter_depends_only_on_telegram_and_core_modules - Verifies Telegram code does not import server or CLI modules
#   test_server_app_is_thin_composition_root - Verifies server app remains a thin composition root
#   test_job_execution_module_has_no_adapter_imports - Verifies job execution contracts avoid adapter imports
#   test_job_execution_module_has_no_local_infra_implementation_imports - Verifies job execution abstractions avoid local infra dependencies
#   test_local_job_execution_adapters_live_in_infrastructure_layer - Verifies local job adapters remain in infrastructure
#   test_job_wiring_in_core_bootstrap_is_config_driven_and_local_default_ready - Verifies bootstrap wiring remains config-driven with local defaults
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Extended adapter boundary checks to cover telegram_bot imports and runtime scans]
# END_CHANGE_SUMMARY

from __future__ import annotations

import ast
from pathlib import Path

import pytest


pytestmark = pytest.mark.architecture


LEGACY_SERVER_MODULES = {
    "server.config",
    "server.services.model_registry",
    "server.services.tts_service",
    "server.infrastructure.audio_io",
}
ADAPTER_IMPORT_PREFIXES = ("server", "cli", "telegram_bot")
FORBIDDEN_JOB_EXECUTION_IMPORT_TOKENS = (
    "collections",
    "contextlib",
    "pathlib",
    "queue",
    "threading",
)


def _collect_import_targets(base_dir: str) -> dict[Path, set[str]]:
    targets: dict[Path, set[str]] = {}
    for path in Path(base_dir).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        file_imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    file_imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                file_imports.add(node.module)
        targets[path] = file_imports
    return targets


def _matches_prefix(module_name: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in prefixes
    )


def test_cli_has_no_other_adapter_imports():
    imports_by_file = _collect_import_targets("cli")
    forbidden = {
        str(path): sorted(
            name
            for name in imports
            if _matches_prefix(name, ("server", "telegram_bot"))
        )
        for path, imports in imports_by_file.items()
        if any(_matches_prefix(name, ("server", "telegram_bot")) for name in imports)
    }
    assert forbidden == {}


def test_core_has_no_adapter_imports():
    imports_by_file = _collect_import_targets("core")
    forbidden = {
        str(path): sorted(
            name for name in imports if _matches_prefix(name, ADAPTER_IMPORT_PREFIXES)
        )
        for path, imports in imports_by_file.items()
        if any(_matches_prefix(name, ADAPTER_IMPORT_PREFIXES) for name in imports)
    }
    assert forbidden == {}


def test_runtime_code_has_no_legacy_server_compatibility_imports():
    runtime_dirs = ("cli", "core", "server", "telegram_bot", "tests")
    forbidden: dict[str, list[str]] = {}
    for base_dir in runtime_dirs:
        for path, imports in _collect_import_targets(base_dir).items():
            disallowed = sorted(
                name for name in imports if name in LEGACY_SERVER_MODULES
            )
            if disallowed:
                forbidden[str(path)] = disallowed
    assert forbidden == {}


def test_server_adapter_depends_only_on_server_and_core_modules():
    imports_by_file = _collect_import_targets("server")
    forbidden = {
        str(path): sorted(
            name for name in imports if _matches_prefix(name, ("cli", "telegram_bot"))
        )
        for path, imports in imports_by_file.items()
        if any(_matches_prefix(name, ("cli", "telegram_bot")) for name in imports)
    }
    assert forbidden == {}


def test_telegram_adapter_depends_only_on_telegram_and_core_modules():
    imports_by_file = _collect_import_targets("telegram_bot")
    forbidden = {
        str(path): sorted(
            name for name in imports if _matches_prefix(name, ("server", "cli"))
        )
        for path, imports in imports_by_file.items()
        if any(_matches_prefix(name, ("server", "cli")) for name in imports)
    }
    assert forbidden == {}


def test_server_app_is_thin_composition_root():
    content = Path("server/app.py").read_text(encoding="utf-8")
    assert "register_health_routes" in content
    assert "register_models_routes" in content
    assert "register_tts_routes" in content
    assert "job_execution" in content
    assert "job_manager.start" in content
    assert "job_manager.stop" in content
    assert '@app.get("/health/live"' not in content
    assert '@app.post("/v1/audio/speech"' not in content


def test_job_execution_module_has_no_adapter_imports():
    imports_by_file = _collect_import_targets("core/application")
    forbidden = {
        str(path): sorted(
            name for name in imports if _matches_prefix(name, ADAPTER_IMPORT_PREFIXES)
        )
        for path, imports in imports_by_file.items()
        if path.name == "job_execution.py"
        and any(_matches_prefix(name, ADAPTER_IMPORT_PREFIXES) for name in imports)
    }
    assert forbidden == {}


def test_job_execution_module_has_no_local_infra_implementation_imports():
    imports = _collect_import_targets("core/application")[
        Path("core/application/job_execution.py")
    ]
    forbidden = sorted(
        name
        for name in imports
        if _matches_prefix(name, FORBIDDEN_JOB_EXECUTION_IMPORT_TOKENS)
    )
    assert forbidden == []


def test_local_job_execution_adapters_live_in_infrastructure_layer():
    content = Path("core/infrastructure/job_execution_local.py").read_text(
        encoding="utf-8"
    )
    assert "class LocalInMemoryJobStore" in content
    assert "class LocalBoundedExecutionManager" in content
    assert "class LocalJobArtifactStore" in content


def test_job_wiring_in_core_bootstrap_is_config_driven_and_local_default_ready():
    content = Path("core/bootstrap.py").read_text(encoding="utf-8")
    assert "def build_job_artifact_store" in content
    assert "def build_job_metadata_store" in content
    assert "def build_job_execution_backend" in content
    assert 'settings.job_artifact_backend == "local"' in content
    assert 'settings.job_metadata_backend == "local"' in content
    assert 'settings.job_execution_backend == "local"' in content
