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
ADAPTER_IMPORT_PREFIXES = ("server", "cli")


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
    return any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in prefixes)


def test_cli_has_no_server_imports():
    imports_by_file = _collect_import_targets("cli")
    forbidden = {
        str(path): sorted(name for name in imports if _matches_prefix(name, ("server",)))
        for path, imports in imports_by_file.items()
        if any(_matches_prefix(name, ("server",)) for name in imports)
    }
    assert forbidden == {}


def test_core_has_no_adapter_imports():
    imports_by_file = _collect_import_targets("core")
    forbidden = {
        str(path): sorted(name for name in imports if _matches_prefix(name, ADAPTER_IMPORT_PREFIXES))
        for path, imports in imports_by_file.items()
        if any(_matches_prefix(name, ADAPTER_IMPORT_PREFIXES) for name in imports)
    }
    assert forbidden == {}


def test_runtime_code_has_no_legacy_server_compatibility_imports():
    runtime_dirs = ("cli", "core", "server", "tests")
    forbidden: dict[str, list[str]] = {}
    for base_dir in runtime_dirs:
        for path, imports in _collect_import_targets(base_dir).items():
            disallowed = sorted(name for name in imports if name in LEGACY_SERVER_MODULES)
            if disallowed:
                forbidden[str(path)] = disallowed
    assert forbidden == {}


def test_server_adapter_depends_only_on_server_and_core_modules():
    imports_by_file = _collect_import_targets("server")
    forbidden = {
        str(path): sorted(name for name in imports if _matches_prefix(name, ("cli",)))
        for path, imports in imports_by_file.items()
        if any(_matches_prefix(name, ("cli",)) for name in imports)
    }
    assert forbidden == {}


def test_server_app_is_thin_composition_root():
    content = Path("server/app.py").read_text(encoding="utf-8")
    assert "register_health_routes" in content
    assert "register_models_routes" in content
    assert "register_tts_routes" in content
    assert '@app.get("/health/live"' not in content
    assert '@app.post("/v1/audio/speech"' not in content
