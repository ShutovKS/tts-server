"""Architecture tests for the Telegram remote-only Phase 1 path."""

# FILE: tests/architecture/test_no_telegram_local_runtime.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Prove the migrated Telegram Phase 1 normal path has no local runtime fallback and remains remote-client-first.
#   SCOPE: Bootstrap import boundaries, startup wiring assertions, dispatcher remote-only command routing, local-runtime forbidden references
#   DEPENDS: M-TELEGRAM, M-SERVER
#   LINKS: V-M-TELEGRAM
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _collect_import_targets - Parse a Python module and collect its import targets for architecture assertions
#   test_telegram_startup_modules_do_not_import_local_runtime_assembly - Verifies Phase 1 startup modules avoid local runtime assembly dependencies
#   test_run_telegram_bot_requires_remote_client_and_wires_dispatcher_remote_only - Verifies startup wiring refuses missing remote client and passes synthesizer=None into the dispatcher
#   test_dispatcher_normal_phase1_commands_route_only_through_remote_job_orchestration - Verifies /tts, /design, and /clone use remote job orchestration gates and contain no direct local synthesizer calls
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added architecture-level anti-local-fallback assertions for Telegram Phase 1 remote-only execution]
# END_CHANGE_SUMMARY

from __future__ import annotations

import ast
from pathlib import Path

import pytest


pytestmark = [pytest.mark.architecture, pytest.mark.integration]

FORBIDDEN_LOCAL_RUNTIME_IMPORT_PREFIXES = (
    "core.application",
    "core.backends",
    "core.bootstrap",
    "core.infrastructure",
    "core.services",
    "server.bootstrap",
)


def _collect_import_targets(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_telegram_startup_modules_do_not_import_local_runtime_assembly():
    startup_files = [
        Path("telegram_bot/bootstrap.py"),
        Path("telegram_bot/__main__.py"),
        Path("telegram_bot/job_orchestrator.py"),
    ]

    forbidden: dict[str, list[str]] = {}
    for path in startup_files:
        imports = _collect_import_targets(path)
        disallowed = sorted(
            name
            for name in imports
            if any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in FORBIDDEN_LOCAL_RUNTIME_IMPORT_PREFIXES
            )
        )
        if disallowed:
            forbidden[str(path)] = disallowed

    assert forbidden == {}


def test_run_telegram_bot_requires_remote_client_and_wires_dispatcher_remote_only():
    contents = Path("telegram_bot/__main__.py").read_text(encoding="utf-8")

    assert "if runtime.remote_server_client is None:" in contents
    assert "Telegram runtime remote server client is required for async command execution" in contents
    assert "dispatcher = CommandDispatcher(" in contents
    assert "synthesizer=None," in contents
    assert "job_orchestrator=job_orchestrator," in contents
    assert "delivery_store=delivery_store," in contents
    assert "TTSSynthesizer" not in contents


def test_dispatcher_normal_phase1_commands_route_only_through_remote_job_orchestration():
    contents = Path("telegram_bot/handlers/dispatcher.py").read_text(encoding="utf-8")

    assert "def _use_job_model(self) -> bool:" in contents
    assert "requires remote async job orchestration but none is configured" in contents
    assert "Clone command requires job model but none is configured" in contents
    assert "await self._handle_tts_via_job(" in contents
    assert "await self._handle_design_via_job(" in contents
    assert "await self._handle_clone_via_job(" in contents
    assert "result = await job_orchestrator.submit_tts_job(" in contents
    assert "result = await job_orchestrator.submit_design_job(" in contents
    assert "result = await job_orchestrator.submit_clone_job(" in contents
    assert "._synthesizer.synthesize(" not in contents
    assert "._synthesizer.synthesize_design(" not in contents
    assert "._synthesizer.synthesize_clone(" not in contents
