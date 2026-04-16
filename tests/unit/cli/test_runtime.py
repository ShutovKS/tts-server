# FILE: tests/unit/cli/test_runtime.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Validate family-aware CLI runtime behavior for multi-family flows.
#   SCOPE: OmniVoice model-id routing and family capability discovery helpers
#   DEPENDS: M-CLI
#   LINKS: V-M-CLI
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_run_design_session_uses_model_metadata_id_for_omnivoice - Ensures CLI design flow routes via capability-specific model id
# END_MODULE_MAP

from __future__ import annotations

from pathlib import Path

import pytest

from cli.runtime import CLI_MODELS, CliRuntime
from core.contracts.results import GenerationResult


pytestmark = pytest.mark.unit


class _ServiceStub:
    def __init__(self):
        self.last_command = None

    def synthesize_design(self, command):
        self.last_command = command
        return GenerationResult(
            audio=b"wav",
            sample_rate=24000,
            mode="design",
            model="OmniVoice-Design",
            backend="torch",
            saved_path=Path(".outputs") / "omnivoice-design.wav",
        )


def test_run_design_session_uses_model_metadata_id_for_omnivoice(monkeypatch: pytest.MonkeyPatch):
    runtime = CliRuntime()
    runtime.service = _ServiceStub()

    monkeypatch.setattr(runtime, "display_saved_output", lambda *args, **kwargs: None)
    inputs = iter(["Design validation sample.", None])
    monkeypatch.setattr(runtime, "get_safe_input", lambda *args, **kwargs: next(inputs))
    monkeypatch.setattr(runtime, "_prompt_instruct", lambda **kwargs: "Warm bilingual narrator")
    monkeypatch.setattr(runtime, "_prompt_language", lambda: "auto")
    monkeypatch.setattr(runtime, "_available_model_specs", lambda: [CLI_MODELS["omnivoice-design-1"]])

    runtime.run_design_session("omnivoice-design-1")

    assert runtime.service.last_command is not None
    assert runtime.service.last_command.model == "OmniVoice-Design"
    assert runtime.service.last_command.voice_description == "Warm bilingual narrator"
