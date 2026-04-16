# FILE: tests/integration/test_cli_launchability.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Verify that the CLI starts from python -m cli, accepts scripted exit input, and retains a transcript artifact.
#   SCOPE: Launchability, startup banner, initial menu interaction, bounded exit, transcript persistence
#   DEPENDS: M-CLI
#   LINKS: V-M-CLI
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_cli_launchability_writes_transcript_and_exits_cleanly - Verifies the CLI launchability baseline, prompt surface, and transcript artifact retention
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added automated CLI launchability coverage with scripted stdin and retained transcript evidence]
# END_CHANGE_SUMMARY

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPT_PATH = PROJECT_ROOT / ".sisyphus" / "evidence" / "cli-launchability-transcript.txt"


def test_cli_launchability_writes_transcript_and_exits_cleanly():
    TRANSCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")

    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, "-m", "cli"],
        input="q\n",
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=env,
        timeout=30,
        check=False,
    )
    duration_seconds = time.monotonic() - started

    transcript = (completed.stdout or "") + (completed.stderr or "")
    transcript_body = "\n".join(
        [
            "# CLI launchability transcript",
            f"timestamp_utc: {time.time()}",
            f"command: {sys.executable} -m cli",
            f"cwd: {PROJECT_ROOT}",
            f"exit_code: {completed.returncode}",
            f"duration_seconds: {duration_seconds:.3f}",
            "",
            "--- transcript ---",
            transcript.rstrip(),
            "",
        ]
    )
    TRANSCRIPT_PATH.write_text(transcript_body, encoding="utf-8")

    assert completed.returncode == 0
    assert duration_seconds < 30.0
    assert TRANSCRIPT_PATH.exists()

    assert "Backend:" in transcript
    assert "Selection:" in transcript
    assert "Multi-Family TTS CLI" in transcript
    assert "Select family:" in transcript
    assert "Qwen3-TTS" in transcript
    assert "Piper" in transcript
    assert "OmniVoice" in transcript
    assert "q. Exit" in transcript
    assert "Exiting..." not in transcript

    saved = TRANSCRIPT_PATH.read_text(encoding="utf-8")
    assert "# CLI launchability transcript" in saved
    assert "exit_code: 0" in saved
    assert "command: " in saved
