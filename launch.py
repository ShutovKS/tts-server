# FILE: launch.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide a root-level cross-platform launcher entrypoint that dispatches to the existing platform-specific guided launch wrappers.
#   SCOPE: project-root resolution, platform detection, subprocess delegation into launcher launch, and exit-code passthrough for operator-facing startup
#   DEPENDS: M-LAUNCHER
#   LINKS: M-ROOT-LAUNCHER
#   ROLE: SCRIPT
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   PROJECT_ROOT - Repository root resolved from the script location.
#   main - Execute the shared launcher dispatch flow through `python -m launcher launch` from the repository root.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added a root-level cross-platform Python launcher entrypoint that delegates to the shared launcher dispatch command]
# END_CHANGE_SUMMARY

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


# START_CONTRACT: main
#   PURPOSE: Execute the shared launcher dispatch flow from the repository root and return its exit code.
#   INPUTS: none
#   OUTPUTS: { int - process exit code from the delegated launcher command }
#   SIDE_EFFECTS: Spawns `python -m launcher launch` in a subprocess rooted at the repository path
#   LINKS: M-ROOT-LAUNCHER, M-LAUNCHER
# END_CONTRACT: main
def main() -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "launcher", "--project-root", str(PROJECT_ROOT), "launch"],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
