# FILE: cli/main.py
# VERSION: 1.0.1
# START_MODULE_CONTRACT
#   PURPOSE: Provide a thin explicit module entry point that delegates to the interactive CLI runtime.
#   SCOPE: module-level handoff to cli.runtime.run_cli
#   DEPENDS: M-CLI
#   LINKS: M-CLI
#   ROLE: SCRIPT
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   run_cli - Imported runtime entrypoint delegated by the module execution guard
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.1 - Downgraded file role to a thin wrapper so its contract matches the actual explicit entrypoint behavior]
# END_CHANGE_SUMMARY

from __future__ import annotations

from cli.runtime import run_cli

if __name__ == "__main__":
    run_cli()
