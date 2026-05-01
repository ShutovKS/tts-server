# FILE: cli/__main__.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Package entry point for running the CLI via python -m cli.
#   SCOPE: CLI bootstrap and interactive loop launch
#   DEPENDS: M-CLI
#   LINKS: M-CLI
#   ROLE: SCRIPT
#   MAP_MODE: NONE
# END_MODULE_CONTRACT
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

from cli.runtime import run_cli

if __name__ == "__main__":
    run_cli()
