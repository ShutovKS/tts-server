#!/usr/bin/env bash
# FILE: launch.sh
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide a root-level Unix shell wrapper for the cross-platform Python launcher entrypoint.
#   SCOPE: repository-root resolution and delegation into launch.py with python3 for Linux and macOS operator startup
#   DEPENDS: M-ROOT-LAUNCHER
#   LINKS: M-ROOT-LAUNCHER-SH
#   ROLE: SCRIPT
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   SCRIPT_DIR - Repository root resolved from the shell wrapper location.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added a root-level Unix shell wrapper that delegates to the shared Python launcher entrypoint]
# END_CHANGE_SUMMARY

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/launch.py"
