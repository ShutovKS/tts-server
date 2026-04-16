# FILE: launcher/__main__.py
# VERSION: 1.0.1
# START_MODULE_CONTRACT
#   PURPOSE: Execute the launcher package entrypoint when invoked with `python -m launcher`.
#   SCOPE: module-level import and process exit handoff to launcher.main.main
#   DEPENDS: M-LAUNCHER
#   LINKS: M-LAUNCHER
#   ROLE: SCRIPT
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   main - Imported launcher entrypoint delegated to the module execution guard
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.1 - Added full GRACE module anchors for the package execution entrypoint]
# END_CHANGE_SUMMARY

from launcher.main import main


# START_BLOCK_MODULE_EXECUTION_GUARD
if __name__ == "__main__":
    raise SystemExit(main())
# END_BLOCK_MODULE_EXECUTION_GUARD
