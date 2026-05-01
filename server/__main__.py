# FILE: server/__main__.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Package entry point for running the server via python -m server.
#   SCOPE: uvicorn launch with server settings
#   DEPENDS: M-SERVER
#   LINKS: M-SERVER
#   ROLE: SCRIPT
#   MAP_MODE: NONE
# END_MODULE_CONTRACT
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

import uvicorn

from server.bootstrap import get_server_settings

if __name__ == "__main__":
    settings = get_server_settings()
    uvicorn.run(
        "server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        factory=False,
    )
