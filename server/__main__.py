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
