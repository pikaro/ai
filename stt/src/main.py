from __future__ import annotations

import uvicorn

from stt.src.api import SETTINGS, app

__all__ = ['SETTINGS', 'app']


if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',  # noqa: S104
        port=SETTINGS.listen_port,
        log_config=None,
    )
