"""TTS application entrypoint.

The public FastAPI transports live in :mod:`tts.src.api`; processing and model
code remain independent of application construction.
"""

import uvicorn

from tts.src.api import SETTINGS, app

if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',  # noqa: S104
        port=SETTINGS.listen_port,
        log_config=None,
    )
