FROM python:3.12-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/tmp \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    XDG_CACHE_HOME=/tmp/ai-cache \
    ASSISTANT_PORT=8080

RUN apt-get update \
    && apt-get install --yes --no-install-recommends dumb-init \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --home-dir /tmp --no-create-home app

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.22 /uv /bin/uv

COPY pyproject.toml uv.lock ./
COPY assistant/pyproject.toml assistant/pyproject.toml
COPY stt/pyproject.toml stt/pyproject.toml
COPY tts/pyproject.toml tts/pyproject.toml

RUN uv sync --frozen --no-dev --package ai-assistant --no-install-local

COPY --chown=app:app assistant/__init__.py assistant/__init__.py
COPY --chown=app:app assistant/src/ assistant/src/

FROM runtime AS verification

USER 65532:65532

RUN python -c "from pathlib import Path; import assistant.src.main; import mcp; Path('/tmp/runtime-verified').touch()"

FROM runtime

COPY --from=verification /tmp/runtime-verified /usr/local/share/runtime-verified

USER 10001:10001

EXPOSE 8080

ENTRYPOINT ["dumb-init", "--"]

CMD ["sh", "-c", "exec uvicorn assistant.src.main:app --host 0.0.0.0 --port \"$ASSISTANT_PORT\" --workers 1"]
