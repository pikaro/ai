FROM python:3.12-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    DO_NOT_TRACK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    WANDB_DISABLED=true

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.22 /uv /bin/uv

COPY pyproject.toml uv.lock ./
COPY assistant/pyproject.toml assistant/pyproject.toml
COPY stt/pyproject.toml stt/pyproject.toml
COPY tts/pyproject.toml tts/pyproject.toml

RUN uv sync --frozen --no-dev --package ai-stt --no-install-local \
    && python -c "import importlib.util; assert importlib.util.find_spec('nv_one_logger') is None; assert importlib.util.find_spec('wandb') is None"

FROM python:3.12-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    DO_NOT_TRACK=1 \
    HOME=/tmp \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WANDB_DISABLED=true \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    HF_HOME=/models/huggingface \
    MPLCONFIGDIR=/tmp/ai-cache/matplotlib \
    NUMBA_CACHE_DIR=/tmp/ai-cache/numba \
    XDG_CACHE_HOME=/tmp/ai-cache \
    LISTEN_PORT=8080

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
    dumb-init \
    ffmpeg \
    libsndfile1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --home-dir /tmp --no-create-home app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY --chown=app:app stt/src/ ./

FROM runtime AS verification

USER 65532:65532

RUN python -c "import importlib.util; from pathlib import Path; import librosa.filters; import nemo.collections.asr; import main; assert importlib.util.find_spec('nv_one_logger') is None; assert importlib.util.find_spec('wandb') is None; Path('/tmp/runtime-verified').touch()"

FROM runtime

COPY --from=verification /tmp/runtime-verified /usr/local/share/runtime-verified

USER 10001:10001

EXPOSE 8080

ENTRYPOINT ["dumb-init", "--"]

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port \"$LISTEN_PORT\" --workers 1"]
