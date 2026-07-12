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
COPY stt/pyproject.toml stt/pyproject.toml
COPY tts/pyproject.toml tts/pyproject.toml

RUN uv sync --frozen --no-dev --package ai-stt --no-install-local \
    && python -c "import importlib.util; assert importlib.util.find_spec('nv_one_logger') is None; assert importlib.util.find_spec('wandb') is None"

FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    DO_NOT_TRACK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WANDB_DISABLED=true \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    HF_HOME=/models/huggingface \
    STT_PORT=8080

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
    dumb-init \
    ffmpeg \
    libsndfile1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY --chown=app:app stt/src/ ./

RUN python -c "import importlib.util; import nemo.collections.asr; import main; assert importlib.util.find_spec('nv_one_logger') is None; assert importlib.util.find_spec('wandb') is None"

USER 10001:10001

EXPOSE 8080

ENTRYPOINT ["dumb-init", "--"]

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port \"$STT_PORT\" --workers 1"]
