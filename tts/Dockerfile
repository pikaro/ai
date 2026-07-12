FROM python:3.14-slim-bookworm

ARG PYTORCH_VERSION=2.12.1

ENV DEBIAN_FRONTEND=noninteractive \
    DO_NOT_TRACK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    TTS_PORT=8080

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libsndfile1 dumb-init \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==${PYTORCH_VERSION}" \
    && python -m pip install --no-cache-dir --requirement requirements.txt

COPY --chown=app:app src/ ./

RUN python -c "import importlib.util; import main; import pocket_tts; assert importlib.util.find_spec('nv_one_logger') is None; assert importlib.util.find_spec('wandb') is None"

USER 10001:10001

EXPOSE 8080

ENTRYPOINT ["dumb-init", "--"]

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port \"$TTS_PORT\" --workers 1"]
