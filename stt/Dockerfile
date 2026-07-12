FROM python:3.14-slim-bookworm AS builder

ARG PYTORCH_VERSION=2.11.0

ENV DEBIAN_FRONTEND=noninteractive \
    DO_NOT_TRACK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WANDB_DISABLED=true

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${PYTORCH_VERSION}" \
        "torchaudio==${PYTORCH_VERSION}"

RUN python -m pip install --no-cache-dir --requirement requirements.txt \
    && python -c "import importlib.util; assert importlib.util.find_spec('nv_one_logger') is None; assert importlib.util.find_spec('wandb') is None"

# editdistance 0.8.1's Python 3.14 build omits its C++ runtime dependency.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends patchelf \
    && find /opt/venv -path '*/editdistance/bycython*.so' \
        -exec patchelf --add-needed libstdc++.so.6 {} + \
    && rm -rf /var/lib/apt/lists/*

FROM python:3.14-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    DO_NOT_TRACK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WANDB_DISABLED=true

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg libsndfile1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY --chown=app:app src/ ./

RUN python -c "import importlib.util; import nemo.collections.asr; import main; assert importlib.util.find_spec('nv_one_logger') is None; assert importlib.util.find_spec('wandb') is None"

USER 10001:10001

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
