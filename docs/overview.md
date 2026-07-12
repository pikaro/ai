# AI services

This repository builds independently deployable speech services. The LLM and
assistant deployments are intentionally outside the current source tree.

## STT

`stt/src/main.py` serves NeMo FastConformer through:

- `GET /health/live` and `GET /health/ready`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/audio/transcriptions`
- `WS /v1/realtime`

The realtime endpoint accepts `session.update`, `input_audio_buffer.append`, and
`input_audio_buffer.commit` events. Appended audio is base64-encoded PCM16. It
emits partial, stable delta, and completed transcription events.

The model is single-worker and internally serialized. Relevant settings use the
`ASR_` prefix; defaults are declared in `stt/src/main.py`.

The STT service uses NeMo 2.4, which provides the required streaming API without
requiring OneLogger. NeMo's published aggregate extras are not used because they
make W&B (and, in newer releases, OneLogger) mandatory; the non-tracking ASR
runtime dependencies are declared in `stt/pyproject.toml` instead. NVIDIA
OneLogger and W&B are not present in the final image. Both service images set
`DO_NOT_TRACK=1` and `HF_HUB_DISABLE_TELEMETRY=1` for transitive model-download
libraries.

## TTS

`tts/src/main.py` keeps Pocket TTS and the configured voice in memory and serves:

- `GET /health/live` and `GET /health/ready`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/audio/speech`
- compatibility endpoints `POST /synthesize` and `POST /synthesize_stream`

`response_format=wav` returns a complete WAV file. `response_format=pcm` streams
PCM16 and includes format headers. Pocket TTS is not thread-safe, so generation
is serialized within the single worker. Relevant settings use the `TTS_` prefix;
defaults are declared in `tts/src/main.py`.

Both `/metrics` endpoints use Prometheus' text exposition format. They include
the Python process collectors and service gauges for model readiness and load
duration; TTS also reports voice load duration. Metrics stay local until a
Prometheus server is configured to scrape them.

## Dependencies

The repository is a uv workspace targeting Python 3.12. Shared HTTP,
configuration, and Prometheus dependencies are declared in `pyproject.toml`.
Model-specific dependencies are declared in `stt/pyproject.toml` and
`tts/pyproject.toml`; `uv.lock` is the only resolved version lock.

Synchronize the shared development environment with:

```sh
uv sync --frozen
```

To create a service-specific environment, select its workspace package:

```sh
uv sync --frozen --package ai-stt
uv sync --frozen --package ai-tts
```

## Local checks

```sh
ruff check .
.venv/bin/pyright
.venv/bin/python -m unittest discover
```

## Images

Build each service from the repository root so uv can access the workspace
metadata and lockfile:

```sh
docker build -f stt/Dockerfile -t ai-stt .
docker build -f tts/Dockerfile -t ai-tts .
```

GitHub Actions publishes `ghcr.io/<owner>/<repository>-stt` and
`ghcr.io/<owner>/<repository>-tts` after checks pass on the default branch or a
version tag. Pull requests and other branches build without publishing.
