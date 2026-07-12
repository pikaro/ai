# AI services

This repository builds independently deployable speech services. The LLM and
assistant deployments are intentionally outside the current source tree.

## STT

`stt/src/main.py` serves NeMo FastConformer through:

- `GET /health/live` and `GET /health/ready`
- `GET /v1/models`
- `POST /v1/audio/transcriptions`
- `WS /v1/realtime`

The realtime endpoint accepts `session.update`, `input_audio_buffer.append`, and
`input_audio_buffer.commit` events. Appended audio is base64-encoded PCM16. It
emits partial, stable delta, and completed transcription events.

The model is single-worker and internally serialized. Relevant settings use the
`ASR_` prefix; defaults are declared in `stt/src/main.py`.

The STT image pins NeMo 2.4, which provides the required streaming API without
OneLogger. NVIDIA OneLogger and W&B are not present in the final image. Both
service images set `DO_NOT_TRACK=1` and
`HF_HUB_DISABLE_TELEMETRY=1` for transitive model-download libraries.

## TTS

`tts/src/main.py` keeps Pocket TTS and the configured voice in memory and serves:

- `GET /health/live` and `GET /health/ready`
- `GET /v1/models`
- `POST /v1/audio/speech`
- compatibility endpoints `POST /synthesize` and `POST /synthesize_stream`

`response_format=wav` returns a complete WAV file. `response_format=pcm` streams
PCM16 and includes format headers. Pocket TTS is not thread-safe, so generation
is serialized within the single worker. Relevant settings use the `TTS_` prefix;
defaults are declared in `tts/src/main.py`.

## Local checks

```sh
ruff check .
.venv/bin/pyright
.venv/bin/python -m unittest discover
```

## Images

Build each service with its own directory as the Docker build context:

```sh
docker build -t ai-stt stt
docker build -t ai-tts tts
```

GitHub Actions publishes `ghcr.io/<owner>/<repository>-stt` and
`ghcr.io/<owner>/<repository>-tts` after checks pass on the default branch or a
version tag. Pull requests and other branches build without publishing.
