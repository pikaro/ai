# AI services

This repository builds independently deployable speech and assistant services.
The LLM remains a separately deployed llama.cpp server.

All three service images listen on port 8080 by default. Override the listener
with the unprefixed `LISTEN_PORT` environment variable; service-specific
`*_PORT` names are intentionally avoided because Kubernetes reserves those for
service-link variables.

## Logging

Every application, dependency, and Uvicorn logging record is emitted as one
JSON object. Common fields are `timestamp`, `level`, `logger`, and the minimal
`message`; event data remains structured in additional fields. Uvicorn access
records expose `client_address`, `method`, `path`, `http_version`, and
`status_code`.

All services log operational activity and stage durations at `INFO` without
request payloads. Set the unprefixed `LOG_LEVEL=DEBUG` environment variable to
also log transcripts, generated responses, text sent to speech synthesis, LLM
request payloads, tool schemas/arguments/results, and MCP request data. LLM
request payloads contain the exact submitted prompt, including tool definitions
and prior tool results, so DEBUG logs can contain sensitive user or MCP data.
Authentication headers are not logged.

Successful `200` responses from `/health`, `/health/live`, and `/health/ready`
are omitted from Uvicorn access logs and the assistant's HTTPX upstream request
logs; failed health checks and all other requests remain visible.

## Assistant

`assistant/src/main.py` combines the STT, llama.cpp, and TTS services through:

- `GET /health/live` and `GET /health/ready`
- `GET /metrics`
- `GET /v1/models`
- `WS /v1/realtime`

One WebSocket carries one utterance. It accepts the same `session.update`,
`input_audio_buffer.append`, and `input_audio_buffer.commit` events as STT. It
forwards STT transcription events and then emits:

- `response.created`
- `response.text.delta` and `response.text.done`
- `response.audio.started`, `response.audio.delta`, and `response.audio.done`
- `response.done`

Audio deltas are base64-encoded PCM16. `response.audio.started` declares the
sample rate, sample width, and channel count.

Every stable STT word delta builds a complete prompt and warms llama.cpp with
`cache_prompt=true`. An utterance leases one configured llama.cpp slot from its
first stable delta through final text generation, so another assistant session
cannot replace that cache. The final request uses the same prompt and slot. A
final corrective warm is sent only when STT's completed transcript or selected
tool set differs from the last warm. Configure every llama.cpp slot dedicated
to this assistant in `llm_slots`; other clients must not concurrently address
those slot IDs.

LLM output is streamed immediately. Complete sentences are sent to TTS while
the LLM continues decoding, and PCM is streamed to the caller without storing a
WAV file.

The assistant retires idle pooled upstream HTTP connections after four seconds,
before the five-second idle timeout used by Uvicorn and llama.cpp. This avoids
reusing a connection while its peer is closing it; the next request establishes
a clean in-cluster connection instead.

### Tools and MCP

Repo-local tools live in `assistant/src/tools/*.py` and export a
`ToolDefinition` as `TOOL` (or multiple definitions as `TOOLS`). The included
`time.py` tool supports `rough` time by default, `exact` time, and `date`, with
optional IANA timezone selection.

Tools are included in the LLM prompt only when a configured trigger matches a
whole word in the stable transcript. MCP servers use Streamable HTTP by default
and may opt into legacy SSE. Server-level triggers expose that server's tools;
`tool_triggers` can activate individual remote tools without exposing the rest
of the server. An enabled MCP server with no triggers is never presented to the
model. MCP discovery is lazy, so an unavailable optional server does not make
assistant readiness fail.

Tool requests use one or more non-streaming `tool_decision` LLM operations. If
the configured tool-iteration limit is exhausted, a final `tool_answer`
operation forces a spoken answer. INFO records include the duration of each LLM,
local tool, and MCP operation. DEBUG records include the corresponding request
and response data, making it possible to distinguish model-generation latency
from tool execution latency.

Set `ASSISTANT_CONFIG_FILE=/config/assistant.yaml` (or `.yml` / `.json`) to load
a mounted configuration file. Initialization arguments and `ASSISTANT_`
environment variables override file values. Nested environment settings use
`__`, for example:

```sh
ASSISTANT_MCP__calendar__TOKEN=secret
ASSISTANT_MCP__calendar__ENABLED=true
```

See `assistant/config.example.yaml` for a complete non-secret example. Prefer
injecting tokens through a Secret-backed environment variable instead of the
mounted file.

### Assistant metrics

Prometheus metrics cover active/completed sessions, slot use and wait time,
stable STT deltas, first-delta/final-transcript latency, cache-warm counts and
latency, cache prompt sizes and tool-set changes, llama.cpp request latency,
TTFT, prompt/cached/generated tokens, cache reuse, prompt/decode throughput,
tool and MCP outcomes/latency, TTS first-audio latency and real-time factor,
audio bytes, per-stage end-to-end latency, and upstream readiness. Labels are
limited to configured services, operations, tools, stages, and outcomes.

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
`STT_` prefix; defaults are declared in `stt/src/main.py`.

Set `STT_SAVE_LATEST_WAV=true` to atomically overwrite the most recently
committed realtime input recording. `STT_LATEST_WAV_PATH` defaults to
`/tmp/latest.wav`; point it at mounted storage when the recording must be read
after a pod replacement. The WAV header preserves the client-declared sample
rate and channel count, and the saved PCM excludes the silence used internally
to flush the streaming decoder. Capture failures are logged without failing the
transcription request.

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
- `POST /v1/voices`
- `POST /v1/audio/speech`
- compatibility endpoints `POST /synthesize` and `POST /synthesize_stream`

`response_format=wav` returns a complete WAV file. `response_format=pcm` streams
PCM16 and includes format headers. Pocket TTS is not thread-safe, so generation
is serialized within the single worker. Relevant settings use the `TTS_` prefix;
defaults are declared in `tts/src/main.py`.

Upload a Pocket TTS voice-state file as multipart form data. The endpoint does
not require authentication:

```sh
curl -F name=foo -F file=@foo.safetensors http://localhost:8080/v1/voices
```

Uploads are stored atomically as `<name>.safetensors` in
`TTS_DATA_DIRECTORY`, which defaults to `/data`. `TTS_MAXIMUM_VOICE_UPLOAD_BYTES`
limits each upload and defaults to 100 MiB. When the service next starts with
`TTS_VOICE=foo`, `/data/foo.safetensors` (or the corresponding file in the
configured data directory) is loaded in place of the canned `foo` voice. Mount
the data directory on persistent storage when uploads must survive container
replacement.

The STT and TTS `/metrics` endpoints use Prometheus' text exposition format.
They include the Python process collectors and service gauges for model
readiness and load duration; TTS also reports voice load duration. Metrics stay
local until a Prometheus server is configured to scrape them.

## Dependencies

The repository is a uv workspace targeting Python 3.12. Shared HTTP,
configuration, and Prometheus dependencies are declared in `pyproject.toml`.
Model-specific dependencies are declared in `stt/pyproject.toml` and
`tts/pyproject.toml`; assistant orchestration dependencies are declared in
`assistant/pyproject.toml`. `uv.lock` is the only resolved version lock.

Synchronize the shared development environment with:

```sh
uv sync --frozen
```

To create a service-specific environment, select its workspace package:

```sh
uv sync --frozen --package ai-stt
uv sync --frozen --package ai-tts
uv sync --frozen --package ai-assistant
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
docker build -f assistant/Dockerfile -t ai-assistant .
```

GitHub Actions publishes `ghcr.io/<owner>/<repository>-assistant`,
`ghcr.io/<owner>/<repository>-stt`, and `ghcr.io/<owner>/<repository>-tts` after
checks pass on the default branch or a version tag. Pull requests and other
branches build without publishing.
