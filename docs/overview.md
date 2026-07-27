# AI services

This repository builds independently deployable speech and assistant services.
The LLM remains a separately deployed llama.cpp server.

All three service images listen on port 8080 by default. Override the listener
with the unprefixed `LISTEN_PORT` environment variable; service-specific
`*_PORT` names are intentionally avoided because Kubernetes reserves those for
service-link variables.

All three services expose `GET /config` and `PATCH /config`. `PATCH /config`
accepts a JSON object containing a top-level subset of the service's settings,
validates the complete resulting configuration, and applies it only in memory.
It returns the changed field names and `ephemeral: true`; restart or pod
replacement restores the file and environment configuration. Nested values such
as the assistant's `mcp` map are replaced as a unit rather than recursively
merged.

Configuration updates and inference are mutually exclusive and are never
queued. A patch or HTTP inference request received while that service is busy
returns `409 Conflict`; a concurrent realtime WebSocket is closed with code
`1013`. Listener and loaded-model identity settings cannot be safely replaced
in place and return a `409` response listing the fields that require restart.
These are `listen_port` for the assistant; `model_id`, `device`, and
`listen_port` for STT; and `model_id`, `language`, and `listen_port` for TTS.
Clearing a live STT `attention_context_size` also requires restart, while
changing one explicit context pair to another is reloadable.

For example, the following temporary update changes cache-warm cadence without
modifying the mounted configuration:

```sh
curl -X PATCH http://assistant/config \
  -H 'Content-Type: application/json' \
  -d '{"llm_cache_warm_min_interval_seconds":0.75,"llm_cache_warm_min_new_characters":5}'
```

The assistant also exposes `GET /dashboard`, a small internal configuration
page for all three services. Its highlighted JSON editors compare edits with
the last loaded configuration and PATCH only the changed top-level fields.
Assistant changes use its local configuration endpoint; STT and TTS changes are
proxied to their existing `/config` endpoints, so their normal validation, busy
rejection, and restart restrictions still apply. The dashboard also edits the
assistant system prompt. Its audio debug controls record mono PCM16 at the
browser's actual audio-context sample rate and can submit those exact bytes to
STT or the assistant realtime endpoint. The TTS debug form selects a voice,
toggles the server-side pipeline, and requests either WAV or PCM. PCM stays
streamed through the dashboard proxy and is scheduled through the browser audio
context as chunks arrive. A small playback-ahead bound propagates backpressure
instead of buffering an arbitrary response. The page reports first-chunk and
completion timing, then adds a WAV container locally for replay. Assistant PCM
responses and TTS audio can therefore both be played in the browser. Browser
microphone access requires a secure context, such as HTTPS or localhost.

## Logging

Every application, dependency, and Uvicorn logging record is emitted as one
JSON object. Common fields are `timestamp`, `level`, `logger`, `event_id`, and
the minimal `message`; event data remains structured in additional fields.
Project event IDs use stable, descriptive `ID_snake_case` names and can be
found in raw logs with `\bID_[a-z_]+\b`. Uvicorn access and HTTPX
request records use `ID_http_server_request` and `ID_http_client_request`;
uncatalogued dependency records use `ID_dependency_log`. Uvicorn access
records expose `client_address`, `method`, `path`, `http_version`, and
`status_code`.

All services log operational activity and stage durations at `INFO` without
request payloads. Set the unprefixed `LOG_LEVEL=DEBUG` environment variable to
also log transcripts, generated responses, text sent to speech synthesis, LLM
request payloads, tool schemas/arguments/results, and MCP request data. LLM
request payloads contain the exact submitted prompt, including tool definitions
and prior tool results, so DEBUG logs can contain sensitive user or MCP data.
Authentication headers are not logged.

The assistant's `WS /v1/realtime` endpoint accepts optional `X-Request-Id` and
`X-Request-Timestamp` headers. When either is present, the assistant logs an
`ID_assistant_request_correlation_received` record immediately on receipt with
the verbatim values in `request_id` and `request_timestamp`. The values are for
log correlation only; they are not validated or forwarded to upstream services.

Successful `200` responses from `/health`, `/health/live`, and `/health/ready`
are omitted from Uvicorn access logs and the assistant's HTTPX upstream request
logs; failed health checks and all other requests remain visible.

## Assistant

`assistant/src/main.py` combines the STT, llama.cpp, and TTS services through:

- `GET /health/live` and `GET /health/ready`
- `GET /metrics`
- `GET /dashboard`
- `POST /dashboard/stt` and `POST /dashboard/tts` (dashboard debug helpers)
- `GET /config` and `PATCH /config`
- `GET /system-prompt` and `PUT /system-prompt`
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
sample rate, sample width, and channel count. If STT produces an empty
transcript, the assistant emits no response event and closes with WebSocket code
1008 so clients can treat the empty utterance as an expected outcome.

Before the application starts serving, it discovers the tool catalog and warms
every configured llama.cpp slot with the immutable system/tool prefix. Startup
does not complete until these requests succeed, so this warm also gates
readiness.

Stable STT word deltas feed a latest-wins llama.cpp cache warmer using
`cache_prompt=true`. Warm requests end immediately after the stable transcript
prefix. They intentionally omit the closing user marker and assistant
generation marker because each transcript extension would move and invalidate
that otherwise immutable suffix. At most one warm is active and only one pending
transcript is retained; a newer partial replaces that pending value. The STT
receive loop never waits for a partial warm. On final transcription the pending
partial is dropped, the one active warm is allowed to finish, and the generation
request appends and evaluates the final chat suffix. No speculative LLM requests
run concurrently and no partially cancelled cache state is assumed to be
reusable.

The prompt uses the non-thinking Qwen3 Instruct chat format. It starts with the
user-editable system prompt followed by the complete, deterministically sorted
tool catalog; it does not add `/no_think` or `<think>` prefill tokens. The
current transcript follows that stable prefix, so ordinary command changes
preserve the system and tool KV cache. On startup the assistant creates
`/tmp/system-prompt` with the default prompt if the file is absent. It checks
the configured `system_prompt_path` before each prompt build and reloads a
stable file revision after an in-place edit or atomic replacement. `GET
/system-prompt` returns the active text and `PUT /system-prompt` atomically
replaces it; updates are rejected while the assistant is busy. Existing
contents are never overwritten during startup. `/tmp` is ephemeral across pod
replacement; configure a mounted path when edits must persist.

Set `llm_cache_warm_enabled` to disable incremental warming,
`llm_cache_warm_min_interval_seconds` to limit warm start frequency, and
`llm_cache_warm_min_new_characters` to ignore very small append-only updates.
An utterance leases one configured llama.cpp slot from its first actual warm
through final text generation. The assistant admits one WebSocket session at a
time and closes a concurrent session with code `1013`, so additional configured
slots cannot introduce concurrent LLM inference. Other clients must not
concurrently address the assistant's llama.cpp slot.

LLM output is streamed immediately. For tool-aware responses, leading
whitespace is ignored while classifying the first output character: `{` buffers
a JSON tool request for validation and execution, while any other character
starts the spoken-answer stream. The assistant forwards those text deltas over
one TTS pipeline WebSocket while continuing LLM decoding, then translates the
returned binary PCM frames into `response.audio.delta` events. Text and audio
flow concurrently. Sentence detection, sequential synthesis, pauses, fades, and
composite WAV capture belong to TTS rather than the assistant orchestration
layer.

The assistant retires idle pooled upstream HTTP connections after four seconds,
before the five-second idle timeout used by Uvicorn and llama.cpp. This avoids
reusing a connection while its peer is closing it; the next request establishes
a clean in-cluster connection instead.

### Tools and MCP

Repo-local tools live in `assistant/src/tools/*.py` and export a
`ToolDefinition` as `TOOL` (or multiple definitions as `TOOLS`). The included
`time.py` tool supports `rough` time by default, `exact` time, and `date`, with
optional IANA timezone selection. Its catalog declares `rough` and
`Europe/Berlin` as defaults. The model is instructed to omit arguments that
match catalog defaults and include only non-default overrides, while the tool
executor applies the same defaults when fields are absent.

Every repo-local tool and every discovered tool from an enabled MCP server is
included in the stable prompt prefix. Transcript triggers do not filter or
whitelist tools. The legacy `triggers` and `tool_triggers` configuration fields
are accepted for compatibility but ignored. MCP servers use Streamable HTTP by
default and may opt into legacy SSE. Discovery starts during the startup cache
warm, runs concurrently across enabled servers, and is cached for the process
lifetime. An unavailable optional server does not make assistant readiness
fail; discovery is retried after its configured interval.

Tool decisions use the same streamed LLM operation as spoken answers. JSON is
buffered only when the response begins with `{`; ordinary text reaches sentence
TTS without waiting for the complete answer. Subsequent tool iterations keep
the system prompt and catalog unchanged, replay the model's generated tool JSON
exactly so llama.cpp can reuse that KV suffix, and append compact tool-result
JSON afterward. If the configured tool-iteration limit is exhausted, the
instruction to produce a final spoken answer is appended as conversation
history rather than changing the system prompt. INFO records include the
duration of each LLM, local tool, and MCP operation. DEBUG records include the
corresponding request and response data, making it possible to distinguish
model-generation latency from tool execution latency.

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

Prometheus metrics cover active/completed/rejected sessions, slot use and wait
time, stable STT deltas, first-delta/final-transcript latency, cache scheduler
dispositions and final-wait time, cache-warm counts and latency, cache prompt
sizes and tool-set changes, llama.cpp request and server-reported phase latency,
TTFT, total/evaluated/cached/generated tokens, correctly denominatored cache
reuse, prompt/decode throughput, tool and MCP outcomes/latency, TTS first-audio
latency and real-time factor, audio bytes, configuration updates, and
end-to-end stages from final transcript through LLM request, first token, first
TTS text, first audio, and completion. Labels are limited to
configured services, operations, tools, stages, dispositions, and outcomes.

## STT

`stt/src/main.py` serves NeMo FastConformer through:

- `GET /health/live` and `GET /health/ready`
- `GET /metrics`
- `GET /config` and `PATCH /config`
- `GET /v1/models`
- `POST /v1/audio/transcriptions`
- `WS /v1/realtime`

The realtime endpoint accepts `session.update`, `input_audio_buffer.append`, and
`input_audio_buffer.commit` events. Appended audio is base64-encoded PCM16. It
emits partial, stable delta, and completed transcription events.

The model is single-worker. Concurrent HTTP or WebSocket inference is rejected
instead of waiting behind the active session. Relevant settings use the `STT_`
prefix; defaults are declared in `stt/src/main.py`.

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

`tts/src/main.py` keeps one Pocket TTS model and reusable voice states in memory
and serves:

- `GET /health/live` and `GET /health/ready`
- `GET /metrics`
- `GET /config` and `PATCH /config`
- `GET /v1/models`
- `POST /v1/voices`
- `POST /v1/audio/speech`
- `POST /v1/audio/speech/pipeline`
- `WS /v1/audio/speech/pipeline`
- compatibility endpoints `POST /synthesize` and `POST /synthesize_stream`

`response_format=wav` returns a complete WAV file. `response_format=pcm` streams
PCM16 and includes format headers. Pocket TTS is not thread-safe, so only one
generation is admitted and concurrent requests receive `409 Conflict` instead
of waiting on its model lock. Relevant settings use the `TTS_` prefix; defaults
are declared in `tts/src/main.py`.

Speech requests select a voice with the OpenAI-compatible `voice` body field.
An omitted value or `"voice": "default"` selects `TTS_VOICE`; a named value
such as `"voice": "bender"` selects the corresponding uploaded voice or Pocket
TTS canned voice. Voice states are loaded on first use and retained while the
process is running, sharing the single base model.

The server-side voice pipeline accepts one complete speech request at `POST
/v1/audio/speech/pipeline`. The regular `POST /v1/audio/speech` enables the same
behavior when `X-Pipeline: true` is present; without the header its behavior is
unchanged. A complete paragraph is split inside TTS, so a PCM response can begin
with the first completed segment without sender-managed sentence requests. WAV
responses use the same synthesis path but are buffered until a complete WAV can
be returned.

The incremental `WS /v1/audio/speech/pipeline` interface preserves overlap with
a text generator such as the assistant LLM. The client first sends a
`session.start` JSON event, followed by `input_text.delta` events and one
`input_text.done`. TTS replies with `session.ready`, streams raw PCM as binary
frames, and finishes with `response.audio.done`. While one segment is being
synthesized the server deliberately stops reading more text; WebSocket/TCP flow
control therefore propagates backpressure to the producer without another
unbounded application queue. The exclusive TTS gate is acquired on the first
text delta and retained through completion. An incomplete client has
`pipeline_idle_timeout_seconds` to provide the next delta before its session is
closed.

Pipeline text normally splits on `.!?`. Configure this with
`pipeline_sentence_terminators`. With
`pipeline_first_segment_comma_delimiter=true`, the default, a comma may
additionally finish only the first segment so audio can begin with the opening
clause. TTS inserts a short silence and linear fades between synthesized
segments; configure these with `pipeline_sentence_pause_seconds` and
`pipeline_sentence_crossfade_seconds`. The corresponding environment variables
are prefixed with `TTS_`, for example
`TTS_PIPELINE_FIRST_SEGMENT_COMMA_DELIMITER=false`. Set either duration to zero
to disable that transition component.

Set `TTS_SAVE_LATEST_WAV=true` to atomically overwrite the most recently
completed synthesized recording. `TTS_LATEST_WAV_PATH` defaults to
`/tmp/latest.wav`; point it at mounted storage when the recording must be read
after a pod replacement. A WAV response is saved verbatim. For a PCM response,
the saved WAV frame data is the exact concatenation of the PCM chunks emitted to
the TTS client; synthesis is not repeated. Pipeline requests save one stitched
recording containing all segment fades and inserted silence, rather than
overwriting the file for each internal segment. An interrupted stream does not
replace the previous recording, and capture failures are logged without failing
the speech request.

Deploy a pipeline-capable TTS image before an assistant image that uses the
incremental endpoint; the services roll independently and the assistant does
not retain its former per-sentence fallback.

Upload a Pocket TTS voice-state file as multipart form data. The endpoint does
not require authentication:

```sh
curl -F name=foo -F file=@foo.safetensors http://localhost:8080/v1/voices
```

Uploads are stored atomically as `<name>.safetensors` in
`TTS_DATA_DIRECTORY`, which defaults to `/data`. `TTS_MAXIMUM_VOICE_UPLOAD_BYTES`
limits each upload and defaults to 100 MiB. A request with `"voice": "foo"`
selects `/data/foo.safetensors` (or the corresponding file in the configured
data directory); setting `TTS_VOICE=foo` makes it the default. Replacing a voice
through the upload endpoint invalidates its cached state so the next request
loads the new file. Mount the data directory on persistent storage when uploads
must survive container replacement.

The STT and TTS `/metrics` endpoints use Prometheus' text exposition format.
They include Python process collectors, model readiness/load gauges, request
counts and latency, active requests, busy rejections, and configuration updates.
STT additionally reports audio duration, chunk and commit processing latency,
and time to first stable delta. TTS additionally reports voice load duration,
time to first PCM audio, output duration, real-time factor, and pipeline
requests and synthesized segment counts by transport. Metrics stay local until
a Prometheus server is configured to scrape them.

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
