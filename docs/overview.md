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
records expose `client_address`, `method`, the query-free `path`,
`http_version`, and `status_code`. HTTP client records likewise omit URL
credentials, queries, and fragments.

All services log only allowlisted operational metadata at levels above
`DEBUG`: counts, timing, status, operation names, configured model/voice/tool
identifiers, exception types, and traceback locations. Exception messages,
formatting arguments, arbitrary dependency messages, request and response
bodies, prompts, transcripts, generated speech text, speaker labels, tool
arguments/results, and conversation history are suppressed at those levels.
Set the unprefixed `LOG_LEVEL=DEBUG` environment variable to emit the explicit
application DEBUG records containing transcripts, generated responses, text
sent to speech synthesis, LLM request payloads, tool schemas/arguments/results,
and MCP request data. LLM request payloads contain the exact submitted prompt,
including tool definitions and prior tool results, so DEBUG logs can contain
sensitive user or MCP data. Authentication headers are not logged.

The assistant's `WS /v1/realtime` endpoint accepts optional `X-Request-Id` and
`X-Request-Timestamp` headers. When either is present, the assistant logs an
`ID_assistant_request_correlation_received` record immediately on receipt with
bounded token-like values in `request_id` and `request_timestamp`; other values
are omitted from the record. The headers are for log correlation only and are
not forwarded to upstream services.

Successful `200` responses from `/health`, `/health/live`, `/health/ready`, and
`/metrics` are omitted from Uvicorn access logs. Successful HTTPX access
records are also omitted because the surrounding application event records the
operation and duration; non-success responses remain visible. Uvicorn's
successful WebSocket accept/open/close records and PocketTTS's per-segment
internal timers are suppressed in favor of the corresponding service session
and segment events.

Routine incremental work stays out of INFO logs. STT records the first partial
for latency correlation and then summarizes the completed session; at DEBUG it
records changed partials, stable deltas, and the final recognized text.
At DEBUG the assistant combines each stable STT update with its cache-warm
scheduler disposition, records one timing event for every warm request that
actually runs, and records the final cache-barrier wait. Cache-warm failures
remain visible above DEBUG. TTS emits one completion event per synthesized
pipeline segment with generation and silence-processing timings. This keeps
one request readable as a chronological service narrative at INFO while
retaining transcript revisions and incremental-warming diagnostics at DEBUG.

## Code structure

Each deployable package follows the same one-way dependency direction:
entrypoint and API transports call a transport-neutral runtime, which composes
model engines, processors, and persistence helpers. Expected failures are typed
in each service's `domain.py` and translated to HTTP or WebSocket behavior only
at the API boundary. `main.py` files only expose the application and Uvicorn
entrypoint.

| Service | API transports and schemas | Application boundary | Processing and integrations |
| --- | --- | --- | --- |
| Assistant | `api.py`, `realtime.py`, `dashboard.py`, `configuration_api.py`, `dependencies.py`, `schemas.py` | `runtime.py` | `pipeline.py`, `upstream.py`, `tooling.py` |
| STT | `api.py`, `schemas.py` | `runtime.py` | `engine.py`, `streaming.py`, `transcript.py`, `uploads.py`, `recording.py` |
| TTS | `api.py`, `schemas.py` | `runtime.py` | `engine.py`, `streaming.py`, `pipeline.py`, `voices.py`, `recording.py` |

Models shared across independently deployed services live in
`service_contracts/`. This package owns the model-list, realtime audio-input,
STT event, and incremental TTS wire contracts without depending on any service
implementation or web framework. In particular, the assistant validates the
same STT/TTS models the servers serialize rather than maintaining parallel
dictionaries. Input contracts remain strict, while server-output parsing
retains unknown fields and event envelopes so a new metadata event does not
break an older assistant.

The normal, complete-input pipeline, incremental pipeline, and multi-speaker
TTS route adapters all delegate validation and lifecycle work to `TtsRuntime`.
That runtime reuses one `PocketTtsEngine` and one set of direct-generator
`PcmPipeline` primitives. This keeps API expansion independent of audio
processing while preserving streaming flow control and avoiding additional
queues or hot-path serialization.

## Assistant

The assistant runtime combines the STT, llama.cpp, and TTS services through:

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
also accepts an optional `session.voice` in `session.update`; the value is
substituted for the configured default character's voice for that response.
Omitting it uses that character's configured voice, while `"voice": "default"`
explicitly selects the TTS service default. Multi-voice mode is enabled when
`session.multi_voice` is omitted. Sending `"multi_voice": false` skips the
multi-voice prompt instructions and tagged header, and forwards `session.voice`
through the ordinary single-voice TTS request path. The assistant forwards STT
transcription events and then emits:

```json
{"type":"session.update","session":{"voice":"bender","multi_voice":false}}
```

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

Startup issues one warm request per configured llama.cpp slot. During an
utterance, each actual incremental warm is also an HTTP request even though it
normally requests zero generated tokens; an empty response is therefore
expected and does not mean the prompt was empty. If the server rejects a
zero-token request, the client retries that warm once with the configured
one-token fallback.

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

Spoken responses use multi-voice marker mode by default. `multi_voice` defines a
default character and a non-empty map of character names to Pydantic-validated
voice and one-symbol marker settings. The built-in safe configuration has one
`assistant` character using `voice: default` and marker `§`; deployments can
add voices without changing prompt or transport code. Assistant markers cannot
use `{`, which is reserved for streamed tool-call classification:

```yaml
multi_voice:
  default_character: narrator
  characters:
    narrator: {voice: attenborough, marker: "§"}
    bandit: {voice: bender, marker: "¶"}
```

The prompt suffix explains each available character and tells the model to emit
a raw marker only when the speaker changes. The assistant selects
`default_character` itself by appending that character's marker to the generated
`<multi>` and `<char>` header, so a normal one-character response costs the LLM
no marker token. A session voice, when present, is substituted into the default
character declaration. The TTS parser therefore receives an explicit initial
speaker while the LLM spends output tokens only on actual speaker changes.
Generated text events retain model-generated change markers; the automatically
inserted initial marker exists only in the TTS stream.

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

The STT API serves NeMo FastConformer through:

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
prefix; defaults are declared in `stt/src/config.py`.

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

`TtsRuntime` keeps one Pocket TTS model and reusable voice states in memory.
The TTS API serves:

- `GET /health/live` and `GET /health/ready`
- `GET /metrics`
- `GET /config` and `PATCH /config`
- `GET /v1/models`
- `POST /v1/voices`
- `POST /v1/audio/speech`
- `POST /v1/audio/speech/pipeline`
- `POST /v1/audio/speech/multi-speaker`
- `WS /v1/audio/speech/pipeline`
- compatibility endpoints `POST /synthesize` and `POST /synthesize_stream`

`response_format=wav` returns a complete WAV file. `response_format=pcm` streams
PCM16 and includes format headers. Pocket TTS is not thread-safe, so only one
generation is admitted and concurrent requests receive `409 Conflict` instead
of waiting on its model lock. Relevant settings use the `TTS_` prefix; defaults
are declared in `tts/src/config.py`.

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

Both ordinary HTTP speech endpoints also recognize strict tagged multi-speaker
text when `input` starts exactly with `<multi>`. Tagged input always uses the
multi-speaker pipeline, independent of `X-Pipeline`; the request-level `voice`
must be omitted because every declared character owns its voice. Plain strings
remain backward compatible. A tagged request can use compact raw markers,
character tags, or both:

```text
<multi>
<char narrator voice=attenborough marker=§>
<char bandit voice=bender marker=¶>
§The story begins.
¶Not so fast.
<narrator>And so it continued.
```

`multi` and `char` are reserved names. Character names use letters, numbers,
underscores, and hyphens; a compact marker is one non-alphanumeric,
non-whitespace symbol. Declarations must precede the body and have unique names
and markers. Every body tag must name a declared character, every turn must
contain spoken text, nesting and closing tags are invalid, and unknown
declaration attributes are rejected. These rules intentionally make `<...>` in
spoken content an error rather than implementing error-tolerant HTML behavior.

The structured `POST /v1/audio/speech/multi-speaker` endpoint maps speaker names
to existing voices and streams all segments as one pipeline response:

```json
{
  "model": "kyutai/pocket-tts",
  "input": {
    "speakers": {
      "narrator": {"voice": "attenborough"},
      "bandit": {"voice": "bender"}
    },
    "segments": [
      {"speaker": "narrator", "text": "The story begins."},
      {"speaker": "bandit", "text": "Not so fast."}
    ]
  },
  "response_format": "pcm"
}
```

All declared voices are resolved before response streaming begins, so a missing
voice fails before any audio bytes are sent. Adjacent segments assigned to the
same speaker are joined before sentence segmentation. Voice changes reuse the
same request gate, pipeline, backpressure path, and latest-recording capture;
PCM streams continuously and WAV buffers the same combined PCM. Structured and
tagged inputs both normalize to the same internal multi-speaker command.

The incremental `WS /v1/audio/speech/pipeline` interface preserves overlap with
a text generator such as the assistant LLM. The client first sends a
`session.start` JSON event with an optional `voice`, followed by
`input_text.delta` events and one `input_text.done`. TTS replies with
`session.ready`, streams raw PCM as binary frames, and finishes with
`response.audio.done`. While one segment is being synthesized the server
deliberately stops reading more text; WebSocket/TCP flow control therefore
propagates backpressure to the producer without another
unbounded application queue. The exclusive TTS gate is acquired on the first
text delta and retained through completion. An initial `<multi>` selects the
same tagged parser incrementally; otherwise the WebSocket retains its existing
plain single-voice behavior. Tagged sessions must omit the request-level
`session.start.voice` selector. An incomplete client has
`pipeline_idle_timeout_seconds` to provide the next delta before its session is
closed.

Pipeline text normally splits on `.!?`. Configure this with
`pipeline_sentence_terminators`. With
`pipeline_first_segment_comma_delimiter=true`, the default, a comma may
additionally finish only the first segment so audio can begin with the opening
clause. A blank line is a paragraph boundary.

Smart chunking keeps the first available clause or sentence immediate for low
time-to-first-audio. After playback has started, it may hold later completed
sentences as text and synthesize several together, preserving Pocket TTS
context and natural cadence across their sentence boundaries. A prediction is
accepted only when the remaining emitted-PCM playback reserve covers the
estimated arrival of another sentence, TTS first-audio latency, and a safety
margin. Because TTS generation is assumed to remain faster than playback, the
budget covers first audio rather than completion of the whole queued chunk;
queued words and predicted audio duration remain part of the decision
telemetry. Paragraph boundaries always flush the held text. The WebSocket
receiver uses an absolute forced-flush deadline, so an incomplete next sentence
cannot turn the held text into an unbounded queue or wait past its playback budget.
The complete HTTP pipeline already knows which text is available, so after its
immediate first segment it groups the rest of each paragraph without a
prediction wait.

The predictor persists Welford running distributions beneath
`TTS_DATA_DIRECTORY/.pipeline-knowledge`. `voice-<voice>.json` records audio
seconds per character and word plus first-audio latency for the selected model
and voice. `llm-<id>.json` records characters and words per sentence plus
observed sentence-arrival rates. Writes are atomic and occur after eight new
observations by default, with a final write during clean shutdown. Configure
the source identity with `pipeline_smart_chunk_llm_id`. Until arrival timing has
been observed, `pipeline_smart_chunk_cold_start_speedup` assumes text is
generated three times faster than it is spoken.

`pipeline_smart_chunk_enabled` controls the behavior.
`pipeline_smart_chunk_confidence` selects the upper statistical estimate
(90 percent by default), and `pipeline_smart_chunk_safety_seconds` reserves an
additional 100 ms. `pipeline_smart_chunk_knowledge_flush_observations` controls
disk-write cadence. A missed forced-flush or first-audio prediction logs
`ID_tts_pipeline_smart_chunk_misprediction` with all effective estimates,
learned-observation counts, queued text size, playback reserve, and tuning
settings.

Every completed sentence logs `ID_tts_pipeline_smart_chunk_decision` at DEBUG
with its `queue` or `flush` decision, the reason, queued text size, and the
effective prediction values when the statistical predictor was consulted. This
also covers deterministic complete-input, paragraph, input-end, first-segment,
and disabled-smart-chunk paths. Optimistic prediction misses log
`ID_tts_pipeline_smart_chunk_misprediction` at WARNING. When a sentence arrives
early enough to prove that a conservative flush could instead have queued it,
the same event ID is logged at INFO with the counterfactual playback margin.

`pipeline_clause_pause_seconds` (40 ms by default),
`pipeline_sentence_pause_seconds` (120 ms by default), and
`pipeline_paragraph_pause_seconds` (240 ms by default) specify the target total
silence at their respective joins.
`pipeline_speaker_switch_pause_seconds` (200 ms by default) does the same when a
multi-speaker request changes speaker. It replaces rather than adds to the
ordinary sentence-boundary target. Model-generated trailing silence counts
toward each target; excess trailing silence and leading silence from the next
segment are removed while voiced PCM continues streaming immediately. A
paragraph marker received in a later WebSocket delta upgrades the still-pending
sentence tail to the paragraph target. Detection uses short-frame RMS rather
than peak amplitude. `pipeline_silence_threshold_dbfs` defaults to -43 dBFS and
is the configurable voice/noise-floor boundary.
`pipeline_speech_hysteresis_db` (6 dB by default) places the speech-resume
threshold above it, and `pipeline_speech_confirmation_seconds` (40 ms by
default) requires sustained speech before leaving silence.
`pipeline_silence_confirmation_seconds` controls when silence becomes a tail
candidate. Linear boundary fades are configured with
`pipeline_sentence_crossfade_seconds`.

Tail candidates remain reversible: if sustained speech resumes while synthesis
is ahead of playback, the candidate is emitted unchanged as an internal pause.
Irreversible clipping occurs only when model EOF confirms the terminal tail, or
when playback reaches the candidate before EOF is available. The latter logs a
warning and continued sustained speech then logs an error. The stitcher also
warns when the next segment or its first voiced frame misses the projected
playback deadline. Detection and confirmation buffers do not add PCM silence,
so setting clause pause, sentence pause, paragraph pause, and crossfade to zero
produces no configured transition floor. Speaker-switch pause can likewise be
set to zero for multi-speaker requests. The corresponding environment variables
are prefixed with `TTS_`, for example
`TTS_PIPELINE_FIRST_SEGMENT_COMMA_DELIMITER=false`. Set any transition duration
to zero to disable that component.

Set `TTS_SAVE_LATEST_WAV=true` to atomically overwrite the most recently
completed synthesized recording. `TTS_LATEST_WAV_PATH` defaults to
`/tmp/latest.wav`; point it at mounted storage when the recording must be read
after a pod replacement. A WAV response is saved verbatim. For a PCM response,
the saved WAV frame data is the exact concatenation of the PCM chunks emitted to
the TTS client; synthesis is not repeated. Pipeline requests save one stitched
recording containing all segment fades and normalized boundary silence, rather
than overwriting the file for each internal segment or speaker turn. An
interrupted stream does not replace the previous recording, and capture failures
are logged without failing the speech request.

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
