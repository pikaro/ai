from prometheus_client import Counter, Gauge, Histogram

SESSIONS = Counter(
    'assistant_sessions_total',
    'Completed assistant WebSocket sessions',
    ['outcome'],
)
ACTIVE_SESSIONS = Gauge('assistant_active_sessions', 'Active assistant WebSocket sessions')
SLOT_WAITERS = Gauge('assistant_llm_slot_waiters', 'Sessions waiting for a llama.cpp slot')
SLOTS_IN_USE = Gauge('assistant_llm_slots_in_use', 'Leased llama.cpp slots')
SLOT_WAIT_SECONDS = Histogram(
    'assistant_llm_slot_wait_duration_seconds',
    'Time an assistant session waits for a llama.cpp slot',
)

AUDIO_INPUT_BYTES = Counter('assistant_audio_input_bytes_total', 'PCM bytes received from clients')
AUDIO_OUTPUT_BYTES = Counter('assistant_audio_output_bytes_total', 'PCM bytes sent to clients')
TRANSCRIPTION_DELTAS = Counter(
    'assistant_stt_deltas_total',
    'Stable transcription deltas received from STT',
)
TRANSCRIPT_CHARACTERS = Histogram(
    'assistant_transcript_characters',
    'Final transcript size in characters',
    buckets=(16, 32, 64, 128, 256, 512, 1024, 2048),
)

CACHE_WARMS = Counter(
    'assistant_llm_cache_warm_requests_total',
    'Incremental llama.cpp prompt-cache warm requests',
    ['reason', 'outcome'],
)
CACHE_WARM_SECONDS = Histogram(
    'assistant_llm_cache_warm_duration_seconds',
    'Incremental llama.cpp prompt-cache warm latency',
    ['reason'],
)
CACHE_WARMS_PER_SESSION = Histogram(
    'assistant_llm_cache_warms_per_session',
    'Prompt-cache warm requests issued per assistant session',
    buckets=(0, 1, 2, 3, 5, 8, 13, 21, 34, 55),
)
CACHE_PROMPT_CHARACTERS = Histogram(
    'assistant_llm_cache_prompt_characters',
    'Prompt size submitted to llama.cpp cache warming',
    buckets=(128, 256, 512, 1024, 2048, 4096, 8192, 16384),
)
CACHE_TOOLSET_CHANGES = Counter(
    'assistant_llm_cache_toolset_changes_total',
    'Cache prompt revisions caused by newly triggered tools',
)

LLM_REQUESTS = Counter(
    'assistant_llm_requests_total',
    'Requests to llama.cpp',
    ['operation', 'outcome'],
)
LLM_REQUEST_SECONDS = Histogram(
    'assistant_llm_request_duration_seconds',
    'llama.cpp request latency',
    ['operation'],
)
LLM_TIME_TO_FIRST_TOKEN = Histogram(
    'assistant_llm_time_to_first_token_seconds',
    'Final transcript to first spoken response token',
)
LLM_OUTPUT_CHARACTERS = Counter(
    'assistant_llm_output_characters_total',
    'Characters generated for spoken responses',
)
LLM_TOKENS = Histogram(
    'assistant_llm_tokens',
    'llama.cpp tokens reported per request',
    ['kind'],
    buckets=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
)
LLM_TOKENS_PER_SECOND = Histogram(
    'assistant_llm_tokens_per_second',
    'llama.cpp prompt processing and decode throughput',
    ['phase'],
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
)
LLM_CACHE_REUSE_RATIO = Histogram(
    'assistant_llm_cache_reuse_ratio',
    'Fraction of prompt tokens reported as cached by llama.cpp',
    buckets=(0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0),
)

TTS_REQUESTS = Counter('assistant_tts_requests_total', 'Requests to TTS', ['outcome'])
TTS_REQUEST_SECONDS = Histogram(
    'assistant_tts_request_duration_seconds',
    'TTS request streaming duration',
)
TTS_TIME_TO_FIRST_AUDIO = Histogram(
    'assistant_tts_time_to_first_audio_seconds',
    'Final transcript to first response audio bytes',
)
TTS_AUDIO_SECONDS = Histogram(
    'assistant_tts_audio_seconds',
    'Audio duration produced per TTS segment',
)
TTS_REALTIME_FACTOR = Histogram(
    'assistant_tts_realtime_factor',
    'TTS generation wall time divided by output audio duration',
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 5),
)

TOOLS_SELECTED = Counter(
    'assistant_tools_selected_total',
    'Tools selected by prompt triggers',
    ['source'],
)
TOOL_CALLS = Counter(
    'assistant_tool_calls_total',
    'Assistant tool calls',
    ['source', 'tool', 'outcome'],
)
TOOL_CALL_SECONDS = Histogram(
    'assistant_tool_call_duration_seconds',
    'Assistant tool call duration',
    ['source', 'tool'],
)
MCP_REQUESTS = Counter(
    'assistant_mcp_requests_total',
    'MCP discovery and tool requests',
    ['server', 'operation', 'outcome'],
)
MCP_REQUEST_SECONDS = Histogram(
    'assistant_mcp_request_duration_seconds',
    'MCP request duration',
    ['server', 'operation'],
)

PIPELINE_SECONDS = Histogram(
    'assistant_pipeline_stage_duration_seconds',
    'End-to-end assistant stage latency',
    ['stage'],
)
UPSTREAM_READY = Gauge(
    'assistant_upstream_ready',
    'Whether a required upstream service passed its latest readiness check',
    ['service'],
)
