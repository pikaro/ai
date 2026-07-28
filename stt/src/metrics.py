from prometheus_client import Counter, Gauge, Histogram

MODEL_READY = Gauge('stt_model_ready', 'Whether the STT model is loaded and ready')
MODEL_LOAD_SECONDS = Gauge('stt_model_load_seconds', 'Time spent loading the STT model')
ACTIVE_REQUESTS = Gauge('stt_active_requests', 'Active STT inference requests')
BUSY_REJECTIONS = Counter(
    'stt_busy_rejections_total',
    'STT inference and configuration requests rejected instead of queued',
)
CONFIGURATION_UPDATES = Counter(
    'stt_configuration_updates_total',
    'Successful ephemeral STT configuration updates',
)
REQUESTS = Counter('stt_requests_total', 'Completed STT requests', ['mode', 'outcome'])
REQUEST_SECONDS = Histogram('stt_request_duration_seconds', 'STT request latency', ['mode'])
STREAM_CHUNK_SECONDS = Histogram(
    'stt_stream_chunk_processing_duration_seconds',
    'Model processing latency for one realtime audio update',
)
STREAM_COMMIT_SECONDS = Histogram(
    'stt_stream_commit_duration_seconds',
    'Model processing latency after realtime audio commit',
)
STREAM_TIME_TO_FIRST_DELTA = Histogram(
    'stt_stream_time_to_first_delta_seconds',
    'Time from first realtime audio bytes to first stable transcript delta',
)
STREAM_AUDIO_SECONDS = Histogram(
    'stt_stream_input_audio_seconds',
    'Client audio duration submitted to a realtime STT session',
)
