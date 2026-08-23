from prometheus_client import Counter, Gauge, Histogram

MODEL_READY = Gauge(
    'tts_model_ready',
    'Whether the TTS model and default voice are loaded and ready',
    ['model'],
)
MODEL_LOAD_SECONDS = Gauge(
    'tts_model_load_seconds',
    'Time spent loading the TTS model',
    ['model'],
)
VOICE_LOAD_SECONDS = Gauge(
    'tts_voice_load_seconds',
    'Time spent loading a TTS voice',
    ['model'],
)
ACTIVE_REQUESTS = Gauge('tts_active_requests', 'Active TTS inference requests')
BUSY_REJECTIONS = Counter(
    'tts_busy_rejections_total',
    'TTS inference and configuration requests rejected instead of queued',
)
CONFIGURATION_UPDATES = Counter(
    'tts_configuration_updates_total',
    'Successful ephemeral TTS configuration updates',
)
REQUESTS = Counter('tts_requests_total', 'Completed TTS requests', ['format', 'outcome'])
REQUEST_SECONDS = Histogram('tts_request_duration_seconds', 'TTS request latency', ['format'])
TIME_TO_FIRST_AUDIO = Histogram(
    'tts_time_to_first_audio_seconds',
    'Time from a PCM synthesis request to its first audio bytes',
)
AUDIO_SECONDS = Histogram('tts_output_audio_seconds', 'Audio duration produced by TTS')
REALTIME_FACTOR = Histogram(
    'tts_realtime_factor',
    'TTS generation wall time divided by output audio duration',
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 5),
)
PIPELINE_REQUESTS = Counter(
    'tts_pipeline_requests_total',
    'Completed server-side text segmentation and stitching pipelines',
    ['transport', 'outcome'],
)
PIPELINE_SEGMENTS = Counter(
    'tts_pipeline_segments_total',
    'Text segments synthesized by the TTS pipeline',
    ['transport'],
)
