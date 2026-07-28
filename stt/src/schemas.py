from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from service_contracts.realtime import AudioInputSessionOptions, RealtimeAudioInputEvent
from service_contracts.stt import STREAM_STRATEGY


class HealthResponse(BaseModel):
    status: Literal['ok']
    backend: Literal['nemo'] = 'nemo'
    model: str
    device: str
    decoder_type: str
    attention_context_size: tuple[int, int] | None
    streaming: Literal[True] = True
    stream_endpoint: Literal['/v1/realtime'] = '/v1/realtime'
    stream_strategy: Literal['cache_aware_conformer_stream_step_continuous_features'] = (
        STREAM_STRATEGY
    )
    sample_rate: int | None
    load_seconds: float


class TranscriptionResponse(BaseModel):
    text: str


SessionOptions = AudioInputSessionOptions
RealtimeEvent = RealtimeAudioInputEvent[SessionOptions]
