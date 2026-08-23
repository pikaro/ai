from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from service_contracts.realtime import AudioInputSessionOptions, RealtimeAudioInputEvent


class SessionOptions(AudioInputSessionOptions):
    voice: str | None = Field(default=None, min_length=1)
    multi_voice: bool | None = None


RealtimeEvent = RealtimeAudioInputEvent[SessionOptions]


class DashboardSpeechRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    text: str = Field(min_length=1)
    model: str | None = Field(default=None, min_length=1)
    voice: str | None = Field(default=None, min_length=1)
    pipeline: bool = False
    response_format: Literal['pcm', 'wav'] = 'wav'


class HealthResponse(BaseModel):
    status: Literal['ok']
    model: str
    streaming: Literal[True] = True
    stream_endpoint: Literal['/v1/realtime'] = '/v1/realtime'
    upstream: dict[str, bool]
    llm_slots: int
    local_tools: int
    enabled_mcp_servers: int
