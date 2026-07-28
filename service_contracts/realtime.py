from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AudioInputSessionOptions(BaseModel):
    """Audio properties shared by realtime STT and assistant inputs."""

    model_config = ConfigDict(extra='ignore')

    input_audio_sample_rate: int | None = Field(default=None, ge=8_000)
    input_audio_channels: int | None = Field(default=None, ge=1, le=2)


class RealtimeAudioInputEvent[SessionOptionsT: AudioInputSessionOptions](BaseModel):
    """Permissive event envelope retained for realtime API compatibility."""

    model_config = ConfigDict(extra='forbid')

    type: Literal[
        'session.update',
        'input_audio_buffer.append',
        'input_audio_buffer.commit',
    ]
    session: SessionOptionsT | None = None
    audio: str | None = None
