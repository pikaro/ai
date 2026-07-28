from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from service_contracts.tts import MODEL_ID


class HealthResponse(BaseModel):
    status: Literal['ok']
    model: str
    voice: str
    language: str
    streaming: Literal[True] = True
    stream_endpoint: Literal['/v1/audio/speech'] = '/v1/audio/speech'
    pipeline_endpoint: Literal['/v1/audio/speech/pipeline'] = '/v1/audio/speech/pipeline'
    multi_speaker_endpoint: Literal['/v1/audio/speech/multi-speaker'] = (
        '/v1/audio/speech/multi-speaker'
    )
    sample_rate: int
    load_seconds: float
    voice_load_seconds: float


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    model: str = MODEL_ID
    input: str = Field(min_length=1)
    voice: str | None = None
    response_format: Literal['pcm', 'wav'] = 'wav'
    speed: float = Field(default=1.0, gt=0)


class SpeakerDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid')

    voice: str = Field(min_length=1)


class SpeakerSegment(BaseModel):
    model_config = ConfigDict(extra='forbid')

    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)


class MultiSpeakerInput(BaseModel):
    model_config = ConfigDict(extra='forbid')

    speakers: dict[str, SpeakerDefinition] = Field(min_length=1)
    segments: list[SpeakerSegment] = Field(min_length=1)


class MultiSpeakerSpeechRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    model: str = MODEL_ID
    input: MultiSpeakerInput
    response_format: Literal['pcm', 'wav'] = 'wav'
    speed: float = Field(default=1.0, gt=0)


class LegacySpeechRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    text: str = Field(min_length=1)


class VoiceUploadResponse(BaseModel):
    name: str
    filename: str
    replaced: bool
