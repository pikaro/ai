from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

MODEL_ID = 'kyutai/pocket-tts'


class PipelineSessionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: Literal['session.start']
    model: str = MODEL_ID
    voice: str | None = None
    speed: float = Field(default=1.0, gt=0)


class PipelineTextDelta(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: Literal['input_text.delta']
    delta: str = Field(min_length=1)


class PipelineTextDone(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: Literal['input_text.done']


PipelineInputEvent = Annotated[
    PipelineTextDelta | PipelineTextDone,
    Field(discriminator='type'),
]
PIPELINE_INPUT_ADAPTER = TypeAdapter(PipelineInputEvent)


class PipelineSessionReady(BaseModel):
    model_config = ConfigDict(extra='ignore')

    type: Literal['session.ready'] = 'session.ready'
    format: Literal['pcm16'] = 'pcm16'
    sample_rate: int = Field(gt=0)
    sample_width: int = Field(gt=0)
    channels: int = Field(gt=0)


class PipelineAudioDone(BaseModel):
    model_config = ConfigDict(extra='ignore')

    type: Literal['response.audio.done'] = 'response.audio.done'


class PipelineError(BaseModel):
    model_config = ConfigDict(extra='ignore')

    type: Literal['error'] = 'error'
    message: str


PipelineOutputEvent = Annotated[
    PipelineSessionReady | PipelineAudioDone | PipelineError,
    Field(discriminator='type'),
]
PIPELINE_OUTPUT_ADAPTER = TypeAdapter(PipelineOutputEvent)
