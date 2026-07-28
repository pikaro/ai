from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

STREAM_STRATEGY: Literal['cache_aware_conformer_stream_step_continuous_features'] = (
    'cache_aware_conformer_stream_step_continuous_features'
)


class _SttEvent(BaseModel):
    model_config = ConfigDict(extra='ignore')


class TranscriptionPartial(_SttEvent):
    type: Literal['conversation.item.input_audio_transcription.partial'] = (
        'conversation.item.input_audio_transcription.partial'
    )
    transcript: str
    source: Literal['cache_aware_conformer_stream_step_continuous_features'] = STREAM_STRATEGY


class TranscriptionDelta(_SttEvent):
    type: Literal['conversation.item.input_audio_transcription.delta'] = (
        'conversation.item.input_audio_transcription.delta'
    )
    delta: str
    transcript: str
    source: Literal['cache_aware_conformer_stream_step_continuous_features'] = STREAM_STRATEGY


class TranscriptionCompleted(_SttEvent):
    type: Literal['conversation.item.input_audio_transcription.completed'] = (
        'conversation.item.input_audio_transcription.completed'
    )
    transcript: str
    source: Literal['cache_aware_conformer_stream_step_continuous_features'] = STREAM_STRATEGY


SttTranscriptEvent = TranscriptionPartial | TranscriptionDelta | TranscriptionCompleted
STT_TRANSCRIPT_EVENT_ADAPTER: TypeAdapter[SttTranscriptEvent] = TypeAdapter(
    Annotated[SttTranscriptEvent, Field(discriminator='type')],
)


class SttError(_SttEvent):
    type: Literal['error'] = 'error'
    message: str


class SttSessionUpdated(_SttEvent):
    type: Literal['session.updated'] = 'session.updated'
    session: dict[str, object]


SttServerEvent = SttTranscriptEvent | SttError | SttSessionUpdated
STT_SERVER_EVENT_ADAPTER: TypeAdapter[SttServerEvent] = TypeAdapter(
    Annotated[SttServerEvent, Field(discriminator='type')],
)
_STT_SERVER_PAYLOAD_ADAPTER = TypeAdapter(dict[str, object])


def parse_stt_server_event(
    raw: str,
) -> tuple[dict[str, object], SttServerEvent | None]:
    """Return the original event plus its typed form when the protocol knows it."""
    payload = _STT_SERVER_PAYLOAD_ADAPTER.validate_json(raw)
    try:
        event = STT_SERVER_EVENT_ADAPTER.validate_python(payload)
    except ValidationError:
        event = None
    return payload, event
