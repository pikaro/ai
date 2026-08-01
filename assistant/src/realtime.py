from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Never

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
    status,
)
from pydantic import ValidationError
from starlette.websockets import WebSocketState
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from assistant.src import metrics
from assistant.src.dependencies import runtime_from_websocket
from assistant.src.schemas import RealtimeEvent
from service_contracts.stt import (
    SttError,
    TranscriptionCompleted,
    TranscriptionDelta,
    parse_stt_server_event,
)

if TYPE_CHECKING:
    from starlette.datastructures import Headers

    from assistant.src.pipeline import AssistantUtterance
    from assistant.src.runtime import AssistantRuntime
    from assistant.src.tooling import ToolDefinition

LOGGER = logging.getLogger('assistant')
router = APIRouter()


class SttEmptyTranscriptError(RuntimeError):
    """Raised when the STT service produces an empty transcript."""


class EventSink:
    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.lock = asyncio.Lock()

    async def __call__(self, payload: dict[str, object]) -> None:
        async with self.lock:
            await self.websocket.send_json(payload)


def log_request_correlation(headers: Headers) -> None:
    """Log optional client correlation headers before starting request work."""
    request_id = headers.get('x-request-id')
    request_timestamp = headers.get('x-request-timestamp')
    if request_id is None and request_timestamp is None:
        return
    LOGGER.info(
        'Assistant request correlation received',
        extra={
            'event_id': 'ID_assistant_request_correlation_received',
            'request_id': request_id,
            'request_timestamp': request_timestamp,
        },
    )


async def _forward_client_audio(  # noqa: C901
    websocket: WebSocket,
    stt: ClientConnection,
    utterance: AssistantUtterance,
    send: EventSink,
    maximum_message_bytes: int,
) -> None:
    while True:
        raw = await websocket.receive_text()
        if len(raw.encode()) > maximum_message_bytes:
            await websocket.close(code=status.WS_1009_MESSAGE_TOO_BIG)
            raise WebSocketDisconnect(code=status.WS_1009_MESSAGE_TOO_BIG)
        try:
            event = RealtimeEvent.model_validate_json(raw)
        except ValidationError as error:
            await send(
                {'type': 'error', 'message': f'invalid event: {error.errors(include_url=False)}'},
            )
            continue
        if event.type == 'session.update' and event.session is not None:
            if event.session.voice is not None:
                utterance.select_voice(event.session.voice)
            if event.session.multi_voice is not None:
                utterance.set_multi_voice(enabled=event.session.multi_voice)
        await stt.send(event.model_dump_json(exclude_none=True))
        if event.type == 'input_audio_buffer.append' and event.audio:
            padding = len(event.audio) - len(event.audio.rstrip('='))
            utterance.note_audio(max(0, len(event.audio) * 3 // 4 - padding))
        if event.type == 'input_audio_buffer.commit':
            return


def _raise_stt_error(message: str) -> Never:
    raise RuntimeError(message)


async def _consume_transcription(  # noqa: C901, PLR0912
    stt: ClientConnection,
    utterance: AssistantUtterance,
    send: EventSink,
) -> tuple[str, list[ToolDefinition]]:
    transcript = ''
    async for raw in stt:
        if not isinstance(raw, str):
            continue
        try:
            payload, message = parse_stt_server_event(raw)
        except ValidationError:
            continue
        received_at = time.perf_counter()
        message_type = str(payload.get('type', ''))
        if message_type.endswith('transcription.completed'):
            utterance.note_final_transcript(received_at)
        await send(payload)
        if isinstance(message, SttError):
            _raise_stt_error(message.message)
        if message_type == 'error':
            _raise_stt_error(str(payload.get('message', 'STT request failed')))
        if isinstance(message, TranscriptionDelta) or message_type.endswith(
            'transcription.delta',
        ):
            metrics.TRANSCRIPTION_DELTAS.inc()
            now = received_at
            if utterance.first_delta_at is None:
                utterance.first_delta_at = now
                if utterance.first_audio_at is not None:
                    metrics.PIPELINE_SECONDS.labels(stage='stt_first_delta').observe(
                        now - utterance.first_audio_at,
                    )
            if isinstance(message, TranscriptionDelta):
                current = message.transcript
                delta = message.delta
            else:
                current = payload.get('transcript')
                delta = payload.get('delta')
            if isinstance(current, str) and current.strip():
                transcript = current.strip()
            elif isinstance(delta, str):
                transcript = f'{transcript} {delta}'.strip()
            if transcript:
                cache_warm_disposition = utterance.schedule_cache_warm(transcript)
                LOGGER.debug(
                    'STT transcription delta received',
                    extra={
                        'event_id': 'ID_assistant_stt_transcription_delta_received',
                        'stage': 'update',
                        'characters': len(transcript),
                        'delta_characters': len(delta) if isinstance(delta, str) else 0,
                        'cache_warm_disposition': cache_warm_disposition,
                        'delta': delta,
                        'transcript': transcript,
                    },
                )
        elif isinstance(message, TranscriptionCompleted) or message_type.endswith(
            'transcription.completed',
        ):
            completed = (
                message.transcript
                if isinstance(message, TranscriptionCompleted)
                else payload.get('transcript')
            )
            if isinstance(completed, str) and completed.strip():
                transcript = completed.strip()
            if transcript:
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(
                        'STT transcription',
                        extra={
                            'event_id': 'ID_assistant_stt_transcription',
                            'stage': 'completed',
                            'characters': len(transcript),
                            'transcript': transcript,
                        },
                    )
                else:
                    LOGGER.info(
                        'STT transcription received',
                        extra={
                            'event_id': 'ID_assistant_stt_transcription_received',
                            'stage': 'completed',
                            'characters': len(transcript),
                        },
                    )
            break
    if not transcript:
        raise SttEmptyTranscriptError
    utterance.note_final_transcript()
    available_tools = await utterance.finalize_cache_warming(transcript)
    return transcript, available_tools


async def _run_realtime_session(
    websocket: WebSocket,
    runtime: AssistantRuntime,
    utterance: AssistantUtterance,
) -> None:
    send = EventSink(websocket)
    async with connect(
        runtime.stt_websocket_url,
        open_timeout=runtime.settings.connect_timeout_seconds,
        close_timeout=runtime.settings.connect_timeout_seconds,
        max_size=runtime.settings.maximum_websocket_message_bytes,
    ) as stt:
        forward_task = asyncio.create_task(
            _forward_client_audio(
                websocket,
                stt,
                utterance,
                send,
                runtime.settings.maximum_websocket_message_bytes,
            ),
        )
        transcription_task = asyncio.create_task(_consume_transcription(stt, utterance, send))
        try:
            _ = await asyncio.gather(forward_task, transcription_task)
        except SttEmptyTranscriptError as error:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason='STT produced empty transcript',
            ) from error
        except BaseException:
            _ = forward_task.cancel()
            _ = transcription_task.cancel()
            _ = await asyncio.gather(
                forward_task,
                transcription_task,
                return_exceptions=True,
            )
            raise
        transcript, available_tools = transcription_task.result()
    await utterance.generate(transcript, available_tools, send)


@router.websocket('/v1/realtime')
async def realtime(websocket: WebSocket) -> None:  # noqa: C901
    log_request_correlation(websocket.headers)
    runtime = runtime_from_websocket(websocket)
    if not runtime.operations.try_acquire():
        metrics.SESSION_REJECTIONS.inc()
        await websocket.accept()
        await websocket.send_json(
            {'type': 'error', 'message': 'another assistant session is already active'},
        )
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return
    try:
        utterance = runtime.utterance()
    except BaseException:
        runtime.operations.release()
        raise
    outcome = 'success'
    session_started = False
    try:
        await websocket.accept()
        LOGGER.info(
            'realtime assistant session started',
            extra={'event_id': 'ID_assistant_realtime_session_started'},
        )
        metrics.ACTIVE_SESSIONS.inc()
        session_started = True
        await _run_realtime_session(websocket, runtime, utterance)
        await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
        LOGGER.info(
            'realtime assistant session completed',
            extra={'event_id': 'ID_assistant_realtime_session_completed'},
        )
    except WebSocketDisconnect as error:
        outcome = 'disconnected'
        LOGGER.info(
            'Realtime assistant client disconnected',
            extra={'event_id': 'ID_assistant_realtime_client_disconnected', 'code': error.code},
        )
    except ConnectionClosed as error:
        outcome = 'upstream_disconnected'
        LOGGER.warning(
            'Realtime STT connection closed',
            extra={
                'event_id': 'ID_assistant_stt_connection_closed',
                'code': error.code,
            },
        )
    except WebSocketException as error:
        outcome = 'rejected'
        LOGGER.info(
            'Realtime assistant session rejected',
            extra={
                'event_id': 'ID_assistant_realtime_session_rejected',
                'code': error.code,
            },
        )
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=error.code, reason=error.reason)
    except Exception:
        outcome = 'error'
        LOGGER.exception(
            'realtime assistant session failed',
            extra={'event_id': 'ID_assistant_realtime_session_failed'},
        )
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({'type': 'error', 'message': 'assistant request failed'})
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
    finally:
        try:
            await utterance.close()
        finally:
            runtime.operations.release()
            if session_started:
                metrics.ACTIVE_SESSIONS.dec()
                metrics.SESSIONS.labels(outcome=outcome).inc()
