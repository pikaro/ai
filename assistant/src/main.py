from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import urlsplit, urlunsplit

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.websockets import WebSocketState
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from assistant.src import metrics
from assistant.src.config import Settings
from assistant.src.pipeline import AssistantUtterance
from assistant.src.tooling import ToolRegistry, discover_local_tools
from assistant.src.upstream import LlmClient, SlotPool, TtsClient, upstream_health

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from assistant.src.tooling import ToolDefinition

LOGGER = logging.getLogger('assistant')


class SessionOptions(BaseModel):
    model_config = ConfigDict(extra='ignore')

    input_audio_sample_rate: int | None = Field(default=None, ge=8_000)
    input_audio_channels: int | None = Field(default=None, ge=1, le=2)


class RealtimeEvent(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: Literal[
        'session.update',
        'input_audio_buffer.append',
        'input_audio_buffer.commit',
    ]
    session: SessionOptions | None = None
    audio: str | None = None


class HealthResponse(BaseModel):
    status: Literal['ok']
    model: str
    streaming: Literal[True] = True
    stream_endpoint: Literal['/v1/realtime'] = '/v1/realtime'
    upstream: dict[str, bool]
    llm_slots: int
    local_tools: int
    enabled_mcp_servers: int


class ModelDescription(BaseModel):
    id: str
    object: Literal['model'] = 'model'
    owned_by: Literal['local'] = 'local'


class ModelList(BaseModel):
    object: Literal['list'] = 'list'
    data: list[ModelDescription]


class AssistantRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        timeout = httpx.Timeout(
            settings.request_timeout_seconds,
            connect=settings.connect_timeout_seconds,
        )
        self.http = httpx.AsyncClient(timeout=timeout)
        self.slots = SlotPool(settings.llm_slots)
        self.llm = LlmClient(self.http, settings)
        self.tts = TtsClient(self.http, settings)
        self.tools = ToolRegistry(
            discover_local_tools(),
            settings.mcp,
            default_timezone=settings.default_timezone,
            maximum_result_characters=settings.maximum_tool_result_characters,
        )

    async def close(self) -> None:
        await self.http.aclose()

    async def health(self) -> HealthResponse:
        upstream = await upstream_health(self.http, self.settings)
        unhealthy = [name for name, healthy in upstream.items() if not healthy]
        if unhealthy:
            LOGGER.warning('upstream services unhealthy: %s', ', '.join(unhealthy))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={'status': 'unavailable', 'upstream': upstream, 'unhealthy': unhealthy},
            )
        return HealthResponse(
            status='ok',
            model=self.settings.model_id,
            upstream=upstream,
            llm_slots=len(self.settings.llm_slots),
            local_tools=self.tools.local_tool_count,
            enabled_mcp_servers=sum(config.enabled for config in self.settings.mcp.values()),
        )

    def utterance(self) -> AssistantUtterance:
        return AssistantUtterance(self.settings, self.slots, self.llm, self.tts, self.tools)

    @property
    def stt_websocket_url(self) -> str:
        parsed = urlsplit(self.settings.stt_base_url.rstrip('/'))
        scheme = 'wss' if parsed.scheme == 'https' else 'ws'
        return urlunsplit((scheme, parsed.netloc, '/v1/realtime', '', ''))


class EventSink:
    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.lock = asyncio.Lock()

    async def __call__(self, payload: dict[str, object]) -> None:
        async with self.lock:
            await self.websocket.send_json(payload)


SETTINGS = Settings()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
    runtime = AssistantRuntime(SETTINGS)
    application.state.runtime = runtime
    try:
        yield
    finally:
        await runtime.close()


app = FastAPI(title='Assistant', version='1.0.0', lifespan=lifespan)


def _runtime_from_request(request: Request) -> AssistantRuntime:
    return cast('AssistantRuntime', request.app.state.runtime)


def _runtime_from_websocket(websocket: WebSocket) -> AssistantRuntime:
    return cast('AssistantRuntime', websocket.app.state.runtime)


@app.get('/health/live')
async def live() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/metrics', include_in_schema=False)
async def prometheus_metrics() -> Response:
    return Response(generate_latest(), headers={'Content-Type': CONTENT_TYPE_LATEST})


@app.get('/health', response_model=HealthResponse)
@app.get('/health/ready', response_model=HealthResponse)
async def ready(request: Request) -> HealthResponse:
    return await _runtime_from_request(request).health()


@app.get('/v1/models', response_model=ModelList)
async def models(request: Request) -> ModelList:
    runtime = _runtime_from_request(request)
    return ModelList(data=[ModelDescription(id=runtime.settings.model_id)])


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
        await stt.send(event.model_dump_json(exclude_none=True))
        if event.type == 'input_audio_buffer.append' and event.audio:
            padding = len(event.audio) - len(event.audio.rstrip('='))
            utterance.note_audio(max(0, len(event.audio) * 3 // 4 - padding))
        if event.type == 'input_audio_buffer.commit':
            return


async def _consume_transcription(  # noqa: C901, PLR0912
    stt: ClientConnection,
    utterance: AssistantUtterance,
    send: EventSink,
) -> tuple[str, list[ToolDefinition]]:
    transcript = ''
    selected_tools = []
    async for raw in stt:
        if not isinstance(raw, str):
            continue
        try:
            message = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(message, dict):
            continue
        await send(message)
        message_type = str(message.get('type', ''))
        if message_type == 'error':
            raise RuntimeError(str(message.get('message', 'STT request failed')))
        if message_type.endswith('transcription.delta'):
            metrics.TRANSCRIPTION_DELTAS.inc()
            now = time.perf_counter()
            if utterance.first_delta_at is None:
                utterance.first_delta_at = now
                if utterance.first_audio_at is not None:
                    metrics.PIPELINE_SECONDS.labels(stage='stt_first_delta').observe(
                        now - utterance.first_audio_at,
                    )
            current = message.get('transcript')
            if isinstance(current, str) and current.strip():
                transcript = current.strip()
            else:
                delta = message.get('delta')
                if isinstance(delta, str):
                    transcript = f'{transcript} {delta}'.strip()
            if transcript:
                selected_tools = await utterance.select_and_warm(transcript, reason='delta')
        elif message_type.endswith('transcription.completed'):
            completed = message.get('transcript')
            if isinstance(completed, str) and completed.strip():
                transcript = completed.strip()
            break
    if not transcript:
        message = 'STT produced an empty transcript'
        raise RuntimeError(message)
    selected_tools = await utterance.select_and_warm(transcript, reason='final')
    return transcript, selected_tools


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
        except BaseException:
            _ = forward_task.cancel()
            _ = transcription_task.cancel()
            _ = await asyncio.gather(
                forward_task,
                transcription_task,
                return_exceptions=True,
            )
            raise
        transcript, selected_tools = transcription_task.result()
    await utterance.generate(transcript, selected_tools, send)


@app.websocket('/v1/realtime')
async def realtime(websocket: WebSocket) -> None:
    runtime = _runtime_from_websocket(websocket)
    utterance = runtime.utterance()
    outcome = 'success'
    await websocket.accept()
    metrics.ACTIVE_SESSIONS.inc()
    try:
        await _run_realtime_session(websocket, runtime, utterance)
        await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
    except (WebSocketDisconnect, ConnectionClosed):
        outcome = 'disconnected'
        LOGGER.info('realtime assistant client disconnected')
    except Exception:
        outcome = 'error'
        LOGGER.exception('realtime assistant session failed')
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({'type': 'error', 'message': 'assistant request failed'})
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
    finally:
        await utterance.close()
        metrics.ACTIVE_SESSIONS.dec()
        metrics.SESSIONS.labels(outcome=outcome).inc()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host='0.0.0.0', port=SETTINGS.listen_port)  # noqa: S104
