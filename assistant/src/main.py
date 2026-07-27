from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal, cast
from urllib.parse import urlsplit, urlunsplit

import httpx
import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
    status,
)
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError
from starlette.websockets import WebSocketState
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, WebSocketException as UpstreamWebSocketException

from assistant.src import metrics
from assistant.src.config import Settings
from assistant.src.pipeline import (
    AssistantUtterance,
    SystemPromptFile,
    build_prompt_prefix,
    warm_llm_cache,
)
from assistant.src.tooling import ToolRegistry, discover_local_tools
from assistant.src.upstream import LlmClient, SlotPool, TtsClient, upstream_health
from runtime_config import (
    ConfigurationUpdateResponse,
    ExclusiveOperationGate,
    reject_if_busy,
    validated_settings_patch,
)
from service_logging import configure_logging

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from starlette.datastructures import Headers

    from assistant.src.tooling import ToolDefinition

LOGGER = logging.getLogger('assistant')
UPSTREAM_KEEPALIVE_EXPIRY_SECONDS: Final = 4.0
RESTART_REQUIRED_SETTINGS: Final = frozenset({'listen_port'})
DASHBOARD_PATH: Final = Path(__file__).with_name('dashboard.html')
DASHBOARD_PCM_CHUNK_BYTES: Final = 64 * 1024


class SttEmptyTranscriptError(RuntimeError):
    """Raised when the STT service produces an empty transcript."""


class SessionOptions(BaseModel):
    model_config = ConfigDict(extra='ignore')

    input_audio_sample_rate: int | None = Field(default=None, ge=8_000)
    input_audio_channels: int | None = Field(default=None, ge=1, le=2)
    voice: str | None = Field(default=None, min_length=1)


class RealtimeEvent(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: Literal[
        'session.update',
        'input_audio_buffer.append',
        'input_audio_buffer.commit',
    ]
    session: SessionOptions | None = None
    audio: str | None = None


class DashboardSpeechRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    text: str = Field(min_length=1)
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


class ModelDescription(BaseModel):
    id: str
    object: Literal['model'] = 'model'
    owned_by: Literal['local'] = 'local'


class ModelList(BaseModel):
    object: Literal['list'] = 'list'
    data: list[ModelDescription]


class AssistantRuntime:
    def __init__(
        self,
        settings: Settings,
        operations: ExclusiveOperationGate | None = None,
    ) -> None:
        self.settings = settings
        self.operations = operations or ExclusiveOperationGate()
        timeout = httpx.Timeout(
            settings.request_timeout_seconds,
            connect=settings.connect_timeout_seconds,
        )
        # Every deployed upstream closes idle HTTP connections after five seconds.
        # Retire pooled connections first to avoid racing a peer-initiated close.
        limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=UPSTREAM_KEEPALIVE_EXPIRY_SECONDS,
        )
        self.http = httpx.AsyncClient(timeout=timeout, limits=limits)
        self.slots = SlotPool(settings.llm_slots)
        self.llm = LlmClient(self.http, settings)
        self.tts = TtsClient(self.http, settings)
        self.tools = ToolRegistry(
            discover_local_tools(),
            settings.mcp,
            default_timezone=settings.default_timezone,
            maximum_result_characters=settings.maximum_tool_result_characters,
        )
        self.system_prompt = SystemPromptFile(settings.system_prompt_path)

    async def start(self) -> None:
        """Warm every configured LLM slot before the application becomes ready."""
        available_tools = await self.tools.available()
        prompt = build_prompt_prefix(
            '',
            available_tools,
            system_prompt=self.system_prompt.read(),
        )
        for slot in self.settings.llm_slots:
            await warm_llm_cache(self.llm, prompt, slot, reason='startup')

    async def close(self) -> None:
        await self.http.aclose()

    async def health(self) -> HealthResponse:
        upstream = await upstream_health(self.http, self.settings)
        unhealthy = [name for name, healthy in upstream.items() if not healthy]
        if unhealthy:
            LOGGER.warning(
                'Upstream services unhealthy',
                extra={
                    'event_id': 'ID_assistant_upstreams_unhealthy',
                    'unhealthy_services': unhealthy,
                    'upstream': upstream,
                },
            )
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
        return AssistantUtterance(
            self.settings,
            self.slots,
            self.llm,
            self.tts,
            self.tools,
            self.system_prompt,
        )

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
configure_logging(SETTINGS.log_level, 'assistant')


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
    runtime = AssistantRuntime(SETTINGS)
    application.state.runtime = runtime
    try:
        await runtime.start()
        yield
    finally:
        await cast('AssistantRuntime', application.state.runtime).close()


app = FastAPI(title='Assistant', version='1.0.0', lifespan=lifespan)


def _runtime_from_request(request: Request) -> AssistantRuntime:
    return cast('AssistantRuntime', request.app.state.runtime)


def _runtime_from_websocket(websocket: WebSocket) -> AssistantRuntime:
    return cast('AssistantRuntime', websocket.app.state.runtime)


def _upstream_configuration_url(
    settings: Settings,
    service: Literal['stt', 'tts'],
) -> str:
    base_url = settings.stt_base_url if service == 'stt' else settings.tts_base_url
    return f'{base_url.rstrip("/")}/config'


async def _proxy_configuration_request(
    runtime: AssistantRuntime,
    service: Literal['stt', 'tts'],
    method: Literal['GET', 'PATCH'],
    patch: dict[str, object] | None = None,
) -> Response:
    url = _upstream_configuration_url(runtime.settings, service)
    try:
        if patch is None:
            upstream_response = await runtime.http.request(method, url)
        else:
            upstream_response = await runtime.http.request(method, url, json=patch)
    except httpx.RequestError as error:
        LOGGER.warning(
            'Configuration proxy request failed',
            extra={
                'event_id': 'ID_assistant_configuration_proxy_failed',
                'service': service,
                'error_type': type(error).__name__,
                'error': str(error),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f'{service} configuration request failed',
        ) from error

    return _forward_upstream_response(upstream_response)


def _forward_upstream_headers(upstream_response: httpx.Response) -> dict[str, str]:
    return {
        name: value
        for name in (
            'content-type',
            'retry-after',
            'x-audio-format',
            'x-audio-sample-rate',
            'x-audio-sample-width',
            'x-audio-channels',
        )
        if (value := upstream_response.headers.get(name)) is not None
    }


def _forward_upstream_response(upstream_response: httpx.Response) -> Response:
    return Response(
        upstream_response.content,
        status_code=upstream_response.status_code,
        headers=_forward_upstream_headers(upstream_response),
    )


async def _stream_upstream_response(
    upstream_response: httpx.Response,
) -> AsyncGenerator[bytes]:
    try:
        async for chunk in upstream_response.aiter_raw():
            yield chunk
    finally:
        await upstream_response.aclose()


def _update_dashboard_transcript(  # noqa: C901
    transcript: str,
    message: dict[str, object],
) -> tuple[str, bool]:
    message_type = str(message.get('type', ''))
    if message_type == 'error':
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(message.get('message', 'STT request failed')),
        )
    if message_type.endswith('transcription.delta'):
        current = message.get('transcript')
        if isinstance(current, str) and current.strip():
            return current.strip(), False
        delta = message.get('delta')
        if isinstance(delta, str):
            return f'{transcript} {delta}'.strip(), False
    if message_type.endswith('transcription.completed'):
        completed = message.get('transcript')
        if isinstance(completed, str) and completed.strip():
            transcript = completed.strip()
        return transcript, True
    return transcript, False


async def _transcribe_dashboard_pcm(  # noqa: C901
    runtime: AssistantRuntime,
    pcm: bytes,
    sample_rate: int,
    channels: int,
) -> str:
    transcript = ''
    try:
        async with asyncio.timeout(runtime.settings.request_timeout_seconds):
            async with connect(
                runtime.stt_websocket_url,
                open_timeout=runtime.settings.connect_timeout_seconds,
                close_timeout=runtime.settings.connect_timeout_seconds,
                max_size=runtime.settings.maximum_websocket_message_bytes,
            ) as stt:
                await stt.send(
                    RealtimeEvent(
                        type='session.update',
                        session=SessionOptions(
                            input_audio_sample_rate=sample_rate,
                            input_audio_channels=channels,
                        ),
                    ).model_dump_json(exclude_none=True),
                )
                for offset in range(0, len(pcm), DASHBOARD_PCM_CHUNK_BYTES):
                    encoded = base64.b64encode(
                        pcm[offset : offset + DASHBOARD_PCM_CHUNK_BYTES],
                    ).decode('ascii')
                    await stt.send(
                        RealtimeEvent(
                            type='input_audio_buffer.append',
                            audio=encoded,
                        ).model_dump_json(exclude_none=True),
                    )
                await stt.send(
                    RealtimeEvent(type='input_audio_buffer.commit').model_dump_json(
                        exclude_none=True,
                    ),
                )

                async for raw in stt:
                    if not isinstance(raw, str):
                        continue
                    try:
                        message = json.loads(raw)
                    except ValueError:
                        continue
                    if not isinstance(message, dict):
                        continue
                    transcript, complete = _update_dashboard_transcript(transcript, message)
                    if complete:
                        break
    except HTTPException:
        raise
    except (OSError, TimeoutError, UpstreamWebSocketException) as error:
        LOGGER.warning(
            'Dashboard STT request failed',
            extra={
                'event_id': 'ID_assistant_dashboard_stt_failed',
                'error_type': type(error).__name__,
                'error': str(error),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail='STT request failed',
        ) from error
    return transcript


def _restore_dashboard_secrets(current: object, proposed: object) -> object:
    if isinstance(current, SecretStr) and proposed == str(current):
        return current
    if isinstance(current, dict) and isinstance(proposed, dict):
        restored = {
            key: _restore_dashboard_secrets(current.get(key), value)
            for key, value in proposed.items()
        }
        restored.update(
            {
                key: value
                for key, value in current.items()
                if key not in proposed and isinstance(value, SecretStr)
            },
        )
        return restored
    if isinstance(current, (list, tuple)) and isinstance(proposed, list):
        return [
            _restore_dashboard_secrets(current[index], value) if index < len(current) else value
            for index, value in enumerate(proposed)
        ]
    return proposed


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


@app.get('/health/live')
async def live() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/metrics', include_in_schema=False)
async def prometheus_metrics() -> Response:
    return Response(generate_latest(), headers={'Content-Type': CONTENT_TYPE_LATEST})


@app.get('/dashboard', include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(DASHBOARD_PATH, media_type='text/html')


@app.get('/health', response_model=HealthResponse)
@app.get('/health/ready', response_model=HealthResponse)
async def ready(request: Request) -> HealthResponse:
    return await _runtime_from_request(request).health()


@app.get('/v1/models', response_model=ModelList)
async def models(request: Request) -> ModelList:
    runtime = _runtime_from_request(request)
    return ModelList(data=[ModelDescription(id=runtime.settings.model_id)])


@app.get('/config', response_model=Settings, response_model_by_alias=False)
async def configuration(request: Request) -> Settings:
    return _runtime_from_request(request).settings


@app.patch('/config', response_model=ConfigurationUpdateResponse)
async def update_configuration(
    request: Request,
    patch: dict[str, object],
) -> ConfigurationUpdateResponse:
    runtime = _runtime_from_request(request)
    reject_if_busy(runtime.operations, 'assistant')
    changed: tuple[str, ...] = ()
    try:
        settings, changed = validated_settings_patch(
            runtime.settings,
            patch,
            restart_required=RESTART_REQUIRED_SETTINGS,
        )
        if changed:
            replacement = AssistantRuntime(settings, runtime.operations)
            configure_logging(settings.log_level, 'assistant')
            request.app.state.runtime = replacement
            metrics.CONFIGURATION_UPDATES.inc()
            LOGGER.info(
                'Assistant configuration updated',
                extra={
                    'event_id': 'ID_assistant_configuration_updated',
                    'changed_fields': changed,
                    'ephemeral': True,
                },
            )
            await runtime.close()
        return ConfigurationUpdateResponse(changed=changed)
    finally:
        runtime.operations.release()


@app.get('/system-prompt', response_class=PlainTextResponse)
async def system_prompt(request: Request) -> str:
    return _runtime_from_request(request).system_prompt.read()


@app.put('/system-prompt', response_class=PlainTextResponse)
async def update_system_prompt(request: Request) -> str:
    try:
        prompt = (await request.body()).decode('utf-8')
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='system prompt must be UTF-8 text',
        ) from error

    runtime = _runtime_from_request(request)
    reject_if_busy(runtime.operations, 'assistant')
    try:
        try:
            updated_prompt = runtime.system_prompt.write(prompt)
        except (OSError, UnicodeError) as error:
            LOGGER.exception(
                'System prompt update failed',
                extra={
                    'event_id': 'ID_assistant_system_prompt_update_failed',
                    'path': str(runtime.settings.system_prompt_path),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='system prompt update failed',
            ) from error
        LOGGER.info(
            'System prompt updated',
            extra={
                'event_id': 'ID_assistant_system_prompt_updated',
                'path': str(runtime.settings.system_prompt_path),
                'characters': len(updated_prompt),
            },
        )
        return updated_prompt
    finally:
        runtime.operations.release()


@app.get('/dashboard/config/{service}', include_in_schema=False)
async def dashboard_configuration(
    request: Request,
    service: Literal['assistant', 'stt', 'tts'],
) -> Response:
    runtime = _runtime_from_request(request)
    if service == 'assistant':
        return Response(runtime.settings.model_dump_json(), media_type='application/json')
    return await _proxy_configuration_request(runtime, service, 'GET')


@app.patch('/dashboard/config/{service}', include_in_schema=False)
async def update_dashboard_configuration(
    request: Request,
    service: Literal['assistant', 'stt', 'tts'],
    patch: dict[str, object],
) -> Response:
    if service == 'assistant':
        runtime = _runtime_from_request(request)
        current = runtime.settings.model_dump(round_trip=True)
        restored_patch = {
            key: _restore_dashboard_secrets(current.get(key), value) for key, value in patch.items()
        }
        result = await update_configuration(request, restored_patch)
        return Response(result.model_dump_json(), media_type='application/json')
    runtime = _runtime_from_request(request)
    return await _proxy_configuration_request(runtime, service, 'PATCH', patch)


@app.post('/dashboard/stt', include_in_schema=False)
async def dashboard_transcription(
    request: Request,
    sample_rate: Annotated[int, Query(ge=8_000)],
    channels: Annotated[int, Query(ge=1, le=2)] = 1,
) -> dict[str, str]:
    pcm = await request.body()
    if not pcm:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail='PCM16 audio is empty',
        )
    frame_bytes = channels * 2
    if len(pcm) % frame_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail='PCM16 audio must contain complete sample frames',
        )
    runtime = _runtime_from_request(request)
    transcript = await _transcribe_dashboard_pcm(
        runtime,
        pcm,
        sample_rate,
        channels,
    )
    return {'transcript': transcript}


@app.post('/dashboard/tts', include_in_schema=False)
async def dashboard_speech(
    request: Request,
    speech_request: DashboardSpeechRequest,
) -> Response:
    runtime = _runtime_from_request(request)
    url = f'{runtime.settings.tts_base_url.rstrip("/")}/v1/audio/speech'
    request_body: dict[str, object] = {
        'model': runtime.settings.tts_model,
        'input': speech_request.text,
        'response_format': speech_request.response_format,
    }
    if speech_request.voice is not None:
        request_body['voice'] = speech_request.voice
    pipeline_headers = {'X-Pipeline': 'true'} if speech_request.pipeline else None
    try:
        if speech_request.response_format == 'pcm':
            upstream_request = runtime.http.build_request(
                'POST',
                url,
                headers=pipeline_headers,
                json=request_body,
            )
            upstream_response = await runtime.http.send(upstream_request, stream=True)
            return StreamingResponse(
                _stream_upstream_response(upstream_response),
                status_code=upstream_response.status_code,
                headers=_forward_upstream_headers(upstream_response),
            )
        if pipeline_headers is None:
            upstream_response = await runtime.http.request('POST', url, json=request_body)
        else:
            upstream_response = await runtime.http.request(
                'POST',
                url,
                headers=pipeline_headers,
                json=request_body,
            )
    except httpx.RequestError as error:
        LOGGER.warning(
            'Dashboard TTS request failed',
            extra={
                'event_id': 'ID_assistant_dashboard_tts_failed',
                'error_type': type(error).__name__,
                'error': str(error),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail='TTS request failed',
        ) from error
    return _forward_upstream_response(upstream_response)


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
        if (
            event.type == 'session.update'
            and event.session is not None
            and event.session.voice is not None
        ):
            utterance.select_voice(event.session.voice)
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
    async for raw in stt:
        if not isinstance(raw, str):
            continue
        try:
            message = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(message, dict):
            continue
        received_at = time.perf_counter()
        message_type = str(message.get('type', ''))
        if message_type.endswith('transcription.completed'):
            utterance.note_final_transcript(received_at)
        await send(message)
        if message_type == 'error':
            raise RuntimeError(str(message.get('message', 'STT request failed')))
        if message_type.endswith('transcription.delta'):
            metrics.TRANSCRIPTION_DELTAS.inc()
            now = received_at
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
                LOGGER.info(
                    'STT transcription received',
                    extra={
                        'event_id': 'ID_assistant_stt_transcription_received',
                        'stage': 'update',
                        'characters': len(transcript),
                    },
                )
                LOGGER.debug(
                    'STT transcription',
                    extra={
                        'event_id': 'ID_assistant_stt_transcription',
                        'stage': 'update',
                        'transcript': transcript,
                    },
                )
                utterance.schedule_cache_warm(transcript)
        elif message_type.endswith('transcription.completed'):
            completed = message.get('transcript')
            if isinstance(completed, str) and completed.strip():
                transcript = completed.strip()
            if transcript:
                LOGGER.info(
                    'STT transcription received',
                    extra={
                        'event_id': 'ID_assistant_stt_transcription_received',
                        'stage': 'completed',
                        'characters': len(transcript),
                    },
                )
                LOGGER.debug(
                    'STT transcription',
                    extra={
                        'event_id': 'ID_assistant_stt_transcription',
                        'stage': 'completed',
                        'transcript': transcript,
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
        except SttEmptyTranscriptError as e:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION, reason='STT produced empty transcript'
            ) from e
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


@app.websocket('/v1/realtime')
async def realtime(websocket: WebSocket) -> None:  # noqa: C901
    log_request_correlation(websocket.headers)
    runtime = _runtime_from_websocket(websocket)
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
                'reason': error.reason or None,
            },
        )
    except WebSocketException as error:
        outcome = 'rejected'
        LOGGER.info(
            'Realtime assistant session rejected',
            extra={
                'event_id': 'ID_assistant_realtime_session_rejected',
                'code': error.code,
                'reason': error.reason or None,
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


if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',  # noqa: S104
        port=SETTINGS.listen_port,
        log_config=None,
    )
