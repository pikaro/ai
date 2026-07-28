from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import SecretStr, ValidationError
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException as UpstreamWebSocketException

from assistant.src.configuration_api import apply_configuration_patch
from assistant.src.dependencies import runtime_from_request
from assistant.src.schemas import DashboardSpeechRequest, RealtimeEvent, SessionOptions
from service_contracts.stt import (
    SttError,
    SttServerEvent,
    TranscriptionCompleted,
    TranscriptionDelta,
    parse_stt_server_event,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from assistant.src.runtime import AssistantRuntime

LOGGER = logging.getLogger('assistant')
DASHBOARD_PATH: Final = Path(__file__).with_name('dashboard.html')
DASHBOARD_PCM_CHUNK_BYTES: Final = 64 * 1024
router = APIRouter()


def _upstream_configuration_url(
    runtime: AssistantRuntime,
    service: Literal['stt', 'tts'],
) -> str:
    base_url = runtime.settings.stt_base_url if service == 'stt' else runtime.settings.tts_base_url
    return f'{base_url.rstrip("/")}/config'


async def _proxy_configuration_request(
    runtime: AssistantRuntime,
    service: Literal['stt', 'tts'],
    method: Literal['GET', 'PATCH'],
    patch: dict[str, object] | None = None,
) -> Response:
    url = _upstream_configuration_url(runtime, service)
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


def _update_dashboard_payload(  # noqa: C901
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


def _update_dashboard_transcript(  # noqa: C901
    transcript: str,
    message: SttServerEvent | dict[str, object],
) -> tuple[str, bool]:
    if isinstance(message, dict):
        return _update_dashboard_payload(transcript, message)
    if isinstance(message, SttError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=message.message,
        )
    if isinstance(message, TranscriptionDelta):
        if message.transcript.strip():
            return message.transcript.strip(), False
        return f'{transcript} {message.delta}'.strip(), False
    if isinstance(message, TranscriptionCompleted):
        if message.transcript.strip():
            transcript = message.transcript.strip()
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
                        payload, message = parse_stt_server_event(raw)
                    except ValidationError:
                        continue
                    transcript, complete = _update_dashboard_transcript(
                        transcript,
                        payload if message is None else message,
                    )
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


@router.get('/dashboard', include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(DASHBOARD_PATH, media_type='text/html')


@router.get('/dashboard/config/{service}', include_in_schema=False)
async def dashboard_configuration(
    request: Request,
    service: Literal['assistant', 'stt', 'tts'],
) -> Response:
    runtime = runtime_from_request(request)
    if service == 'assistant':
        return Response(runtime.settings.model_dump_json(), media_type='application/json')
    return await _proxy_configuration_request(runtime, service, 'GET')


@router.patch('/dashboard/config/{service}', include_in_schema=False)
async def update_dashboard_configuration(
    request: Request,
    service: Literal['assistant', 'stt', 'tts'],
    patch: dict[str, object],
) -> Response:
    if service == 'assistant':
        runtime = runtime_from_request(request)
        current = runtime.settings.model_dump(round_trip=True)
        restored_patch = {
            key: _restore_dashboard_secrets(current.get(key), value) for key, value in patch.items()
        }
        result = await apply_configuration_patch(request, restored_patch)
        return Response(result.model_dump_json(), media_type='application/json')
    runtime = runtime_from_request(request)
    return await _proxy_configuration_request(runtime, service, 'PATCH', patch)


@router.post('/dashboard/stt', include_in_schema=False)
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
    runtime = runtime_from_request(request)
    transcript = await _transcribe_dashboard_pcm(
        runtime,
        pcm,
        sample_rate,
        channels,
    )
    return {'transcript': transcript}


@router.post('/dashboard/tts', include_in_schema=False)
async def dashboard_speech(
    request: Request,
    speech_request: DashboardSpeechRequest,
) -> Response:
    runtime = runtime_from_request(request)
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
            },
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail='TTS request failed',
        ) from error
    return _forward_upstream_response(upstream_response)
