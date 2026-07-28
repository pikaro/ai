from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import TYPE_CHECKING, Annotated, Final, cast

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError

from runtime_config import (
    ConfigurationUpdateResponse,
    reject_if_busy,
    validated_settings_patch,
)
from service_contracts.models import ModelDescription, ModelList
from service_contracts.stt import SttError, SttSessionUpdated, TranscriptionDelta
from service_logging import configure_logging
from stt.src.config import Settings
from stt.src.domain import (
    AsrServiceError,
    AttentionContextRestartRequiredError,
    AudioStreamTooLargeError,
    AudioUploadTooLargeError,
    ModelMismatchError,
    ModelNotLoadedError,
    UnsupportedAudioExtensionError,
)
from stt.src.metrics import (
    ACTIVE_REQUESTS,
    BUSY_REJECTIONS,
    CONFIGURATION_UPDATES,
    REQUEST_SECONDS,
    REQUESTS,
    STREAM_AUDIO_SECONDS,
    STREAM_CHUNK_SECONDS,
    STREAM_COMMIT_SECONDS,
    STREAM_TIME_TO_FIRST_DELTA,
)
from stt.src.runtime import AsrRuntime
from stt.src.schemas import (
    HealthResponse,
    RealtimeEvent,
    SessionOptions,
    TranscriptionResponse,
)
from stt.src.transcript import log_transcription
from stt.src.uploads import save_temporary_upload

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from stt.src.streaming import CacheAwareStreamingSession

LOGGER = logging.getLogger('stt')
RESTART_REQUIRED_SETTINGS: Final = frozenset({'model_id', 'device', 'listen_port'})

SETTINGS = Settings()
configure_logging(SETTINGS.log_level, 'stt')


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
    runtime = AsrRuntime(SETTINGS)
    application.state.runtime = runtime
    await asyncio.to_thread(runtime.load)
    try:
        yield
    finally:
        await asyncio.to_thread(runtime.close)


app = FastAPI(title='STT', version='1.0.0', lifespan=lifespan)

_ASR_ERROR_STATUS: dict[type[AsrServiceError], int] = {
    AudioUploadTooLargeError: status.HTTP_413_CONTENT_TOO_LARGE,
    UnsupportedAudioExtensionError: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    ModelMismatchError: status.HTTP_400_BAD_REQUEST,
}


@app.exception_handler(AsrServiceError)
async def _asr_service_error(_request: Request, error: AsrServiceError) -> JSONResponse:
    if isinstance(error, AttentionContextRestartRequiredError):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                'detail': {
                    'message': str(error),
                    'fields': ['attention_context_size'],
                },
            },
        )
    if isinstance(error, ModelNotLoadedError):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={'detail': 'Service Unavailable'},
        )
    status_code = _ASR_ERROR_STATUS.get(type(error), status.HTTP_400_BAD_REQUEST)
    return JSONResponse(status_code=status_code, content={'detail': str(error)})


def _runtime_from_request(request: Request) -> AsrRuntime:
    return cast('AsrRuntime', request.app.state.runtime)


def _runtime_from_websocket(websocket: WebSocket) -> AsrRuntime:
    return cast('AsrRuntime', websocket.app.state.runtime)


def _health_response(runtime: AsrRuntime) -> HealthResponse:
    return HealthResponse(status='ok', **asdict(runtime.status()))


@app.get('/health/live')
async def live() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/metrics', include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), headers={'Content-Type': CONTENT_TYPE_LATEST})


@app.get('/health', response_model=HealthResponse)
@app.get('/health/ready', response_model=HealthResponse)
async def ready(request: Request) -> HealthResponse:
    return _health_response(_runtime_from_request(request))


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
    try:
        reject_if_busy(runtime.operations, 'STT')
    except HTTPException:
        BUSY_REJECTIONS.inc()
        raise
    try:
        settings, changed = validated_settings_patch(
            runtime.settings,
            patch,
            restart_required=RESTART_REQUIRED_SETTINGS,
        )
        if changed:
            await asyncio.to_thread(runtime.apply_settings, settings)
            configure_logging(settings.log_level, 'stt')
            CONFIGURATION_UPDATES.inc()
            LOGGER.info(
                'STT configuration updated',
                extra={
                    'event_id': 'ID_stt_configuration_updated',
                    'changed_fields': changed,
                    'ephemeral': True,
                },
            )
        return ConfigurationUpdateResponse(changed=changed)
    finally:
        runtime.operations.release()


@app.post('/v1/audio/transcriptions', response_model=TranscriptionResponse)
async def transcribe(
    request: Request,
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()] = '',
) -> TranscriptionResponse:
    runtime = _runtime_from_request(request)
    try:
        reject_if_busy(runtime.operations, 'STT')
    except HTTPException:
        BUSY_REJECTIONS.inc()
        raise
    started = time.perf_counter()
    outcome = 'success'
    ACTIVE_REQUESTS.inc()
    try:
        runtime.validate_model(model)
        path = await asyncio.to_thread(
            save_temporary_upload,
            file.filename,
            file.file,
            runtime.settings.maximum_upload_bytes,
        )
        try:
            text = await asyncio.to_thread(runtime.transcribe_file, path)
            log_transcription('file', text, announce=True)
        finally:
            await asyncio.to_thread(path.unlink, missing_ok=True)
        return TranscriptionResponse(text=text)
    except Exception:
        outcome = 'error'
        raise
    finally:
        await file.close()
        runtime.operations.release()
        ACTIVE_REQUESTS.dec()
        REQUESTS.labels(mode='file', outcome=outcome).inc()
        REQUEST_SECONDS.labels(mode='file').observe(time.perf_counter() - started)


async def _send_error(websocket: WebSocket, message: str) -> None:
    await websocket.send_json(SttError(message=message).model_dump())


@app.websocket('/v1/realtime')
async def realtime(websocket: WebSocket) -> None:  # noqa: C901, PLR0912, PLR0915
    runtime = _runtime_from_websocket(websocket)
    if not runtime.operations.try_acquire():
        BUSY_REJECTIONS.inc()
        await websocket.accept()
        await _send_error(websocket, 'another STT session is already active')
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return
    started = time.perf_counter()
    outcome = 'success'
    request_started = False
    first_audio_at: float | None = None
    first_delta_observed = False
    input_pcm_bytes = 0
    sample_rate = runtime.settings.stream_sample_rate
    channels = 1
    stream: CacheAwareStreamingSession | None = None
    try:
        await websocket.accept()
        ACTIVE_REQUESTS.inc()
        request_started = True
        LOGGER.info(
            'realtime transcription session started',
            extra={'event_id': 'ID_stt_realtime_session_started'},
        )
        while True:
            raw = await websocket.receive_text()
            if len(raw.encode()) > runtime.settings.maximum_websocket_message_bytes:
                outcome = 'rejected'
                await websocket.close(code=status.WS_1009_MESSAGE_TOO_BIG)
                return
            try:
                event = RealtimeEvent.model_validate_json(raw)
            except ValidationError as error:
                await _send_error(websocket, f'invalid event: {error.errors(include_url=False)}')
                continue

            if event.type == 'session.update':
                if stream is not None:
                    await _send_error(websocket, 'session.update is only supported before audio')
                    continue
                options = event.session or SessionOptions()
                sample_rate = options.input_audio_sample_rate or sample_rate
                channels = options.input_audio_channels or channels
                await websocket.send_json(
                    SttSessionUpdated(
                        session=_health_response(runtime).model_dump(),
                    ).model_dump(),
                )
                continue

            if event.type == 'input_audio_buffer.append':
                if event.audio is None:
                    await _send_error(websocket, 'audio is required')
                    continue
                try:
                    pcm = base64.b64decode(event.audio, validate=True)
                except (binascii.Error, ValueError):
                    await _send_error(websocket, 'audio must be valid base64')
                    continue
                if first_audio_at is None:
                    first_audio_at = time.perf_counter()
                input_pcm_bytes += len(pcm)
                stream = stream or runtime.create_stream(sample_rate, channels)
                chunk_started = time.perf_counter()
                try:
                    events = await asyncio.to_thread(stream.append_pcm, pcm)
                except AudioStreamTooLargeError as error:
                    outcome = 'rejected'
                    await _send_error(websocket, str(error))
                    await websocket.close(code=status.WS_1009_MESSAGE_TOO_BIG)
                    return
                finally:
                    STREAM_CHUNK_SECONDS.observe(time.perf_counter() - chunk_started)
                for transcript_event in events:
                    if not first_delta_observed and isinstance(
                        transcript_event,
                        TranscriptionDelta,
                    ):
                        first_delta_observed = True
                        STREAM_TIME_TO_FIRST_DELTA.observe(
                            time.perf_counter() - first_audio_at,
                        )
                    await websocket.send_json(transcript_event.model_dump())
                continue

            if stream is None:
                await _send_error(websocket, 'cannot commit an empty audio buffer')
                continue
            commit_started = time.perf_counter()
            try:
                events = await asyncio.to_thread(stream.finish)
            finally:
                STREAM_COMMIT_SECONDS.observe(time.perf_counter() - commit_started)
            for transcript_event in events:
                if (
                    not first_delta_observed
                    and isinstance(transcript_event, TranscriptionDelta)
                    and first_audio_at is not None
                ):
                    first_delta_observed = True
                    STREAM_TIME_TO_FIRST_DELTA.observe(
                        time.perf_counter() - first_audio_at,
                    )
                await websocket.send_json(transcript_event.model_dump())
            await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
            duration_seconds = time.perf_counter() - started
            audio_seconds = (
                input_pcm_bytes / (sample_rate * channels * 2)
                if sample_rate > 0 and channels > 0
                else 0.0
            )
            completed_transcript = stream.latest_transcript or stream.emitted_text
            LOGGER.info(
                'realtime transcription session completed',
                extra={
                    'event_id': 'ID_stt_realtime_session_completed',
                    'duration_seconds': duration_seconds,
                    'audio_seconds': audio_seconds,
                    'audio_bytes': input_pcm_bytes,
                    'characters': len(completed_transcript),
                    'partial_events': stream.partial_event_count,
                    'delta_events': stream.delta_event_count,
                    'inference_steps': stream.step_number,
                },
            )
            return
    except WebSocketDisconnect as error:
        outcome = 'disconnected'
        LOGGER.info(
            'Realtime transcription client disconnected',
            extra={'event_id': 'ID_stt_realtime_client_disconnected', 'code': error.code},
        )
    except BaseException:
        outcome = 'error'
        raise
    finally:
        runtime.operations.release()
        if request_started:
            ACTIVE_REQUESTS.dec()
            REQUESTS.labels(mode='realtime', outcome=outcome).inc()
            REQUEST_SECONDS.labels(mode='realtime').observe(time.perf_counter() - started)
            bytes_per_second = sample_rate * channels * 2
            if bytes_per_second > 0:
                STREAM_AUDIO_SECONDS.observe(input_pcm_bytes / bytes_per_second)
