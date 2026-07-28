from __future__ import annotations

import asyncio
import io
import logging
import time
import wave
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from typing import TYPE_CHECKING, Annotated, Never, cast

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from runtime_config import (
    ConfigurationUpdateResponse,
    reject_if_busy,
    validated_settings_patch,
)
from service_contracts.models import ModelDescription, ModelList
from service_contracts.tts import (
    PIPELINE_INPUT_ADAPTER,
    PipelineAudioDone,
    PipelineError,
    PipelineSessionReady,
    PipelineSessionRequest,
    PipelineTextDone,
)
from service_logging import configure_logging
from tts.src.config import Settings
from tts.src.domain import (
    EmptyInputError,
    EmptySegmentError,
    EmptyVoiceUploadError,
    InputTooLongError,
    InvalidVoiceNameError,
    InvalidVoiceSelectorError,
    ModelMismatchError,
    ModelNotLoadedError,
    MultiSpeakerCommand,
    SpeakerSegment,
    SpeechCommand,
    TtsServiceError,
    UndefinedSpeakerError,
    UnsupportedSpeedError,
    UnsupportedVoiceFileError,
    VoiceUnavailableError,
    VoiceUploadTooLargeError,
)
from tts.src.metrics import (
    ACTIVE_REQUESTS,
    AUDIO_SECONDS,
    BUSY_REJECTIONS,
    CONFIGURATION_UPDATES,
    PIPELINE_REQUESTS,
    REALTIME_FACTOR,
    REQUEST_SECONDS,
    REQUESTS,
    TIME_TO_FIRST_AUDIO,
)
from tts.src.runtime import TtsRuntime
from tts.src.schemas import (
    HealthResponse,
    LegacySpeechRequest,
    MultiSpeakerSpeechRequest,
    SpeechRequest,
    VoiceUploadResponse,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator, Iterator

    from starlette.types import Receive, Scope, Send

    from tts.src.pipeline import SpeakerTurn
    from tts.src.streaming import IncrementalPipelineSession

LOGGER = logging.getLogger('tts')
RESTART_REQUIRED_SETTINGS = frozenset({'model_id', 'language', 'listen_port'})


class _ClosingStreamingResponse(StreamingResponse):
    """Close a synchronous body iterator promptly when its client disconnects."""

    def __init__(
        self,
        content: Generator[bytes, None, None],
        *,
        media_type: str,
        headers: dict[str, str],
    ) -> None:
        self._content = content
        super().__init__(content, media_type=media_type, headers=headers)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await asyncio.to_thread(self._content.close)


SETTINGS = Settings()
configure_logging(SETTINGS.log_level, 'tts')


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
    runtime = TtsRuntime(SETTINGS)
    application.state.runtime = runtime
    await asyncio.to_thread(runtime.load)
    try:
        yield
    finally:
        await asyncio.to_thread(runtime.close)


app = FastAPI(title='TTS', version='1.0.0', lifespan=lifespan)

_TTS_ERROR_STATUS: dict[type[TtsServiceError], int] = {
    ModelNotLoadedError: status.HTTP_503_SERVICE_UNAVAILABLE,
    InvalidVoiceNameError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    UnsupportedVoiceFileError: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    VoiceUploadTooLargeError: status.HTTP_413_CONTENT_TOO_LARGE,
    InputTooLongError: status.HTTP_413_CONTENT_TOO_LARGE,
    EmptyVoiceUploadError: status.HTTP_400_BAD_REQUEST,
    InvalidVoiceSelectorError: status.HTTP_400_BAD_REQUEST,
    VoiceUnavailableError: status.HTTP_400_BAD_REQUEST,
    ModelMismatchError: status.HTTP_400_BAD_REQUEST,
    UnsupportedSpeedError: status.HTTP_400_BAD_REQUEST,
    EmptyInputError: status.HTTP_400_BAD_REQUEST,
    EmptySegmentError: status.HTTP_400_BAD_REQUEST,
    UndefinedSpeakerError: status.HTTP_400_BAD_REQUEST,
}


@app.exception_handler(TtsServiceError)
async def _tts_service_error(_request: Request, error: TtsServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=_TTS_ERROR_STATUS.get(type(error), status.HTTP_400_BAD_REQUEST),
        content={'detail': str(error)},
    )


def _runtime(request: Request) -> TtsRuntime:
    return cast('TtsRuntime', request.app.state.runtime)


def _runtime_from_websocket(websocket: WebSocket) -> TtsRuntime:
    return cast('TtsRuntime', websocket.app.state.runtime)


def _observe_request(
    response_format: str,
    outcome: str,
    started: float,
    audio_bytes: int,
    sample_rate: int,
) -> None:
    wall_seconds = time.perf_counter() - started
    audio_seconds = audio_bytes / (sample_rate * 2) if sample_rate > 0 else 0
    REQUESTS.labels(format=response_format, outcome=outcome).inc()
    REQUEST_SECONDS.labels(format=response_format).observe(wall_seconds)
    if audio_seconds > 0:
        AUDIO_SECONDS.observe(audio_seconds)
        REALTIME_FACTOR.observe(wall_seconds / audio_seconds)


def _stream_speech(  # noqa: C901
    runtime: TtsRuntime,
    text: str,
    voice: str,
    started: float,
    *,
    pipeline: bool,
) -> Generator[bytes, None, None]:
    outcome = 'success'
    output_bytes = 0
    first_audio = True
    sample_rate = 0
    chunks: Generator[bytes, None, None] | None = None
    try:
        sample_rate = runtime.sample_rate()
        chunks = (
            runtime.stream_pipeline_pcm(text, voice, transport='http')
            if pipeline
            else runtime.stream_pcm(text, voice)
        )
        for chunk in chunks:
            if first_audio:
                first_audio = False
                TIME_TO_FIRST_AUDIO.observe(time.perf_counter() - started)
            output_bytes += len(chunk)
            yield chunk
    except GeneratorExit:
        outcome = 'cancelled'
        raise
    except Exception:
        outcome = 'error'
        raise
    finally:
        if chunks is not None:
            chunks.close()
        runtime.operations.release()
        ACTIVE_REQUESTS.dec()
        _observe_request('pcm', outcome, started, output_bytes, sample_rate)
        if pipeline:
            PIPELINE_REQUESTS.labels(transport='http', outcome=outcome).inc()
            LOGGER.info(
                'TTS HTTP pipeline finished',
                extra={
                    'event_id': 'ID_tts_pipeline_http_finished',
                    'outcome': outcome,
                    'audio_bytes': output_bytes,
                },
            )


def _stream_multi_speaker_speech(  # noqa: C901
    runtime: TtsRuntime,
    turns: list[SpeakerTurn],
    started: float,
) -> Generator[bytes, None, None]:
    outcome = 'success'
    output_bytes = 0
    first_audio = True
    sample_rate = 0
    chunks: Generator[bytes, None, None] | None = None
    try:
        sample_rate = runtime.sample_rate()
        chunks = runtime.stream_multi_speaker_pcm(turns)
        for chunk in chunks:
            if first_audio:
                first_audio = False
                TIME_TO_FIRST_AUDIO.observe(time.perf_counter() - started)
            output_bytes += len(chunk)
            yield chunk
    except GeneratorExit:
        outcome = 'cancelled'
        raise
    except Exception:
        outcome = 'error'
        raise
    finally:
        if chunks is not None:
            chunks.close()
        runtime.operations.release()
        ACTIVE_REQUESTS.dec()
        _observe_request('pcm', outcome, started, output_bytes, sample_rate)
        PIPELINE_REQUESTS.labels(
            transport='http_multi_speaker',
            outcome=outcome,
        ).inc()
        LOGGER.info(
            'TTS multi-speaker stream finished',
            extra={
                'event_id': 'ID_tts_multi_speaker_finished',
                'response_format': 'pcm',
                'outcome': outcome,
                'audio_bytes': output_bytes,
                'turn_count': len(turns),
            },
        )


def _speech_command(request: SpeechRequest) -> SpeechCommand:
    return SpeechCommand(
        model=request.model,
        text=request.input,
        voice=request.voice,
        speed=request.speed,
    )


def _multi_speaker_command(request: MultiSpeakerSpeechRequest) -> MultiSpeakerCommand:
    return MultiSpeakerCommand(
        model=request.model,
        speakers={
            speaker: definition.voice for speaker, definition in request.input.speakers.items()
        },
        segments=tuple(
            SpeakerSegment(speaker=segment.speaker, text=segment.text)
            for segment in request.input.segments
        ),
        speed=request.speed,
    )


def _speech_response(  # noqa: C901
    runtime: TtsRuntime,
    speech_request: SpeechRequest,
    *,
    pipeline: bool,
) -> Response:
    try:
        reject_if_busy(runtime.operations, 'TTS')
    except HTTPException:
        BUSY_REJECTIONS.inc()
        raise
    started = time.perf_counter()
    ACTIVE_REQUESTS.inc()
    stream_response = False
    request_observed = False
    outcome = 'success'
    try:
        prepared = runtime.prepare_speech(_speech_command(speech_request))
        text = prepared.text
        voice = prepared.voice
        LOGGER.info(
            'Speech synthesis requested',
            extra={
                'event_id': 'ID_tts_synthesis_requested',
                'response_format': speech_request.response_format,
                'pipeline': pipeline,
                'characters': len(text),
                'voice': voice,
            },
        )
        LOGGER.debug(
            'Speech synthesis input',
            extra={'event_id': 'ID_tts_synthesis_input', 'text': text},
        )
        if speech_request.response_format == 'pcm':
            response = _ClosingStreamingResponse(
                _stream_speech(runtime, text, voice, started, pipeline=pipeline),
                media_type='application/octet-stream',
                headers=runtime.pcm_headers(),
            )
            stream_response = True
            return response
        if pipeline:
            wav, pcm_bytes = runtime.generate_pipeline_wav(text, voice)
            PIPELINE_REQUESTS.labels(transport='http', outcome=outcome).inc()
        else:
            wav = runtime.generate_wav(text, voice)
            with wave.open(io.BytesIO(wav), 'rb') as wav_file:
                pcm_bytes = (
                    wav_file.getnframes() * wav_file.getnchannels() * wav_file.getsampwidth()
                )
        LOGGER.info(
            'Speech synthesis completed',
            extra={
                'event_id': 'ID_tts_synthesis_completed',
                'response_format': 'wav',
                'audio_bytes': len(wav),
            },
        )
        sample_rate = runtime.sample_rate()
        response = Response(wav, media_type='audio/wav')
        _observe_request('wav', outcome, started, pcm_bytes, sample_rate)
        request_observed = True
        return response  # noqa: TRY300
    except Exception:
        outcome = 'error'
        if pipeline and speech_request.response_format == 'wav':
            PIPELINE_REQUESTS.labels(transport='http', outcome=outcome).inc()
        raise
    finally:
        if not stream_response:
            runtime.operations.release()
            ACTIVE_REQUESTS.dec()
            if not request_observed:
                _observe_request(speech_request.response_format, outcome, started, 0, 1)


def _multi_speaker_response(  # noqa: C901
    runtime: TtsRuntime,
    speech_request: MultiSpeakerSpeechRequest,
) -> Response:
    try:
        reject_if_busy(runtime.operations, 'TTS')
    except HTTPException:
        BUSY_REJECTIONS.inc()
        raise
    started = time.perf_counter()
    ACTIVE_REQUESTS.inc()
    stream_response = False
    request_observed = False
    outcome = 'success'
    try:
        turns = runtime.prepare_multi_speaker_turns(_multi_speaker_command(speech_request))
        LOGGER.info(
            'Multi-speaker speech synthesis requested',
            extra={
                'event_id': 'ID_tts_multi_speaker_requested',
                'response_format': speech_request.response_format,
                'characters': sum(len(turn.text) for turn in turns),
                'speakers': len({turn.speaker for turn in turns}),
                'turn_count': len(turns),
                'voices': sorted({turn.voice for turn in turns}),
                'speaker_switch_pause_seconds': (
                    runtime.settings.pipeline_speaker_switch_pause_seconds
                ),
            },
        )
        LOGGER.debug(
            'Multi-speaker speech synthesis input',
            extra={
                'event_id': 'ID_tts_multi_speaker_input',
                'turns': [
                    {
                        'speaker': turn.speaker,
                        'voice': turn.voice,
                        'text': turn.text,
                    }
                    for turn in turns
                ],
            },
        )
        if speech_request.response_format == 'pcm':
            response = _ClosingStreamingResponse(
                _stream_multi_speaker_speech(runtime, turns, started),
                media_type='application/octet-stream',
                headers=runtime.pcm_headers(),
            )
            stream_response = True
            return response

        wav, pcm_bytes = runtime.generate_multi_speaker_wav(turns)
        PIPELINE_REQUESTS.labels(
            transport='http_multi_speaker',
            outcome=outcome,
        ).inc()
        LOGGER.info(
            'TTS multi-speaker synthesis completed',
            extra={
                'event_id': 'ID_tts_multi_speaker_finished',
                'response_format': 'wav',
                'outcome': outcome,
                'audio_bytes': len(wav),
                'turn_count': len(turns),
            },
        )
        sample_rate = runtime.sample_rate()
        response = Response(wav, media_type='audio/wav')
        _observe_request('wav', outcome, started, pcm_bytes, sample_rate)
        request_observed = True
        return response  # noqa: TRY300
    except Exception:
        outcome = 'error'
        if speech_request.response_format == 'wav':
            PIPELINE_REQUESTS.labels(
                transport='http_multi_speaker',
                outcome=outcome,
            ).inc()
        raise
    finally:
        if not stream_response:
            runtime.operations.release()
            ACTIVE_REQUESTS.dec()
            if not request_observed:
                _observe_request(speech_request.response_format, outcome, started, 0, 1)


def _next_pcm_chunk(chunks: Iterator[bytes]) -> bytes | None:
    try:
        return next(chunks)
    except StopIteration:
        return None


class _WebSocketPcmSender:
    def __init__(self, websocket: WebSocket, sample_rate: int, started: float) -> None:
        self.websocket = websocket
        self.sample_rate = sample_rate
        self.started = started
        self.output_bytes = 0
        self.first_audio = True

    async def send(self, chunks: Generator[bytes, None, None]) -> None:
        try:
            while chunk := await asyncio.to_thread(_next_pcm_chunk, chunks):
                if self.first_audio:
                    self.first_audio = False
                    TIME_TO_FIRST_AUDIO.observe(time.perf_counter() - self.started)
                await self.websocket.send_bytes(chunk)
                self.output_bytes += len(chunk)
        finally:
            await asyncio.to_thread(chunks.close)


async def _receive_pipeline_text(websocket: WebSocket, timeout_seconds: float) -> str:
    async with asyncio.timeout(timeout_seconds):
        return await websocket.receive_text()


async def _close_pipeline_websocket(
    websocket: WebSocket,
    *,
    code: int,
    message: str,
) -> None:
    if websocket.client_state != WebSocketState.CONNECTED:
        return
    with suppress(RuntimeError):
        await websocket.send_json(PipelineError(message=message).model_dump())
    with suppress(RuntimeError):
        await websocket.close(code=code, reason=message)


def _pipeline_validation_message(error: TtsServiceError | ValidationError) -> str:
    if isinstance(error, TtsServiceError):
        return str(error)
    return 'invalid pipeline event'


def _raise_empty_pipeline() -> Never:
    message = 'pipeline input is empty'
    raise ValueError(message)


def _enforce_pipeline_input_limit(input_characters: int, maximum: int) -> None:
    if input_characters <= maximum:
        return
    raise InputTooLongError


async def _stream_pipeline_websocket(  # noqa: C901, PLR0912, PLR0915
    websocket: WebSocket,
    runtime: TtsRuntime,
) -> None:
    started = time.perf_counter()
    outcome = 'success'
    gate_acquired = False
    active_request = False
    completed = False
    pipeline: IncrementalPipelineSession | None = None
    sender: _WebSocketPcmSender | None = None
    try:
        await websocket.accept()
        session_raw = await _receive_pipeline_text(
            websocket,
            runtime.settings.pipeline_idle_timeout_seconds,
        )
        session = PipelineSessionRequest.model_validate_json(session_raw)
        runtime.validate_options(session.model, session.speed)
        await websocket.send_json(
            PipelineSessionReady(
                sample_rate=runtime.sample_rate(),
                sample_width=2,
                channels=1,
            ).model_dump(),
        )
        LOGGER.info(
            'TTS pipeline WebSocket started',
            extra={'event_id': 'ID_tts_pipeline_websocket_started'},
        )

        input_characters = 0
        input_has_text = False
        input_done = False
        while not input_done:
            receive_timeout_seconds = runtime.settings.pipeline_idle_timeout_seconds
            smart_deadline = pipeline.smart_flush_deadline if pipeline is not None else None
            smart_timeout = False
            if smart_deadline is not None:
                remaining_seconds = smart_deadline - time.perf_counter()
                if remaining_seconds <= 0:
                    if pipeline is None or sender is None:
                        message = 'smart chunk pipeline sender is unavailable'
                        raise RuntimeError(message)  # noqa: TRY301
                    await sender.send(
                        pipeline.flush_smart_queue(
                            observed_at=time.perf_counter(),
                        ),
                    )
                    continue
                if remaining_seconds <= receive_timeout_seconds:
                    receive_timeout_seconds = remaining_seconds
                    smart_timeout = True
            try:
                raw = await _receive_pipeline_text(
                    websocket,
                    receive_timeout_seconds,
                )
            except TimeoutError:
                if not smart_timeout or pipeline is None or sender is None:
                    raise
                await sender.send(
                    pipeline.flush_smart_queue(
                        observed_at=time.perf_counter(),
                    ),
                )
                continue
            event_observed_at = time.perf_counter()
            if pipeline is not None and sender is not None:
                smart_deadline = pipeline.smart_flush_deadline
                if smart_deadline is not None and event_observed_at >= smart_deadline:
                    await sender.send(
                        pipeline.flush_smart_queue(
                            observed_at=event_observed_at,
                        ),
                    )
            event = PIPELINE_INPUT_ADAPTER.validate_json(raw)
            if isinstance(event, PipelineTextDone):
                if pipeline is None or not input_has_text:
                    _raise_empty_pipeline()
                input_done = True
            else:
                input_characters += len(event.delta)
                input_has_text = input_has_text or bool(event.delta.strip())
                _enforce_pipeline_input_limit(
                    input_characters,
                    runtime.settings.maximum_input_characters,
                )
                if pipeline is None:
                    if not runtime.operations.try_acquire():
                        BUSY_REJECTIONS.inc()
                        outcome = 'busy'
                        await _close_pipeline_websocket(
                            websocket,
                            code=status.WS_1013_TRY_AGAIN_LATER,
                            message='TTS is busy',
                        )
                        return
                    gate_acquired = True
                    ACTIVE_REQUESTS.inc()
                    active_request = True
                    pipeline = await asyncio.to_thread(
                        runtime.create_incremental_pipeline,
                        session.voice,
                    )
                    sender = _WebSocketPcmSender(websocket, pipeline.sample_rate, started)

            if sender is None:
                continue
            if isinstance(event, PipelineTextDone):
                await sender.send(
                    pipeline.finish(observed_at=event_observed_at),
                )
            else:
                await sender.send(
                    pipeline.append_text(
                        event.delta,
                        observed_at=event_observed_at,
                    ),
                )

        if pipeline is None or sender is None:
            _raise_empty_pipeline()
        pipeline.close(completed=True)
        completed = True
        await websocket.send_json(PipelineAudioDone().model_dump())
        await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
        LOGGER.info(
            'TTS pipeline WebSocket completed',
            extra={
                'event_id': 'ID_tts_pipeline_websocket_completed',
                'characters': input_characters,
                'audio_bytes': sender.output_bytes,
            },
        )
    except WebSocketDisconnect:
        outcome = 'cancelled'
        LOGGER.info(
            'TTS pipeline WebSocket disconnected',
            extra={'event_id': 'ID_tts_pipeline_websocket_disconnected'},
        )
    except TimeoutError:
        outcome = 'timeout'
        await _close_pipeline_websocket(
            websocket,
            code=status.WS_1008_POLICY_VIOLATION,
            message='pipeline input timed out',
        )
    except (TtsServiceError, ValidationError) as error:
        outcome = 'rejected'
        await _close_pipeline_websocket(
            websocket,
            code=status.WS_1008_POLICY_VIOLATION,
            message=_pipeline_validation_message(error),
        )
    except ValueError as error:
        outcome = 'rejected'
        await _close_pipeline_websocket(
            websocket,
            code=status.WS_1008_POLICY_VIOLATION,
            message=str(error),
        )
    except Exception:
        outcome = 'error'
        LOGGER.exception(
            'TTS pipeline WebSocket failed',
            extra={'event_id': 'ID_tts_pipeline_websocket_failed'},
        )
        await _close_pipeline_websocket(
            websocket,
            code=status.WS_1011_INTERNAL_ERROR,
            message='TTS pipeline failed',
        )
    finally:
        if pipeline is not None and not completed:
            pipeline.close(completed=False)
        if gate_acquired:
            runtime.operations.release()
        if active_request:
            ACTIVE_REQUESTS.dec()
            output_bytes = sender.output_bytes if sender is not None else 0
            _observe_request('pcm', outcome, started, output_bytes, runtime.sample_rate())
        PIPELINE_REQUESTS.labels(transport='websocket', outcome=outcome).inc()


@app.get('/health/live')
async def live() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/metrics', include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), headers={'Content-Type': CONTENT_TYPE_LATEST})


@app.get('/health', response_model=HealthResponse)
@app.get('/health/ready', response_model=HealthResponse)
async def ready(request: Request) -> HealthResponse:
    runtime_status = _runtime(request).status()
    return HealthResponse(status='ok', **asdict(runtime_status))


@app.get('/v1/models', response_model=ModelList)
async def models(request: Request) -> ModelList:
    runtime = _runtime(request)
    return ModelList(data=[ModelDescription(id=runtime.settings.model_id)])


@app.get('/config', response_model=Settings, response_model_by_alias=False)
async def configuration(request: Request) -> Settings:
    return _runtime(request).settings


@app.patch('/config', response_model=ConfigurationUpdateResponse)
async def update_configuration(
    request: Request,
    patch: dict[str, object],
) -> ConfigurationUpdateResponse:
    runtime = _runtime(request)
    try:
        reject_if_busy(runtime.operations, 'TTS')
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
            configure_logging(settings.log_level, 'tts')
            CONFIGURATION_UPDATES.inc()
            LOGGER.info(
                'TTS configuration updated',
                extra={
                    'event_id': 'ID_tts_configuration_updated',
                    'changed_fields': changed,
                    'ephemeral': True,
                },
            )
        return ConfigurationUpdateResponse(changed=changed)
    finally:
        runtime.operations.release()


@app.post('/v1/audio/speech')
def speech(
    request: Request,
    speech_request: SpeechRequest,
    *,
    x_pipeline: Annotated[bool, Header(alias='X-Pipeline')] = False,
) -> Response:
    return _speech_response(_runtime(request), speech_request, pipeline=x_pipeline)


@app.post('/v1/audio/speech/pipeline')
def speech_pipeline(request: Request, speech_request: SpeechRequest) -> Response:
    return _speech_response(_runtime(request), speech_request, pipeline=True)


@app.post('/v1/audio/speech/multi-speaker')
def multi_speaker_speech(
    request: Request,
    speech_request: MultiSpeakerSpeechRequest,
) -> Response:
    return _multi_speaker_response(_runtime(request), speech_request)


@app.websocket('/v1/audio/speech/pipeline')
async def speech_pipeline_websocket(websocket: WebSocket) -> None:
    await _stream_pipeline_websocket(websocket, _runtime_from_websocket(websocket))


@app.post('/v1/voices', response_model=VoiceUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_voice(
    request: Request,
    name: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
) -> VoiceUploadResponse:
    runtime = _runtime(request)
    try:
        reject_if_busy(runtime.operations, 'TTS')
    except HTTPException:
        BUSY_REJECTIONS.inc()
        await file.close()
        raise
    try:
        stored = await asyncio.to_thread(
            runtime.save_voice,
            name,
            file.filename,
            file.file,
        )
        return VoiceUploadResponse(
            name=stored.name,
            filename=stored.filename,
            replaced=stored.replaced,
        )
    finally:
        await file.close()
        runtime.operations.release()


@app.post('/synthesize')
def synthesize(request: Request, speech_request: LegacySpeechRequest) -> Response:
    runtime = _runtime(request)
    return _speech_response(
        runtime,
        SpeechRequest(model=runtime.settings.model_id, input=speech_request.text),
        pipeline=False,
    )


@app.post('/synthesize_stream')
def synthesize_stream(request: Request, speech_request: LegacySpeechRequest) -> Response:
    runtime = _runtime(request)
    return _speech_response(
        runtime,
        SpeechRequest(
            model=runtime.settings.model_id,
            input=speech_request.text,
            response_format='pcm',
        ),
        pipeline=False,
    )
