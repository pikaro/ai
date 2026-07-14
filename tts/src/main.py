from __future__ import annotations

import asyncio
import importlib
import io
import logging
import os
import re
import tempfile
import threading
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, cast

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator

LOGGER = logging.getLogger('tts')
MODEL_ID: Final = 'kyutai/pocket-tts'
READ_CHUNK_BYTES: Final = 1024 * 1024
VOICE_NAME_PATTERN: Final = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}')
MODEL_READY = Gauge('tts_model_ready', 'Whether the TTS model and voice are loaded and ready')
MODEL_LOAD_SECONDS = Gauge('tts_model_load_seconds', 'Time spent loading the TTS model')
VOICE_LOAD_SECONDS = Gauge('tts_voice_load_seconds', 'Time spent loading the TTS voice')
HEALTH_ENDPOINTS = frozenset({'/health', '/health/live', '/health/ready'})


class SuccessfulHealthCheckFilter(logging.Filter):
    """Suppress successful health checks while retaining failures and other requests."""

    def filter(self, record: logging.LogRecord) -> bool:
        arguments = record.args
        if not isinstance(arguments, tuple) or len(arguments) < 5:  # noqa: PLR2004
            return True
        method, path, status_code = arguments[1], arguments[2], arguments[4]
        request_path = path.partition('?')[0] if isinstance(path, str) else path
        return not (
            method == 'GET'
            and request_path in HEALTH_ENDPOINTS
            and status_code == status.HTTP_200_OK
        )


def _configure_logging(level: str) -> None:
    LOGGER.setLevel(level)
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(levelname)s: %(name)s: %(message)s'))
        LOGGER.addHandler(handler)
    LOGGER.propagate = False
    logging.getLogger('uvicorn.access').addFilter(SuccessfulHealthCheckFilter())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='TTS_',
        case_sensitive=False,
        frozen=True,
        populate_by_name=True,
        extra='ignore',
    )

    model_id: str = MODEL_ID
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
        default='INFO',
        validation_alias='LOG_LEVEL',
    )
    language: str = 'english'
    voice: str = 'alba'
    data_directory: Path = Path('/data')
    torch_threads: int = Field(default=2, ge=1)
    maximum_input_characters: int = Field(default=4_000, ge=1)
    maximum_voice_upload_bytes: int = Field(default=100 * 1024**2, ge=1)
    listen_port: int = Field(
        default=8080,
        ge=1,
        le=65_535,
        validation_alias='LISTEN_PORT',
    )


class HealthResponse(BaseModel):
    status: Literal['ok']
    model: str
    voice: str
    language: str
    streaming: Literal[True] = True
    stream_endpoint: Literal['/v1/audio/speech'] = '/v1/audio/speech'
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


class LegacySpeechRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    text: str = Field(min_length=1)


class ModelDescription(BaseModel):
    id: str
    object: Literal['model'] = 'model'
    owned_by: Literal['local'] = 'local'


class ModelList(BaseModel):
    object: Literal['list'] = 'list'
    data: list[ModelDescription]


class VoiceUploadResponse(BaseModel):
    name: str
    filename: str
    replaced: bool


class TtsRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Any | None = None
        self.voice_state: Any | None = None
        self.torch: Any | None = None
        self.load_seconds = 0.0
        self.voice_load_seconds = 0.0
        self.lock = threading.Lock()

    def load(self) -> None:
        LOGGER.info('loading TTS model %s', self.settings.model_id)
        self.settings.data_directory.mkdir(parents=True, exist_ok=True)
        self.torch = importlib.import_module('torch')
        pocket_tts = importlib.import_module('pocket_tts')
        self.torch.set_num_threads(self.settings.torch_threads)

        started = time.perf_counter()
        model = pocket_tts.TTSModel.load_model(language=self.settings.language)
        self.model = model
        self.load_seconds = time.perf_counter() - started

        started = time.perf_counter()
        voice_source = self.voice_source()
        self.voice_state = model.get_state_for_audio_prompt(voice_source)
        self.voice_load_seconds = time.perf_counter() - started
        MODEL_LOAD_SECONDS.set(self.load_seconds)
        VOICE_LOAD_SECONDS.set(self.voice_load_seconds)
        MODEL_READY.set(1)
        LOGGER.info(
            'TTS model ready in %.3f seconds (voice=%s loaded in %.3f seconds)',
            self.load_seconds,
            voice_source,
            self.voice_load_seconds,
        )

    def voice_source(self) -> str:
        if VOICE_NAME_PATTERN.fullmatch(self.settings.voice):
            voice_path = self.settings.data_directory / f'{self.settings.voice}.safetensors'
            if voice_path.is_file():
                return str(voice_path)
        return self.settings.voice

    def save_voice(self, name: str, upload: UploadFile) -> VoiceUploadResponse:
        if VOICE_NAME_PATTERN.fullmatch(name) is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='voice name must contain only letters, numbers, underscores, and hyphens',
            )
        if Path(upload.filename or '').suffix.casefold() != '.safetensors':
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail='voice upload must be a .safetensors file',
            )

        destination = self.settings.data_directory / f'{name}.safetensors'
        replaced = destination.exists()
        _save_upload_atomic(upload, destination, self.settings.maximum_voice_upload_bytes)
        LOGGER.info('stored TTS voice %s (replaced=%s)', name, replaced)
        return VoiceUploadResponse(name=name, filename=destination.name, replaced=replaced)

    def close(self) -> None:
        MODEL_READY.set(0)
        self.voice_state = None
        self.model = None

    def health(self) -> HealthResponse:
        if self.model is None or self.voice_state is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        return HealthResponse(
            status='ok',
            model=self.settings.model_id,
            voice=self.settings.voice,
            language=self.settings.language,
            sample_rate=int(self.model.sample_rate),
            load_seconds=self.load_seconds,
            voice_load_seconds=self.voice_load_seconds,
        )

    def validate_request(self, request: SpeechRequest) -> str:
        text = request.input.strip()
        if request.model != self.settings.model_id:
            detail = f'loaded model is {self.settings.model_id}, not {request.model}'
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
        if request.voice is not None and request.voice != self.settings.voice:
            detail = f'loaded voice is {self.settings.voice}, not {request.voice}'
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
        if request.speed != 1.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Pocket TTS does not support speed adjustment',
            )
        if len(text) > self.settings.maximum_input_characters:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail='input exceeds TTS_MAX_INPUT_CHARACTERS',
            )
        return text

    def generate_wav(self, text: str) -> bytes:
        model, voice_state = self._loaded_model()
        with self.lock:
            audio = model.generate_audio(voice_state, text)
        return self._wav_bytes(int(model.sample_rate), audio)

    def stream_pcm(self, text: str) -> Iterator[bytes]:
        model, voice_state = self._loaded_model()
        output_bytes = 0
        with self.lock:
            for audio_chunk in model.generate_audio_stream(voice_state, text):
                chunk = self._pcm16_bytes(audio_chunk)
                if chunk:
                    output_bytes += len(chunk)
                    yield chunk
        LOGGER.info('speech synthesis completed (format=pcm, audio_bytes=%d)', output_bytes)

    def pcm_headers(self) -> dict[str, str]:
        model, _ = self._loaded_model()
        return {
            'X-Audio-Format': 'pcm_s16le',
            'X-Audio-Sample-Rate': str(model.sample_rate),
            'X-Audio-Sample-Width': '2',
            'X-Audio-Channels': '1',
        }

    def _loaded_model(self) -> tuple[Any, Any]:
        if self.model is None or self.voice_state is None:
            message = 'TTS model is not loaded'
            raise RuntimeError(message)
        return self.model, self.voice_state

    def _pcm16_bytes(self, audio: Any) -> bytes:  # noqa: ANN401
        torch_module = self.torch
        if torch_module is None:
            message = 'PyTorch is not loaded'
            raise RuntimeError(message)
        tensor = audio.detach().cpu().flatten()
        if tensor.dtype.is_floating_point:
            tensor = tensor.clamp(-1.0, 1.0).mul(32767.0)
        tensor = tensor.to(torch_module.int16).contiguous()
        return cast('bytes', tensor.numpy().tobytes())

    def _wav_bytes(self, sample_rate: int, audio: Any) -> bytes:  # noqa: ANN401
        output = io.BytesIO()
        with wave.open(output, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(self._pcm16_bytes(audio))
        return output.getvalue()


SETTINGS = Settings()
_configure_logging(SETTINGS.log_level)


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


def _runtime(request: Request) -> TtsRuntime:
    return cast('TtsRuntime', request.app.state.runtime)


def _speech_response(runtime: TtsRuntime, speech_request: SpeechRequest) -> Response:
    text = runtime.validate_request(speech_request)
    LOGGER.info(
        'speech synthesis requested (format=%s, characters=%d)',
        speech_request.response_format,
        len(text),
    )
    LOGGER.debug('speech synthesis input: %r', text)
    if speech_request.response_format == 'pcm':
        return StreamingResponse(
            runtime.stream_pcm(text),
            media_type='application/octet-stream',
            headers=runtime.pcm_headers(),
        )
    wav = runtime.generate_wav(text)
    LOGGER.info('speech synthesis completed (format=wav, audio_bytes=%d)', len(wav))
    return Response(wav, media_type='audio/wav')


def _save_upload_atomic(upload: UploadFile, destination: Path, maximum_bytes: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f'.{destination.name}.',
            suffix='.tmp',
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            while chunk := upload.file.read(READ_CHUNK_BYTES):
                total_bytes += len(chunk)
                _check_voice_upload_size(total_bytes, maximum_bytes)
                _ = temporary.write(chunk)
            _check_voice_upload_not_empty(total_bytes)
            temporary.flush()
            os.fsync(temporary.fileno())
        _ = temporary_path.replace(destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _check_voice_upload_size(total_bytes: int, maximum_bytes: int) -> None:
    if total_bytes > maximum_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail='voice upload exceeds TTS_MAXIMUM_VOICE_UPLOAD_BYTES',
        )


def _check_voice_upload_not_empty(total_bytes: int) -> None:
    if total_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='voice upload is empty',
        )


@app.get('/health/live')
async def live() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/metrics', include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), headers={'Content-Type': CONTENT_TYPE_LATEST})


@app.get('/health', response_model=HealthResponse)
@app.get('/health/ready', response_model=HealthResponse)
async def ready(request: Request) -> HealthResponse:
    return _runtime(request).health()


@app.get('/v1/models', response_model=ModelList)
async def models(request: Request) -> ModelList:
    runtime = _runtime(request)
    return ModelList(data=[ModelDescription(id=runtime.settings.model_id)])


@app.post('/v1/audio/speech')
def speech(request: Request, speech_request: SpeechRequest) -> Response:
    return _speech_response(_runtime(request), speech_request)


@app.post('/v1/voices', response_model=VoiceUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_voice(
    request: Request,
    name: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
) -> VoiceUploadResponse:
    try:
        return await asyncio.to_thread(_runtime(request).save_voice, name, file)
    finally:
        await file.close()


@app.post('/synthesize')
def synthesize(request: Request, speech_request: LegacySpeechRequest) -> Response:
    runtime = _runtime(request)
    return _speech_response(
        runtime,
        SpeechRequest(model=runtime.settings.model_id, input=speech_request.text),
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
    )


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=SETTINGS.listen_port)  # noqa: S104
