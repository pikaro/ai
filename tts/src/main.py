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
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, Protocol, cast

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from runtime_config import (
    ConfigurationUpdateResponse,
    ExclusiveOperationGate,
    reject_if_busy,
    validated_settings_patch,
)
from service_logging import configure_logging

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from starlette.types import Receive, Scope, Send

LOGGER = logging.getLogger('tts')
MODEL_ID: Final = 'kyutai/pocket-tts'
READ_CHUNK_BYTES: Final = 1024 * 1024
VOICE_NAME_PATTERN: Final = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}')
MODEL_READY = Gauge('tts_model_ready', 'Whether the TTS model and voice are loaded and ready')
MODEL_LOAD_SECONDS = Gauge('tts_model_load_seconds', 'Time spent loading the TTS model')
VOICE_LOAD_SECONDS = Gauge('tts_voice_load_seconds', 'Time spent loading the TTS voice')
ACTIVE_REQUESTS = Gauge('tts_active_requests', 'Active TTS inference requests')
BUSY_REJECTIONS = Counter(
    'tts_busy_rejections_total',
    'TTS inference and configuration requests rejected instead of queued',
)
CONFIGURATION_UPDATES = Counter(
    'tts_configuration_updates_total',
    'Successful ephemeral TTS configuration updates',
)
REQUESTS = Counter('tts_requests_total', 'Completed TTS requests', ['format', 'outcome'])
REQUEST_SECONDS = Histogram('tts_request_duration_seconds', 'TTS request latency', ['format'])
TIME_TO_FIRST_AUDIO = Histogram(
    'tts_time_to_first_audio_seconds',
    'Time from a PCM synthesis request to its first audio bytes',
)
AUDIO_SECONDS = Histogram('tts_output_audio_seconds', 'Audio duration produced by TTS')
REALTIME_FACTOR = Histogram(
    'tts_realtime_factor',
    'TTS generation wall time divided by output audio duration',
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 5),
)
RESTART_REQUIRED_SETTINGS: Final = frozenset({'model_id', 'language', 'listen_port'})


class _TorchModule(Protocol):
    int16: object

    def set_num_threads(self, threads: int, /) -> None: ...


class _SampleRateModel(Protocol):
    sample_rate: int


class _AtomicWavWriter:
    """Build a WAV incrementally and publish it only after successful completion."""

    def __init__(self, destination: Path, sample_rate: int) -> None:
        self.destination = destination
        self._temporary_path: Path | None = None
        self._temporary: Any | None = None
        self._wav_file: wave.Wave_write | None = None
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = tempfile.NamedTemporaryFile(  # noqa: SIM115
                dir=destination.parent,
                prefix=f'.{destination.name}.',
                suffix='.tmp',
                delete=False,
            )
            self._temporary = temporary
            self._temporary_path = Path(temporary.name)
            self._wav_file = wave.open(temporary, 'wb')  # noqa: SIM115
            self._wav_file.setnchannels(1)
            self._wav_file.setsampwidth(2)
            self._wav_file.setframerate(sample_rate)
        except (OSError, wave.Error):
            self.abort()
            raise

    def write(self, pcm: bytes) -> None:
        if self._wav_file is None:
            message = 'WAV capture is not open'
            raise RuntimeError(message)
        self._wav_file.writeframesraw(pcm)

    def commit(self) -> None:
        temporary = self._temporary
        temporary_path = self._temporary_path
        wav_file = self._wav_file
        if temporary is None or temporary_path is None or wav_file is None:
            message = 'WAV capture is not open'
            raise RuntimeError(message)

        wav_file.close()
        self._wav_file = None
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary.close()
        self._temporary = None
        _ = temporary_path.replace(self.destination)
        self._temporary_path = None

    def abort(self) -> None:
        wav_file = self._wav_file
        self._wav_file = None
        if wav_file is not None:
            with suppress(OSError, wave.Error):
                wav_file.close()

        temporary = self._temporary
        self._temporary = None
        if temporary is not None:
            with suppress(OSError):
                temporary.close()

        temporary_path = self._temporary_path
        self._temporary_path = None
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


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
    save_latest_wav: bool = False
    latest_wav_path: Path = Path(tempfile.gettempdir()) / 'latest.wav'
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
        self.operations = ExclusiveOperationGate()
        self.model: Any | None = None
        self.voice_states: dict[str, Any] = {}
        self.torch: _TorchModule | None = None
        self.load_seconds = 0.0
        self.voice_load_seconds = 0.0
        self.lock = threading.Lock()

    @staticmethod
    def _apply_torch_threads(
        torch_module: _TorchModule | None,
        previous_threads: int,
        new_threads: int,
    ) -> None:
        if torch_module is not None and new_threads != previous_threads:
            torch_module.set_num_threads(new_threads)

    def _replacement_voice(
        self,
        settings: Settings,
        *,
        voice_changed: bool,
    ) -> tuple[dict[str, Any], float]:
        if not voice_changed or self.model is None:
            return self.voice_states, self.voice_load_seconds

        data_directory_changed = settings.data_directory != self.settings.data_directory
        if not data_directory_changed and settings.voice in self.voice_states:
            return self.voice_states, self.voice_load_seconds

        started = time.perf_counter()
        voice_state = self.model.get_state_for_audio_prompt(
            self.voice_source(settings.voice, settings),
        )
        voice_states = {} if data_directory_changed else dict(self.voice_states)
        voice_states[settings.voice] = voice_state
        return voice_states, time.perf_counter() - started

    def apply_settings(self, settings: Settings) -> None:
        """Apply settings and load a replacement voice before publishing it."""
        previous = self.settings
        voice_changed = (
            settings.voice != previous.voice or settings.data_directory != previous.data_directory
        )
        with self.lock:
            torch_module = self.torch
            try:
                self._apply_torch_threads(
                    torch_module,
                    previous.torch_threads,
                    settings.torch_threads,
                )
                replacement_voices, replacement_voice_load_seconds = self._replacement_voice(
                    settings,
                    voice_changed=voice_changed,
                )
            except BaseException:
                self._apply_torch_threads(
                    torch_module,
                    settings.torch_threads,
                    previous.torch_threads,
                )
                raise
            self.settings = settings
            self.voice_states = replacement_voices
            self.voice_load_seconds = replacement_voice_load_seconds
            if voice_changed:
                VOICE_LOAD_SECONDS.set(self.voice_load_seconds)

    def load(self) -> None:
        LOGGER.info(
            'Loading TTS model',
            extra={'event_id': 'ID_tts_model_loading', 'model': self.settings.model_id},
        )
        self.settings.data_directory.mkdir(parents=True, exist_ok=True)
        self.torch = cast('_TorchModule', importlib.import_module('torch'))
        pocket_tts = importlib.import_module('pocket_tts')
        self.torch.set_num_threads(self.settings.torch_threads)

        started = time.perf_counter()
        model = pocket_tts.TTSModel.load_model(language=self.settings.language)
        self.model = model
        self.load_seconds = time.perf_counter() - started

        started = time.perf_counter()
        voice_source = self.voice_source(self.settings.voice)
        voice_state = model.get_state_for_audio_prompt(voice_source)
        self.voice_states = {self.settings.voice: voice_state}
        self.voice_load_seconds = time.perf_counter() - started
        MODEL_LOAD_SECONDS.set(self.load_seconds)
        VOICE_LOAD_SECONDS.set(self.voice_load_seconds)
        MODEL_READY.set(1)
        LOGGER.info(
            'TTS model ready',
            extra={
                'event_id': 'ID_tts_model_ready',
                'duration_seconds': self.load_seconds,
                'voice': voice_source,
                'voice_load_seconds': self.voice_load_seconds,
            },
        )

    def voice_source(self, voice: str, settings: Settings | None = None) -> str:
        selected = settings or self.settings
        if VOICE_NAME_PATTERN.fullmatch(voice):
            voice_path = selected.data_directory / f'{voice}.safetensors'
            if voice_path.is_file():
                return str(voice_path)
        return voice

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
        LOGGER.info(
            'Stored TTS voice',
            extra={'event_id': 'ID_tts_voice_stored', 'voice': name, 'replaced': replaced},
        )
        self.voice_states.pop(name, None)
        return VoiceUploadResponse(name=name, filename=destination.name, replaced=replaced)

    def close(self) -> None:
        MODEL_READY.set(0)
        self.voice_states.clear()
        self.model = None

    def health(self) -> HealthResponse:
        if self.model is None:
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

    def prepare_voice(self, requested_voice: str | None) -> str:
        """Resolve and cache a request voice before response streaming begins."""
        voice = self._selected_voice(requested_voice)
        model = self.model
        if model is None:
            message = 'TTS model is not loaded'
            raise RuntimeError(message)
        with self.lock:
            if voice in self.voice_states:
                return voice
            started = time.perf_counter()
            try:
                voice_state = model.get_state_for_audio_prompt(self.voice_source(voice))
            except (FileNotFoundError, ValueError) as error:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f'voice {voice!r} is not available',
                ) from error
            self.voice_load_seconds = time.perf_counter() - started
            self.voice_states[voice] = voice_state
            VOICE_LOAD_SECONDS.set(self.voice_load_seconds)
            LOGGER.info(
                'TTS voice loaded',
                extra={
                    'event_id': 'ID_tts_voice_loaded',
                    'voice': voice,
                    'duration_seconds': self.voice_load_seconds,
                    'cached_voices': len(self.voice_states),
                },
            )
        return voice

    def _selected_voice(self, requested_voice: str | None) -> str:
        if requested_voice is None:
            return self.settings.voice
        requested = requested_voice.strip()
        if not requested or requested.casefold() == 'default':
            return self.settings.voice
        if VOICE_NAME_PATTERN.fullmatch(requested) is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='voice must be "default" or a valid voice name',
            )
        return requested

    def validate_request(self, request: SpeechRequest) -> str:
        text = request.input.strip()
        if request.model != self.settings.model_id:
            detail = f'loaded model is {self.settings.model_id}, not {request.model}'
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

    def generate_wav(self, text: str, voice: str) -> bytes:
        model, voice_state = self._loaded_model(voice)
        with self.lock:
            audio = model.generate_audio(voice_state, text)
        wav = self._wav_bytes(int(model.sample_rate), audio)
        if self.settings.save_latest_wav:
            self._save_latest_wav_response(wav)
        return wav

    def stream_pcm(self, text: str, voice: str) -> Generator[bytes, None, None]:
        model, voice_state = self._loaded_model(voice)
        output_bytes = 0
        capture: _AtomicWavWriter | None = None
        completed = False
        with self.lock:
            if self.settings.save_latest_wav:
                capture = self._open_latest_wav_capture(int(model.sample_rate))
            try:
                for audio_chunk in model.generate_audio_stream(voice_state, text):
                    chunk = self._pcm16_bytes(audio_chunk)
                    if chunk:
                        output_bytes += len(chunk)
                        yield chunk
                        capture = self._write_latest_wav_chunk(capture, chunk)
                completed = True
            finally:
                self._finish_latest_wav_capture(
                    capture,
                    completed=completed,
                    pcm_bytes=output_bytes,
                )
        LOGGER.info(
            'Speech synthesis completed',
            extra={
                'event_id': 'ID_tts_synthesis_completed',
                'response_format': 'pcm',
                'audio_bytes': output_bytes,
            },
        )

    def _save_latest_wav_response(self, wav: bytes) -> None:
        """Atomically persist the exact WAV response body without regenerating audio."""
        destination = self.settings.latest_wav_path
        temporary_path: Path | None = None
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f'.{destination.name}.',
                suffix='.tmp',
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                _ = temporary.write(wav)
                temporary.flush()
                os.fsync(temporary.fileno())
            _ = temporary_path.replace(destination)
        except OSError:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)
            self._log_latest_wav_failure()
            return
        LOGGER.info(
            'Saved latest synthesized recording',
            extra={
                'event_id': 'ID_tts_latest_recording_saved',
                'response_format': 'wav',
                'audio_bytes': len(wav),
                'path': str(destination),
            },
        )

    def _open_latest_wav_capture(self, sample_rate: int) -> _AtomicWavWriter | None:
        try:
            return _AtomicWavWriter(self.settings.latest_wav_path, sample_rate)
        except (OSError, wave.Error):
            self._log_latest_wav_failure()
            return None

    def _write_latest_wav_chunk(
        self,
        capture: _AtomicWavWriter | None,
        chunk: bytes,
    ) -> _AtomicWavWriter | None:
        if capture is None:
            return None
        try:
            capture.write(chunk)
        except (OSError, wave.Error):
            capture.abort()
            self._log_latest_wav_failure()
            return None
        return capture

    def _finish_latest_wav_capture(
        self,
        capture: _AtomicWavWriter | None,
        *,
        completed: bool,
        pcm_bytes: int,
    ) -> None:
        if capture is None:
            return
        if not completed:
            capture.abort()
            return
        self._commit_latest_wav_capture(capture, pcm_bytes)

    def _commit_latest_wav_capture(
        self,
        capture: _AtomicWavWriter,
        pcm_bytes: int,
    ) -> None:
        try:
            capture.commit()
        except (OSError, wave.Error):
            capture.abort()
            self._log_latest_wav_failure()
            return
        LOGGER.info(
            'Saved latest synthesized recording',
            extra={
                'event_id': 'ID_tts_latest_recording_saved',
                'response_format': 'pcm',
                'pcm_bytes': pcm_bytes,
                'path': str(self.settings.latest_wav_path),
            },
        )

    def _log_latest_wav_failure(self) -> None:
        LOGGER.exception(
            'Failed to save latest synthesized recording',
            extra={
                'event_id': 'ID_tts_latest_recording_save_failed',
                'path': str(self.settings.latest_wav_path),
            },
        )

    def pcm_headers(self) -> dict[str, str]:
        model = self._loaded_model_only()
        return {
            'X-Audio-Format': 'pcm_s16le',
            'X-Audio-Sample-Rate': str(model.sample_rate),
            'X-Audio-Sample-Width': '2',
            'X-Audio-Channels': '1',
        }

    def sample_rate(self) -> int:
        model = self._loaded_model_only()
        return int(model.sample_rate)

    def _loaded_model_only(self) -> _SampleRateModel:
        if self.model is None:
            message = 'TTS model is not loaded'
            raise RuntimeError(message)
        return cast('_SampleRateModel', self.model)

    def _loaded_model(self, voice: str) -> tuple[Any, Any]:
        model = self._loaded_model_only()
        if voice not in self.voice_states:
            message = f'TTS voice {voice!r} is not loaded'
            raise RuntimeError(message)
        return model, self.voice_states[voice]

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


def _runtime(request: Request) -> TtsRuntime:
    return cast('TtsRuntime', request.app.state.runtime)


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


def _stream_speech(
    runtime: TtsRuntime,
    text: str,
    voice: str,
    started: float,
) -> Generator[bytes, None, None]:
    outcome = 'success'
    output_bytes = 0
    first_audio = True
    sample_rate = 0
    try:
        sample_rate = runtime.sample_rate()
        for chunk in runtime.stream_pcm(text, voice):
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
        runtime.operations.release()
        ACTIVE_REQUESTS.dec()
        _observe_request('pcm', outcome, started, output_bytes, sample_rate)


def _speech_response(  # noqa: C901
    runtime: TtsRuntime,
    speech_request: SpeechRequest,
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
        text = runtime.validate_request(speech_request)
        voice = runtime.prepare_voice(speech_request.voice)
        LOGGER.info(
            'Speech synthesis requested',
            extra={
                'event_id': 'ID_tts_synthesis_requested',
                'response_format': speech_request.response_format,
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
                _stream_speech(runtime, text, voice, started),
                media_type='application/octet-stream',
                headers=runtime.pcm_headers(),
            )
            stream_response = True
            return response
        wav = runtime.generate_wav(text, voice)
        LOGGER.info(
            'Speech synthesis completed',
            extra={
                'event_id': 'ID_tts_synthesis_completed',
                'response_format': 'wav',
                'audio_bytes': len(wav),
            },
        )
        with wave.open(io.BytesIO(wav), 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            pcm_bytes = wav_file.getnframes() * wav_file.getnchannels() * wav_file.getsampwidth()
        response = Response(wav, media_type='audio/wav')
        _observe_request('wav', outcome, started, pcm_bytes, sample_rate)
        request_observed = True
        return response  # noqa: TRY300
    except Exception:
        outcome = 'error'
        raise
    finally:
        if not stream_response:
            runtime.operations.release()
            ACTIVE_REQUESTS.dec()
            if not request_observed:
                _observe_request(speech_request.response_format, outcome, started, 0, 1)


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
def speech(request: Request, speech_request: SpeechRequest) -> Response:
    return _speech_response(_runtime(request), speech_request)


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
        return await asyncio.to_thread(runtime.save_voice, name, file)
    finally:
        await file.close()
        runtime.operations.release()


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
    uvicorn.run(
        app,
        host='0.0.0.0',  # noqa: S104
        port=SETTINGS.listen_port,
        log_config=None,
    )
