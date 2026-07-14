from __future__ import annotations

import asyncio
import base64
import binascii
import importlib
import json
import logging
import os
import tempfile
import threading
import time
import wave
from contextlib import asynccontextmanager, nullcontext, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, cast

import uvicorn
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
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from service_logging import configure_logging

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

LOGGER = logging.getLogger('stt')
STREAM_STRATEGY: Final = 'cache_aware_conformer_stream_step_continuous_features'
READ_CHUNK_BYTES: Final = 1024 * 1024
SUPPORTED_UPLOAD_SUFFIXES: Final = frozenset(
    {'.flac', '.m4a', '.mp3', '.mp4', '.mpeg', '.mpga', '.ogg', '.wav', '.webm'},
)
MODEL_READY = Gauge('stt_model_ready', 'Whether the STT model is loaded and ready')
MODEL_LOAD_SECONDS = Gauge('stt_model_load_seconds', 'Time spent loading the STT model')


def _log_transcription(stage: str, text: str) -> None:
    LOGGER.info(
        'Transcription produced',
        extra={
            'event_id': 'ID_stt_transcription_produced',
            'stage': stage,
            'characters': len(text),
        },
    )
    LOGGER.debug(
        'Transcription',
        extra={'event_id': 'ID_stt_transcription', 'stage': stage, 'transcript': text},
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='STT_',
        case_sensitive=False,
        frozen=True,
        populate_by_name=True,
        extra='ignore',
    )

    model_id: str = 'nvidia/stt_en_fastconformer_hybrid_large_streaming_multi'
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
        default='INFO',
        validation_alias='LOG_LEVEL',
    )
    device: str = 'cpu'
    torch_threads: int = Field(default=4, ge=1)
    decoder_type: Literal['rnnt', 'ctc'] = 'rnnt'

    attention_context_size: Annotated[
        tuple[int, int] | None,
        NoDecode,
    ] = (70, 1)

    input_audio_seconds: float = Field(default=0.25, gt=0)
    final_flush_seconds: float = Field(default=0.25, ge=0)
    preprocess_holdback_seconds: float = Field(default=0.1, ge=0)
    minimum_delta_characters: int = Field(default=1, ge=1)

    online_normalization: bool = False
    pad_and_drop_preencoded: bool = False

    stream_sample_rate: int = Field(default=16_000, ge=8_000)
    maximum_upload_bytes: int = Field(default=100 * 1024**2, ge=1)
    maximum_stream_bytes: int = Field(
        default=30 * 60 * 16_000 * 2,
        ge=1,
    )
    maximum_websocket_message_bytes: int = Field(
        default=2 * 1024**2,
        ge=1,
    )
    save_latest_wav: bool = False
    latest_wav_path: Path = Path(tempfile.gettempdir()) / 'latest.wav'
    listen_port: int = Field(
        default=8080,
        ge=1,
        le=65_535,
        validation_alias='LISTEN_PORT',
    )

    @model_validator(mode='before')
    @classmethod
    def default_final_flush_to_input_size(cls, values: object) -> object:
        if not isinstance(values, dict):
            return values

        values = values.copy()

        if 'final_flush_seconds' not in values:
            values['final_flush_seconds'] = values.get(
                'input_audio_seconds',
                cls.model_fields['input_audio_seconds'].default,
            )

        return values

    @field_validator('attention_context_size', mode='before')
    @classmethod
    def parse_attention_context_size(
        cls,
        value: object,
    ) -> object:
        if value is None or value == '':
            return None

        if not isinstance(value, str):
            return value

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [item.strip() for item in value.split(',') if item.strip()]

        if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:  # noqa: PLR2004
            _err = f'expected a JSON array or comma-separated pair of integers, got {value!r}'
            raise ValueError(_err)

        return tuple(int(item) for item in parsed)


class HealthResponse(BaseModel):
    status: Literal['ok']
    backend: Literal['nemo'] = 'nemo'
    model: str
    device: str
    decoder_type: str
    attention_context_size: tuple[int, int] | None
    streaming: Literal[True] = True
    stream_endpoint: Literal['/v1/realtime'] = '/v1/realtime'
    stream_strategy: Literal['cache_aware_conformer_stream_step_continuous_features'] = (
        STREAM_STRATEGY
    )
    sample_rate: int | None
    load_seconds: float


class TranscriptionResponse(BaseModel):
    text: str


class ModelDescription(BaseModel):
    id: str
    object: Literal['model'] = 'model'
    owned_by: Literal['local'] = 'local'


class ModelList(BaseModel):
    object: Literal['list'] = 'list'
    data: list[ModelDescription]


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


def _extract_text(result: object) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        value = result.get('text') or result.get('transcript')
        return str(value or '').strip()
    value = getattr(result, 'text', None)
    return str(value if value is not None else result or '').strip()


def _extract_first_text(result: object) -> str:
    if isinstance(result, (list, tuple)):
        return '' if not result else _extract_text(result[0])
    return _extract_text(result)


def transcript_delta(previous: str, current: str) -> str:
    if not previous:
        return current.strip()
    return current[len(previous) :].strip() if current.startswith(previous) else ''


def stable_word_prefix(previous: str, current: str) -> str:
    common_length = 0
    for previous_character, current_character in zip(previous, current, strict=False):
        if previous_character != current_character:
            break
        common_length += 1
    if common_length == 0:
        return ''

    prefix = current[:common_length]
    if common_length < len(current) and not current[common_length].isspace():
        boundary = prefix.rstrip().rfind(' ')
        prefix = '' if boundary < 0 else prefix[:boundary]
    return prefix.strip()


class AsrRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Any | None = None
        self.torch: Any | None = None
        self.numpy: Any | None = None
        self.streaming_buffer_type: Any | None = None
        self.load_seconds = 0.0
        self.lock = threading.Lock()

    def load(self) -> None:
        started = time.perf_counter()
        LOGGER.info(
            'Loading ASR model',
            extra={'event_id': 'ID_stt_model_loading', 'model': self.settings.model_id},
        )
        self.numpy = importlib.import_module('numpy')
        self.torch = importlib.import_module('torch')
        nemo_asr = importlib.import_module('nemo.collections.asr')
        streaming_utils = importlib.import_module(
            'nemo.collections.asr.parts.utils.streaming_utils',
        )
        self.streaming_buffer_type = streaming_utils.CacheAwareStreamingAudioBuffer

        self.torch.set_num_threads(self.settings.torch_threads)
        self.torch.set_grad_enabled(False)
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.settings.model_id)
        self._configure_model(model)
        self.model = model.to(self.settings.device) if hasattr(model, 'to') else model
        self.load_seconds = time.perf_counter() - started
        MODEL_LOAD_SECONDS.set(self.load_seconds)
        MODEL_READY.set(1)
        LOGGER.info(
            'ASR model ready',
            extra={'event_id': 'ID_stt_model_ready', 'duration_seconds': self.load_seconds},
        )

    def close(self) -> None:
        MODEL_READY.set(0)
        self.model = None
        self.streaming_buffer_type = None

    def health(self) -> HealthResponse:
        if self.model is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        return HealthResponse(
            status='ok',
            model=self.settings.model_id,
            device=self.settings.device,
            decoder_type=self.settings.decoder_type,
            attention_context_size=self.settings.attention_context_size,
            sample_rate=self.model_sample_rate(),
            load_seconds=self.load_seconds,
        )

    def model_sample_rate(self) -> int | None:
        if self.model is None:
            return None
        config = getattr(self.model, 'cfg', None) or getattr(self.model, '_cfg', None)
        if config is None:
            return None
        value = getattr(config, 'sample_rate', None)
        if value is None and hasattr(config, 'get'):
            value = config.get('sample_rate')
        return int(value) if value else None

    def transcribe_file(self, path: Path) -> str:
        if self.model is None:
            message = 'ASR model is not loaded'
            raise RuntimeError(message)
        inference_context = self.torch.inference_mode() if self.torch is not None else nullcontext()
        with self.lock, inference_context:
            result = self.model.transcribe([str(path)], batch_size=1)
        if isinstance(result, tuple):
            result = result[0]
        return _extract_first_text(result)

    def save_latest_wav(self, pcm: bytes, sample_rate: int, channels: int) -> None:
        """Atomically replace the latest diagnostic recording with raw client PCM."""
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
                with wave.open(temporary, 'wb') as wav_file:
                    wav_file.setnchannels(channels)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(pcm)
                temporary.flush()
                os.fsync(temporary.fileno())
            _ = temporary_path.replace(destination)
        except (OSError, wave.Error):
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)
            LOGGER.exception(
                'Failed to save latest input recording',
                extra={'event_id': 'ID_stt_latest_recording_save_failed', 'path': str(destination)},
            )
            return
        LOGGER.info(
            'Saved latest input recording',
            extra={
                'event_id': 'ID_stt_latest_recording_saved',
                'pcm_bytes': len(pcm),
                'sample_rate': sample_rate,
                'channels': channels,
                'path': str(destination),
            },
        )

    def pcm16_to_float32(self, pcm: bytes, channels: int) -> Any:  # noqa: ANN401
        if self.numpy is None:
            message = 'NumPy is not loaded'
            raise RuntimeError(message)
        samples = self.numpy.frombuffer(pcm, dtype='<i2').astype(self.numpy.float32) / 32768.0
        if channels > 1:
            usable_samples = samples.size - (samples.size % channels)
            samples = samples[:usable_samples].reshape(-1, channels).mean(axis=1)
        return samples

    def drop_extra_pre_encoded(self, step_number: int) -> int:
        if step_number == 0 and not self.settings.pad_and_drop_preencoded:
            return 0
        encoder = getattr(self.model, 'encoder', None)
        streaming_config = getattr(encoder, 'streaming_cfg', None)
        return int(getattr(streaming_config, 'drop_extra_pre_encoded', 0) or 0)

    def create_stream(self, sample_rate: int, channels: int) -> CacheAwareStreamingSession:
        return CacheAwareStreamingSession(self, sample_rate, channels)

    def _configure_model(self, model: Any) -> None:  # noqa: ANN401
        self._configure_decoder(model)
        self._configure_encoder(model)
        self._configure_preprocessor(model)
        if hasattr(model, 'freeze'):
            model.freeze()
        if hasattr(model, 'eval'):
            model.eval()

    def _configure_decoder(self, model: Any) -> None:  # noqa: ANN401
        decoder_module_name = (
            'nemo.collections.asr.parts.submodules.rnnt_decoding'
            if self.settings.decoder_type == 'rnnt'
            else 'nemo.collections.asr.parts.submodules.ctc_decoding'
        )
        decoder_module = importlib.import_module(decoder_module_name)
        decoder_config = (
            decoder_module.RNNTDecodingConfig(fused_batch_size=-1)
            if self.settings.decoder_type == 'rnnt'
            else decoder_module.CTCDecodingConfig()
        )
        if hasattr(model, 'cur_decoder'):
            model.change_decoding_strategy(
                decoder_config,
                decoder_type=self.settings.decoder_type,
            )
        else:
            model.change_decoding_strategy(decoder_config)

    def _configure_encoder(self, model: Any) -> None:  # noqa: ANN401
        encoder = getattr(model, 'encoder', None)
        if self.settings.attention_context_size and hasattr(
            encoder,
            'set_default_att_context_size',
        ):
            cast('Any', encoder).set_default_att_context_size(
                list(self.settings.attention_context_size),
            )

    @staticmethod
    def _configure_preprocessor(model: Any) -> None:  # noqa: ANN401
        featurizer = getattr(getattr(model, 'preprocessor', None), 'featurizer', None)
        if featurizer is not None:
            if hasattr(featurizer, 'dither'):
                featurizer.dither = 0.0
            if hasattr(featurizer, 'pad_to'):
                featurizer.pad_to = 0


class CacheAwareStreamingSession:
    def __init__(self, runtime: AsrRuntime, sample_rate: int, channels: int) -> None:
        if runtime.model is None or runtime.torch is None or runtime.streaming_buffer_type is None:
            message = 'ASR model is not loaded'
            raise RuntimeError(message)
        expected_sample_rate = runtime.model_sample_rate()
        if expected_sample_rate is not None and sample_rate != expected_sample_rate:
            msg = f'expected {expected_sample_rate} Hz PCM, got {sample_rate} Hz'
            raise ValueError(msg)

        self.runtime = runtime
        self.sample_rate = sample_rate
        self.channels = channels
        self.input_chunk_bytes = max(
            channels * 2,
            int(sample_rate * runtime.settings.input_audio_seconds) * channels * 2,
        )
        self.holdback_feature_frames = self._seconds_to_feature_frames(
            runtime.settings.preprocess_holdback_seconds,
        )
        self.raw_pcm = bytearray()
        self.unprocessed_bytes = 0
        self.appended_feature_frames = 0
        self.stream_id = -1
        self.streaming_buffer = runtime.streaming_buffer_type(
            model=runtime.model,
            online_normalization=runtime.settings.online_normalization,
            pad_and_drop_preencoded=runtime.settings.pad_and_drop_preencoded,
        )
        (
            self.cache_last_channel,
            self.cache_last_time,
            self.cache_last_channel_length,
        ) = runtime.model.encoder.get_initial_cache_state(batch_size=1)
        self.previous_hypotheses: Any | None = None
        self.previous_prediction: Any | None = None
        self.step_number = 0
        self.emitted_text = ''
        self.previous_transcript = ''
        self.latest_transcript = ''

    def append_pcm(self, pcm: bytes) -> list[dict[str, object]]:
        if len(self.raw_pcm) + len(pcm) > self.runtime.settings.maximum_stream_bytes:
            message = 'audio stream exceeds STT_MAXIMUM_STREAM_BYTES'
            raise OverflowError(message)
        self.raw_pcm.extend(pcm)
        self.unprocessed_bytes += len(pcm)
        if self.unprocessed_bytes < self.input_chunk_bytes:
            return []
        with self.runtime.lock:
            self._append_ready_features(final=False)
            messages = self._process_ready(final=False)
        self.unprocessed_bytes = 0
        return messages

    def finish(self) -> list[dict[str, object]]:
        with self.runtime.lock:
            if self.runtime.settings.save_latest_wav:
                self.runtime.save_latest_wav(bytes(self.raw_pcm), self.sample_rate, self.channels)
            if self.runtime.settings.final_flush_seconds > 0 and self.raw_pcm:
                flush_samples = max(
                    1,
                    int(self.sample_rate * self.runtime.settings.final_flush_seconds),
                )
                self.raw_pcm.extend(b'\x00\x00' * flush_samples * self.channels)
            self._append_ready_features(final=True)
            messages = self._process_ready(final=True)
        completed = self.latest_transcript or self.emitted_text
        if completed:
            _log_transcription('completed', completed)
            messages.append(
                {
                    'type': 'conversation.item.input_audio_transcription.completed',
                    'transcript': completed,
                    'source': STREAM_STRATEGY,
                },
            )
        return messages

    def _seconds_to_feature_frames(self, seconds: float) -> int:
        config = getattr(self.runtime.model, 'cfg', None) or getattr(
            self.runtime.model,
            '_cfg',
            None,
        )
        if config is None:
            return 0
        preprocessor_config = getattr(config, 'preprocessor', None)
        if preprocessor_config is None:
            return 0
        window_stride = getattr(preprocessor_config, 'window_stride', None)
        if window_stride is None and hasattr(preprocessor_config, 'get'):
            window_stride = preprocessor_config.get('window_stride')
        return 0 if not window_stride else max(0, int(seconds / float(window_stride)))

    def _append_ready_features(self, *, final: bool) -> None:
        audio = self.runtime.pcm16_to_float32(bytes(self.raw_pcm), self.channels)
        if audio.size == 0:
            return
        processed_signal, processed_length = self.streaming_buffer.preprocess_audio(audio)
        total_frames = int(processed_length.item())
        ready_frames = (
            total_frames if final else max(0, total_frames - self.holdback_feature_frames)
        )
        if ready_frames <= self.appended_feature_frames:
            return
        new_signal = processed_signal[:, :, self.appended_feature_frames : ready_frames]
        _, _, stream_id = self.streaming_buffer.append_processed_signal(
            new_signal,
            stream_id=self.stream_id,
        )
        self.stream_id = max(0, stream_id)
        self.appended_feature_frames = ready_frames

    def _process_ready(self, *, final: bool) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = []
        if getattr(self.streaming_buffer, 'buffer', None) is None:
            return messages
        torch_module = self.runtime.torch
        model = self.runtime.model
        if torch_module is None or model is None:
            message = 'ASR model is not loaded'
            raise RuntimeError(message)
        for chunk_audio, chunk_lengths in self.streaming_buffer:
            with torch_module.inference_mode():
                (
                    self.previous_prediction,
                    transcribed_texts,
                    self.cache_last_channel,
                    self.cache_last_time,
                    self.cache_last_channel_length,
                    self.previous_hypotheses,
                ) = model.conformer_stream_step(
                    processed_signal=chunk_audio.to(torch_module.float32),
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=self.cache_last_channel,
                    cache_last_time=self.cache_last_time,
                    cache_last_channel_len=self.cache_last_channel_length,
                    keep_all_outputs=final and self.streaming_buffer.is_buffer_empty(),
                    previous_hypotheses=self.previous_hypotheses,
                    previous_pred_out=self.previous_prediction,
                    drop_extra_pre_encoded=self.runtime.drop_extra_pre_encoded(self.step_number),
                    return_transcription=True,
                )
            self.step_number += 1
            transcript = _extract_first_text(transcribed_texts)
            if transcript:
                messages.extend(self._transcript_messages(transcript, final=final))
        return messages

    def _transcript_messages(
        self,
        transcript: str,
        *,
        final: bool,
    ) -> list[dict[str, object]]:
        stable_transcript = (
            transcript if final else stable_word_prefix(self.previous_transcript, transcript)
        )
        self.previous_transcript = transcript
        self.latest_transcript = transcript
        delta = transcript_delta(self.emitted_text, stable_transcript)
        if delta and len(delta) >= self.runtime.settings.minimum_delta_characters:
            self.emitted_text = stable_transcript
            _log_transcription('stable delta', delta)
            return [
                {
                    'type': 'conversation.item.input_audio_transcription.delta',
                    'delta': delta,
                    'transcript': stable_transcript,
                    'source': STREAM_STRATEGY,
                },
            ]
        if not final:
            _log_transcription('partial', transcript)
            return [
                {
                    'type': 'conversation.item.input_audio_transcription.partial',
                    'transcript': transcript,
                    'source': STREAM_STRATEGY,
                },
            ]
        return []


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


def _runtime_from_request(request: Request) -> AsrRuntime:
    return cast('AsrRuntime', request.app.state.runtime)


def _runtime_from_websocket(websocket: WebSocket) -> AsrRuntime:
    return cast('AsrRuntime', websocket.app.state.runtime)


def _validate_model(requested_model: str, settings: Settings) -> None:
    if requested_model and requested_model != settings.model_id:
        detail = f'loaded model is {settings.model_id}, not {requested_model}'
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def _check_upload_size(total_bytes: int, maximum_bytes: int) -> None:
    if total_bytes > maximum_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail='audio upload exceeds STT_MAXIMUM_UPLOAD_BYTES',
        )


def _upload_suffix(upload: UploadFile) -> str:
    suffix = Path(upload.filename or 'audio.wav').suffix.casefold() or '.wav'
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f'unsupported audio extension: {suffix}',
        )
    return suffix


def _save_upload(upload: UploadFile, maximum_bytes: int) -> Path:
    suffix = _upload_suffix(upload)
    total_bytes = 0
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        path = Path(temporary.name)
        try:
            while chunk := upload.file.read(READ_CHUNK_BYTES):
                total_bytes += len(chunk)
                _check_upload_size(total_bytes, maximum_bytes)
                _ = temporary.write(chunk)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    return path


@app.get('/health/live')
async def live() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/metrics', include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), headers={'Content-Type': CONTENT_TYPE_LATEST})


@app.get('/health', response_model=HealthResponse)
@app.get('/health/ready', response_model=HealthResponse)
async def ready(request: Request) -> HealthResponse:
    return _runtime_from_request(request).health()


@app.get('/v1/models', response_model=ModelList)
async def models(request: Request) -> ModelList:
    runtime = _runtime_from_request(request)
    return ModelList(data=[ModelDescription(id=runtime.settings.model_id)])


@app.post('/v1/audio/transcriptions', response_model=TranscriptionResponse)
async def transcribe(
    request: Request,
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()] = '',
) -> TranscriptionResponse:
    runtime = _runtime_from_request(request)
    _validate_model(model, runtime.settings)
    path = await asyncio.to_thread(_save_upload, file, runtime.settings.maximum_upload_bytes)
    try:
        text = await asyncio.to_thread(runtime.transcribe_file, path)
        _log_transcription('file', text)
    finally:
        await file.close()
        await asyncio.to_thread(path.unlink, missing_ok=True)
    return TranscriptionResponse(text=text)


async def _send_error(websocket: WebSocket, message: str) -> None:
    await websocket.send_json({'type': 'error', 'message': message})


@app.websocket('/v1/realtime')
async def realtime(websocket: WebSocket) -> None:  # noqa: C901, PLR0912
    runtime = _runtime_from_websocket(websocket)
    await websocket.accept()
    LOGGER.info(
        'realtime transcription session started',
        extra={'event_id': 'ID_stt_realtime_session_started'},
    )
    sample_rate = runtime.settings.stream_sample_rate
    channels = 1
    stream: CacheAwareStreamingSession | None = None
    try:
        while True:
            raw = await websocket.receive_text()
            if len(raw.encode()) > runtime.settings.maximum_websocket_message_bytes:
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
                    {'type': 'session.updated', 'session': runtime.health().model_dump()},
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
                stream = stream or runtime.create_stream(sample_rate, channels)
                try:
                    messages = await asyncio.to_thread(stream.append_pcm, pcm)
                except OverflowError as error:
                    await _send_error(websocket, str(error))
                    await websocket.close(code=status.WS_1009_MESSAGE_TOO_BIG)
                    return
                for message in messages:
                    await websocket.send_json(message)
                continue

            if stream is None:
                await _send_error(websocket, 'cannot commit an empty audio buffer')
                continue
            for message in await asyncio.to_thread(stream.finish):
                await websocket.send_json(message)
            await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
            LOGGER.info(
                'realtime transcription session completed',
                extra={'event_id': 'ID_stt_realtime_session_completed'},
            )
            return
    except WebSocketDisconnect as error:
        LOGGER.info(
            'Realtime transcription client disconnected',
            extra={'event_id': 'ID_stt_realtime_client_disconnected', 'code': error.code},
        )


if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',  # noqa: S104
        port=SETTINGS.listen_port,
        log_config=None,
    )
