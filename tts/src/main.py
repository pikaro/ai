from __future__ import annotations

import asyncio
import importlib
import io
import logging
import math
import os
import re
import sys
import tempfile
import threading
import time
import wave
from array import array
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, Never, Protocol, cast

import uvicorn
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
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.websockets import WebSocketState

from runtime_config import (
    ConfigurationUpdateResponse,
    ExclusiveOperationGate,
    reject_if_busy,
    validated_settings_patch,
)
from service_logging import configure_logging

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Generator, Iterator

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
PIPELINE_REQUESTS = Counter(
    'tts_pipeline_requests_total',
    'Completed server-side text segmentation and stitching pipelines',
    ['transport', 'outcome'],
)
PIPELINE_SEGMENTS = Counter(
    'tts_pipeline_segments_total',
    'Text segments synthesized by the TTS pipeline',
    ['transport'],
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


class _TextSegmenter:
    """Incrementally split text while retaining only the incomplete tail."""

    def __init__(
        self,
        terminators: str,
        *,
        first_segment_comma_delimiter: bool,
    ) -> None:
        self.terminators = terminators
        self.first_segment_comma_delimiter = first_segment_comma_delimiter
        self.pending = ''
        self.first_segment = True

    def append(self, text: str) -> list[tuple[str, Literal['clause', 'sentence', 'input_end']]]:
        self.pending += text
        segments: list[tuple[str, Literal['clause', 'sentence', 'input_end']]] = []
        while match := self._next_boundary():
            segment = self.pending[: match.end()].strip()
            self.pending = self.pending[match.end() :].lstrip()
            if segment:
                boundary: Literal['clause', 'sentence', 'input_end'] = (
                    'clause' if match.group() == ',' else 'sentence'
                )
                segments.append((segment, boundary))
                self.first_segment = False
        return segments

    def finish(self) -> list[tuple[str, Literal['clause', 'sentence', 'input_end']]]:
        segments = self.append('')
        tail = self.pending.strip()
        self.pending = ''
        if tail:
            segments.append((tail, 'input_end'))
            self.first_segment = False
        return segments

    def _next_boundary(self) -> re.Match[str] | None:
        terminators = self.terminators
        if self.first_segment and self.first_segment_comma_delimiter:
            terminators += ','
        return re.search(rf'[{re.escape(terminators)}](?=\s|$)', self.pending)


def _duration_frames(duration_seconds: float, sample_rate: int) -> int:
    return int(duration_seconds * sample_rate + 0.5)


def _linear_fade_pcm16(  # noqa: C901
    pcm: bytes,
    *,
    start_frame: int,
    fade_frames: int,
    fade_in: bool,
) -> bytes:
    if not pcm or fade_frames <= 1 or start_frame >= fade_frames:
        return pcm
    if len(pcm) % 2:
        message = 'TTS returned a partial PCM16 frame'
        raise RuntimeError(message)

    samples = array('h')
    samples.frombytes(pcm)
    if sys.byteorder == 'big':
        samples.byteswap()
    frames_to_fade = min(len(samples), fade_frames - start_frame)
    denominator = fade_frames - 1
    for frame_offset in range(frames_to_fade):
        fade_frame = start_frame + frame_offset
        numerator = fade_frame if fade_in else denominator - fade_frame
        samples[frame_offset] = round(samples[frame_offset] * numerator / denominator)
    if sys.byteorder == 'big':
        samples.byteswap()
    return samples.tobytes()


def _pcm16_peak(pcm: bytes) -> int:
    if len(pcm) % 2:
        message = 'TTS returned a partial PCM16 frame'
        raise RuntimeError(message)
    samples = array('h')
    samples.frombytes(pcm)
    if sys.byteorder == 'big':
        samples.byteswap()
    return max((abs(sample) for sample in samples), default=0)


class _SentencePcmBuffer:
    """Stream speech while normalizing model silence at a stitched boundary."""

    def __init__(  # noqa: PLR0913
        self,
        sample_rate: int,
        settings: Settings,
        *,
        boundary: Literal['clause', 'sentence', 'input_end'],
        fade_in: bool,
        transport: str,
        on_tail_clipped: Callable[[int], None],
        on_false_tail: Callable[[], None],
    ) -> None:
        self.sample_rate = sample_rate
        self.boundary = boundary
        self.transport = transport
        self.crossfade_frames = _duration_frames(
            settings.pipeline_sentence_crossfade_seconds,
            sample_rate,
        )
        boundary_pause_seconds = (
            settings.pipeline_clause_pause_seconds
            if boundary == 'clause'
            else settings.pipeline_sentence_pause_seconds
        )
        self.boundary_pause_frames = _duration_frames(boundary_pause_seconds, sample_rate)
        self.silence_confirmation_frames = max(
            1,
            _duration_frames(settings.pipeline_silence_confirmation_seconds, sample_rate),
        )
        self.silence_threshold = round(
            32_767 * math.pow(10, settings.pipeline_silence_threshold_dbfs / 20),
        )
        self.analysis_frame_bytes = max(2, _duration_frames(0.01, sample_rate) * 2)
        self.fade_in = fade_in
        self.frames_emitted = 0
        self.buffer = bytearray()
        self.boundary_silence = bytearray()
        self.analysis_buffer = bytearray()
        self.silence_candidate = bytearray()
        self.generated_frames = 0
        self.trailing_silence_frames = 0
        self.emitted_trailing_silence_frames = 0
        self.clipped_silence_frames = 0
        self.leading_silence_frames = 0
        self.seen_voice = False
        self.silence_confirmed = False
        self.first_voice_at: float | None = None
        self.model_finished_at: float | None = None
        self.on_tail_clipped = on_tail_clipped
        self.on_false_tail = on_false_tail

    def append(self, pcm: bytes) -> bytes:
        if len(pcm) % 2:
            message = 'TTS returned a partial PCM16 frame'
            raise RuntimeError(message)
        self.generated_frames += len(pcm) // 2
        self.analysis_buffer.extend(pcm)
        output = bytearray()
        while len(self.analysis_buffer) >= self.analysis_frame_bytes:
            frame = bytes(self.analysis_buffer[: self.analysis_frame_bytes])
            del self.analysis_buffer[: self.analysis_frame_bytes]
            output.extend(self._process_frame(frame))
        if (
            self.analysis_buffer
            and _pcm16_peak(bytes(self.analysis_buffer)) > self.silence_threshold
        ):
            frame = bytes(self.analysis_buffer)
            self.analysis_buffer.clear()
            output.extend(self._process_voice(frame))
        return bytes(output)

    def end_model(self) -> bytes:
        output = bytearray()
        if self.analysis_buffer:
            frame = bytes(self.analysis_buffer)
            self.analysis_buffer.clear()
            output.extend(self._process_frame(frame))
        output.extend(
            self._finish_silence_candidate(fade_out=self.boundary != 'input_end'),
        )
        self.model_finished_at = time.perf_counter()
        return bytes(output)

    @property
    def pending_stitch_frames(self) -> int:
        return (
            len(self.buffer) // 2
            + len(self.boundary_silence) // 2
            + max(
                0,
                self.boundary_pause_frames - self.emitted_trailing_silence_frames,
            )
        )

    def finish(self, *, fade_out: bool) -> bytes:
        if self.model_finished_at is None:
            _ = self.end_model()

        output = bytearray(self._finish_silence_candidate(fade_out=fade_out))

        buffered = self._apply_fade_in(bytes(self.buffer))
        self.buffer.clear()
        if fade_out:
            buffered = _linear_fade_pcm16(
                buffered,
                start_frame=0,
                fade_frames=len(buffered) // 2,
                fade_in=False,
            )
        output.extend(buffered)
        output.extend(self.boundary_silence)
        self.boundary_silence.clear()
        if fade_out:
            missing_pause_frames = max(
                0,
                self.boundary_pause_frames - self.emitted_trailing_silence_frames,
            )
            if missing_pause_frames:
                output.extend(b'\0' * (missing_pause_frames * 2))
        return bytes(output)

    def _finish_silence_candidate(self, *, fade_out: bool) -> bytes:
        if not self.silence_candidate:
            return b''
        if fade_out:
            self.silence_confirmed = True
            return self._consume_silence_candidate()

        output = self._append_output(bytes(self.silence_candidate))
        self.emitted_trailing_silence_frames += len(self.silence_candidate) // 2
        self.silence_candidate.clear()
        return output

    def _process_frame(self, pcm: bytes) -> bytes:
        if _pcm16_peak(pcm) <= self.silence_threshold:
            return self._process_silence(pcm)
        return self._process_voice(pcm)

    def _process_silence(self, pcm: bytes) -> bytes:
        frames = len(pcm) // 2
        if not self.seen_voice:
            self.leading_silence_frames += frames
            return b''

        self.trailing_silence_frames += frames
        self.silence_candidate.extend(pcm)
        if (
            not self.silence_confirmed
            and self.trailing_silence_frames < self.silence_confirmation_frames
        ):
            return b''
        self.silence_confirmed = True
        return self._consume_silence_candidate()

    def _process_voice(self, pcm: bytes) -> bytes:
        output = bytearray()
        if not self.seen_voice:
            self.seen_voice = True
            self.first_voice_at = time.perf_counter()
        elif self.trailing_silence_frames:
            if not self.silence_confirmed:
                output.extend(self._append_output(bytes(self.silence_candidate)))
            else:
                output.extend(self._append_output(bytes(self.boundary_silence)))
                if self.clipped_silence_frames:
                    self.on_false_tail()
                    LOGGER.error(
                        'TTS stitcher found voiced audio after clipped silence',
                        extra={
                            'event_id': 'ID_tts_pipeline_stitch_false_tail',
                            'transport': self.transport,
                            'boundary': self.boundary,
                            'clipped_seconds': self.clipped_silence_frames / self.sample_rate,
                            'generated_seconds': self.generated_frames / self.sample_rate,
                        },
                    )
            self._reset_silence()
        output.extend(self._append_output(pcm))
        return bytes(output)

    def _consume_silence_candidate(self) -> bytes:
        tail_was_already_clipped = self.clipped_silence_frames > 0
        candidate_frames = len(self.silence_candidate) // 2
        allowed_frames = min(
            candidate_frames,
            max(
                0,
                self.boundary_pause_frames - self.emitted_trailing_silence_frames,
            ),
        )
        allowed_bytes = allowed_frames * 2
        self.boundary_silence.extend(self.silence_candidate[:allowed_bytes])
        clipped_frames = candidate_frames - allowed_frames
        self.clipped_silence_frames += clipped_frames
        self.emitted_trailing_silence_frames += allowed_frames
        self.silence_candidate.clear()
        if clipped_frames and not tail_was_already_clipped:
            self.on_tail_clipped(self.frames_emitted + self.pending_stitch_frames)
        return b''

    def _append_output(self, pcm: bytes) -> bytes:
        self.buffer.extend(pcm)
        emit_bytes = max(0, len(self.buffer) - self.crossfade_frames * 2)
        emit_bytes -= emit_bytes % 2
        if not emit_bytes:
            return b''
        output = bytes(self.buffer[:emit_bytes])
        del self.buffer[:emit_bytes]
        return self._apply_fade_in(output)

    def _apply_fade_in(self, pcm: bytes) -> bytes:
        output = pcm
        if self.fade_in:
            output = _linear_fade_pcm16(
                pcm,
                start_frame=self.frames_emitted,
                fade_frames=self.crossfade_frames,
                fade_in=True,
            )
        self.frames_emitted += len(pcm) // 2
        return output

    def _reset_silence(self) -> None:
        self.silence_candidate.clear()
        self.boundary_silence.clear()
        self.trailing_silence_frames = 0
        self.emitted_trailing_silence_frames = 0
        self.clipped_silence_frames = 0
        self.silence_confirmed = False


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
    pipeline_clause_pause_seconds: float = Field(default=0.04, ge=0, le=2)
    pipeline_sentence_pause_seconds: float = Field(default=0.12, ge=0, le=2)
    pipeline_sentence_crossfade_seconds: float = Field(default=0.01, ge=0, le=0.25)
    pipeline_silence_confirmation_seconds: float = Field(default=0.02, gt=0, le=0.1)
    pipeline_silence_threshold_dbfs: float = Field(default=-50.0, ge=-100, le=0)
    pipeline_sentence_terminators: str = Field(default='.!?', min_length=1)
    pipeline_first_segment_comma_delimiter: bool = True
    pipeline_idle_timeout_seconds: float = Field(default=30.0, gt=0)
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
    pipeline_endpoint: Literal['/v1/audio/speech/pipeline'] = '/v1/audio/speech/pipeline'
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


class PipelineSessionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: Literal['session.start']
    model: str = MODEL_ID
    voice: str | None = None
    speed: float = Field(default=1.0, gt=0)


class PipelineTextDelta(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: Literal['input_text.delta']
    delta: str = Field(min_length=1)


class PipelineTextDone(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: Literal['input_text.done']


PipelineInputEvent = Annotated[
    PipelineTextDelta | PipelineTextDone,
    Field(discriminator='type'),
]
PIPELINE_INPUT_ADAPTER = TypeAdapter(PipelineInputEvent)


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
        _ = self.settings.data_directory.mkdir(parents=True, exist_ok=True)
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
        self.validate_options(request.model, request.speed)
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='input must contain non-whitespace text',
            )
        if len(text) > self.settings.maximum_input_characters:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail='input exceeds TTS_MAX_INPUT_CHARACTERS',
            )
        return text

    def validate_options(self, model: str, speed: float) -> None:
        if model != self.settings.model_id:
            detail = f'loaded model is {self.settings.model_id}, not {model}'
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
        if speed != 1.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Pocket TTS does not support speed adjustment',
            )

    def generate_wav(self, text: str, voice: str) -> bytes:
        model, voice_state = self._loaded_model(voice)
        with self.lock:
            audio = model.generate_audio(voice_state, text)
        wav = self._wav_bytes(int(model.sample_rate), audio)
        if self.settings.save_latest_wav:
            self._save_latest_wav_response(wav)
        return wav

    def stream_model_pcm(self, text: str, voice: str) -> Generator[bytes, None, None]:
        """Stream raw model PCM without request-level recording semantics."""
        model, voice_state = self._loaded_model(voice)
        with self.lock:
            for audio_chunk in model.generate_audio_stream(voice_state, text):
                chunk = self._pcm16_bytes(audio_chunk)
                if chunk:
                    yield chunk

    def stream_pcm(self, text: str, voice: str) -> Generator[bytes, None, None]:
        output_bytes = 0
        capture: _AtomicWavWriter | None = None
        completed = False
        if self.settings.save_latest_wav:
            capture = self.open_latest_wav_capture(self.sample_rate())
        try:
            for chunk in self.stream_model_pcm(text, voice):
                output_bytes += len(chunk)
                yield chunk
                capture = self.write_latest_wav_chunk(capture, chunk)
            completed = True
        finally:
            self.finish_latest_wav_capture(
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

    def generate_pipeline_wav(self, text: str, voice: str) -> tuple[bytes, int]:
        pcm = b''.join(self.stream_pipeline_pcm(text, voice, capture_latest=False))
        wav = self._wav_from_pcm(self.sample_rate(), pcm)
        if self.settings.save_latest_wav:
            self._save_latest_wav_response(wav)
        return wav, len(pcm)

    def stream_pipeline_pcm(
        self,
        text: str,
        voice: str,
        *,
        capture_latest: bool = True,
        transport: str = 'http',
    ) -> Generator[bytes, None, None]:
        segmenter = _TextSegmenter(
            self.settings.pipeline_sentence_terminators,
            first_segment_comma_delimiter=self.settings.pipeline_first_segment_comma_delimiter,
        )
        segments = [*segmenter.append(text), *segmenter.finish()]
        pipeline = _PcmPipeline(
            self,
            voice,
            capture_latest=capture_latest,
            transport=transport,
        )
        completed = False
        try:
            for segment, boundary in segments:
                yield from pipeline.add_segment(segment, boundary)
            yield from pipeline.finish()
            completed = True
        finally:
            pipeline.close(completed=completed)

    def _save_latest_wav_response(self, wav: bytes) -> None:
        """Atomically persist the exact WAV response body without regenerating audio."""
        destination = self.settings.latest_wav_path
        temporary_path: Path | None = None
        try:
            _ = destination.parent.mkdir(parents=True, exist_ok=True)
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

    def open_latest_wav_capture(self, sample_rate: int) -> _AtomicWavWriter | None:
        try:
            return _AtomicWavWriter(self.settings.latest_wav_path, sample_rate)
        except (OSError, wave.Error):
            self._log_latest_wav_failure()
            return None

    def write_latest_wav_chunk(
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

    def finish_latest_wav_capture(
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
        return self._wav_from_pcm(sample_rate, self._pcm16_bytes(audio))

    @staticmethod
    def _wav_from_pcm(sample_rate: int, pcm: bytes) -> bytes:
        output = io.BytesIO()
        with wave.open(output, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm)
        return output.getvalue()


class _PcmPipeline:
    """Synthesize text segments while streaming one stitched PCM response."""

    def __init__(
        self,
        runtime: TtsRuntime,
        voice: str,
        *,
        capture_latest: bool,
        transport: str,
    ) -> None:
        self.runtime = runtime
        self.voice = voice
        self.transport = transport
        self.sample_rate = runtime.sample_rate()
        self.capture_latest = capture_latest and runtime.settings.save_latest_wav
        self.capture_attempted = False
        self.capture: _AtomicWavWriter | None = None
        self.pending: _SentencePcmBuffer | None = None
        self.pending_stitch_deadline: float | None = None
        self.playback_started_at: float | None = None
        self.previous_segment_had_audio = False
        self.output_bytes = 0
        self.closed = False

    def add_segment(  # noqa: C901, PLR0912, PLR0915
        self,
        text: str,
        boundary: Literal['clause', 'sentence', 'input_end'],
    ) -> Generator[bytes, None, None]:
        next_audio_deadline = self.pending_stitch_deadline
        lookahead_warning_logged = False
        pause_seconds = 0.0
        if self.pending is not None:
            pause_seconds = self.pending.boundary_pause_frames / self.sample_rate
            if next_audio_deadline is not None and time.perf_counter() > next_audio_deadline:
                self._warn_lookahead('next_segment_unavailable', next_audio_deadline)
                lookahead_warning_logged = True
            tail = self.pending.finish(fade_out=True)
            self.previous_segment_had_audio = self.pending.seen_voice
            self.pending = None
            self.pending_stitch_deadline = None
            if tail:
                captured = self._capture(tail)
                if next_audio_deadline is None and self.playback_started_at is not None:
                    next_audio_deadline = self.playback_started_at + self.output_bytes / (
                        self.sample_rate * 2
                    )
                yield captured

        PIPELINE_SEGMENTS.labels(transport=self.transport).inc()
        LOGGER.info(
            'TTS pipeline segment requested',
            extra={
                'event_id': 'ID_tts_pipeline_segment_requested',
                'transport': self.transport,
                'characters': len(text),
                'boundary': boundary,
                'pause_before_seconds': pause_seconds,
            },
        )
        LOGGER.debug(
            'TTS pipeline segment',
            extra={
                'event_id': 'ID_tts_pipeline_segment',
                'transport': self.transport,
                'text': text,
            },
        )

        segment: _SentencePcmBuffer | None = None
        first_voice_checked = False
        segment_base_output_frames = self.output_bytes // 2
        active_clip_output_frames: int | None = None
        sample_end_warning_logged = False

        def warn_if_sample_end_late(observed_at: float) -> None:
            nonlocal sample_end_warning_logged
            if active_clip_output_frames is None or sample_end_warning_logged:
                return
            deadline = self._playback_deadline(
                segment_base_output_frames + active_clip_output_frames,
            )
            if deadline is not None and observed_at > deadline:
                self._warn_lookahead(
                    'sample_end_unavailable',
                    deadline,
                    observed_at=observed_at,
                )
                sample_end_warning_logged = True

        def on_tail_clipped(output_frames: int) -> None:
            nonlocal active_clip_output_frames
            active_clip_output_frames = output_frames
            warn_if_sample_end_late(time.perf_counter())

        def on_false_tail() -> None:
            nonlocal active_clip_output_frames
            warn_if_sample_end_late(time.perf_counter())
            active_clip_output_frames = None

        for chunk in self.runtime.stream_model_pcm(text, self.voice):
            if segment is None:
                segment = _SentencePcmBuffer(
                    self.sample_rate,
                    self.runtime.settings,
                    boundary=boundary,
                    fade_in=self.previous_segment_had_audio,
                    transport=self.transport,
                    on_tail_clipped=on_tail_clipped,
                    on_false_tail=on_false_tail,
                )
            output = segment.append(chunk)
            if segment.first_voice_at is not None and not first_voice_checked:
                first_voice_checked = True
                if (
                    not lookahead_warning_logged
                    and next_audio_deadline is not None
                    and segment.first_voice_at > next_audio_deadline
                ):
                    self._warn_lookahead('next_audio_unavailable', next_audio_deadline)
                    lookahead_warning_logged = True
            if output:
                captured = self._capture(output)
                warn_if_sample_end_late(time.perf_counter())
                yield captured
        if segment is not None:
            output = segment.end_model()
            if segment.model_finished_at is not None:
                warn_if_sample_end_late(segment.model_finished_at)
            if segment.leading_silence_frames or segment.clipped_silence_frames:
                LOGGER.info(
                    'TTS pipeline segment silence normalized',
                    extra={
                        'event_id': 'ID_tts_pipeline_segment_silence_normalized',
                        'transport': self.transport,
                        'boundary': boundary,
                        'generated_seconds': segment.generated_frames / self.sample_rate,
                        'leading_silence_removed_seconds': (
                            segment.leading_silence_frames / self.sample_rate
                        ),
                        'trailing_silence_detected_seconds': (
                            segment.trailing_silence_frames / self.sample_rate
                        ),
                        'trailing_silence_clipped_seconds': (
                            segment.clipped_silence_frames / self.sample_rate
                        ),
                    },
                )
            if segment.first_voice_at is not None and not first_voice_checked:
                first_voice_checked = True
                if (
                    not lookahead_warning_logged
                    and next_audio_deadline is not None
                    and segment.first_voice_at > next_audio_deadline
                ):
                    self._warn_lookahead('next_audio_unavailable', next_audio_deadline)
                    lookahead_warning_logged = True
            if output:
                yield self._capture(output)
        if (
            not lookahead_warning_logged
            and next_audio_deadline is not None
            and (segment is None or segment.first_voice_at is None)
        ):
            observed_at = (
                segment.model_finished_at
                if segment is not None and segment.model_finished_at is not None
                else time.perf_counter()
            )
            self._warn_lookahead(
                'next_audio_unavailable',
                next_audio_deadline,
                observed_at=observed_at,
            )
        self.pending = segment
        if segment is not None:
            self.pending_stitch_deadline = self._projected_playback_deadline(segment)

    def finish(self) -> Generator[bytes, None, None]:
        if self.pending is None:
            return
        output = self.pending.finish(fade_out=False)
        self.pending = None
        self.pending_stitch_deadline = None
        self.previous_segment_had_audio = True
        if output:
            yield self._capture(output)

    def close(self, *, completed: bool) -> None:
        if self.closed:
            return
        self.closed = True
        self.runtime.finish_latest_wav_capture(
            self.capture,
            completed=completed,
            pcm_bytes=self.output_bytes,
        )
        self.capture = None

    def _capture(self, pcm: bytes) -> bytes:
        if pcm and self.playback_started_at is None:
            self.playback_started_at = time.perf_counter()
        if self.capture_latest and not self.capture_attempted:
            self.capture_attempted = True
            self.capture = self.runtime.open_latest_wav_capture(self.sample_rate)
        self.capture = self.runtime.write_latest_wav_chunk(self.capture, pcm)
        self.output_bytes += len(pcm)
        return pcm

    def _projected_playback_deadline(
        self,
        segment: _SentencePcmBuffer,
    ) -> float | None:
        projected_bytes = self.output_bytes + segment.pending_stitch_frames * 2
        return self._playback_deadline(projected_bytes // 2)

    def _playback_deadline(self, output_frames: int) -> float | None:
        if self.playback_started_at is None:
            return None
        return self.playback_started_at + output_frames / self.sample_rate

    def _warn_lookahead(
        self,
        reason: str,
        deadline: float | None,
        *,
        observed_at: float | None = None,
    ) -> None:
        now = time.perf_counter() if observed_at is None else observed_at
        LOGGER.warning(
            'TTS stitching lookahead invariant was not met',
            extra={
                'event_id': 'ID_tts_pipeline_stitch_lookahead_warning',
                'transport': self.transport,
                'reason': reason,
                'deadline_overrun_seconds': max(0.0, now - deadline) if deadline else None,
            },
        )


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
        text = runtime.validate_request(speech_request)
        voice = runtime.prepare_voice(speech_request.voice)
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
        await websocket.send_json({'type': 'error', 'message': message})
    with suppress(RuntimeError):
        await websocket.close(code=code, reason=message)


def _pipeline_validation_message(error: HTTPException | ValidationError) -> str:
    if isinstance(error, HTTPException):
        return str(error.detail)
    return 'invalid pipeline event'


def _raise_empty_pipeline() -> Never:
    message = 'pipeline input is empty'
    raise ValueError(message)


def _enforce_pipeline_input_limit(input_characters: int, maximum: int) -> None:
    if input_characters <= maximum:
        return
    raise HTTPException(
        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        detail='input exceeds TTS_MAX_INPUT_CHARACTERS',
    )


async def _stream_pipeline_websocket(  # noqa: C901, PLR0912, PLR0915
    websocket: WebSocket,
    runtime: TtsRuntime,
) -> None:
    started = time.perf_counter()
    outcome = 'success'
    gate_acquired = False
    active_request = False
    completed = False
    pipeline: _PcmPipeline | None = None
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
            {
                'type': 'session.ready',
                'format': 'pcm16',
                'sample_rate': runtime.sample_rate(),
                'sample_width': 2,
                'channels': 1,
            },
        )
        LOGGER.info(
            'TTS pipeline WebSocket started',
            extra={'event_id': 'ID_tts_pipeline_websocket_started'},
        )

        segmenter: _TextSegmenter | None = None
        input_characters = 0
        input_has_text = False
        input_done = False
        while not input_done:
            raw = await _receive_pipeline_text(
                websocket,
                runtime.settings.pipeline_idle_timeout_seconds,
            )
            event = PIPELINE_INPUT_ADAPTER.validate_json(raw)
            if isinstance(event, PipelineTextDone):
                if segmenter is None or not input_has_text:
                    _raise_empty_pipeline()
                segments = segmenter.finish()
                input_done = True
            else:
                input_characters += len(event.delta)
                input_has_text = input_has_text or bool(event.delta.strip())
                _enforce_pipeline_input_limit(
                    input_characters,
                    runtime.settings.maximum_input_characters,
                )
                if segmenter is None:
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
                    voice = await asyncio.to_thread(runtime.prepare_voice, session.voice)
                    segmenter = _TextSegmenter(
                        runtime.settings.pipeline_sentence_terminators,
                        first_segment_comma_delimiter=(
                            runtime.settings.pipeline_first_segment_comma_delimiter
                        ),
                    )
                    pipeline = _PcmPipeline(
                        runtime,
                        voice,
                        capture_latest=True,
                        transport='websocket',
                    )
                    sender = _WebSocketPcmSender(websocket, pipeline.sample_rate, started)
                segments = segmenter.append(event.delta)

            if pipeline is None or sender is None:
                continue
            for segment, boundary in segments:
                await sender.send(pipeline.add_segment(segment, boundary))

        if pipeline is None or sender is None:
            _raise_empty_pipeline()
        await sender.send(pipeline.finish())
        pipeline.close(completed=True)
        completed = True
        await websocket.send_json({'type': 'response.audio.done'})
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
    except (HTTPException, ValidationError) as error:
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


if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',  # noqa: S104
        port=SETTINGS.listen_port,
        log_config=None,
    )
