from __future__ import annotations

import asyncio
import importlib
import io
import json
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
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
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
SMART_CHUNK_KNOWLEDGE_VERSION: Final = 1
SMART_CHUNK_MINIMUM_OBSERVATIONS: Final = 5
SMART_CHUNK_DEFAULT_CHARACTERS_PER_SENTENCE: Final = 72.0
SMART_CHUNK_DEFAULT_WORDS_PER_SENTENCE: Final = 12.0
SMART_CHUNK_DEFAULT_AUDIO_SECONDS_PER_CHARACTER: Final = 0.055
SMART_CHUNK_DEFAULT_AUDIO_SECONDS_PER_WORD: Final = 0.33
SMART_CHUNK_DEFAULT_FIRST_AUDIO_SECONDS: Final = 0.12
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

    def append(
        self,
        text: str,
    ) -> list[tuple[str, Literal['clause', 'sentence', 'paragraph', 'input_end']]]:
        self.pending += text
        segments: list[tuple[str, Literal['clause', 'sentence', 'paragraph', 'input_end']]] = []
        while boundary := self._next_boundary():
            segment_end, consumed_end, boundary_type = boundary
            segment = self.pending[:segment_end].strip()
            self.pending = self.pending[consumed_end:].lstrip(' \t')
            if segment:
                segments.append((segment, boundary_type))
                self.first_segment = False
            elif boundary_type == 'paragraph':
                segments.append(('', 'paragraph'))
        return segments

    def finish(
        self,
    ) -> list[tuple[str, Literal['clause', 'sentence', 'paragraph', 'input_end']]]:
        segments = self.append('')
        tail = self.pending.strip()
        self.pending = ''
        if tail:
            segments.append((tail, 'input_end'))
            self.first_segment = False
        return segments

    def _next_boundary(
        self,
    ) -> tuple[int, int, Literal['clause', 'sentence', 'paragraph']] | None:
        terminators = self.terminators
        if self.first_segment and self.first_segment_comma_delimiter:
            terminators += ','
        sentence = re.search(rf'[{re.escape(terminators)}](?=\s|$)', self.pending)
        paragraph = re.search(r'\r?\n[ \t]*\r?\n+', self.pending)
        if paragraph is None:
            if sentence is None:
                return None
            boundary: Literal['clause', 'sentence', 'paragraph'] = (
                'clause' if sentence.group() == ',' else 'sentence'
            )
            return sentence.end(), sentence.end(), boundary

        paragraph_content_end = len(self.pending[: paragraph.start()].rstrip())
        if sentence is not None and sentence.end() < paragraph_content_end:
            boundary = 'clause' if sentence.group() == ',' else 'sentence'
            return sentence.end(), sentence.end(), boundary
        return paragraph_content_end, paragraph.end(), 'paragraph'


class _RunningStats:
    """Persistable Welford running statistics."""

    __slots__ = ('count', 'm2', 'mean')

    def __init__(self, count: int = 0, mean: float = 0.0, m2: float = 0.0) -> None:
        self.count = count
        self.mean = mean
        self.m2 = m2

    @classmethod
    def from_payload(cls, payload: object) -> _RunningStats:
        if not isinstance(payload, dict):
            message = 'running statistics must be an object'
            raise TypeError(message)
        count = payload.get('count')
        mean = payload.get('mean')
        m2 = payload.get('m2')
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            or not isinstance(mean, (int, float))
            or isinstance(mean, bool)
            or not math.isfinite(mean)
            or not isinstance(m2, (int, float))
            or isinstance(m2, bool)
            or not math.isfinite(m2)
            or m2 < 0
        ):
            message = 'running statistics contain invalid values'
            raise ValueError(message)
        return cls(count=count, mean=float(mean), m2=float(m2))

    def observe(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def upper_bound(self, fallback: float, standard_deviations: float) -> float:
        if self.count == 0:
            return fallback
        deviation = math.sqrt(self.m2 / (self.count - 1)) if self.count > 1 else 0.0
        estimate = self.mean + standard_deviations * deviation
        if self.count < SMART_CHUNK_MINIMUM_OBSERVATIONS:
            estimate = max(fallback, estimate)
        return max(0.0, estimate)

    def payload(self) -> dict[str, int | float]:
        return {'count': self.count, 'mean': self.mean, 'm2': self.m2}


def _write_json_atomic(destination: Path, payload: dict[str, object]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=destination.parent,
            prefix=f'.{destination.name}.',
            suffix='.tmp',
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, sort_keys=True, separators=(',', ':'))
            temporary.flush()
            os.fsync(temporary.fileno())
        _ = temporary_path.replace(destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


class _SmartChunkKnowledge:
    """Learn text-arrival and per-voice duration distributions on disk."""

    _VOICE_KEYS: Final = (
        'audio_seconds_per_character',
        'audio_seconds_per_word',
        'first_audio_seconds',
    )
    _LLM_KEYS: Final = (
        'characters_per_sentence',
        'words_per_sentence',
        'seconds_per_character',
        'seconds_per_word',
    )

    def __init__(self, settings: Settings, voice: str) -> None:
        self.settings = settings
        self.voice = voice
        directory = settings.data_directory / '.pipeline-knowledge'
        self.voice_path = directory / f'voice-{voice}.json'
        self.llm_path = directory / f'llm-{settings.pipeline_smart_chunk_llm_id}.json'
        self.voice_stats = {key: _RunningStats() for key in self._VOICE_KEYS}
        self.llm_stats = {key: _RunningStats() for key in self._LLM_KEYS}
        self.voice_dirty = 0
        self.llm_dirty = 0
        self._load(
            self.voice_path,
            {'model_id': settings.model_id, 'voice': voice},
            self.voice_stats,
        )
        self._load(
            self.llm_path,
            {'llm_id': settings.pipeline_smart_chunk_llm_id},
            self.llm_stats,
        )

    def observe_voice(
        self,
        text: str,
        *,
        audio_seconds: float,
        first_audio_seconds: float | None,
    ) -> None:
        characters, words = _text_units(text)
        if characters:
            self.voice_stats['audio_seconds_per_character'].observe(
                audio_seconds / characters,
            )
        if words:
            self.voice_stats['audio_seconds_per_word'].observe(audio_seconds / words)
        if first_audio_seconds is not None:
            self.voice_stats['first_audio_seconds'].observe(first_audio_seconds)
        self.voice_dirty += 1

    def observe_llm_sentence(  # noqa: C901
        self,
        text: str,
        *,
        arrival_seconds: float | None,
        timing_text: str,
    ) -> None:
        characters, words = _text_units(text)
        timing_characters, timing_words = _text_units(timing_text)
        if characters:
            self.llm_stats['characters_per_sentence'].observe(float(characters))
        if words:
            self.llm_stats['words_per_sentence'].observe(float(words))
        if arrival_seconds is not None:
            if timing_characters:
                self.llm_stats['seconds_per_character'].observe(
                    arrival_seconds / timing_characters,
                )
            if timing_words:
                self.llm_stats['seconds_per_word'].observe(arrival_seconds / timing_words)
        self.llm_dirty += 1

    def prediction(
        self,
        settings: Settings,
        *,
        queued_text: str,
    ) -> dict[str, int | float | str]:
        standard_deviations = NormalDist().inv_cdf(
            settings.pipeline_smart_chunk_confidence,
        )
        expected_characters = self.llm_stats['characters_per_sentence'].upper_bound(
            SMART_CHUNK_DEFAULT_CHARACTERS_PER_SENTENCE,
            standard_deviations,
        )
        expected_words = self.llm_stats['words_per_sentence'].upper_bound(
            SMART_CHUNK_DEFAULT_WORDS_PER_SENTENCE,
            standard_deviations,
        )
        seconds_per_character = self.voice_stats['audio_seconds_per_character'].upper_bound(
            SMART_CHUNK_DEFAULT_AUDIO_SECONDS_PER_CHARACTER,
            standard_deviations,
        )
        seconds_per_word = self.voice_stats['audio_seconds_per_word'].upper_bound(
            SMART_CHUNK_DEFAULT_AUDIO_SECONDS_PER_WORD,
            standard_deviations,
        )
        expected_spoken_seconds = max(
            expected_characters * seconds_per_character,
            expected_words * seconds_per_word,
        )
        llm_character_timing = self.llm_stats['seconds_per_character']
        llm_word_timing = self.llm_stats['seconds_per_word']
        llm_seconds_per_character = llm_character_timing.upper_bound(
            seconds_per_character / settings.pipeline_smart_chunk_cold_start_speedup,
            standard_deviations,
        )
        llm_seconds_per_word = llm_word_timing.upper_bound(
            seconds_per_word / settings.pipeline_smart_chunk_cold_start_speedup,
            standard_deviations,
        )
        expected_arrival_seconds = max(
            expected_characters * llm_seconds_per_character,
            expected_words * llm_seconds_per_word,
        )
        arrival_source = (
            'observed'
            if llm_character_timing.count or llm_word_timing.count
            else 'speech_speed_prior'
        )

        first_audio_seconds = self.voice_stats['first_audio_seconds'].upper_bound(
            SMART_CHUNK_DEFAULT_FIRST_AUDIO_SECONDS,
            standard_deviations,
        )
        queued_characters, queued_words = _text_units(queued_text)
        queued_audio_seconds = max(
            queued_characters * seconds_per_character,
            queued_words * seconds_per_word,
        )
        return {
            'confidence': settings.pipeline_smart_chunk_confidence,
            'standard_deviations': standard_deviations,
            'safety_seconds': settings.pipeline_smart_chunk_safety_seconds,
            'cold_start_speedup': settings.pipeline_smart_chunk_cold_start_speedup,
            'clause_pause_seconds': settings.pipeline_clause_pause_seconds,
            'sentence_pause_seconds': settings.pipeline_sentence_pause_seconds,
            'paragraph_pause_seconds': settings.pipeline_paragraph_pause_seconds,
            'crossfade_seconds': settings.pipeline_sentence_crossfade_seconds,
            'llm_id': settings.pipeline_smart_chunk_llm_id,
            'arrival_estimate_source': arrival_source,
            'expected_next_characters': expected_characters,
            'expected_next_words': expected_words,
            'expected_next_spoken_seconds': expected_spoken_seconds,
            'expected_next_arrival_seconds': expected_arrival_seconds,
            'expected_first_audio_seconds': first_audio_seconds,
            'queued_characters': queued_characters,
            'queued_words': queued_words,
            'queued_audio_seconds': queued_audio_seconds,
            'voice_seconds_per_character': seconds_per_character,
            'voice_seconds_per_word': seconds_per_word,
            'llm_seconds_per_character': llm_seconds_per_character,
            'llm_seconds_per_word': llm_seconds_per_word,
            'voice_observations': self.voice_stats['audio_seconds_per_character'].count,
            'llm_sentence_observations': self.llm_stats['characters_per_sentence'].count,
            'llm_timing_observations': max(
                llm_character_timing.count,
                llm_word_timing.count,
            ),
            'knowledge_directory': str(self.voice_path.parent),
        }

    def save(self, *, force: bool = False) -> None:
        threshold = self.settings.pipeline_smart_chunk_knowledge_flush_observations
        if (
            self.voice_dirty
            and (force or self.voice_dirty >= threshold)
            and self._save(
                self.voice_path,
                {'model_id': self.settings.model_id, 'voice': self.voice},
                self.voice_stats,
            )
        ):
            self.voice_dirty = 0
        if (
            self.llm_dirty
            and (force or self.llm_dirty >= threshold)
            and self._save(
                self.llm_path,
                {'llm_id': self.settings.pipeline_smart_chunk_llm_id},
                self.llm_stats,
            )
        ):
            self.llm_dirty = 0

    @staticmethod
    def _load(  # noqa: C901
        path: Path,
        identity: dict[str, str],
        statistics: dict[str, _RunningStats],
    ) -> None:
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            if (
                not isinstance(payload, dict)
                or payload.get('version') != SMART_CHUNK_KNOWLEDGE_VERSION
                or any(payload.get(key) != value for key, value in identity.items())
            ):
                return
            raw_statistics = payload.get('statistics')
            if not isinstance(raw_statistics, dict):
                message = 'knowledge statistics must be an object'
                raise TypeError(message)  # noqa: TRY301
            for key in statistics:
                if key in raw_statistics:
                    statistics[key] = _RunningStats.from_payload(raw_statistics[key])
        except FileNotFoundError:
            return
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            LOGGER.warning(
                'TTS smart chunk knowledge could not be loaded',
                extra={
                    'event_id': 'ID_tts_pipeline_smart_chunk_knowledge_load_failed',
                    'path': str(path),
                    'error_type': type(error).__name__,
                },
            )

    @staticmethod
    def _save(
        path: Path,
        identity: dict[str, str],
        statistics: dict[str, _RunningStats],
    ) -> bool:
        payload: dict[str, object] = {
            'version': SMART_CHUNK_KNOWLEDGE_VERSION,
            **identity,
            'statistics': {key: estimate.payload() for key, estimate in statistics.items()},
        }
        try:
            _write_json_atomic(path, payload)
        except (OSError, TypeError, ValueError) as error:
            LOGGER.warning(
                'TTS smart chunk knowledge could not be saved',
                extra={
                    'event_id': 'ID_tts_pipeline_smart_chunk_knowledge_save_failed',
                    'path': str(path),
                    'error_type': type(error).__name__,
                },
            )
            return False
        return True


def _text_units(text: str) -> tuple[int, int]:
    return len(text), len(text.split())


@dataclass(slots=True)
class _SmartChunkDecision:
    made_at: float
    flush_deadline: float
    playback_deadline: float
    should_queue: bool
    reason: str
    parameters: dict[str, int | float | str]


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


def _pcm16_rms(pcm: bytes) -> float:
    if len(pcm) % 2:
        message = 'TTS returned a partial PCM16 frame'
        raise RuntimeError(message)
    samples = array('h')
    samples.frombytes(pcm)
    if sys.byteorder == 'big':
        samples.byteswap()
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


class _SentencePcmBuffer:
    """Stream speech while normalizing model silence at a stitched boundary."""

    def __init__(  # noqa: PLR0913
        self,
        sample_rate: int,
        settings: Settings,
        *,
        boundary: Literal['clause', 'sentence', 'paragraph', 'input_end'],
        fade_in: bool,
        transport: str,
        should_speculatively_clip: Callable[[int], bool],
    ) -> None:
        self.sample_rate = sample_rate
        self.boundary = boundary
        self.boundary_pause_frames = 0
        self.transport = transport
        self.crossfade_frames = _duration_frames(
            settings.pipeline_sentence_crossfade_seconds,
            sample_rate,
        )
        self.set_boundary(boundary, settings)
        self.silence_confirmation_frames = max(
            1,
            _duration_frames(settings.pipeline_silence_confirmation_seconds, sample_rate),
        )
        self.speech_confirmation_frames = max(
            1,
            _duration_frames(settings.pipeline_speech_confirmation_seconds, sample_rate),
        )
        self.silence_threshold_dbfs = settings.pipeline_silence_threshold_dbfs
        self.speech_threshold_dbfs = min(
            0.0,
            self.silence_threshold_dbfs + settings.pipeline_speech_hysteresis_db,
        )
        self.silence_threshold = round(
            32_767 * math.pow(10, self.silence_threshold_dbfs / 20),
        )
        self.speech_threshold = round(
            32_767 * math.pow(10, self.speech_threshold_dbfs / 20),
        )
        self.analysis_frame_bytes = max(2, _duration_frames(0.01, sample_rate) * 2)
        self.fade_in = fade_in
        self.frames_emitted = 0
        self.buffer = bytearray()
        self.boundary_silence = bytearray()
        self.analysis_buffer = bytearray()
        self.silence_candidate = bytearray()
        self.speech_candidate = bytearray()
        self.generated_frames = 0
        self.trailing_silence_frames = 0
        self.emitted_trailing_silence_frames = 0
        self.clipped_silence_frames = 0
        self.leading_silence_frames = 0
        self.ignored_noise_frames = 0
        self.preserved_internal_silence_frames = 0
        self.seen_voice = False
        self.silence_confirmed = False
        self.tail_was_speculatively_clipped = False
        self.false_tail_error_logged = False
        self.first_voice_at: float | None = None
        self.model_finished_at: float | None = None
        self.should_speculatively_clip = should_speculatively_clip

    def set_boundary(
        self,
        boundary: Literal['clause', 'sentence', 'paragraph', 'input_end'],
        settings: Settings,
    ) -> None:
        """Set the target pause, including a paragraph revealed by a later delta."""
        self.boundary = boundary
        if boundary == 'clause':
            pause_seconds = settings.pipeline_clause_pause_seconds
        elif boundary == 'paragraph':
            pause_seconds = settings.pipeline_paragraph_pause_seconds
        else:
            pause_seconds = settings.pipeline_sentence_pause_seconds
        self.boundary_pause_frames = _duration_frames(pause_seconds, self.sample_rate)

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
            and _pcm16_rms(bytes(self.analysis_buffer)) >= self.speech_threshold
        ):
            frame = bytes(self.analysis_buffer)
            self.analysis_buffer.clear()
            output.extend(self._process_frame(frame))
        return bytes(output)

    def end_model(self) -> bytes:  # noqa: C901
        output = bytearray()
        if self.analysis_buffer:
            frame = bytes(self.analysis_buffer)
            self.analysis_buffer.clear()
            output.extend(self._process_frame(frame))
        if not self.seen_voice:
            output.extend(self._finish_leading_speech())
        elif self.speech_candidate:
            self._reject_speech_candidate()
        if self.silence_candidate or self.tail_was_speculatively_clipped:
            self.silence_confirmed = True
            self._maybe_speculatively_clip()
            if not self.tail_was_speculatively_clipped:
                self._normalize_terminal_silence()
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

        output = bytearray()
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

    def _process_frame(self, pcm: bytes) -> bytes:
        if not self.seen_voice:
            return self._process_leading_frame(pcm)

        rms = _pcm16_rms(pcm)
        if self.trailing_silence_frames or self.speech_candidate:
            return self._process_possible_speech_resume(pcm, rms)
        if rms <= self.silence_threshold:
            self._append_silence(pcm)
            return b''
        return self._append_output(pcm)

    def _process_leading_frame(self, pcm: bytes) -> bytes:
        rms = _pcm16_rms(pcm)
        if self.speech_candidate:
            if rms > self.silence_threshold:
                self.speech_candidate.extend(pcm)
            else:
                rejected_frames = len(self.speech_candidate) // 2
                self.leading_silence_frames += rejected_frames
                self.ignored_noise_frames += rejected_frames
                self.speech_candidate.clear()
                self.leading_silence_frames += len(pcm) // 2
        elif rms >= self.speech_threshold:
            self.speech_candidate.extend(pcm)
        else:
            self.leading_silence_frames += len(pcm) // 2

        if len(self.speech_candidate) // 2 < self.speech_confirmation_frames:
            return b''
        return self._confirm_initial_speech()

    def _confirm_initial_speech(self) -> bytes:
        self.seen_voice = True
        self.first_voice_at = time.perf_counter()
        speech = bytes(self.speech_candidate)
        self.speech_candidate.clear()
        return self._append_output(speech)

    def _finish_leading_speech(self) -> bytes:
        if not self.speech_candidate:
            return b''
        return self._confirm_initial_speech()

    def _process_possible_speech_resume(self, pcm: bytes, rms: float) -> bytes:
        if rms >= self.speech_threshold or (self.speech_candidate and rms > self.silence_threshold):
            self.speech_candidate.extend(pcm)
            if len(self.speech_candidate) // 2 >= self.speech_confirmation_frames:
                return self._confirm_speech_resume()
            return b''

        if self.speech_candidate:
            self._reject_speech_candidate()
        self._append_silence(pcm)
        return b''

    def _confirm_speech_resume(self) -> bytes:
        output = bytearray()
        returning_speech = bytes(self.speech_candidate)
        self.speech_candidate.clear()
        self._maybe_speculatively_clip()
        if self.tail_was_speculatively_clipped:
            if not self.false_tail_error_logged:
                self.false_tail_error_logged = True
                LOGGER.error(
                    'TTS stitcher found sustained speech after a speculative tail cut',
                    extra={
                        'event_id': 'ID_tts_pipeline_stitch_false_tail',
                        'transport': self.transport,
                        'boundary': self.boundary,
                        'clipped_seconds': (self.clipped_silence_frames / self.sample_rate),
                        'returning_speech_seconds': (
                            len(returning_speech) / (self.sample_rate * 2)
                        ),
                        'generated_seconds': self.generated_frames / self.sample_rate,
                        'silence_threshold_dbfs': self.silence_threshold_dbfs,
                        'speech_threshold_dbfs': self.speech_threshold_dbfs,
                        'speech_confirmation_seconds': (
                            self.speech_confirmation_frames / self.sample_rate
                        ),
                    },
                )
            output.extend(self._append_output(bytes(self.boundary_silence)))
        else:
            self.preserved_internal_silence_frames += len(self.silence_candidate) // 2
            output.extend(self._append_output(bytes(self.silence_candidate)))
        output.extend(self._append_output(returning_speech))
        self._reset_silence()
        return bytes(output)

    def _reject_speech_candidate(self) -> None:
        noise = bytes(self.speech_candidate)
        self.speech_candidate.clear()
        noise_frames = len(noise) // 2
        self.ignored_noise_frames += noise_frames
        self._append_silence(noise)

    def _append_silence(self, pcm: bytes) -> None:
        frames = len(pcm) // 2
        self.trailing_silence_frames += frames
        if self.tail_was_speculatively_clipped:
            self.clipped_silence_frames += frames
            return
        self.silence_candidate.extend(pcm)
        if self.trailing_silence_frames >= self.silence_confirmation_frames:
            self.silence_confirmed = True
            self._maybe_speculatively_clip()

    def _maybe_speculatively_clip(self) -> None:
        if (
            self.tail_was_speculatively_clipped
            or not self.silence_confirmed
            or len(self.silence_candidate) // 2 <= self.boundary_pause_frames
        ):
            return
        stitch_output_frames = (
            self.frames_emitted + len(self.buffer) // 2 + self.boundary_pause_frames
        )
        if not self.should_speculatively_clip(stitch_output_frames):
            return
        self.tail_was_speculatively_clipped = True
        self._normalize_terminal_silence()

    def _normalize_terminal_silence(self) -> None:
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
        self.speech_candidate.clear()
        self.trailing_silence_frames = 0
        self.emitted_trailing_silence_frames = 0
        self.clipped_silence_frames = 0
        self.silence_confirmed = False
        self.tail_was_speculatively_clipped = False


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
    pipeline_paragraph_pause_seconds: float = Field(default=0.24, ge=0, le=2)
    pipeline_sentence_crossfade_seconds: float = Field(default=0.01, ge=0, le=0.25)
    pipeline_silence_confirmation_seconds: float = Field(default=0.02, gt=0, le=0.1)
    pipeline_silence_threshold_dbfs: float = Field(default=-43.0, ge=-100, le=0)
    pipeline_speech_hysteresis_db: float = Field(default=6.0, ge=0, le=30)
    pipeline_speech_confirmation_seconds: float = Field(default=0.04, gt=0, le=0.25)
    pipeline_sentence_terminators: str = Field(default='.!?', min_length=1)
    pipeline_first_segment_comma_delimiter: bool = True
    pipeline_smart_chunk_enabled: bool = True
    pipeline_smart_chunk_confidence: float = Field(default=0.9, ge=0.5, lt=1)
    pipeline_smart_chunk_safety_seconds: float = Field(default=0.1, ge=0, le=5)
    pipeline_smart_chunk_cold_start_speedup: float = Field(default=3.0, ge=1, le=100)
    pipeline_smart_chunk_knowledge_flush_observations: int = Field(
        default=8,
        ge=1,
        le=10_000,
    )
    pipeline_smart_chunk_llm_id: str = Field(
        default='default',
        pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$',
    )
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
        self._smart_chunk_knowledge_cache: dict[
            tuple[Path, str, str, str],
            _SmartChunkKnowledge,
        ] = {}

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
        for knowledge in self._smart_chunk_knowledge_cache.values():
            knowledge.save(force=True)
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
                yield from pipeline.add_segment(
                    segment,
                    boundary,
                    input_complete=True,
                )
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

    def smart_chunk_knowledge(self, voice: str) -> _SmartChunkKnowledge:
        settings = self.settings
        key = (
            settings.data_directory,
            settings.model_id,
            voice,
            settings.pipeline_smart_chunk_llm_id,
        )
        knowledge = self._smart_chunk_knowledge_cache.get(key)
        if knowledge is None:
            knowledge = _SmartChunkKnowledge(settings, voice)
            self._smart_chunk_knowledge_cache[key] = knowledge
        else:
            knowledge.settings = settings
        return knowledge

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
        self.queued_text: list[str] = []
        self.queued_boundary: Literal['sentence', 'paragraph', 'input_end'] = 'sentence'
        self.queued_sentence_count = 0
        self.smart_waiting: _SmartChunkDecision | None = None
        self.smart_decisions: list[_SmartChunkDecision] = []
        self.smart_flushed: _SmartChunkDecision | None = None
        self.knowledge: _SmartChunkKnowledge | None = None
        self.sentence_prefix = ''
        self.next_pause_seconds = 0.0

    def add_segment(  # noqa: C901, PLR0911, PLR0912
        self,
        text: str,
        boundary: Literal['clause', 'sentence', 'paragraph', 'input_end'],
        *,
        input_complete: bool = False,
        observed_at: float | None = None,
    ) -> Generator[bytes, None, None]:
        observed_at = time.perf_counter() if observed_at is None else observed_at
        self._log_conservative_misprediction(text, boundary, observed_at)
        if boundary == 'paragraph' and not text:
            self.sentence_prefix = ''
            if self.queued_text:
                self.queued_boundary = 'paragraph'
                self._log_smart_chunk_decision(
                    'flush',
                    'paragraph_boundary',
                    boundary,
                    input_complete=input_complete,
                )
            elif self.pending is not None:
                self.pending.set_boundary('paragraph', self.runtime.settings)
                self.pending_stitch_deadline = self._projected_playback_deadline(self.pending)
            yield from self._flush_queued_text()
            return
        self._observe_source_boundary(text, boundary, observed_at)
        if boundary == 'clause':
            if self.queued_text:
                yield from self._flush_queued_text()
            yield from self._synthesize_chunk(
                text,
                'clause',
                source_sentence_count=0,
                smart_decision=None,
            )
            return

        if self.smart_waiting is not None:
            self.smart_waiting = None
        self.queued_text.append(text)
        self.queued_sentence_count += 1
        self.queued_boundary = boundary
        if boundary == 'paragraph':
            self._log_smart_chunk_decision(
                'flush',
                'paragraph_boundary',
                boundary,
                input_complete=input_complete,
            )
            yield from self._flush_queued_text()
            return
        if boundary == 'input_end':
            self._log_smart_chunk_decision(
                'flush',
                'input_end',
                boundary,
                input_complete=input_complete,
            )
            yield from self._flush_queued_text()
            return
        if not self.runtime.settings.pipeline_smart_chunk_enabled:
            self._log_smart_chunk_decision(
                'flush',
                'smart_chunk_disabled',
                boundary,
                input_complete=input_complete,
            )
            yield from self._flush_queued_text()
            return
        if self.playback_started_at is None and self.pending is None:
            self._log_smart_chunk_decision(
                'flush',
                'first_segment_immediate',
                boundary,
                input_complete=input_complete,
            )
            yield from self._flush_queued_text()
            return

        yield from self._release_pending_boundary()
        if input_complete:
            self._log_smart_chunk_decision(
                'queue',
                'complete_input_available',
                boundary,
                input_complete=True,
            )
            return
        decision = self._smart_chunk_decision(time.perf_counter())
        if decision is None:
            self._log_smart_chunk_decision(
                'flush',
                'playback_buffer_unavailable',
                boundary,
                input_complete=False,
            )
            yield from self._flush_queued_text()
            return

        self._log_smart_chunk_decision(
            'queue' if decision.should_queue else 'flush',
            decision.reason,
            boundary,
            input_complete=False,
            parameters=decision.parameters,
        )
        if not decision.should_queue:
            self.smart_flushed = decision if decision.flush_deadline > decision.made_at else None
            yield from self._flush_queued_text()
            return

        self.smart_waiting = decision
        self.smart_decisions.append(decision)
        LOGGER.info(
            'TTS smart chunker is holding completed text for another sentence',
            extra={
                'event_id': 'ID_tts_pipeline_smart_chunk_held',
                'transport': self.transport,
                'voice': self.voice,
                **decision.parameters,
            },
        )

    @property
    def smart_flush_deadline(self) -> float | None:
        decision = self.smart_waiting
        return decision.flush_deadline if decision is not None else None

    def flush_smart_queue(
        self,
        *,
        misprediction: bool,
        observed_at: float | None = None,
    ) -> Generator[bytes, None, None]:
        decision = self.smart_waiting
        if decision is None:
            return
        observed_at = time.perf_counter() if observed_at is None else observed_at
        if misprediction:
            self._warn_smart_chunk_misprediction(
                'next_sentence_unavailable',
                decision,
                observed_at,
            )
            self.smart_decisions.clear()
        self._log_smart_chunk_decision(
            'flush',
            'forced_flush_deadline' if misprediction else 'smart_queue_flush',
            self.queued_boundary,
            input_complete=False,
        )
        self.smart_waiting = None
        yield from self._flush_queued_text()

    def _flush_queued_text(self) -> Generator[bytes, None, None]:
        if not self.queued_text:
            return
        text = ' '.join(self.queued_text)
        boundary = self.queued_boundary
        source_sentence_count = self.queued_sentence_count
        smart_decision = self.smart_decisions[-1] if self.smart_decisions else None
        self.queued_text.clear()
        self.queued_boundary = 'sentence'
        self.queued_sentence_count = 0
        self.smart_waiting = None
        self.smart_decisions.clear()
        yield from self._synthesize_chunk(
            text,
            boundary,
            source_sentence_count=source_sentence_count,
            smart_decision=smart_decision,
        )

    def _smart_chunk_decision(self, observed_at: float) -> _SmartChunkDecision | None:
        playback_deadline = self._playback_deadline(self.output_bytes // 2)
        if playback_deadline is None or playback_deadline <= observed_at:
            return None
        settings = self.runtime.settings
        prediction = self._knowledge().prediction(
            settings,
            queued_text=' '.join(self.queued_text),
        )
        required_seconds = (
            float(prediction['expected_next_arrival_seconds'])
            + float(prediction['expected_first_audio_seconds'])
            + settings.pipeline_smart_chunk_safety_seconds
        )
        playback_buffer_seconds = playback_deadline - observed_at
        flush_deadline = (
            playback_deadline
            - float(prediction['expected_first_audio_seconds'])
            - settings.pipeline_smart_chunk_safety_seconds
        )
        should_queue = required_seconds < playback_buffer_seconds and flush_deadline > observed_at
        reason = (
            'prediction_fits_playback_buffer'
            if should_queue
            else (
                'synthesis_start_deadline_reached'
                if flush_deadline <= observed_at
                else 'predicted_next_sentence_misses_playback_buffer'
            )
        )
        parameters = {
            **prediction,
            'queued_sentences': self.queued_sentence_count,
            'playback_buffer_seconds': playback_buffer_seconds,
            'required_buffer_seconds': required_seconds,
            'flush_in_seconds': flush_deadline - observed_at,
        }
        return _SmartChunkDecision(
            made_at=observed_at,
            flush_deadline=flush_deadline,
            playback_deadline=playback_deadline,
            should_queue=should_queue,
            reason=reason,
            parameters=parameters,
        )

    def _observe_source_boundary(
        self,
        text: str,
        boundary: Literal['clause', 'sentence', 'paragraph', 'input_end'],
        observed_at: float,
    ) -> None:
        if self.transport != 'websocket' or not self.runtime.settings.pipeline_smart_chunk_enabled:
            return
        if boundary == 'clause':
            self.sentence_prefix = text
            return

        complete_sentence = (
            f'{self.sentence_prefix} {text}'.strip() if self.sentence_prefix else text
        )
        arrival_seconds = (
            observed_at - self.smart_waiting.made_at
            if (self.smart_waiting is not None and observed_at > self.smart_waiting.made_at)
            else None
        )
        self._knowledge().observe_llm_sentence(
            complete_sentence,
            arrival_seconds=arrival_seconds,
            timing_text=text,
        )
        self.sentence_prefix = ''

    def _knowledge(self) -> _SmartChunkKnowledge:
        if self.knowledge is None:
            self.knowledge = self.runtime.smart_chunk_knowledge(self.voice)
        return self.knowledge

    def _log_smart_chunk_decision(
        self,
        decision: Literal['queue', 'flush'],
        reason: str,
        boundary: Literal['sentence', 'paragraph', 'input_end'],
        *,
        input_complete: bool,
        parameters: dict[str, int | float | str] | None = None,
    ) -> None:
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return
        queued_characters, queued_words = _text_units(' '.join(self.queued_text))
        LOGGER.debug(
            'TTS smart chunk decision',
            extra={
                'event_id': 'ID_tts_pipeline_smart_chunk_decision',
                'transport': self.transport,
                'voice': self.voice,
                'decision': decision,
                'reason': reason,
                'boundary': boundary,
                'input_complete': input_complete,
                'smart_chunk_enabled': self.runtime.settings.pipeline_smart_chunk_enabled,
                'queued_sentences': self.queued_sentence_count,
                'queued_characters': queued_characters,
                'queued_words': queued_words,
                'playback_started': self.playback_started_at is not None,
                'pending_audio': self.pending is not None,
                **(parameters or {}),
            },
        )

    def _log_conservative_misprediction(
        self,
        text: str,
        boundary: Literal['clause', 'sentence', 'paragraph', 'input_end'],
        observed_at: float,
    ) -> None:
        decision = self.smart_flushed
        self.smart_flushed = None
        if decision is None or not text or boundary == 'clause':
            return
        if observed_at >= decision.flush_deadline:
            return
        expected_first_audio_seconds = float(
            decision.parameters['expected_first_audio_seconds'],
        )
        safety_seconds = float(decision.parameters['safety_seconds'])
        required_seconds = expected_first_audio_seconds + safety_seconds
        playback_seconds_remaining = decision.playback_deadline - observed_at
        margin_seconds = playback_seconds_remaining - required_seconds
        if margin_seconds <= 0:
            return
        LOGGER.info(
            'TTS smart chunker could have queued the next sentence',
            extra={
                'event_id': 'ID_tts_pipeline_smart_chunk_misprediction',
                'transport': self.transport,
                'voice': self.voice,
                'reason': 'next_sentence_could_have_been_queued',
                'original_decision_reason': decision.reason,
                'next_boundary': boundary,
                'decision_age_seconds': max(0.0, observed_at - decision.made_at),
                'playback_seconds_remaining': playback_seconds_remaining,
                'counterfactual_required_seconds': required_seconds,
                'counterfactual_margin_seconds': margin_seconds,
                **decision.parameters,
            },
        )

    def _release_pending_boundary(self) -> Generator[bytes, None, None]:
        pending = self.pending
        if pending is None:
            return
        self.next_pause_seconds = pending.boundary_pause_frames / self.sample_rate
        tail = pending.finish(fade_out=True)
        self.previous_segment_had_audio = pending.seen_voice
        self.pending = None
        if tail:
            yield self._capture(tail)
        if self.pending_stitch_deadline is None and self.playback_started_at is not None:
            self.pending_stitch_deadline = self.playback_started_at + self.output_bytes / (
                self.sample_rate * 2
            )

    def _warn_smart_chunk_misprediction(
        self,
        reason: str,
        decision: _SmartChunkDecision,
        observed_at: float,
    ) -> None:
        LOGGER.warning(
            'TTS smart chunk prediction missed its playback budget',
            extra={
                'event_id': 'ID_tts_pipeline_smart_chunk_misprediction',
                'transport': self.transport,
                'voice': self.voice,
                'reason': reason,
                'decision_age_seconds': max(0.0, observed_at - decision.made_at),
                'flush_deadline_overrun_seconds': max(
                    0.0,
                    observed_at - decision.flush_deadline,
                ),
                'playback_deadline_overrun_seconds': max(
                    0.0,
                    observed_at - decision.playback_deadline,
                ),
                'playback_seconds_remaining': max(
                    0.0,
                    decision.playback_deadline - observed_at,
                ),
                **decision.parameters,
            },
        )

    def _synthesize_chunk(  # noqa: C901, PLR0912, PLR0915
        self,
        text: str,
        boundary: Literal['clause', 'sentence', 'paragraph', 'input_end'],
        *,
        source_sentence_count: int,
        smart_decision: _SmartChunkDecision | None,
    ) -> Generator[bytes, None, None]:
        lookahead_warning_logged = False
        pause_seconds = self.next_pause_seconds
        if self.pending is not None:
            yield from self._release_pending_boundary()
            pause_seconds = self.next_pause_seconds
        next_audio_deadline = self.pending_stitch_deadline
        self.pending_stitch_deadline = None
        self.next_pause_seconds = 0.0
        observed_at = time.perf_counter()
        if next_audio_deadline is not None and observed_at > next_audio_deadline:
            if smart_decision is not None:
                self._warn_smart_chunk_misprediction(
                    'next_sentence_arrived_late',
                    smart_decision,
                    observed_at,
                )
            else:
                self._warn_lookahead(
                    'next_segment_unavailable',
                    next_audio_deadline,
                    observed_at=observed_at,
                )
            lookahead_warning_logged = True

        PIPELINE_SEGMENTS.labels(transport=self.transport).inc()
        LOGGER.info(
            'TTS pipeline segment requested',
            extra={
                'event_id': 'ID_tts_pipeline_segment_requested',
                'transport': self.transport,
                'characters': len(text),
                'boundary': boundary,
                'pause_before_seconds': pause_seconds,
                'source_sentences': source_sentence_count,
                'smart_chunked': source_sentence_count > 1,
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
        sample_end_warning_logged = False
        synthesis_started_at = time.perf_counter()

        def should_speculatively_clip(output_frames: int) -> bool:
            nonlocal sample_end_warning_logged
            deadline = self._playback_deadline(
                segment_base_output_frames + output_frames,
            )
            observed_at = time.perf_counter()
            if deadline is None or observed_at <= deadline:
                return False
            if not sample_end_warning_logged:
                self._warn_lookahead(
                    'sample_end_unavailable',
                    deadline,
                    observed_at=observed_at,
                    boundary=boundary,
                    speculative_tail_cut=True,
                )
                sample_end_warning_logged = True
            return True

        for chunk in self.runtime.stream_model_pcm(text, self.voice):
            if segment is None:
                segment = _SentencePcmBuffer(
                    self.sample_rate,
                    self.runtime.settings,
                    boundary=boundary,
                    fade_in=self.previous_segment_had_audio,
                    transport=self.transport,
                    should_speculatively_clip=should_speculatively_clip,
                )
            output = segment.append(chunk)
            if segment.first_voice_at is not None and not first_voice_checked:
                first_voice_checked = True
                if (
                    not lookahead_warning_logged
                    and next_audio_deadline is not None
                    and segment.first_voice_at > next_audio_deadline
                ):
                    if smart_decision is not None:
                        self._warn_smart_chunk_misprediction(
                            'first_audio_late',
                            smart_decision,
                            segment.first_voice_at,
                        )
                    else:
                        self._warn_lookahead(
                            'next_audio_unavailable',
                            next_audio_deadline,
                            observed_at=segment.first_voice_at,
                        )
                    lookahead_warning_logged = True
            if output:
                yield self._capture(output)
        if segment is not None:
            output = segment.end_model()
            if (
                segment.leading_silence_frames
                or segment.clipped_silence_frames
                or segment.ignored_noise_frames
            ):
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
                        'internal_silence_preserved_seconds': (
                            segment.preserved_internal_silence_frames / self.sample_rate
                        ),
                        'ignored_noise_seconds': (segment.ignored_noise_frames / self.sample_rate),
                        'speculative_tail_cut': (segment.tail_was_speculatively_clipped),
                        'silence_threshold_dbfs': segment.silence_threshold_dbfs,
                        'speech_threshold_dbfs': segment.speech_threshold_dbfs,
                    },
                )
            if segment.first_voice_at is not None and not first_voice_checked:
                first_voice_checked = True
                if (
                    not lookahead_warning_logged
                    and next_audio_deadline is not None
                    and segment.first_voice_at > next_audio_deadline
                ):
                    if smart_decision is not None:
                        self._warn_smart_chunk_misprediction(
                            'first_audio_late',
                            smart_decision,
                            segment.first_voice_at,
                        )
                    else:
                        self._warn_lookahead(
                            'next_audio_unavailable',
                            next_audio_deadline,
                            observed_at=segment.first_voice_at,
                        )
                    lookahead_warning_logged = True
            if self.runtime.settings.pipeline_smart_chunk_enabled:
                self._knowledge().observe_voice(
                    text,
                    audio_seconds=segment.generated_frames / self.sample_rate,
                    first_audio_seconds=(
                        max(0.0, segment.first_voice_at - synthesis_started_at)
                        if segment.first_voice_at is not None
                        else None
                    ),
                )
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
            if smart_decision is not None:
                self._warn_smart_chunk_misprediction(
                    'first_audio_unavailable',
                    smart_decision,
                    observed_at,
                )
            else:
                self._warn_lookahead(
                    'next_audio_unavailable',
                    next_audio_deadline,
                    observed_at=observed_at,
                )
        self.pending = segment
        if segment is not None:
            self.pending_stitch_deadline = self._projected_playback_deadline(segment)

    def finish(self) -> Generator[bytes, None, None]:
        if self.queued_text:
            self.queued_boundary = 'input_end'
            self._log_smart_chunk_decision(
                'flush',
                'input_end',
                'input_end',
                input_complete=True,
            )
            yield from self._flush_queued_text()
        self.smart_waiting = None
        self.smart_decisions.clear()
        self.smart_flushed = None
        if self.pending is None:
            return
        output = self.pending.finish(fade_out=False)
        self.previous_segment_had_audio = self.pending.seen_voice
        self.pending = None
        self.pending_stitch_deadline = None
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
        if self.knowledge is not None:
            self.knowledge.save()

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
        boundary: Literal['clause', 'sentence', 'paragraph', 'input_end'] | None = None,
        speculative_tail_cut: bool = False,
    ) -> None:
        now = time.perf_counter() if observed_at is None else observed_at
        message = (
            'TTS tail cut became irreversible before model EOF'
            if speculative_tail_cut
            else 'TTS stitching lookahead invariant was not met'
        )
        LOGGER.warning(
            message,
            extra={
                'event_id': 'ID_tts_pipeline_stitch_lookahead_warning',
                'transport': self.transport,
                'reason': reason,
                'deadline_overrun_seconds': max(0.0, now - deadline) if deadline else None,
                'boundary': boundary,
                'speculative_tail_cut': speculative_tail_cut,
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
                            misprediction=True,
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
                        misprediction=True,
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
                            misprediction=True,
                            observed_at=event_observed_at,
                        ),
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
                await sender.send(
                    pipeline.add_segment(
                        segment,
                        boundary,
                        observed_at=event_observed_at,
                    ),
                )

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
