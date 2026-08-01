from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from array import array
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import TYPE_CHECKING, Final, Literal, Protocol

from tts.src.metrics import PIPELINE_SEGMENTS

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from tts.src.config import Settings
    from tts.src.recording import AtomicWavWriter

LOGGER = logging.getLogger('tts')
SMART_CHUNK_KNOWLEDGE_VERSION = 1
SMART_CHUNK_MINIMUM_OBSERVATIONS = 5
SMART_CHUNK_DEFAULT_CHARACTERS_PER_SENTENCE = 72.0
SMART_CHUNK_DEFAULT_WORDS_PER_SENTENCE = 12.0
SMART_CHUNK_DEFAULT_AUDIO_SECONDS_PER_CHARACTER = 0.055
SMART_CHUNK_DEFAULT_AUDIO_SECONDS_PER_WORD = 0.33
SMART_CHUNK_DEFAULT_FIRST_AUDIO_SECONDS = 0.12


class TextSegmenter:
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


class SmartChunkKnowledge:
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
            'speaker_switch_pause_seconds': (settings.pipeline_speaker_switch_pause_seconds),
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


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    speaker: str
    voice: str
    text: str


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
        boundary: Literal[
            'clause',
            'sentence',
            'paragraph',
            'speaker_switch',
            'input_end',
        ],
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
        boundary: Literal[
            'clause',
            'sentence',
            'paragraph',
            'speaker_switch',
            'input_end',
        ],
        settings: Settings,
    ) -> None:
        """Set the target pause, including a paragraph revealed by a later delta."""
        self.boundary = boundary
        if boundary == 'clause':
            pause_seconds = settings.pipeline_clause_pause_seconds
        elif boundary == 'paragraph':
            pause_seconds = settings.pipeline_paragraph_pause_seconds
        elif boundary == 'speaker_switch':
            pause_seconds = settings.pipeline_speaker_switch_pause_seconds
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


class PipelineRuntime(Protocol):
    @property
    def settings(self) -> Settings: ...

    def sample_rate(self) -> int: ...

    def prepare_voice(self, requested_voice: str | None) -> str: ...

    def stream_model_pcm(self, text: str, voice: str) -> Generator[bytes, None, None]: ...

    def smart_chunk_knowledge(self, voice: str) -> SmartChunkKnowledge: ...

    def open_latest_wav_capture(self, sample_rate: int) -> AtomicWavWriter | None: ...

    def write_latest_wav_chunk(
        self,
        capture: AtomicWavWriter | None,
        chunk: bytes,
    ) -> AtomicWavWriter | None: ...

    def finish_latest_wav_capture(
        self,
        capture: AtomicWavWriter | None,
        *,
        completed: bool,
        pcm_bytes: int,
    ) -> None: ...


class PcmPipeline:
    """Synthesize text segments while streaming one stitched PCM response."""

    def __init__(
        self,
        runtime: PipelineRuntime,
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
        self.capture: AtomicWavWriter | None = None
        self.pending: _SentencePcmBuffer | None = None
        self.pending_stitch_deadline: float | None = None
        self.playback_started_at: float | None = None
        self.previous_segment_had_audio = False
        self.output_bytes = 0
        self.closed = False
        self.queued_text: list[str] = []
        self.queued_boundary: Literal[
            'sentence',
            'paragraph',
            'speaker_switch',
            'input_end',
        ] = 'sentence'
        self.queued_sentence_count = 0
        self.smart_waiting: _SmartChunkDecision | None = None
        self.smart_decisions: list[_SmartChunkDecision] = []
        self.smart_flushed: _SmartChunkDecision | None = None
        self.knowledge: SmartChunkKnowledge | None = None
        self.sentence_prefix = ''
        self.next_pause_seconds = 0.0

    def select_voice(self, voice: str) -> None:
        """Select the next turn's preloaded voice without releasing pending audio."""
        if voice == self.voice:
            return
        if self.queued_text or self.smart_waiting is not None:
            message = 'cannot switch TTS voice while text is queued'
            raise RuntimeError(message)
        if self.knowledge is not None:
            self.knowledge.save()
        self.voice = voice
        self.knowledge = None
        self.sentence_prefix = ''

    def add_segment(  # noqa: C901, PLR0911, PLR0912
        self,
        text: str,
        boundary: Literal[
            'clause',
            'sentence',
            'paragraph',
            'speaker_switch',
            'input_end',
        ],
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
        if boundary == 'speaker_switch':
            self._log_smart_chunk_decision(
                'flush',
                'speaker_switch',
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
        LOGGER.debug(
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

    def finish_speaker_turn(self) -> Generator[bytes, None, None]:
        """Flush queued text and mark retained audio as ending at a speaker switch."""
        self.smart_waiting = None
        self.smart_decisions.clear()
        self.smart_flushed = None
        if self.queued_text:
            self.queued_boundary = 'speaker_switch'
            yield from self._flush_queued_text()
            return
        if self.pending is not None:
            self.pending.set_boundary('speaker_switch', self.runtime.settings)
            self.pending_stitch_deadline = self._projected_playback_deadline(self.pending)

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
        boundary: Literal[
            'clause',
            'sentence',
            'paragraph',
            'speaker_switch',
            'input_end',
        ],
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

    def _knowledge(self) -> SmartChunkKnowledge:
        if self.knowledge is None:
            self.knowledge = self.runtime.smart_chunk_knowledge(self.voice)
        return self.knowledge

    def _log_smart_chunk_decision(
        self,
        decision: Literal['queue', 'flush'],
        reason: str,
        boundary: Literal['sentence', 'paragraph', 'speaker_switch', 'input_end'],
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
        boundary: Literal[
            'clause',
            'sentence',
            'paragraph',
            'speaker_switch',
            'input_end',
        ],
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
        LOGGER.debug(
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
        boundary: Literal[
            'clause',
            'sentence',
            'paragraph',
            'speaker_switch',
            'input_end',
        ],
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
        completion_fields: dict[str, object] = {
            'transport': self.transport,
            'voice': self.voice,
            'characters': len(text),
            'boundary': boundary,
            'pause_before_seconds': pause_seconds,
            'source_sentences': source_sentence_count,
            'smart_chunked': source_sentence_count > 1,
            'duration_seconds': time.perf_counter() - synthesis_started_at,
        }
        if segment is not None:
            completion_fields.update(
                {
                    'generated_seconds': segment.generated_frames / self.sample_rate,
                    'first_voice_seconds': (
                        max(0.0, segment.first_voice_at - synthesis_started_at)
                        if segment.first_voice_at is not None
                        else None
                    ),
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
                    'ignored_noise_seconds': segment.ignored_noise_frames / self.sample_rate,
                    'speculative_tail_cut': segment.tail_was_speculatively_clipped,
                },
            )
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                'TTS pipeline segment completed',
                extra={
                    'event_id': 'ID_tts_pipeline_segment_completed',
                    **completion_fields,
                    'text': text,
                },
            )
        else:
            LOGGER.info(
                'TTS pipeline segment completed',
                extra={'event_id': 'ID_tts_pipeline_segment_completed', **completion_fields},
            )
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
        boundary: Literal[
            'clause',
            'sentence',
            'paragraph',
            'speaker_switch',
            'input_end',
        ]
        | None = None,
        speculative_tail_cut: bool = False,
    ) -> None:
        now = time.perf_counter() if observed_at is None else observed_at
        LOGGER.warning(
            'TTS stitching lookahead invariant was not met',
            extra={
                'event_id': 'ID_tts_pipeline_stitch_lookahead_warning',
                'transport': self.transport,
                'reason': reason,
                'deadline_overrun_seconds': max(0.0, now - deadline) if deadline else None,
                'boundary': boundary,
                'speculative_tail_cut': speculative_tail_cut,
            },
        )
