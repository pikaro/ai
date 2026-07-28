from __future__ import annotations

import logging
import os
import tempfile
import wave
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from runtime_config import ExclusiveOperationGate
from tts.src.domain import (
    EmptyInputError,
    EmptySegmentError,
    InputTooLongError,
    ModelMismatchError,
    MultiSpeakerCommand,
    PreparedSpeech,
    RuntimeStatus,
    SpeechCommand,
    UndefinedSpeakerError,
    UnsupportedSpeedError,
)
from tts.src.engine import PocketTtsEngine, wav_from_pcm
from tts.src.pipeline import PcmPipeline, SmartChunkKnowledge, SpeakerTurn, TextSegmenter
from tts.src.recording import AtomicWavWriter
from tts.src.streaming import IncrementalPipelineSession
from tts.src.voices import StoredVoice, VoiceRepository

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import BinaryIO, Literal

    from tts.src.config import Settings

LOGGER = logging.getLogger('tts')


class TtsRuntime:
    def __init__(self, settings: Settings) -> None:
        self.operations = ExclusiveOperationGate()
        self.engine = PocketTtsEngine(settings)
        self._smart_chunk_knowledge_cache: dict[
            tuple[Path, str, str, str],
            SmartChunkKnowledge,
        ] = {}

    @property
    def settings(self) -> Settings:
        return self.engine.settings

    def apply_settings(self, settings: Settings) -> None:
        self.engine.apply_settings(settings)

    def load(self) -> None:
        self.engine.load()

    def save_voice(
        self,
        name: str,
        source_filename: str | None,
        source: BinaryIO,
    ) -> StoredVoice:
        stored = VoiceRepository(
            self.settings.data_directory,
            self.settings.maximum_voice_upload_bytes,
        ).store(name, source_filename, source)
        LOGGER.info(
            'Stored TTS voice',
            extra={
                'event_id': 'ID_tts_voice_stored',
                'voice': stored.name,
                'replaced': stored.replaced,
            },
        )
        self.engine.invalidate_voice(stored.name)
        return stored

    def close(self) -> None:
        for knowledge in self._smart_chunk_knowledge_cache.values():
            knowledge.save(force=True)
        self.engine.close()

    def status(self) -> RuntimeStatus:
        return self.engine.status()

    def prepare_voice(self, requested_voice: str | None) -> str:
        return self.engine.prepare_voice(requested_voice)

    def create_incremental_pipeline(
        self,
        requested_voice: str | None,
    ) -> IncrementalPipelineSession:
        return IncrementalPipelineSession(
            self,
            self.prepare_voice(requested_voice),
        )

    def validate_speech(self, command: SpeechCommand) -> str:
        text = command.text.strip()
        self.validate_options(command.model, command.speed)
        if not text:
            raise EmptyInputError
        if len(text) > self.settings.maximum_input_characters:
            raise InputTooLongError
        return text

    def prepare_speech(self, command: SpeechCommand) -> PreparedSpeech:
        text = self.validate_speech(command)
        return PreparedSpeech(text=text, voice=self.prepare_voice(command.voice))

    def prepare_multi_speaker_turns(  # noqa: C901
        self,
        command: MultiSpeakerCommand,
    ) -> list[SpeakerTurn]:
        """Validate structured input and preload every voice before streaming."""
        self.validate_options(command.model, command.speed)
        normalized_segments: list[tuple[str, str]] = []
        total_characters = 0
        for segment in command.segments:
            text = segment.text.strip()
            if not text:
                raise EmptySegmentError
            if segment.speaker not in command.speakers:
                raise UndefinedSpeakerError(segment.speaker)
            same_speaker = bool(
                normalized_segments and normalized_segments[-1][0] == segment.speaker,
            )
            separator_characters = int(same_speaker)
            total_characters += len(text) + separator_characters
            if total_characters > self.settings.maximum_input_characters:
                raise InputTooLongError
            if same_speaker:
                speaker, previous_text = normalized_segments[-1]
                normalized_segments[-1] = (speaker, f'{previous_text} {text}')
            else:
                normalized_segments.append((segment.speaker, text))

        voices = {speaker: self.prepare_voice(voice) for speaker, voice in command.speakers.items()}
        return [
            SpeakerTurn(speaker, voices[speaker], text) for speaker, text in normalized_segments
        ]

    def validate_options(self, model: str, speed: float) -> None:
        if model != self.settings.model_id:
            raise ModelMismatchError(self.settings.model_id, model)
        if speed != 1.0:
            raise UnsupportedSpeedError

    def generate_wav(self, text: str, voice: str) -> bytes:
        wav = self.engine.generate_wav(text, voice)
        if self.settings.save_latest_wav:
            self._save_latest_wav_response(wav)
        return wav

    def stream_model_pcm(self, text: str, voice: str) -> Generator[bytes, None, None]:
        """Stream raw model PCM without request-level recording semantics."""
        yield from self.engine.stream_pcm(text, voice)

    def stream_pcm(self, text: str, voice: str) -> Generator[bytes, None, None]:
        output_bytes = 0
        capture: AtomicWavWriter | None = None
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
        wav = wav_from_pcm(self.sample_rate(), pcm)
        if self.settings.save_latest_wav:
            self._save_latest_wav_response(wav)
        return wav, len(pcm)

    def generate_multi_speaker_wav(
        self,
        turns: list[SpeakerTurn],
    ) -> tuple[bytes, int]:
        pcm = b''.join(self.stream_multi_speaker_pcm(turns, capture_latest=False))
        wav = wav_from_pcm(self.sample_rate(), pcm)
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
        segmenter = TextSegmenter(
            self.settings.pipeline_sentence_terminators,
            first_segment_comma_delimiter=self.settings.pipeline_first_segment_comma_delimiter,
        )
        segments = [*segmenter.append(text), *segmenter.finish()]
        pipeline = PcmPipeline(
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

    def stream_multi_speaker_pcm(
        self,
        turns: list[SpeakerTurn],
        *,
        capture_latest: bool = True,
    ) -> Generator[bytes, None, None]:
        pipeline = PcmPipeline(
            self,
            turns[0].voice,
            capture_latest=capture_latest,
            transport='http_multi_speaker',
        )
        completed = False
        try:
            for turn_index, turn in enumerate(turns):
                pipeline.select_voice(turn.voice)
                segmenter = TextSegmenter(
                    self.settings.pipeline_sentence_terminators,
                    first_segment_comma_delimiter=(
                        self.settings.pipeline_first_segment_comma_delimiter
                    ),
                )
                segments = [*segmenter.append(turn.text), *segmenter.finish()]
                LOGGER.info(
                    'TTS multi-speaker turn started',
                    extra={
                        'event_id': 'ID_tts_multi_speaker_turn_started',
                        'turn_index': turn_index,
                        'voice': turn.voice,
                        'characters': len(turn.text),
                    },
                )
                for segment_index, (segment, boundary) in enumerate(segments):
                    pipeline_boundary: Literal[
                        'clause',
                        'sentence',
                        'paragraph',
                        'speaker_switch',
                        'input_end',
                    ] = boundary
                    if turn_index < len(turns) - 1 and segment_index == len(segments) - 1:
                        pipeline_boundary = 'speaker_switch'
                    yield from pipeline.add_segment(
                        segment,
                        pipeline_boundary,
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

    def open_latest_wav_capture(self, sample_rate: int) -> AtomicWavWriter | None:
        try:
            return AtomicWavWriter(self.settings.latest_wav_path, sample_rate)
        except (OSError, wave.Error):
            self._log_latest_wav_failure()
            return None

    def write_latest_wav_chunk(
        self,
        capture: AtomicWavWriter | None,
        chunk: bytes,
    ) -> AtomicWavWriter | None:
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
        capture: AtomicWavWriter | None,
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
        capture: AtomicWavWriter,
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
        sample_rate = self.sample_rate()
        return {
            'X-Audio-Format': 'pcm_s16le',
            'X-Audio-Sample-Rate': str(sample_rate),
            'X-Audio-Sample-Width': '2',
            'X-Audio-Channels': '1',
        }

    def sample_rate(self) -> int:
        return self.engine.sample_rate()

    def smart_chunk_knowledge(self, voice: str) -> SmartChunkKnowledge:
        settings = self.settings
        key = (
            settings.data_directory,
            settings.model_id,
            voice,
            settings.pipeline_smart_chunk_llm_id,
        )
        knowledge = self._smart_chunk_knowledge_cache.get(key)
        if knowledge is None:
            knowledge = SmartChunkKnowledge(settings, voice)
            self._smart_chunk_knowledge_cache[key] = knowledge
        else:
            knowledge.settings = settings
        return knowledge
