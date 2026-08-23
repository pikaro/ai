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
    VoiceModelConflictError,
)
from tts.src.engine import PocketTtsEngine, wav_from_pcm
from tts.src.pipeline import PcmPipeline, SmartChunkKnowledge, SpeakerTurn, TextSegmenter
from tts.src.recording import AtomicWavWriter
from tts.src.streaming import IncrementalPipelineSession, IncrementalTaggedPipelineSession
from tts.src.voices import StoredVoice, VoiceRepository

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from typing import BinaryIO, Literal

    from tts.src.config import Settings

LOGGER = logging.getLogger('tts')


class _ModelRuntime:
    """Bind pipeline operations to one preloaded model for a complete request."""

    def __init__(self, runtime: TtsRuntime, engine: PocketTtsEngine) -> None:
        self._runtime = runtime
        self.engine = engine

    @property
    def settings(self) -> Settings:
        return self.engine.settings

    def sample_rate(self) -> int:
        return self.engine.sample_rate()

    def prepare_voice(self, requested_voice: str | None) -> str:
        return self.engine.prepare_voice(requested_voice)

    def stream_model_pcm(self, text: str, voice: str) -> Generator[bytes, None, None]:
        yield from self.engine.stream_pcm(text, voice)

    def smart_chunk_knowledge(self, voice: str) -> SmartChunkKnowledge:
        return self._runtime.smart_chunk_knowledge(
            voice,
            model=self.settings.model_id,
        )

    def open_latest_wav_capture(self, sample_rate: int) -> AtomicWavWriter | None:
        return self._runtime.open_latest_wav_capture(sample_rate)

    def write_latest_wav_chunk(
        self,
        capture: AtomicWavWriter | None,
        chunk: bytes,
    ) -> AtomicWavWriter | None:
        return self._runtime.write_latest_wav_chunk(capture, chunk)

    def finish_latest_wav_capture(
        self,
        capture: AtomicWavWriter | None,
        *,
        completed: bool,
        pcm_bytes: int,
    ) -> None:
        self._runtime.finish_latest_wav_capture(
            capture,
            completed=completed,
            pcm_bytes=pcm_bytes,
        )


class TtsRuntime:
    def __init__(self, settings: Settings) -> None:
        self.operations = ExclusiveOperationGate()
        self._settings = settings
        self._models = {
            model_id: _ModelRuntime(self, PocketTtsEngine(model_settings))
            for model_id, model_settings in self._configured_model_settings(settings).items()
        }
        # Preserve the existing default-engine access used by focused engine tests and callers.
        self.engine = self._models[settings.model_id].engine
        self._smart_chunk_knowledge_cache: dict[
            tuple[Path, str, str, str],
            SmartChunkKnowledge,
        ] = {}

    @staticmethod
    def _configured_model_settings(settings: Settings) -> dict[str, Settings]:
        configured = {settings.model_id: settings}
        configured.update(
            {
                model_id: settings.model_copy(
                    update={
                        'model_id': model_id,
                        'language': model.language,
                        'voice': model.voice,
                        'additional_models': {},
                        'model_by_voice': {},
                    },
                )
                for model_id, model in settings.additional_models.items()
            },
        )
        return configured

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(self._models)

    def apply_settings(self, settings: Settings) -> None:
        replacements = self._configured_model_settings(settings)
        previous = {model_id: model.engine.settings for model_id, model in self._models.items()}
        applied: list[str] = []
        try:
            for model_id, model in self._models.items():
                model.engine.apply_settings(replacements[model_id])
                applied.append(model_id)
        except BaseException:
            for model_id in reversed(applied):
                self._models[model_id].engine.apply_settings(previous[model_id])
            raise
        self._settings = settings

    def load(self) -> None:
        try:
            for model in self._models.values():
                model.engine.load()
        except BaseException:
            for model in self._models.values():
                model.engine.close()
            raise

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
        for model in self._models.values():
            model.engine.invalidate_voice(stored.name)
        return stored

    def list_voices(self) -> list[StoredVoice]:
        return VoiceRepository(
            self.settings.data_directory,
            self.settings.maximum_voice_upload_bytes,
        ).list()

    def close(self) -> None:
        for knowledge in self._smart_chunk_knowledge_cache.values():
            knowledge.save(force=True)
        for model in self._models.values():
            model.engine.close()

    def status(self) -> RuntimeStatus:
        return self.engine.status()

    def _model(self, model: str | None = None) -> _ModelRuntime:
        selected = self.settings.model_id if model is None else model
        try:
            return self._models[selected]
        except KeyError as error:
            raise ModelMismatchError(self.model_ids, selected) from error

    def _select_model(
        self,
        requested_model: str | None,
        voices: Iterable[str | None],
    ) -> _ModelRuntime:
        selected = self._model(requested_model)
        if selected.settings.model_id != self.settings.model_id:
            return selected
        routed_models: set[str] = set()
        for voice in voices:
            voice_name = voice.strip() if voice is not None else 'default'
            model_id = (
                self.settings.model_id
                if not voice_name or voice_name == 'default'
                else self.settings.model_by_voice.get(voice_name, self.settings.model_id)
            )
            routed_models.add(model_id)
        if len(routed_models) > 1:
            raise VoiceModelConflictError
        if routed_models:
            return self._model(routed_models.pop())
        return selected

    def prepare_voice(self, requested_voice: str | None, *, model: str | None = None) -> str:
        return self._model(model).prepare_voice(requested_voice)

    def create_incremental_pipeline(
        self,
        requested_voice: str | None,
        *,
        model: str | None = None,
    ) -> IncrementalPipelineSession:
        selected = self._select_model(model, (requested_voice,))
        return IncrementalPipelineSession(
            selected,
            selected.prepare_voice(requested_voice),
        )

    def create_incremental_tagged_pipeline(
        self,
        *,
        model: str | None = None,
    ) -> IncrementalTaggedPipelineSession:
        return IncrementalTaggedPipelineSession(self._model(model))

    def _validated_speech(self, command: SpeechCommand) -> tuple[str, _ModelRuntime]:
        text = command.text.strip()
        selected = self.validate_options(command.model, command.speed, voices=(command.voice,))
        if not text:
            raise EmptyInputError
        if len(text) > self.settings.maximum_input_characters:
            raise InputTooLongError
        return text, selected

    def validate_speech(self, command: SpeechCommand) -> str:
        return self._validated_speech(command)[0]

    def prepare_speech(self, command: SpeechCommand) -> PreparedSpeech:
        text, selected = self._validated_speech(command)
        return PreparedSpeech(
            model=selected.settings.model_id,
            text=text,
            voice=selected.prepare_voice(command.voice),
        )

    def prepare_multi_speaker(
        self,
        command: MultiSpeakerCommand,
    ) -> tuple[str, list[SpeakerTurn]]:
        """Validate structured input and bind every turn to one selected model."""
        selected = self.validate_options(
            command.model,
            command.speed,
            voices=command.speakers.values(),
        )
        turns = self._prepare_multi_speaker_turns(command, selected)
        return selected.settings.model_id, turns

    def prepare_multi_speaker_turns(self, command: MultiSpeakerCommand) -> list[SpeakerTurn]:
        """Return validated turns for callers that do not need the routed model ID."""
        return self.prepare_multi_speaker(command)[1]

    def _prepare_multi_speaker_turns(  # noqa: C901
        self,
        command: MultiSpeakerCommand,
        selected: _ModelRuntime,
    ) -> list[SpeakerTurn]:
        """Validate structured input and preload every voice before streaming."""
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

        voices = {
            speaker: selected.prepare_voice(voice) for speaker, voice in command.speakers.items()
        }
        return [
            SpeakerTurn(speaker, voices[speaker], text) for speaker, text in normalized_segments
        ]

    def validate_options(
        self,
        model: str,
        speed: float,
        *,
        voices: Iterable[str | None] = (),
    ) -> _ModelRuntime:
        selected = self._select_model(model, voices)
        if speed != 1.0:
            raise UnsupportedSpeedError
        return selected

    def generate_wav(self, text: str, voice: str, *, model: str | None = None) -> bytes:
        wav = self._model(model).engine.generate_wav(text, voice)
        if self.settings.save_latest_wav:
            self._save_latest_wav_response(wav)
        return wav

    def stream_model_pcm(
        self,
        text: str,
        voice: str,
        *,
        model: str | None = None,
    ) -> Generator[bytes, None, None]:
        """Stream raw model PCM without request-level recording semantics."""
        yield from self._model(model).stream_model_pcm(text, voice)

    def stream_pcm(
        self,
        text: str,
        voice: str,
        *,
        model: str | None = None,
    ) -> Generator[bytes, None, None]:
        selected = self._model(model)
        output_bytes = 0
        capture: AtomicWavWriter | None = None
        completed = False
        if self.settings.save_latest_wav:
            capture = self.open_latest_wav_capture(selected.sample_rate())
        try:
            for chunk in selected.stream_model_pcm(text, voice):
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
                'model': selected.settings.model_id,
                'response_format': 'pcm',
                'audio_bytes': output_bytes,
            },
        )

    def generate_pipeline_wav(
        self,
        text: str,
        voice: str,
        *,
        model: str | None = None,
    ) -> tuple[bytes, int]:
        selected = self._model(model)
        pcm = b''.join(
            self.stream_pipeline_pcm(
                text,
                voice,
                model=selected.settings.model_id,
                capture_latest=False,
            ),
        )
        wav = wav_from_pcm(selected.sample_rate(), pcm)
        if self.settings.save_latest_wav:
            self._save_latest_wav_response(wav)
        return wav, len(pcm)

    def generate_multi_speaker_wav(
        self,
        turns: list[SpeakerTurn],
        *,
        model: str | None = None,
    ) -> tuple[bytes, int]:
        selected = self._model(model)
        pcm = b''.join(
            self.stream_multi_speaker_pcm(
                turns,
                model=selected.settings.model_id,
                capture_latest=False,
            ),
        )
        wav = wav_from_pcm(selected.sample_rate(), pcm)
        if self.settings.save_latest_wav:
            self._save_latest_wav_response(wav)
        return wav, len(pcm)

    def stream_pipeline_pcm(
        self,
        text: str,
        voice: str,
        *,
        model: str | None = None,
        capture_latest: bool = True,
        transport: str = 'http',
    ) -> Generator[bytes, None, None]:
        selected = self._model(model)
        segmenter = TextSegmenter(
            selected.settings.pipeline_sentence_terminators,
            first_segment_comma_delimiter=(
                selected.settings.pipeline_first_segment_comma_delimiter
            ),
        )
        segments = [*segmenter.append(text), *segmenter.finish()]
        pipeline = PcmPipeline(
            selected,
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
        model: str | None = None,
        capture_latest: bool = True,
    ) -> Generator[bytes, None, None]:
        selected = self._model(model)
        pipeline = PcmPipeline(
            selected,
            turns[0].voice,
            capture_latest=capture_latest,
            transport='http_multi_speaker',
        )
        completed = False
        try:
            for turn_index, turn in enumerate(turns):
                pipeline.select_voice(turn.voice)
                segmenter = TextSegmenter(
                    selected.settings.pipeline_sentence_terminators,
                    first_segment_comma_delimiter=(
                        selected.settings.pipeline_first_segment_comma_delimiter
                    ),
                )
                segments = [*segmenter.append(turn.text), *segmenter.finish()]
                LOGGER.debug(
                    'TTS multi-speaker turn started',
                    extra={
                        'event_id': 'ID_tts_multi_speaker_turn_started',
                        'model': selected.settings.model_id,
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
        LOGGER.debug(
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
        LOGGER.debug(
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

    def pcm_headers(self, model: str | None = None) -> dict[str, str]:
        sample_rate = self.sample_rate(model)
        return {
            'X-Audio-Format': 'pcm_s16le',
            'X-Audio-Sample-Rate': str(sample_rate),
            'X-Audio-Sample-Width': '2',
            'X-Audio-Channels': '1',
        }

    def sample_rate(self, model: str | None = None) -> int:
        return self._model(model).sample_rate()

    def smart_chunk_knowledge(
        self,
        voice: str,
        *,
        model: str | None = None,
    ) -> SmartChunkKnowledge:
        settings = self._model(model).settings
        key = (
            settings.data_directory,
            settings.model_id,
            voice,
            settings.pipeline_smart_chunk_llm_id,
        )
        knowledge = self._smart_chunk_knowledge_cache.get(key)
        if knowledge is None:
            knowledge = SmartChunkKnowledge(
                settings,
                voice,
                model_scope=(
                    settings.model_id if settings.model_id != self.settings.model_id else None
                ),
            )
            self._smart_chunk_knowledge_cache[key] = knowledge
        else:
            knowledge.settings = settings
        return knowledge
