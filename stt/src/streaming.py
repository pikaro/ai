from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from service_contracts.stt import (
    SttTranscriptEvent,
    TranscriptionCompleted,
    TranscriptionDelta,
    TranscriptionPartial,
)
from stt.src.domain import AudioStreamTooLargeError, ModelNotLoadedError
from stt.src.transcript import (
    extract_first_text,
    log_transcription,
    stable_word_prefix,
    transcript_delta,
)

if TYPE_CHECKING:
    from stt.src.engine import AsrEngine

RecordingSink = Callable[[bytes, int, int], None]


class CacheAwareStreamingSession:
    """Incrementally decode one PCM stream using a loaded cache-aware engine."""

    def __init__(
        self,
        engine: AsrEngine,
        sample_rate: int,
        channels: int,
        recording_sink: RecordingSink,
    ) -> None:
        if engine.model is None or engine.torch is None or engine.streaming_buffer_type is None:
            raise ModelNotLoadedError
        expected_sample_rate = engine.model_sample_rate()
        if expected_sample_rate is not None and sample_rate != expected_sample_rate:
            message = f'expected {expected_sample_rate} Hz PCM, got {sample_rate} Hz'
            raise ValueError(message)

        self.engine = engine
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording_sink = recording_sink
        self.input_chunk_bytes = max(
            channels * 2,
            int(sample_rate * engine.settings.input_audio_seconds) * channels * 2,
        )
        self.holdback_feature_frames = self._seconds_to_feature_frames(
            engine.settings.preprocess_holdback_seconds,
        )
        self.raw_pcm = bytearray()
        self.unprocessed_bytes = 0
        self.appended_feature_frames = 0
        self.stream_id = -1
        self.streaming_buffer = engine.streaming_buffer_type(
            model=engine.model,
            online_normalization=engine.settings.online_normalization,
            pad_and_drop_preencoded=engine.settings.pad_and_drop_preencoded,
        )
        (
            self.cache_last_channel,
            self.cache_last_time,
            self.cache_last_channel_length,
        ) = engine.model.encoder.get_initial_cache_state(batch_size=1)
        self.previous_hypotheses: Any | None = None
        self.previous_prediction: Any | None = None
        self.step_number = 0
        self.emitted_text = ''
        self.previous_transcript = ''
        self.latest_transcript = ''

    def append_pcm(self, pcm: bytes) -> list[SttTranscriptEvent]:
        if len(self.raw_pcm) + len(pcm) > self.engine.settings.maximum_stream_bytes:
            raise AudioStreamTooLargeError
        self.raw_pcm.extend(pcm)
        self.unprocessed_bytes += len(pcm)
        if self.unprocessed_bytes < self.input_chunk_bytes:
            return []
        with self.engine.lock:
            self._append_ready_features(final=False)
            events = self._process_ready(final=False)
        self.unprocessed_bytes = 0
        return events

    def finish(self) -> list[SttTranscriptEvent]:
        with self.engine.lock:
            if self.engine.settings.save_latest_wav:
                self.recording_sink(bytes(self.raw_pcm), self.sample_rate, self.channels)
            if self.engine.settings.final_flush_seconds > 0 and self.raw_pcm:
                flush_samples = max(
                    1,
                    int(self.sample_rate * self.engine.settings.final_flush_seconds),
                )
                self.raw_pcm.extend(b'\x00\x00' * flush_samples * self.channels)
            self._append_ready_features(final=True)
            events = self._process_ready(final=True)
        completed = self.latest_transcript or self.emitted_text
        if completed:
            log_transcription('completed', completed)
            events.append(TranscriptionCompleted(transcript=completed))
        return events

    def _seconds_to_feature_frames(self, seconds: float) -> int:
        config = getattr(self.engine.model, 'cfg', None) or getattr(
            self.engine.model,
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
        audio = self.engine.pcm16_to_float32(bytes(self.raw_pcm), self.channels)
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

    def _process_ready(self, *, final: bool) -> list[SttTranscriptEvent]:
        events: list[SttTranscriptEvent] = []
        if getattr(self.streaming_buffer, 'buffer', None) is None:
            return events
        torch_module = self.engine.torch
        model = self.engine.model
        if torch_module is None or model is None:
            raise ModelNotLoadedError
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
                    drop_extra_pre_encoded=self.engine.drop_extra_pre_encoded(self.step_number),
                    return_transcription=True,
                )
            self.step_number += 1
            transcript = extract_first_text(transcribed_texts)
            if transcript:
                events.extend(self._transcript_events(transcript, final=final))
        return events

    def _transcript_events(
        self,
        transcript: str,
        *,
        final: bool,
    ) -> list[SttTranscriptEvent]:
        stable_transcript = (
            transcript if final else stable_word_prefix(self.previous_transcript, transcript)
        )
        self.previous_transcript = transcript
        self.latest_transcript = transcript
        delta = transcript_delta(self.emitted_text, stable_transcript)
        if delta and len(delta) >= self.engine.settings.minimum_delta_characters:
            self.emitted_text = stable_transcript
            log_transcription('stable delta', delta)
            return [
                TranscriptionDelta(
                    delta=delta,
                    transcript=stable_transcript,
                ),
            ]
        if not final:
            log_transcription('partial', transcript)
            return [TranscriptionPartial(transcript=transcript)]
        return []
