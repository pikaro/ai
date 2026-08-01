from __future__ import annotations

from typing import TYPE_CHECKING

from tts.src.domain import InputTooLongError
from tts.src.markup import (
    MarkupHeader,
    MarkupSpeaker,
    MarkupText,
    MultiSpeakerMarkupParser,
)
from tts.src.pipeline import PcmPipeline, TextSegmenter

if TYPE_CHECKING:
    from collections.abc import Generator

    from tts.src.pipeline import PipelineRuntime


class IncrementalPipelineSession:
    """Incrementally segment text and synthesize one stitched PCM stream."""

    def __init__(self, runtime: PipelineRuntime, voice: str) -> None:
        settings = runtime.settings
        self._segmenter = TextSegmenter(
            settings.pipeline_sentence_terminators,
            first_segment_comma_delimiter=settings.pipeline_first_segment_comma_delimiter,
        )
        self._pipeline = PcmPipeline(
            runtime,
            voice,
            capture_latest=True,
            transport='websocket',
        )

    @property
    def sample_rate(self) -> int:
        return self._pipeline.sample_rate

    @property
    def smart_flush_deadline(self) -> float | None:
        return self._pipeline.smart_flush_deadline

    def append_text(
        self,
        delta: str,
        *,
        observed_at: float,
    ) -> Generator[bytes, None, None]:
        for segment, boundary in self._segmenter.append(delta):
            yield from self._pipeline.add_segment(
                segment,
                boundary,
                observed_at=observed_at,
            )

    def flush_smart_queue(
        self,
        *,
        observed_at: float,
    ) -> Generator[bytes, None, None]:
        yield from self._pipeline.flush_smart_queue(
            misprediction=True,
            observed_at=observed_at,
        )

    def switch_voice(
        self,
        voice: str,
        *,
        observed_at: float,
    ) -> Generator[bytes, None, None]:
        segments = self._segmenter.finish()
        if segments:
            for index, (segment, boundary) in enumerate(segments):
                selected_boundary = 'speaker_switch' if index == len(segments) - 1 else boundary
                if segment:
                    yield from self._pipeline.add_segment(
                        segment,
                        selected_boundary,
                        observed_at=observed_at,
                    )
        yield from self._pipeline.finish_speaker_turn()
        self._pipeline.select_voice(voice)
        settings = self._pipeline.runtime.settings
        self._segmenter = TextSegmenter(
            settings.pipeline_sentence_terminators,
            first_segment_comma_delimiter=settings.pipeline_first_segment_comma_delimiter,
        )

    def finish(self, *, observed_at: float) -> Generator[bytes, None, None]:
        for segment, boundary in self._segmenter.finish():
            yield from self._pipeline.add_segment(
                segment,
                boundary,
                observed_at=observed_at,
            )
        yield from self._pipeline.finish()

    def close(self, *, completed: bool) -> None:
        self._pipeline.close(completed=completed)


class IncrementalTaggedPipelineSession:
    """Parse tagged deltas and synthesize their turns without buffering the response."""

    def __init__(self, runtime: PipelineRuntime) -> None:
        self._runtime = runtime
        self._parser = MultiSpeakerMarkupParser()
        self._voices: dict[str, str] = {}
        self._pipeline: IncrementalPipelineSession | None = None
        self._input_characters = 0

    @property
    def sample_rate(self) -> int:
        return self._runtime.sample_rate()

    @property
    def smart_flush_deadline(self) -> float | None:
        return self._pipeline.smart_flush_deadline if self._pipeline is not None else None

    def append_text(
        self,
        delta: str,
        *,
        observed_at: float,
    ) -> Generator[bytes, None, None]:
        yield from self._apply_events(
            self._parser.append(delta),
            observed_at=observed_at,
        )

    def flush_smart_queue(
        self,
        *,
        observed_at: float,
    ) -> Generator[bytes, None, None]:
        if self._pipeline is not None:
            yield from self._pipeline.flush_smart_queue(observed_at=observed_at)

    def finish(self, *, observed_at: float) -> Generator[bytes, None, None]:
        yield from self._apply_events(
            self._parser.finish(),
            observed_at=observed_at,
        )
        if self._pipeline is None:
            message = 'tagged pipeline input has no speaker turns'
            raise RuntimeError(message)
        yield from self._pipeline.finish(observed_at=observed_at)

    def close(self, *, completed: bool) -> None:
        if self._pipeline is not None:
            self._pipeline.close(completed=completed)

    def _apply_events(  # noqa: C901
        self,
        events: list[MarkupHeader | MarkupSpeaker | MarkupText],
        *,
        observed_at: float,
    ) -> Generator[bytes, None, None]:
        for event in events:
            if isinstance(event, MarkupHeader):
                self._voices = {
                    name: self._runtime.prepare_voice(character.voice)
                    for name, character in event.characters.items()
                }
            elif isinstance(event, MarkupSpeaker):
                voice = self._voices[event.name]
                if self._pipeline is None:
                    self._pipeline = IncrementalPipelineSession(self._runtime, voice)
                else:
                    yield from self._pipeline.switch_voice(
                        voice,
                        observed_at=observed_at,
                    )
            elif self._pipeline is not None:
                self._input_characters += len(event.text)
                if self._input_characters > self._runtime.settings.maximum_input_characters:
                    raise InputTooLongError
                yield from self._pipeline.append_text(
                    event.text,
                    observed_at=observed_at,
                )
