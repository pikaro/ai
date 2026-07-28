from __future__ import annotations

from typing import TYPE_CHECKING

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
