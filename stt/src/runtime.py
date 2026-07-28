from __future__ import annotations

from typing import TYPE_CHECKING

from runtime_config import ExclusiveOperationGate
from stt.src.domain import ModelMismatchError, RuntimeStatus
from stt.src.engine import AsrEngine
from stt.src.recording import save_wav_atomic
from stt.src.streaming import CacheAwareStreamingSession

if TYPE_CHECKING:
    from pathlib import Path

    from stt.src.config import Settings


class AsrRuntime:
    """Transport-neutral STT application service."""

    def __init__(self, settings: Settings) -> None:
        self.operations = ExclusiveOperationGate()
        self.engine = AsrEngine(settings)

    @property
    def settings(self) -> Settings:
        return self.engine.settings

    def apply_settings(self, settings: Settings) -> None:
        self.engine.apply_settings(settings)

    def load(self) -> None:
        self.engine.load()

    def close(self) -> None:
        self.engine.close()

    def status(self) -> RuntimeStatus:
        return self.engine.status()

    def validate_model(self, requested_model: str) -> None:
        if requested_model and requested_model != self.settings.model_id:
            raise ModelMismatchError(self.settings.model_id, requested_model)

    def transcribe_file(self, path: Path) -> str:
        return self.engine.transcribe_file(path)

    def save_latest_wav(self, pcm: bytes, sample_rate: int, channels: int) -> None:
        save_wav_atomic(self.settings.latest_wav_path, pcm, sample_rate, channels)

    def create_stream(self, sample_rate: int, channels: int) -> CacheAwareStreamingSession:
        return CacheAwareStreamingSession(
            self.engine,
            sample_rate,
            channels,
            self.save_latest_wav,
        )
