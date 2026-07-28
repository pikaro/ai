from __future__ import annotations

from dataclasses import dataclass


class AsrServiceError(Exception):
    """Base class for expected ASR service failures."""


class ModelNotLoadedError(AsrServiceError):
    def __init__(self) -> None:
        super().__init__('ASR model is not loaded')


class AttentionContextRestartRequiredError(AsrServiceError):
    def __init__(self) -> None:
        super().__init__('clearing attention_context_size requires a service restart')


class ModelMismatchError(AsrServiceError):
    def __init__(self, loaded_model: str, requested_model: str) -> None:
        super().__init__(f'loaded model is {loaded_model}, not {requested_model}')


class AudioUploadTooLargeError(AsrServiceError):
    def __init__(self) -> None:
        super().__init__('audio upload exceeds STT_MAXIMUM_UPLOAD_BYTES')


class UnsupportedAudioExtensionError(AsrServiceError):
    def __init__(self, suffix: str) -> None:
        super().__init__(f'unsupported audio extension: {suffix}')


class AudioStreamTooLargeError(AsrServiceError):
    def __init__(self) -> None:
        super().__init__('audio stream exceeds STT_MAXIMUM_STREAM_BYTES')


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    model: str
    device: str
    decoder_type: str
    attention_context_size: tuple[int, int] | None
    sample_rate: int | None
    load_seconds: float
