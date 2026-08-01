from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class TtsServiceError(Exception):
    """Base class for expected TTS request and service failures."""


class ModelNotLoadedError(TtsServiceError):
    def __init__(self) -> None:
        super().__init__('TTS model is not loaded')


class InvalidVoiceNameError(TtsServiceError):
    def __init__(self) -> None:
        super().__init__(
            'voice name must contain only letters, numbers, underscores, and hyphens',
        )


class UnsupportedVoiceFileError(TtsServiceError):
    def __init__(self) -> None:
        super().__init__('voice upload must be a .safetensors file')


class VoiceUploadTooLargeError(TtsServiceError):
    def __init__(self) -> None:
        super().__init__('voice upload exceeds TTS_MAXIMUM_VOICE_UPLOAD_BYTES')


class EmptyVoiceUploadError(TtsServiceError):
    def __init__(self) -> None:
        super().__init__('voice upload is empty')


class InvalidVoiceSelectorError(TtsServiceError):
    def __init__(self) -> None:
        super().__init__('voice must be "default" or a valid voice name')


class VoiceUnavailableError(TtsServiceError):
    def __init__(self, voice: str) -> None:
        super().__init__(f'voice {voice!r} is not available')


class ModelMismatchError(TtsServiceError):
    def __init__(self, loaded_model: str, requested_model: str) -> None:
        super().__init__(f'loaded model is {loaded_model}, not {requested_model}')


class UnsupportedSpeedError(TtsServiceError):
    def __init__(self) -> None:
        super().__init__('Pocket TTS does not support speed adjustment')


class EmptyInputError(TtsServiceError):
    def __init__(self) -> None:
        super().__init__('input must contain non-whitespace text')


class EmptySegmentError(TtsServiceError):
    def __init__(self) -> None:
        super().__init__('segment text must contain non-whitespace text')


class InputTooLongError(TtsServiceError):
    def __init__(self) -> None:
        super().__init__('input exceeds TTS_MAX_INPUT_CHARACTERS')


class UndefinedSpeakerError(TtsServiceError):
    def __init__(self, speaker: str) -> None:
        super().__init__(f'speaker {speaker!r} is not defined')


class InvalidMultiSpeakerMarkupError(TtsServiceError):
    def __init__(self, message: str) -> None:
        super().__init__(f'invalid multi-speaker markup: {message}')


@dataclass(frozen=True, slots=True)
class SpeechCommand:
    model: str
    text: str
    voice: str | None
    speed: float


@dataclass(frozen=True, slots=True)
class PreparedSpeech:
    text: str
    voice: str


@dataclass(frozen=True, slots=True)
class SpeakerSegment:
    speaker: str
    text: str


@dataclass(frozen=True, slots=True)
class MultiSpeakerCommand:
    model: str
    speakers: Mapping[str, str]
    segments: Sequence[SpeakerSegment]
    speed: float


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    model: str
    voice: str
    language: str
    sample_rate: int
    load_seconds: float
    voice_load_seconds: float
