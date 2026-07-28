from __future__ import annotations

import importlib
import io
import logging
import threading
import time
import wave
from typing import TYPE_CHECKING, Any, Protocol, cast

from tts.src.domain import (
    InvalidVoiceSelectorError,
    ModelNotLoadedError,
    RuntimeStatus,
    VoiceUnavailableError,
)
from tts.src.metrics import MODEL_LOAD_SECONDS, MODEL_READY, VOICE_LOAD_SECONDS
from tts.src.voices import is_voice_name

if TYPE_CHECKING:
    from collections.abc import Generator

    from tts.src.config import Settings

LOGGER = logging.getLogger('tts')


class _TorchModule(Protocol):
    int16: object

    def set_num_threads(self, threads: int, /) -> None: ...


class _SampleRateModel(Protocol):
    sample_rate: int


class PocketTtsEngine:
    """Own the PocketTTS model, voice-state cache, and tensor conversion."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Any | None = None
        self.voice_states: dict[str, Any] = {}
        self.torch: _TorchModule | None = None
        self.load_seconds = 0.0
        self.voice_load_seconds = 0.0
        self.lock = threading.Lock()

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
        """Load replacement engine state before publishing changed settings."""
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

    def close(self) -> None:
        MODEL_READY.set(0)
        self.voice_states.clear()
        self.model = None

    def voice_source(self, voice: str, settings: Settings | None = None) -> str:
        selected = settings or self.settings
        if is_voice_name(voice):
            voice_path = selected.data_directory / f'{voice}.safetensors'
            if voice_path.is_file():
                return str(voice_path)
        return voice

    def invalidate_voice(self, voice: str) -> None:
        self.voice_states.pop(voice, None)

    def prepare_voice(self, requested_voice: str | None) -> str:
        """Resolve and cache a request voice before response streaming begins."""
        voice = self._selected_voice(requested_voice)
        model = self.model
        if model is None:
            raise ModelNotLoadedError
        with self.lock:
            if voice in self.voice_states:
                return voice
            started = time.perf_counter()
            try:
                voice_state = model.get_state_for_audio_prompt(self.voice_source(voice))
            except (FileNotFoundError, ValueError) as error:
                raise VoiceUnavailableError(voice) from error
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

    def status(self) -> RuntimeStatus:
        model = self._loaded_model_only()
        return RuntimeStatus(
            model=self.settings.model_id,
            voice=self.settings.voice,
            language=self.settings.language,
            sample_rate=int(model.sample_rate),
            load_seconds=self.load_seconds,
            voice_load_seconds=self.voice_load_seconds,
        )

    def sample_rate(self) -> int:
        return int(self._loaded_model_only().sample_rate)

    def generate_wav(self, text: str, voice: str) -> bytes:
        model, voice_state = self._loaded_model(voice)
        with self.lock:
            audio = model.generate_audio(voice_state, text)
        return self._wav_bytes(int(model.sample_rate), audio)

    def stream_pcm(self, text: str, voice: str) -> Generator[bytes, None, None]:
        model, voice_state = self._loaded_model(voice)
        with self.lock:
            for audio_chunk in model.generate_audio_stream(voice_state, text):
                chunk = self._pcm16_bytes(audio_chunk)
                if chunk:
                    yield chunk

    def _selected_voice(self, requested_voice: str | None) -> str:
        if requested_voice is None:
            return self.settings.voice
        requested = requested_voice.strip()
        if not requested or requested.casefold() == 'default':
            return self.settings.voice
        if not is_voice_name(requested):
            raise InvalidVoiceSelectorError
        return requested

    def _loaded_model_only(self) -> _SampleRateModel:
        if self.model is None:
            raise ModelNotLoadedError
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
        return wav_from_pcm(sample_rate, self._pcm16_bytes(audio))


def wav_from_pcm(sample_rate: int, pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()
