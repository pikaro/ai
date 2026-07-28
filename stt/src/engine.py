from __future__ import annotations

import importlib
import logging
import threading
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, cast

from stt.src.domain import (
    AttentionContextRestartRequiredError,
    ModelNotLoadedError,
    RuntimeStatus,
)
from stt.src.metrics import MODEL_LOAD_SECONDS, MODEL_READY
from stt.src.transcript import extract_first_text

if TYPE_CHECKING:
    from pathlib import Path

    from stt.src.config import Settings

LOGGER = logging.getLogger('stt')


class AsrEngine:
    """Own the NeMo model, decoder configuration, and inference lock."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Any | None = None
        self.torch: Any | None = None
        self.numpy: Any | None = None
        self.streaming_buffer_type: Any | None = None
        self.load_seconds = 0.0
        self.lock = threading.Lock()

    def apply_settings(self, settings: Settings) -> None:  # noqa: C901
        """Apply settings that are safe between exclusive STT sessions."""
        previous = self.settings
        if (
            settings.attention_context_size is None
            and settings.attention_context_size != previous.attention_context_size
        ):
            raise AttentionContextRestartRequiredError
        with self.lock:
            try:
                self.settings = settings
                if self.torch is not None and settings.torch_threads != previous.torch_threads:
                    self.torch.set_num_threads(settings.torch_threads)
                if self.model is not None and settings.decoder_type != previous.decoder_type:
                    self._configure_decoder(self.model)
                if (
                    self.model is not None
                    and settings.attention_context_size != previous.attention_context_size
                ):
                    self._configure_encoder(self.model)
            except BaseException:
                self.settings = previous
                if self.torch is not None and settings.torch_threads != previous.torch_threads:
                    self.torch.set_num_threads(previous.torch_threads)
                if self.model is not None and settings.decoder_type != previous.decoder_type:
                    self._configure_decoder(self.model)
                if (
                    self.model is not None
                    and settings.attention_context_size != previous.attention_context_size
                ):
                    self._configure_encoder(self.model)
                raise

    def load(self) -> None:
        started = time.perf_counter()
        LOGGER.info(
            'Loading ASR model',
            extra={'event_id': 'ID_stt_model_loading', 'model': self.settings.model_id},
        )
        self.numpy = importlib.import_module('numpy')
        self.torch = importlib.import_module('torch')
        nemo_asr = importlib.import_module('nemo.collections.asr')
        streaming_utils = importlib.import_module(
            'nemo.collections.asr.parts.utils.streaming_utils',
        )
        self.streaming_buffer_type = streaming_utils.CacheAwareStreamingAudioBuffer

        self.torch.set_num_threads(self.settings.torch_threads)
        self.torch.set_grad_enabled(False)
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.settings.model_id)
        self._configure_model(model)
        self.model = model.to(self.settings.device) if hasattr(model, 'to') else model
        self.load_seconds = time.perf_counter() - started
        MODEL_LOAD_SECONDS.set(self.load_seconds)
        MODEL_READY.set(1)
        LOGGER.info(
            'ASR model ready',
            extra={'event_id': 'ID_stt_model_ready', 'duration_seconds': self.load_seconds},
        )

    def close(self) -> None:
        MODEL_READY.set(0)
        self.model = None
        self.streaming_buffer_type = None

    def status(self) -> RuntimeStatus:
        if self.model is None:
            raise ModelNotLoadedError
        return RuntimeStatus(
            model=self.settings.model_id,
            device=self.settings.device,
            decoder_type=self.settings.decoder_type,
            attention_context_size=self.settings.attention_context_size,
            sample_rate=self.model_sample_rate(),
            load_seconds=self.load_seconds,
        )

    def model_sample_rate(self) -> int | None:
        if self.model is None:
            return None
        config = getattr(self.model, 'cfg', None) or getattr(self.model, '_cfg', None)
        if config is None:
            return None
        value = getattr(config, 'sample_rate', None)
        if value is None and hasattr(config, 'get'):
            value = config.get('sample_rate')
        return int(value) if value else None

    def transcribe_file(self, path: Path) -> str:
        if self.model is None:
            raise ModelNotLoadedError
        inference_context = self.torch.inference_mode() if self.torch is not None else nullcontext()
        with self.lock, inference_context:
            result = self.model.transcribe([str(path)], batch_size=1)
        if isinstance(result, tuple):
            result = result[0]
        return extract_first_text(result)

    def pcm16_to_float32(self, pcm: bytes, channels: int) -> Any:  # noqa: ANN401
        if self.numpy is None:
            message = 'NumPy is not loaded'
            raise RuntimeError(message)
        samples = self.numpy.frombuffer(pcm, dtype='<i2').astype(self.numpy.float32) / 32768.0
        if channels > 1:
            usable_samples = samples.size - (samples.size % channels)
            samples = samples[:usable_samples].reshape(-1, channels).mean(axis=1)
        return samples

    def drop_extra_pre_encoded(self, step_number: int) -> int:
        if step_number == 0 and not self.settings.pad_and_drop_preencoded:
            return 0
        encoder = getattr(self.model, 'encoder', None)
        streaming_config = getattr(encoder, 'streaming_cfg', None)
        return int(getattr(streaming_config, 'drop_extra_pre_encoded', 0) or 0)

    def _configure_model(self, model: Any) -> None:  # noqa: ANN401
        self._configure_decoder(model)
        self._configure_encoder(model)
        self._configure_preprocessor(model)
        if hasattr(model, 'freeze'):
            model.freeze()
        if hasattr(model, 'eval'):
            model.eval()

    def _configure_decoder(self, model: Any) -> None:  # noqa: ANN401
        decoder_module_name = (
            'nemo.collections.asr.parts.submodules.rnnt_decoding'
            if self.settings.decoder_type == 'rnnt'
            else 'nemo.collections.asr.parts.submodules.ctc_decoding'
        )
        decoder_module = importlib.import_module(decoder_module_name)
        decoder_config = (
            decoder_module.RNNTDecodingConfig(fused_batch_size=-1)
            if self.settings.decoder_type == 'rnnt'
            else decoder_module.CTCDecodingConfig()
        )
        if hasattr(model, 'cur_decoder'):
            model.change_decoding_strategy(
                decoder_config,
                decoder_type=self.settings.decoder_type,
            )
        else:
            model.change_decoding_strategy(decoder_config)

    def _configure_encoder(self, model: Any) -> None:  # noqa: ANN401
        encoder = getattr(model, 'encoder', None)
        if self.settings.attention_context_size and hasattr(
            encoder,
            'set_default_att_context_size',
        ):
            cast('Any', encoder).set_default_att_context_size(
                list(self.settings.attention_context_size),
            )

    @staticmethod
    def _configure_preprocessor(model: Any) -> None:  # noqa: ANN401
        featurizer = getattr(getattr(model, 'preprocessor', None), 'featurizer', None)
        if featurizer is not None:
            if hasattr(featurizer, 'dither'):
                featurizer.dither = 0.0
            if hasattr(featurizer, 'pad_to'):
                featurizer.pad_to = 0
