from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='STT_',
        case_sensitive=False,
        frozen=True,
        populate_by_name=True,
        extra='ignore',
    )

    model_id: str = 'nvidia/stt_en_fastconformer_hybrid_large_streaming_multi'
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
        default='INFO',
        validation_alias='LOG_LEVEL',
    )
    device: str = 'cpu'
    torch_threads: int = Field(default=4, ge=1)
    decoder_type: Literal['rnnt', 'ctc'] = 'rnnt'

    attention_context_size: Annotated[
        tuple[int, int] | None,
        NoDecode,
    ] = (70, 1)

    input_audio_seconds: float = Field(default=0.25, gt=0)
    final_flush_seconds: float = Field(default=0.25, ge=0)
    preprocess_holdback_seconds: float = Field(default=0.1, ge=0)
    minimum_delta_characters: int = Field(default=1, ge=1)

    online_normalization: bool = False
    pad_and_drop_preencoded: bool = False

    stream_sample_rate: int = Field(default=16_000, ge=8_000)
    maximum_upload_bytes: int = Field(default=100 * 1024**2, ge=1)
    maximum_stream_bytes: int = Field(
        default=30 * 60 * 16_000 * 2,
        ge=1,
    )
    maximum_websocket_message_bytes: int = Field(
        default=2 * 1024**2,
        ge=1,
    )
    save_latest_wav: bool = False
    latest_wav_path: Path = Path(tempfile.gettempdir()) / 'latest.wav'
    listen_port: int = Field(
        default=8080,
        ge=1,
        le=65_535,
        validation_alias='LISTEN_PORT',
    )

    @model_validator(mode='before')
    @classmethod
    def default_final_flush_to_input_size(cls, values: object) -> object:
        if not isinstance(values, dict):
            return values

        values = values.copy()

        if 'final_flush_seconds' not in values:
            values['final_flush_seconds'] = values.get(
                'input_audio_seconds',
                cls.model_fields['input_audio_seconds'].default,
            )

        return values

    @field_validator('attention_context_size', mode='before')
    @classmethod
    def parse_attention_context_size(
        cls,
        value: object,
    ) -> object:
        if value is None or value == '':
            return None

        if not isinstance(value, str):
            return value

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [item.strip() for item in value.split(',') if item.strip()]

        if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:  # noqa: PLR2004
            error = f'expected a JSON array or comma-separated pair of integers, got {value!r}'
            raise ValueError(error)

        return tuple(int(item) for item in parsed)
