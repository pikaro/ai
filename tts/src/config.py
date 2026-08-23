import tempfile
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from service_contracts.tts import MODEL_ID


class AdditionalModelSettings(BaseModel):
    """Configure one additional preloaded Pocket TTS language model."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    language: str = Field(min_length=1)
    voice: str = Field(min_length=1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='TTS_',
        case_sensitive=False,
        frozen=True,
        populate_by_name=True,
        extra='ignore',
    )

    model_id: str = MODEL_ID
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
        default='INFO',
        validation_alias='LOG_LEVEL',
    )
    language: str = 'english'
    voice: str = 'alba'
    additional_models: dict[str, AdditionalModelSettings] = Field(default_factory=dict)
    data_directory: Path = Path('/data')
    torch_threads: int = Field(default=2, ge=1)
    maximum_input_characters: int = Field(default=50_000, ge=1)
    maximum_voice_upload_bytes: int = Field(default=100 * 1024**2, ge=1)
    pipeline_clause_pause_seconds: float = Field(default=0.04, ge=0, le=2)
    pipeline_sentence_pause_seconds: float = Field(default=0.12, ge=0, le=2)
    pipeline_paragraph_pause_seconds: float = Field(default=0.24, ge=0, le=2)
    pipeline_speaker_switch_pause_seconds: float = Field(default=0.2, ge=0, le=2)
    pipeline_sentence_crossfade_seconds: float = Field(default=0.01, ge=0, le=0.25)
    pipeline_silence_confirmation_seconds: float = Field(default=0.02, gt=0, le=0.1)
    pipeline_silence_threshold_dbfs: float = Field(default=-43.0, ge=-100, le=0)
    pipeline_speech_hysteresis_db: float = Field(default=6.0, ge=0, le=30)
    pipeline_speech_confirmation_seconds: float = Field(default=0.04, gt=0, le=0.25)
    pipeline_sentence_terminators: str = Field(default='.!?', min_length=1)
    pipeline_first_segment_comma_delimiter: bool = True
    pipeline_smart_chunk_enabled: bool = True
    pipeline_smart_chunk_confidence: float = Field(default=0.9, ge=0.5, lt=1)
    pipeline_smart_chunk_safety_seconds: float = Field(default=0.1, ge=0, le=5)
    pipeline_smart_chunk_cold_start_speedup: float = Field(default=3.0, ge=1, le=100)
    pipeline_smart_chunk_knowledge_flush_observations: int = Field(
        default=8,
        ge=1,
        le=10_000,
    )
    pipeline_smart_chunk_llm_id: str = Field(
        default='default',
        pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$',
    )
    pipeline_idle_timeout_seconds: float = Field(default=30.0, gt=0)
    save_latest_wav: bool = False
    latest_wav_path: Path = Path(tempfile.gettempdir()) / 'latest.wav'
    listen_port: int = Field(
        default=8080,
        ge=1,
        le=65_535,
        validation_alias='LISTEN_PORT',
    )

    @model_validator(mode='after')
    def validate_additional_models(self) -> Self:
        if self.model_id in self.additional_models:
            message = 'additional_models must not repeat the default model_id'
            raise ValueError(message)
        empty_model_ids = [model_id for model_id in self.additional_models if not model_id.strip()]
        if empty_model_ids:
            message = 'additional_models keys must be non-empty model ids'
            raise ValueError(message)
        return self
