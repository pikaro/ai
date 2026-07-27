from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class MCPConfig(BaseModel):
    """Connection settings for one MCP server."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    host: str | None = None
    port: int = Field(default=8080, ge=1, le=65_535)
    scheme: Literal['http', 'https'] = 'http'
    path: str = '/mcp'
    url: str | None = None
    enabled: bool = True
    token: SecretStr | None = None
    transport: Literal['streamable-http', 'sse'] = 'streamable-http'
    # Retained so existing mounted configurations remain valid. Every discovered tool is now
    # exposed to the model, independent of these former prompt-selection fields.
    triggers: frozenset[str] = frozenset()
    tool_triggers: dict[str, frozenset[str]] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=10.0, gt=0)
    retry_seconds: float = Field(default=30.0, ge=0)

    @model_validator(mode='after')
    def require_address(self) -> MCPConfig:
        if self.enabled and self.url is None and self.host is None:
            message = 'enabled MCP servers require either url or host'
            raise ValueError(message)
        return self

    @property
    def endpoint(self) -> str:
        if self.url is not None:
            return self.url.rstrip('/')
        path = self.path if self.path.startswith('/') else f'/{self.path}'
        return f'{self.scheme}://{self.host}:{self.port}{path}'

    @property
    def headers(self) -> dict[str, str]:
        if self.token is None:
            return {}
        return {'Authorization': f'Bearer {self.token.get_secret_value()}'}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='ASSISTANT_',
        env_nested_delimiter='__',
        case_sensitive=False,
        frozen=True,
        populate_by_name=True,
        extra='ignore',
    )

    model_id: str = 'qwen3-4b-instruct'
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
        default='INFO',
        validation_alias='LOG_LEVEL',
    )
    llm_base_url: str = 'http://llama-server.llama-server'
    stt_base_url: str = 'http://nemo-asr.nemo-asr'
    tts_base_url: str = 'http://pockettts.pockettts'
    tts_model: str = 'kyutai/pocket-tts'
    system_prompt_path: Path = Path('/tmp/system-prompt')  # noqa: S108
    llm_slots: tuple[int, ...] = (0,)
    llm_max_tokens: int = Field(default=128, ge=1)
    llm_tool_tokens: int = Field(default=128, ge=1)
    llm_cache_warm_tokens: int = Field(default=0, ge=0, le=1)
    llm_cache_warm_fallback_tokens: int = Field(default=1, ge=1, le=2)
    llm_cache_warm_enabled: bool = True
    llm_cache_warm_min_interval_seconds: float = Field(default=0.5, ge=0)
    llm_cache_warm_min_new_characters: int = Field(default=1, ge=1)
    llm_temperature: float = Field(default=0.0, ge=0)
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    connect_timeout_seconds: float = Field(default=10.0, gt=0)
    health_timeout_seconds: float = Field(default=3.0, gt=0)
    maximum_websocket_message_bytes: int = Field(default=2 * 1024**2, ge=1)
    maximum_tool_iterations: int = Field(default=2, ge=1, le=5)
    maximum_tool_result_characters: int = Field(default=8_000, ge=128)
    default_timezone: str = 'Europe/Berlin'
    listen_port: int = Field(
        default=8080,
        ge=1,
        le=65_535,
        validation_alias='LISTEN_PORT',
    )
    mcp: dict[str, MCPConfig] = Field(default_factory=dict)

    @model_validator(mode='after')
    def require_unique_slots(self) -> Settings:
        if not self.llm_slots:
            message = 'llm_slots must contain at least one llama.cpp slot id'
            raise ValueError(message)
        if len(set(self.llm_slots)) != len(self.llm_slots):
            message = 'llm_slots must not contain duplicates'
            raise ValueError(message)
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        config_file = os.getenv('ASSISTANT_CONFIG_FILE')
        if config_file:
            path = Path(config_file)
            if path.suffix.casefold() in {'.yaml', '.yml'}:
                sources.append(YamlConfigSettingsSource(settings_cls, path))
            elif path.suffix.casefold() == '.json':
                sources.append(JsonConfigSettingsSource(settings_cls, path))
            else:
                message = 'ASSISTANT_CONFIG_FILE must end in .json, .yaml, or .yml'
                raise ValueError(message)
        sources.extend((dotenv_settings, file_secret_settings))
        return tuple(sources)
