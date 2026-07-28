from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit, urlunsplit

import httpx

from assistant.src.domain import RuntimeStatus, UpstreamUnavailableError
from assistant.src.pipeline import (
    AssistantUtterance,
    SystemPromptFile,
    build_prompt_prefix,
    warm_llm_cache,
)
from assistant.src.tooling import ToolRegistry, discover_local_tools
from assistant.src.upstream import LlmClient, SlotPool, TtsClient, upstream_health
from runtime_config import ExclusiveOperationGate

if TYPE_CHECKING:
    from assistant.src.config import Settings

LOGGER = logging.getLogger('assistant')
UPSTREAM_KEEPALIVE_EXPIRY_SECONDS: Final = 4.0


class AssistantRuntime:
    """Compose assistant application services independently of API transports."""

    def __init__(
        self,
        settings: Settings,
        operations: ExclusiveOperationGate | None = None,
    ) -> None:
        self.settings = settings
        self.operations = operations or ExclusiveOperationGate()
        timeout = httpx.Timeout(
            settings.request_timeout_seconds,
            connect=settings.connect_timeout_seconds,
        )
        # Every deployed upstream closes idle HTTP connections after five seconds.
        # Retire pooled connections first to avoid racing a peer-initiated close.
        limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=UPSTREAM_KEEPALIVE_EXPIRY_SECONDS,
        )
        self.http = httpx.AsyncClient(timeout=timeout, limits=limits)
        self.slots = SlotPool(settings.llm_slots)
        self.llm = LlmClient(self.http, settings)
        self.tts = TtsClient(self.http, settings)
        self.tools = ToolRegistry(
            discover_local_tools(),
            settings.mcp,
            default_timezone=settings.default_timezone,
            maximum_result_characters=settings.maximum_tool_result_characters,
        )
        self.system_prompt = SystemPromptFile(settings.system_prompt_path)

    async def start(self) -> None:
        """Warm every configured LLM slot before the application becomes ready."""
        available_tools = await self.tools.available()
        prompt = build_prompt_prefix(
            '',
            available_tools,
            system_prompt=self.system_prompt.read(),
        )
        for slot in self.settings.llm_slots:
            await warm_llm_cache(self.llm, prompt, slot, reason='startup')

    async def close(self) -> None:
        await self.http.aclose()

    async def status(self) -> RuntimeStatus:
        upstream = await upstream_health(self.http, self.settings)
        unhealthy = [name for name, healthy in upstream.items() if not healthy]
        if unhealthy:
            LOGGER.warning(
                'Upstream services unhealthy',
                extra={
                    'event_id': 'ID_assistant_upstreams_unhealthy',
                    'unhealthy_services': unhealthy,
                    'upstream': upstream,
                },
            )
            raise UpstreamUnavailableError(upstream, unhealthy)
        return RuntimeStatus(
            model=self.settings.model_id,
            upstream=upstream,
            llm_slots=len(self.settings.llm_slots),
            local_tools=self.tools.local_tool_count,
            enabled_mcp_servers=sum(config.enabled for config in self.settings.mcp.values()),
        )

    def utterance(self) -> AssistantUtterance:
        return AssistantUtterance(
            self.settings,
            self.slots,
            self.llm,
            self.tts,
            self.tools,
            self.system_prompt,
        )

    @property
    def stt_websocket_url(self) -> str:
        parsed = urlsplit(self.settings.stt_base_url.rstrip('/'))
        scheme = 'wss' if parsed.scheme == 'https' else 'ws'
        return urlunsplit((scheme, parsed.netloc, '/v1/realtime', '', ''))
