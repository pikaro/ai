from __future__ import annotations

import importlib
import inspect
import json
import logging
import pkgutil
import re
import time
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import PaginatedRequestParams

from assistant.src import metrics

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from assistant.src.config import MCPConfig

LOGGER = logging.getLogger('assistant.tools')
ToolExecutor = Callable[[dict[str, Any], 'ToolContext'], Awaitable[object] | object]


@dataclass(frozen=True, slots=True)
class ToolContext:
    default_timezone: str


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    triggers: frozenset[str]
    executor: ToolExecutor = field(repr=False, compare=False)
    source: str = 'local'

    def prompt_description(self) -> dict[str, object]:
        return {
            'name': self.name,
            'description': self.description,
            'input_schema': self.input_schema,
        }


def _contains_trigger(text: str, triggers: frozenset[str]) -> bool:
    folded = text.casefold()
    return any(
        re.search(rf'(?<!\w){re.escape(trigger.casefold())}(?!\w)', folded) is not None
        for trigger in triggers
        if trigger.strip()
    )


def discover_local_tools(  # noqa: C901
    package_name: str = 'assistant.src.tools',
) -> list[ToolDefinition]:
    package = importlib.import_module(package_name)
    discovered: list[ToolDefinition] = []
    for module_info in pkgutil.iter_modules(package.__path__, f'{package_name}.'):
        module = importlib.import_module(module_info.name)
        definitions = getattr(module, 'TOOLS', None)
        if definitions is None:
            definition = getattr(module, 'TOOL', None)
            definitions = [] if definition is None else [definition]
        for definition in definitions:
            if not isinstance(definition, ToolDefinition):
                message = f'{module_info.name} exports an invalid tool definition'
                raise TypeError(message)
            if not definition.triggers:
                message = f'local tool {definition.name!r} must declare at least one trigger'
                raise ValueError(message)
            discovered.append(definition)
    return discovered


class ToolRegistry:
    def __init__(
        self,
        local_tools: list[ToolDefinition],
        mcp: dict[str, MCPConfig],
        *,
        default_timezone: str,
        maximum_result_characters: int,
    ) -> None:
        self.context = ToolContext(default_timezone=default_timezone)
        self.mcp = mcp
        self.maximum_result_characters = maximum_result_characters
        self._tools = self._unique_tools(local_tools)
        self._mcp_tools: dict[str, list[ToolDefinition]] = {}
        self._mcp_retry_after: dict[str, float] = {}

    @property
    def local_tool_count(self) -> int:
        return len(self._tools)

    @staticmethod
    def _unique_tools(tools: list[ToolDefinition]) -> dict[str, ToolDefinition]:
        result: dict[str, ToolDefinition] = {}
        for tool in tools:
            if tool.name in result:
                message = f'duplicate assistant tool name: {tool.name}'
                raise ValueError(message)
            result[tool.name] = tool
        return result

    async def select(self, prompt: str) -> list[ToolDefinition]:  # noqa: C901
        selected = [
            tool for tool in self._tools.values() if _contains_trigger(prompt, tool.triggers)
        ]
        for server_name, config in self.mcp.items():
            if not config.enabled or not self._server_triggered(prompt, config):
                continue
            tools = self._mcp_tools.get(server_name)
            if tools is None:
                if time.monotonic() < self._mcp_retry_after.get(server_name, 0):
                    continue
                try:
                    tools = await self._discover_mcp_tools(server_name, config)
                except Exception:
                    LOGGER.exception('MCP tool discovery failed', extra={'server': server_name})
                    self._mcp_retry_after[server_name] = time.monotonic() + config.retry_seconds
                    continue
                self._mcp_tools[server_name] = tools
                _ = self._mcp_retry_after.pop(server_name, None)
            server_selected = _contains_trigger(prompt, config.triggers)
            selected.extend(
                tool
                for tool in tools
                if server_selected or _contains_trigger(prompt, tool.triggers)
            )
        for tool in selected:
            metrics.TOOLS_SELECTED.labels(source=tool.source).inc()
        if selected:
            LOGGER.info(
                'Tools selected',
                extra={
                    'tool_count': len(selected),
                    'tools': [tool.name for tool in selected],
                    'sources': [tool.source for tool in selected],
                },
            )
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    'Tool selection',
                    extra={
                        'transcript': prompt,
                        'tools': [
                            {'source': tool.source, **tool.prompt_description()}
                            for tool in selected
                        ],
                    },
                )
        return selected

    @staticmethod
    def _server_triggered(prompt: str, config: MCPConfig) -> bool:
        if _contains_trigger(prompt, config.triggers):
            return True
        return any(
            _contains_trigger(prompt, triggers) for triggers in config.tool_triggers.values()
        )

    async def call(self, tool: ToolDefinition, arguments: dict[str, Any]) -> str:
        started = time.perf_counter()
        outcome = 'success'
        LOGGER.info('Tool call started', extra={'tool': tool.name, 'source': tool.source})
        LOGGER.debug(
            'Tool call request',
            extra={'tool': tool.name, 'source': tool.source, 'arguments': arguments},
        )
        try:
            result = tool.executor(arguments, self.context)
            if inspect.isawaitable(result):
                result = await result
            text = self._result_text(result)[: self.maximum_result_characters]
        except Exception:
            outcome = 'error'
            raise
        else:
            LOGGER.debug(
                'Tool call response',
                extra={'tool': tool.name, 'source': tool.source, 'result': text},
            )
            return text
        finally:
            duration_seconds = time.perf_counter() - started
            metrics.TOOL_CALLS.labels(
                source=tool.source,
                tool=tool.name,
                outcome=outcome,
            ).inc()
            metrics.TOOL_CALL_SECONDS.labels(source=tool.source, tool=tool.name).observe(
                duration_seconds,
            )
            LOGGER.info(
                'Tool call completed',
                extra={
                    'tool': tool.name,
                    'source': tool.source,
                    'outcome': outcome,
                    'duration_seconds': duration_seconds,
                },
            )

    @staticmethod
    def _result_text(result: object) -> str:
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, separators=(',', ':'), default=str)

    async def _discover_mcp_tools(  # noqa: C901
        self,
        server_name: str,
        config: MCPConfig,
    ) -> list[ToolDefinition]:
        started = time.perf_counter()
        outcome = 'success'
        tool_count = 0
        LOGGER.info(
            'MCP request started',
            extra={
                'server': server_name,
                'operation': 'list_tools',
                'endpoint': config.endpoint,
            },
        )
        try:
            remote_tools: list[Any] = []
            async with self._mcp_session(config) as session:
                cursor: str | None = None
                while True:
                    response = await session.list_tools(
                        params=PaginatedRequestParams(cursor=cursor),
                    )
                    remote_tools.extend(response.tools)
                    cursor = response.nextCursor
                    if cursor is None:
                        break
            definitions = []
            for remote_tool in remote_tools:
                public_name = self._mcp_tool_name(server_name, remote_tool.name)
                triggers = config.tool_triggers.get(remote_tool.name, frozenset())

                async def execute(
                    arguments: dict[str, Any],
                    _context: ToolContext,
                    *,
                    tool_name: str = remote_tool.name,
                ) -> object:
                    return await self._call_mcp(server_name, config, tool_name, arguments)

                definitions.append(
                    ToolDefinition(
                        name=public_name,
                        description=remote_tool.description or remote_tool.name,
                        input_schema=remote_tool.inputSchema,
                        triggers=triggers,
                        executor=execute,
                        source=f'mcp:{server_name}',
                    ),
                )
            tool_count = len(definitions)
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    'MCP tool discovery response',
                    extra={
                        'server': server_name,
                        'endpoint': config.endpoint,
                        'tools': [
                            {'source': tool.source, **tool.prompt_description()}
                            for tool in definitions
                        ],
                    },
                )
            return definitions  # noqa: TRY300
        except Exception:
            outcome = 'error'
            raise
        finally:
            duration_seconds = time.perf_counter() - started
            metrics.MCP_REQUESTS.labels(
                server=server_name,
                operation='list_tools',
                outcome=outcome,
            ).inc()
            metrics.MCP_REQUEST_SECONDS.labels(
                server=server_name,
                operation='list_tools',
            ).observe(duration_seconds)
            LOGGER.info(
                'MCP request completed',
                extra={
                    'server': server_name,
                    'operation': 'list_tools',
                    'outcome': outcome,
                    'duration_seconds': duration_seconds,
                    'tool_count': tool_count,
                },
            )

    async def _call_mcp(
        self,
        server_name: str,
        config: MCPConfig,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> object:
        started = time.perf_counter()
        outcome = 'success'
        LOGGER.info(
            'MCP request started',
            extra={'server': server_name, 'operation': 'call_tool', 'tool': tool_name},
        )
        LOGGER.debug(
            'MCP tool request',
            extra={
                'server': server_name,
                'endpoint': config.endpoint,
                'tool': tool_name,
                'arguments': arguments,
            },
        )
        try:
            async with self._mcp_session(config) as session:
                result = await session.call_tool(tool_name, arguments)
            if result.isError:
                self._raise_mcp_error(tool_name)
            if result.structuredContent is not None:
                response: object = result.structuredContent
            else:
                content = [item.model_dump(mode='json', by_alias=True) for item in result.content]
                text_items = [item.get('text') for item in content if item.get('type') == 'text']
                response = '\n'.join(str(item) for item in text_items) if text_items else content
        except Exception:
            outcome = 'error'
            raise
        else:
            LOGGER.debug(
                'MCP tool response',
                extra={'server': server_name, 'tool': tool_name, 'result': response},
            )
            return response
        finally:
            duration_seconds = time.perf_counter() - started
            metrics.MCP_REQUESTS.labels(
                server=server_name,
                operation='call_tool',
                outcome=outcome,
            ).inc()
            metrics.MCP_REQUEST_SECONDS.labels(
                server=server_name,
                operation='call_tool',
            ).observe(duration_seconds)
            LOGGER.info(
                'MCP request completed',
                extra={
                    'server': server_name,
                    'operation': 'call_tool',
                    'tool': tool_name,
                    'outcome': outcome,
                    'duration_seconds': duration_seconds,
                },
            )

    @staticmethod
    def _raise_mcp_error(tool_name: str) -> None:
        message = f'MCP tool {tool_name!r} returned an error'
        raise RuntimeError(message)

    @staticmethod
    def _mcp_tool_name(server_name: str, tool_name: str) -> str:
        raw_name = f'{server_name}__{tool_name}'
        return re.sub(r'[^A-Za-z0-9_-]', '_', raw_name)

    @staticmethod
    @asynccontextmanager
    async def _mcp_session(config: MCPConfig) -> AsyncGenerator[ClientSession]:
        timeout = timedelta(seconds=config.timeout_seconds)
        async with AsyncExitStack() as stack:
            if config.transport == 'streamable-http':
                client = await stack.enter_async_context(
                    httpx.AsyncClient(
                        headers=config.headers,
                        timeout=config.timeout_seconds,
                        follow_redirects=True,
                    ),
                )
                read, write, _ = await stack.enter_async_context(
                    streamable_http_client(config.endpoint, http_client=client),
                )
            else:
                read, write = await stack.enter_async_context(
                    sse_client(
                        config.endpoint,
                        headers=config.headers,
                        timeout=config.timeout_seconds,
                        sse_read_timeout=config.timeout_seconds,
                    ),
                )
            session = await stack.enter_async_context(
                ClientSession(read, write, read_timeout_seconds=timeout),
            )
            _ = await session.initialize()
            yield session
