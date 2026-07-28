from __future__ import annotations

from dataclasses import dataclass


class AssistantServiceError(Exception):
    """Base class for expected assistant service failures."""


class UpstreamUnavailableError(AssistantServiceError):
    def __init__(self, upstream: dict[str, bool], unhealthy: list[str]) -> None:
        super().__init__('upstream services are unavailable')
        self.upstream = upstream
        self.unhealthy = unhealthy


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    model: str
    upstream: dict[str, bool]
    llm_slots: int
    local_tools: int
    enabled_mcp_servers: int
