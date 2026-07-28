from typing import cast

from fastapi import Request, WebSocket

from assistant.src.runtime import AssistantRuntime


def runtime_from_request(request: Request) -> AssistantRuntime:
    return cast('AssistantRuntime', request.app.state.runtime)


def runtime_from_websocket(websocket: WebSocket) -> AssistantRuntime:
    return cast('AssistantRuntime', websocket.app.state.runtime)
