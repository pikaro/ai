from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from assistant.src.config import Settings
from assistant.src.configuration_api import apply_configuration_patch
from assistant.src.dashboard import router as dashboard_router
from assistant.src.dependencies import runtime_from_request
from assistant.src.domain import AssistantServiceError, UpstreamUnavailableError
from assistant.src.realtime import router as realtime_router
from assistant.src.runtime import AssistantRuntime
from assistant.src.schemas import HealthResponse
from runtime_config import ConfigurationUpdateResponse, reject_if_busy
from service_contracts.models import ModelDescription, ModelList
from service_logging import configure_logging

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

LOGGER = logging.getLogger('assistant')

SETTINGS = Settings()
configure_logging(SETTINGS.log_level, 'assistant')


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
    runtime = AssistantRuntime(SETTINGS)
    application.state.runtime = runtime
    try:
        await runtime.start()
        yield
    finally:
        await runtime.close()


app = FastAPI(title='Assistant', version='1.0.0', lifespan=lifespan)


@app.exception_handler(AssistantServiceError)
async def _assistant_service_error(
    _request: Request,
    error: AssistantServiceError,
) -> JSONResponse:
    if isinstance(error, UpstreamUnavailableError):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                'detail': {
                    'status': 'unavailable',
                    'upstream': error.upstream,
                    'unhealthy': error.unhealthy,
                },
            },
        )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={'detail': 'assistant service failed'},
    )


app.include_router(dashboard_router)
app.include_router(realtime_router)


@app.get('/health/live')
async def live() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/metrics', include_in_schema=False)
async def prometheus_metrics() -> Response:
    return Response(generate_latest(), headers={'Content-Type': CONTENT_TYPE_LATEST})


@app.get('/health', response_model=HealthResponse)
@app.get('/health/ready', response_model=HealthResponse)
async def ready(request: Request) -> HealthResponse:
    runtime_status = await runtime_from_request(request).status()
    return HealthResponse(status='ok', **asdict(runtime_status))


@app.get('/v1/models', response_model=ModelList)
async def models(request: Request) -> ModelList:
    runtime = runtime_from_request(request)
    return ModelList(data=[ModelDescription(id=runtime.settings.model_id)])


@app.get('/config', response_model=Settings, response_model_by_alias=False)
async def configuration(request: Request) -> Settings:
    return runtime_from_request(request).settings


@app.patch('/config', response_model=ConfigurationUpdateResponse)
async def update_configuration(
    request: Request,
    patch: dict[str, object],
) -> ConfigurationUpdateResponse:
    return await apply_configuration_patch(request, patch)


@app.get('/system-prompt', response_class=PlainTextResponse)
async def system_prompt(request: Request) -> str:
    return runtime_from_request(request).system_prompt.read()


@app.put('/system-prompt', response_class=PlainTextResponse)
async def update_system_prompt(request: Request) -> str:
    try:
        prompt = (await request.body()).decode('utf-8')
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='system prompt must be UTF-8 text',
        ) from error

    runtime = runtime_from_request(request)
    reject_if_busy(runtime.operations, 'assistant')
    try:
        try:
            updated_prompt = runtime.system_prompt.write(prompt)
        except (OSError, UnicodeError) as error:
            LOGGER.exception(
                'System prompt update failed',
                extra={
                    'event_id': 'ID_assistant_system_prompt_update_failed',
                    'path': str(runtime.settings.system_prompt_path),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='system prompt update failed',
            ) from error
        LOGGER.info(
            'System prompt updated',
            extra={
                'event_id': 'ID_assistant_system_prompt_updated',
                'path': str(runtime.settings.system_prompt_path),
                'characters': len(updated_prompt),
            },
        )
        return updated_prompt
    finally:
        runtime.operations.release()
