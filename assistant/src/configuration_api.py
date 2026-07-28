from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from assistant.src import metrics
from assistant.src.dependencies import runtime_from_request
from assistant.src.runtime import AssistantRuntime
from runtime_config import (
    ConfigurationUpdateResponse,
    reject_if_busy,
    validated_settings_patch,
)
from service_logging import configure_logging

if TYPE_CHECKING:
    from fastapi import Request

LOGGER = logging.getLogger('assistant')
RESTART_REQUIRED_SETTINGS: Final = frozenset({'listen_port'})


async def apply_configuration_patch(
    request: Request,
    patch: dict[str, object],
) -> ConfigurationUpdateResponse:
    runtime = runtime_from_request(request)
    reject_if_busy(runtime.operations, 'assistant')
    changed: tuple[str, ...] = ()
    try:
        settings, changed = validated_settings_patch(
            runtime.settings,
            patch,
            restart_required=RESTART_REQUIRED_SETTINGS,
        )
        if changed:
            replacement = AssistantRuntime(settings, runtime.operations)
            configure_logging(settings.log_level, 'assistant')
            request.app.state.runtime = replacement
            metrics.CONFIGURATION_UPDATES.inc()
            LOGGER.info(
                'Assistant configuration updated',
                extra={
                    'event_id': 'ID_assistant_configuration_updated',
                    'changed_fields': changed,
                    'ephemeral': True,
                },
            )
            await runtime.close()
        return ConfigurationUpdateResponse(changed=changed)
    finally:
        runtime.operations.release()
