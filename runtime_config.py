from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Literal

from fastapi import HTTPException, status
from pydantic import BaseModel, ValidationError
from pydantic_settings import BaseSettings

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping


class ConfigurationUpdateResponse(BaseModel):
    status: Literal['updated'] = 'updated'
    changed: tuple[str, ...]
    ephemeral: Literal[True] = True


class ExclusiveOperationGate:
    """Fail-fast single-operation gate shared by requests and configuration updates."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = False

    def try_acquire(self) -> bool:
        with self._lock:
            if self._active:
                return False
            self._active = True
            return True

    def release(self) -> None:
        with self._lock:
            if not self._active:
                message = 'exclusive operation gate is not acquired'
                raise RuntimeError(message)
            self._active = False

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active


def validated_settings_patch[SettingsT: BaseSettings](
    current: SettingsT,
    patch: Mapping[str, object],
    *,
    restart_required: Collection[str],
) -> tuple[SettingsT, tuple[str, ...]]:
    """Validate a top-level in-memory settings patch and identify effective changes."""
    settings_type = type(current)
    unknown = sorted(set(patch) - set(settings_type.model_fields))
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={'message': 'unknown configuration fields', 'fields': unknown},
        )

    values = current.model_dump(round_trip=True)
    values.update(patch)
    try:
        candidate = settings_type.model_validate(values)
    except ValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=error.errors(include_url=False),
        ) from error

    changed = tuple(
        name
        for name in settings_type.model_fields
        if getattr(candidate, name) != getattr(current, name)
    )
    blocked = sorted(set(changed) & set(restart_required))
    if blocked:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                'message': 'configuration fields require a service restart',
                'fields': blocked,
            },
        )
    return candidate, changed


def reject_if_busy(gate: ExclusiveOperationGate, service: str) -> None:
    """Acquire a service gate or fail without queueing behind active inference."""
    if gate.try_acquire():
        return
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=f'{service} is busy; configuration and inference requests are not queued',
        headers={'Retry-After': '1'},
    )
