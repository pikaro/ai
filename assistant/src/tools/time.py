from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from assistant.src.tooling import ToolContext, ToolDefinition


def _now(timezone_name: str, context: ToolContext) -> datetime:
    selected_timezone = timezone_name or context.default_timezone
    if selected_timezone == 'local':
        return datetime.now().astimezone()
    try:
        return datetime.now(ZoneInfo(selected_timezone))
    except ZoneInfoNotFoundError as error:
        message = f'unknown IANA timezone: {selected_timezone}'
        raise ValueError(message) from error


def rough_time(value: datetime) -> str:
    rounded_minutes = ((value.minute + 2) // 5) * 5
    hour = value.hour
    if rounded_minutes == 60:  # noqa: PLR2004
        rounded_minutes = 0
        hour = (hour + 1) % 24

    hour_names = (
        'twelve',
        'one',
        'two',
        'three',
        'four',
        'five',
        'six',
        'seven',
        'eight',
        'nine',
        'ten',
        'eleven',
    )
    phrases = {
        0: f"{hour_names[hour % 12]} o'clock",
        5: f'five past {hour_names[hour % 12]}',
        10: f'ten past {hour_names[hour % 12]}',
        15: f'quarter past {hour_names[hour % 12]}',
        20: f'twenty past {hour_names[hour % 12]}',
        25: f'twenty-five past {hour_names[hour % 12]}',
        30: f'half past {hour_names[hour % 12]}',
        35: f'twenty-five to {hour_names[(hour + 1) % 12]}',
        40: f'twenty to {hour_names[(hour + 1) % 12]}',
        45: f'quarter to {hour_names[(hour + 1) % 12]}',
        50: f'ten to {hour_names[(hour + 1) % 12]}',
        55: f'five to {hour_names[(hour + 1) % 12]}',
    }
    return phrases[rounded_minutes]


def get_time(arguments: dict[str, Any], context: ToolContext) -> str:
    mode = str(arguments.get('mode', 'rough'))
    current = _now(str(arguments.get('timezone', '')), context)
    if mode == 'rough':
        return rough_time(current)
    if mode == 'exact':
        return current.isoformat(timespec='seconds')
    if mode == 'date':
        return current.strftime('%A, %d %B %Y')
    message = f'unsupported time mode: {mode}'
    raise ValueError(message)


TOOL = ToolDefinition(
    name='time',
    description=(
        'Return the current time or date. For time, use rough time unless the user explicitly '
        'asks for the exact or precise time.'
    ),
    input_schema={
        'type': 'object',
        'properties': {
            'mode': {
                'type': 'string',
                'enum': ['rough', 'exact', 'date'],
                'default': 'rough',
            },
            'timezone': {
                'type': 'string',
                'description': 'Optional IANA timezone such as Europe/Berlin.',
                'defaut': 'Europe/Berlin',
            },
        },
        'additionalProperties': False,
    },
    triggers=frozenset({'time', 'date'}),
    executor=get_time,
)
