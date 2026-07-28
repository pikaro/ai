from __future__ import annotations

import json
import logging
import re
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from urllib.parse import SplitResult, urlsplit, urlunsplit

HEALTH_ENDPOINTS: Final = frozenset({'/health', '/health/live', '/health/ready'})
DEFAULT_EVENT_ID: Final = 'ID_dependency_log'
HTTP_SERVER_REQUEST_EVENT_ID: Final = 'ID_http_server_request'
HTTP_CLIENT_REQUEST_EVENT_ID: Final = 'ID_http_client_request'
EVENT_ID_PATTERN: Final = re.compile(r'ID_[a-z][a-z_]*')
_LOG_RECORD_FIELDS: Final = frozenset(logging.makeLogRecord({}).__dict__) | {
    'asctime',
    'color_message',
    'message',
}
_APPLICATION_LOGGERS: Final = ('assistant', 'stt', 'tts')
_SAFE_STRING_FIELDS: Final = frozenset(
    {
        'arrival_estimate_source',
        'boundary',
        'client_address',
        'endpoint',
        'error_type',
        'http_version',
        'knowledge_directory',
        'llm_id',
        'method',
        'model',
        'next_boundary',
        'operation',
        'original_decision_reason',
        'outcome',
        'path',
        'reason',
        'reason_phrase',
        'request_id',
        'request_timestamp',
        'response_format',
        'server',
        'service',
        'source',
        'stage',
        'tool',
        'transport',
        'url',
        'voice',
    },
)
_SAFE_SEQUENCE_FIELDS: Final = frozenset(
    {
        'changed_fields',
        'sources',
        'tools',
        'unhealthy_services',
        'voices',
    },
)
_SAFE_MAPPING_FIELDS: Final = frozenset({'upstream'})
_TOKEN_PATTERN: Final = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,255}')
_CORRELATION_PATTERN: Final = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}')
_REASON_PATTERN: Final = re.compile(r'[a-z][a-z0-9_]{0,127}')
_OMIT: Final = object()


class SuccessfulHealthCheckFilter(logging.Filter):
    """Suppress successful health checks while retaining failures and other requests."""

    def filter(self, record: logging.LogRecord) -> bool:
        arguments = record.args
        if not isinstance(arguments, tuple) or len(arguments) < 5:  # noqa: PLR2004
            return True
        if record.name == 'uvicorn.access':
            method, target, status_code = arguments[1], arguments[2], arguments[4]
        elif record.name == 'httpx':
            method, target, status_code = arguments[0], arguments[1], arguments[3]
        else:
            return True
        request_path = urlsplit(str(target)).path
        return not (
            method == 'GET' and request_path in HEALTH_ENDPOINTS and status_code == 200  # noqa: PLR2004
        )


class JsonFormatter(logging.Formatter):
    """Render logging records with a regex-friendly event ID as one JSON object."""

    def __init__(self, application_logger: str | None = None) -> None:
        super().__init__()
        self._application_loggers = (
            _APPLICATION_LOGGERS if application_logger is None else (application_logger,)
        )

    def format(self, record: logging.LogRecord) -> str:  # noqa: C901
        raw_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _LOG_RECORD_FIELDS and not key.startswith('_')
        }
        event_id = raw_fields.pop('event_id', None)
        verbose = record.levelno <= logging.DEBUG
        application_record = self._is_application_record(record)
        access_fields = self._uvicorn_access_fields(record)
        if access_fields is not None:
            message = 'HTTP request'
            fields = access_fields
            event_id = HTTP_SERVER_REQUEST_EVENT_ID
        elif (httpx_fields := self._httpx_fields(record)) is not None:
            message = 'HTTP request'
            fields = httpx_fields
            event_id = HTTP_CLIENT_REQUEST_EVENT_ID
        elif verbose:
            message = record.getMessage()
            fields = raw_fields
            if record.args:
                fields['arguments'] = record.args
        elif application_record:
            message = record.msg if isinstance(record.msg, str) else 'Application log'
            fields = self._safe_non_debug_fields(raw_fields)
        else:
            message = 'Dependency log'
            fields = {}
            event_id = None
        if not isinstance(event_id, str) or EVENT_ID_PATTERN.fullmatch(event_id) is None:
            event_id = DEFAULT_EVENT_ID

        payload: dict[str, object] = {
            'timestamp': datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'event_id': event_id,
            'message': message,
            **fields,
        }
        payload.update(self._exception_fields(record, verbose=verbose))
        if verbose and record.stack_info:
            payload['stack'] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, separators=(',', ':'), default=str)

    def _exception_fields(
        self,
        record: logging.LogRecord,
        *,
        verbose: bool,
    ) -> dict[str, object]:
        if record.exc_info is None:
            return {}
        if verbose:
            return {'exception': self.formatException(record.exc_info)}
        exception_type, _exception, exception_traceback = record.exc_info
        fields: dict[str, object] = {}
        if exception_type is not None:
            fields['exception_type'] = exception_type.__name__
        if exception_traceback is not None:
            fields['traceback'] = [
                {
                    'file': frame.filename,
                    'line': frame.lineno,
                    'function': frame.name,
                }
                for frame in traceback.extract_tb(exception_traceback)
            ]
        return fields

    def _is_application_record(self, record: logging.LogRecord) -> bool:
        return any(
            record.name == prefix or record.name.startswith(f'{prefix}.')
            for prefix in self._application_loggers
        )

    @staticmethod
    def _safe_non_debug_fields(fields: dict[str, object]) -> dict[str, object]:
        safe: dict[str, object] = {}
        for key, value in fields.items():
            sanitized = _safe_non_debug_field(key, value)
            if sanitized is not _OMIT:
                safe[key] = sanitized
        return safe

    @staticmethod
    def _uvicorn_access_fields(record: logging.LogRecord) -> dict[str, object] | None:
        arguments = record.args
        if record.name != 'uvicorn.access' or not isinstance(arguments, tuple):
            return None
        if len(arguments) < 5:  # noqa: PLR2004
            return None
        client_address, method, path, http_version, status_code = arguments[:5]
        return {
            'client_address': client_address,
            'method': method,
            'path': urlsplit(str(path)).path,
            'http_version': http_version,
            'status_code': status_code,
        }

    @staticmethod
    def _httpx_fields(record: logging.LogRecord) -> dict[str, object] | None:
        arguments = record.args
        if record.name != 'httpx' or not isinstance(arguments, tuple):
            return None
        if len(arguments) < 5:  # noqa: PLR2004
            return None
        method, url, http_version, status_code, reason_phrase = arguments[:5]
        return {
            'method': method,
            'url': _safe_url(str(url)),
            'http_version': http_version,
            'status_code': status_code,
            'reason_phrase': reason_phrase,
        }


def _safe_non_debug_field(key: str, value: object) -> object:
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if key in _SAFE_STRING_FIELDS and isinstance(value, (str, Path)):
        return _safe_string_field(key, str(value))
    if key in _SAFE_SEQUENCE_FIELDS and isinstance(value, (list, tuple, set, frozenset)):
        items = [str(item) for item in value]
        return items if all(_TOKEN_PATTERN.fullmatch(item) for item in items) else _OMIT
    if key in _SAFE_MAPPING_FIELDS and isinstance(value, dict):
        valid = all(
            isinstance(map_key, str)
            and _TOKEN_PATTERN.fullmatch(map_key)
            and isinstance(map_value, bool)
            for map_key, map_value in value.items()
        )
        return value if valid else _OMIT
    return _OMIT


def _safe_string_field(key: str, value: str) -> str | object:
    if key in {'endpoint', 'url'}:
        return _safe_url(value)
    if key == 'reason':
        return value if _REASON_PATTERN.fullmatch(value) else _OMIT
    if key in {'request_id', 'request_timestamp'}:
        return value if _CORRELATION_PATTERN.fullmatch(value) else _OMIT
    return value


def _safe_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        if hostname is None:
            return parsed.path
        host = f'[{hostname}]' if ':' in hostname else hostname
        port = f':{parsed.port}' if parsed.port is not None else ''
        sanitized = SplitResult(parsed.scheme, f'{host}{port}', parsed.path, '', '')
        return urlunsplit(sanitized)
    except ValueError:
        return value.split('?', maxsplit=1)[0].split('#', maxsplit=1)[0]


def configure_logging(level: str, application_logger: str) -> None:
    """Configure application, dependency, and Uvicorn records as JSON."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(application_logger))

    # DEBUG exposes application payloads without enabling verbose transport logs that may
    # duplicate content or headers. Higher configured thresholds still apply globally.
    infrastructure_level = logging.INFO if level == 'DEBUG' else level
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(infrastructure_level)

    for logger_name in ('uvicorn', 'uvicorn.error', 'uvicorn.access'):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.setLevel(infrastructure_level)
        logger.propagate = True
        logger.disabled = False

    logger = logging.getLogger(application_logger)
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = True
    logger.disabled = False

    for logger_name in ('uvicorn.access', 'httpx'):
        logger = logging.getLogger(logger_name)
        if not any(isinstance(item, SuccessfulHealthCheckFilter) for item in logger.filters):
            logger.addFilter(SuccessfulHealthCheckFilter())
