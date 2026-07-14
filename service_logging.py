from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Final
from urllib.parse import urlsplit

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

    def format(self, record: logging.LogRecord) -> str:  # noqa: C901
        message = record.getMessage()
        fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _LOG_RECORD_FIELDS and not key.startswith('_')
        }
        event_id = fields.pop('event_id', None)
        access_fields = self._uvicorn_access_fields(record)
        if access_fields is not None:
            message = 'HTTP request'
            fields.update(access_fields)
            event_id = HTTP_SERVER_REQUEST_EVENT_ID
        elif (httpx_fields := self._httpx_fields(record)) is not None:
            message = 'HTTP request'
            fields.update(httpx_fields)
            event_id = HTTP_CLIENT_REQUEST_EVENT_ID
        elif record.args:
            fields['arguments'] = record.args
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
        if record.exc_info is not None:
            payload['exception'] = self.formatException(record.exc_info)
        if record.stack_info:
            payload['stack'] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, separators=(',', ':'), default=str)

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
            'path': path,
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
            'url': url,
            'http_version': http_version,
            'status_code': status_code,
            'reason_phrase': reason_phrase,
        }


def configure_logging(level: str, application_logger: str) -> None:
    """Configure application, dependency, and Uvicorn records as JSON."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

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
