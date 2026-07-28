from __future__ import annotations

import ast
import json
import logging
import re
import sys
import unittest
from pathlib import Path
from typing import TypeGuard

from service_logging import (
    JsonFormatter,
    SuccessfulHealthCheckFilter,
    SuccessfulHttpClientRequestFilter,
    UvicornWebSocketNoiseFilter,
)


class JsonFormatterTest(unittest.TestCase):
    def test_extra_fields_remain_structured(self) -> None:
        record = logging.makeLogRecord(
            {
                'name': 'assistant.upstream',
                'levelno': logging.DEBUG,
                'levelname': 'DEBUG',
                'msg': 'LLM request',
                'args': (),
                'event_id': 'ID_assistant_llm_request',
                'payload': {
                    'prompt': 'hello',
                    'tools': [{'name': 'time', 'input_schema': {'type': 'object'}}],
                },
            },
        )

        formatted = JsonFormatter().format(record)
        output = json.loads(formatted)

        self.assertRegex(formatted, r'\bID_[a-z_]+\b')
        self.assertEqual(output['message'], 'LLM request')
        self.assertEqual(output['level'], 'DEBUG')
        self.assertEqual(output['event_id'], 'ID_assistant_llm_request')
        self.assertEqual(output['payload']['prompt'], 'hello')
        self.assertEqual(output['payload']['tools'][0]['name'], 'time')

    def test_uvicorn_access_data_uses_named_fields(self) -> None:
        content_sentinel = 'private-conversation-sentinel'
        record = logging.LogRecord(
            'uvicorn.access',
            logging.INFO,
            __file__,
            1,
            '%s - "%s %s HTTP/%s" %d',
            (
                '10.0.0.1:1234',
                'POST',
                f'/v1/audio/speech?input={content_sentinel}',
                '1.1',
                200,
            ),
            None,
        )

        formatted = JsonFormatter().format(record)
        output = json.loads(formatted)

        self.assertNotIn(content_sentinel, formatted)
        self.assertEqual(output['message'], 'HTTP request')
        self.assertEqual(output['event_id'], 'ID_http_server_request')
        self.assertEqual(output['client_address'], '10.0.0.1:1234')
        self.assertEqual(output['method'], 'POST')
        self.assertEqual(output['path'], '/v1/audio/speech')
        self.assertEqual(output['http_version'], '1.1')
        self.assertEqual(output['status_code'], 200)
        self.assertNotIn('arguments', output)

    def test_httpx_request_data_uses_named_fields(self) -> None:
        content_sentinel = 'private-conversation-sentinel'
        record = logging.LogRecord(
            'httpx',
            logging.INFO,
            __file__,
            1,
            'HTTP Request: %s %s "%s %d %s"',
            (
                'POST',
                (
                    'http://service:'
                    f'{content_sentinel}@llm/completion?input={content_sentinel}'
                    f'#{content_sentinel}'
                ),
                'HTTP/1.1',
                200,
                'OK',
            ),
            None,
        )

        formatted = JsonFormatter().format(record)
        output = json.loads(formatted)

        self.assertNotIn(content_sentinel, formatted)
        self.assertEqual(output['message'], 'HTTP request')
        self.assertEqual(output['event_id'], 'ID_http_client_request')
        self.assertEqual(output['method'], 'POST')
        self.assertEqual(output['url'], 'http://llm/completion')
        self.assertEqual(output['http_version'], 'HTTP/1.1')
        self.assertEqual(output['status_code'], 200)
        self.assertEqual(output['reason_phrase'], 'OK')
        self.assertNotIn('arguments', output)

    def test_uncatalogued_dependency_record_uses_fallback_event_id(self) -> None:
        content_sentinel = 'private-conversation-sentinel'
        record = logging.makeLogRecord(
            {
                'name': 'dependency',
                'levelno': logging.WARNING,
                'levelname': 'WARNING',
                'msg': 'Dependency warning: %s',
                'args': (content_sentinel,),
                'body': content_sentinel,
            },
        )

        formatted = JsonFormatter().format(record)
        output = json.loads(formatted)

        self.assertNotIn(content_sentinel, formatted)
        self.assertEqual(output['message'], 'Dependency log')
        self.assertEqual(output['event_id'], 'ID_dependency_log')

    def test_application_error_redacts_content_and_exception_message(self) -> None:
        content_sentinel = 'private conversation and tool arguments'
        try:
            raise RuntimeError(content_sentinel)  # noqa: TRY301
        except RuntimeError:
            record = logging.LogRecord(
                'assistant.pipeline',
                logging.ERROR,
                __file__,
                1,
                'Assistant request failed: %s',
                (content_sentinel,),
                sys.exc_info(),
            )
        record.event_id = 'ID_assistant_request_failed'
        record.operation = 'generation'
        record.transcript = content_sentinel
        record.arguments = {'query': content_sentinel}
        record.error = content_sentinel

        formatted = JsonFormatter().format(record)
        output = json.loads(formatted)

        self.assertNotIn(content_sentinel, formatted)
        self.assertEqual(output['message'], 'Assistant request failed: %s')
        self.assertEqual(output['operation'], 'generation')
        self.assertEqual(output['exception_type'], 'RuntimeError')
        self.assertNotIn('arguments', output)
        self.assertNotIn('error', output)
        self.assertNotIn('transcript', output)
        self.assertNotIn('exception', output)
        self.assertGreater(len(output['traceback']), 0)

    def test_application_debug_record_retains_explicit_content(self) -> None:
        content_sentinel = 'private conversation and tool arguments'
        try:
            raise RuntimeError(content_sentinel)  # noqa: TRY301
        except RuntimeError:
            record = logging.LogRecord(
                'assistant.pipeline',
                logging.DEBUG,
                __file__,
                1,
                'Assistant request: %s',
                (content_sentinel,),
                sys.exc_info(),
            )
        record.event_id = 'ID_assistant_request'
        record.transcript = content_sentinel

        formatted = JsonFormatter().format(record)
        output = json.loads(formatted)

        self.assertIn(content_sentinel, formatted)
        self.assertEqual(output['message'], f'Assistant request: {content_sentinel}')
        self.assertEqual(output['transcript'], content_sentinel)
        self.assertIn(content_sentinel, output['exception'])


class ProjectEventIdTest(unittest.TestCase):
    CONTENT_FIELDS = frozenset(
        {
            'answer',
            'arguments',
            'body',
            'content',
            'conversation',
            'decision',
            'delta',
            'error',
            'history',
            'input',
            'messages',
            'output',
            'payload',
            'prompt',
            'raw_response',
            'request_body',
            'response',
            'result',
            'speaker',
            'text',
            'transcript',
            'turns',
        },
    )

    def test_project_log_calls_declare_valid_event_ids(self) -> None:  # noqa: C901
        repository = Path(__file__).parents[1]
        source_roots = (
            repository / 'assistant' / 'src',
            repository / 'stt' / 'src',
            repository / 'tts' / 'src',
        )
        event_id_pattern = re.compile(r'ID_[a-z][a-z_]*')
        problems: list[str] = []

        for source_root in source_roots:
            for path in source_root.rglob('*.py'):
                tree = ast.parse(path.read_text(), filename=str(path))
                for node in ast.walk(tree):
                    if not self._is_log_call(node):
                        continue
                    event_id = self._event_id(node)
                    if event_id is None or event_id_pattern.fullmatch(event_id) is None:
                        relative_path = path.relative_to(repository)
                        problems.append(f'{relative_path}:{node.lineno}')

        self.assertEqual(problems, [], f'log calls without valid event IDs: {problems}')

    def test_non_debug_log_calls_use_literal_messages_without_content_fields(  # noqa: C901
        self,
    ) -> None:
        repository = Path(__file__).parents[1]
        source_roots = (
            repository / 'assistant' / 'src',
            repository / 'stt' / 'src',
            repository / 'tts' / 'src',
        )
        problems: list[str] = []

        for source_root in source_roots:
            for path in source_root.rglob('*.py'):
                tree = ast.parse(path.read_text(), filename=str(path))
                for node in ast.walk(tree):
                    if not self._is_log_call(node):
                        continue
                    function = node.func
                    if not isinstance(function, ast.Attribute) or function.attr == 'debug':
                        continue
                    relative_path = path.relative_to(repository)
                    location = f'{relative_path}:{node.lineno}'
                    if (
                        not node.args
                        or not isinstance(node.args[0], ast.Constant)
                        or not isinstance(node.args[0].value, str)
                    ):
                        problems.append(f'{location} has a dynamic message')
                    exposed = self.CONTENT_FIELDS.intersection(self._extra_keys(node))
                    if exposed:
                        problems.append(
                            f'{location} exposes content fields {sorted(exposed)}',
                        )

        self.assertEqual(problems, [], f'unsafe non-debug log calls: {problems}')

    @staticmethod
    def _is_log_call(node: ast.AST) -> TypeGuard[ast.Call]:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'LOGGER'
            and node.func.attr
            in {'debug', 'info', 'warning', 'error', 'exception', 'critical', 'log'}
        )

    @staticmethod
    def _event_id(node: ast.Call) -> str | None:
        extra_keyword = next((keyword for keyword in node.keywords if keyword.arg == 'extra'), None)
        if extra_keyword is None or not isinstance(extra_keyword.value, ast.Dict):
            return None
        for key, value in zip(extra_keyword.value.keys, extra_keyword.value.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == 'event_id'
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                return value.value
        return None

    @staticmethod
    def _extra_keys(node: ast.Call) -> set[str]:
        extra_keyword = next((keyword for keyword in node.keywords if keyword.arg == 'extra'), None)
        if extra_keyword is None or not isinstance(extra_keyword.value, ast.Dict):
            return set()
        return {
            key.value
            for key in extra_keyword.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }


class SuccessfulHealthCheckFilterTest(unittest.TestCase):
    @staticmethod
    def _httpx_record(method: str, url: str, status_code: int) -> logging.LogRecord:
        return logging.LogRecord(
            'httpx',
            logging.INFO,
            __file__,
            1,
            'HTTP Request: %s %s "%s %d %s"',
            (method, url, 'HTTP/1.1', status_code, 'status'),
            None,
        )

    def test_successful_upstream_health_requests_are_suppressed(self) -> None:
        health_filter = SuccessfulHealthCheckFilter()

        self.assertFalse(
            health_filter.filter(
                self._httpx_record(
                    'GET',
                    'http://llama-server.llama-server/health?probe=ready',
                    200,
                ),
            ),
        )
        self.assertFalse(
            health_filter.filter(
                self._httpx_record('GET', 'http://nemo-asr.nemo-asr/health/ready', 200),
            ),
        )
        self.assertFalse(
            health_filter.filter(
                self._httpx_record('GET', 'http://nemo-asr.nemo-asr/metrics', 200),
            ),
        )

    def test_failed_health_and_successful_application_requests_are_retained(self) -> None:
        health_filter = SuccessfulHealthCheckFilter()

        self.assertTrue(
            health_filter.filter(
                self._httpx_record('GET', 'http://llama-server.llama-server/health', 503),
            ),
        )
        self.assertTrue(
            health_filter.filter(
                self._httpx_record('POST', 'http://llama-server.llama-server/completion', 200),
            ),
        )


class SuccessfulHttpClientRequestFilterTest(unittest.TestCase):
    @staticmethod
    def _record(status_code: int) -> logging.LogRecord:
        return logging.LogRecord(
            'httpx',
            logging.INFO,
            __file__,
            1,
            'HTTP Request: %s %s "%s %d %s"',
            ('POST', 'http://llama-server/completion', 'HTTP/1.1', status_code, 'status'),
            None,
        )

    def test_success_is_suppressed_and_failure_is_retained(self) -> None:
        request_filter = SuccessfulHttpClientRequestFilter()

        self.assertFalse(request_filter.filter(self._record(200)))
        self.assertTrue(request_filter.filter(self._record(503)))


class UvicornWebSocketNoiseFilterTest(unittest.TestCase):
    @staticmethod
    def _record(message: str) -> logging.LogRecord:
        return logging.LogRecord(
            'uvicorn.error',
            logging.INFO,
            __file__,
            1,
            message,
            (),
            None,
        )

    def test_successful_protocol_lifecycle_is_suppressed(self) -> None:
        protocol_filter = UvicornWebSocketNoiseFilter()

        self.assertFalse(
            protocol_filter.filter(self._record('%s - "WebSocket %s" [accepted]')),
        )
        self.assertFalse(protocol_filter.filter(self._record('connection open')))
        self.assertFalse(protocol_filter.filter(self._record('connection closed')))
        self.assertTrue(protocol_filter.filter(self._record('Application startup complete.')))
