from __future__ import annotations

import ast
import json
import logging
import re
import unittest
from pathlib import Path
from typing import TypeGuard

from service_logging import JsonFormatter, SuccessfulHealthCheckFilter


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
        record = logging.LogRecord(
            'uvicorn.access',
            logging.INFO,
            __file__,
            1,
            '%s - "%s %s HTTP/%s" %d',
            ('10.0.0.1:1234', 'POST', '/v1/audio/speech', '1.1', 200),
            None,
        )

        output = json.loads(JsonFormatter().format(record))

        self.assertEqual(output['message'], 'HTTP request')
        self.assertEqual(output['event_id'], 'ID_http_server_request')
        self.assertEqual(output['client_address'], '10.0.0.1:1234')
        self.assertEqual(output['method'], 'POST')
        self.assertEqual(output['path'], '/v1/audio/speech')
        self.assertEqual(output['http_version'], '1.1')
        self.assertEqual(output['status_code'], 200)
        self.assertNotIn('arguments', output)

    def test_httpx_request_data_uses_named_fields(self) -> None:
        record = logging.LogRecord(
            'httpx',
            logging.INFO,
            __file__,
            1,
            'HTTP Request: %s %s "%s %d %s"',
            ('POST', 'http://llm/completion', 'HTTP/1.1', 200, 'OK'),
            None,
        )

        output = json.loads(JsonFormatter().format(record))

        self.assertEqual(output['message'], 'HTTP request')
        self.assertEqual(output['event_id'], 'ID_http_client_request')
        self.assertEqual(output['method'], 'POST')
        self.assertEqual(output['url'], 'http://llm/completion')
        self.assertEqual(output['http_version'], 'HTTP/1.1')
        self.assertEqual(output['status_code'], 200)
        self.assertEqual(output['reason_phrase'], 'OK')
        self.assertNotIn('arguments', output)

    def test_uncatalogued_dependency_record_uses_fallback_event_id(self) -> None:
        record = logging.makeLogRecord(
            {
                'name': 'dependency',
                'levelno': logging.WARNING,
                'levelname': 'WARNING',
                'msg': 'Dependency warning',
                'args': (),
            },
        )

        output = json.loads(JsonFormatter().format(record))

        self.assertEqual(output['event_id'], 'ID_dependency_log')


class ProjectEventIdTest(unittest.TestCase):
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
