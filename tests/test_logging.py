from __future__ import annotations

import json
import logging
import unittest

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
                'payload': {
                    'prompt': 'hello',
                    'tools': [{'name': 'time', 'input_schema': {'type': 'object'}}],
                },
            },
        )

        output = json.loads(JsonFormatter().format(record))

        self.assertEqual(output['message'], 'LLM request')
        self.assertEqual(output['level'], 'DEBUG')
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
        self.assertEqual(output['method'], 'POST')
        self.assertEqual(output['url'], 'http://llm/completion')
        self.assertEqual(output['http_version'], 'HTTP/1.1')
        self.assertEqual(output['status_code'], 200)
        self.assertEqual(output['reason_phrase'], 'OK')
        self.assertNotIn('arguments', output)


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
