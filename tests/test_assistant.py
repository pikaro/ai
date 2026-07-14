from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import struct
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from assistant.src.config import Settings
from assistant.src.main import (
    AssistantRuntime,
    RealtimeEvent,
    prometheus_metrics,
)
from assistant.src.pipeline import (
    AssistantUtterance,
    build_prompt,
    completed_sentences,
    parse_tool_response,
)
from assistant.src.tooling import ToolDefinition, ToolRegistry, discover_local_tools
from assistant.src.tools.time import rough_time
from assistant.src.upstream import AudioFormat, LlmClient, SlotPool, upstream_health
from service_logging import SuccessfulHealthCheckFilter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class SettingsTest(unittest.TestCase):
    def test_environment_supports_nested_mcp_servers(self) -> None:
        environment = {
            'LISTEN_PORT': '9003',
            'ASSISTANT_LLM_SLOTS': '[1,2]',
            'ASSISTANT_MCP__CALENDAR__HOST': 'calendar-mcp',
            'ASSISTANT_MCP__CALENDAR__TRIGGERS': '["calendar","meeting"]',
            'ASSISTANT_TTS_SENTENCE_PAUSE_SECONDS': '0.2',
            'ASSISTANT_TTS_SENTENCE_CROSSFADE_SECONDS': '0.02',
            'LOG_LEVEL': 'DEBUG',
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings()

        self.assertEqual(settings.llm_slots, (1, 2))
        self.assertEqual(settings.listen_port, 9003)
        self.assertEqual(settings.log_level, 'DEBUG')
        self.assertEqual(settings.tts_sentence_pause_seconds, 0.2)
        self.assertEqual(settings.tts_sentence_crossfade_seconds, 0.02)
        self.assertEqual(settings.mcp['calendar'].endpoint, 'http://calendar-mcp:8080/mcp')
        self.assertEqual(settings.mcp['calendar'].triggers, {'calendar', 'meeting'})

    def test_yaml_file_is_overridden_by_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'assistant.yaml'
            yaml_lines = [
                'model_id: yaml-model',
                'listen_port: 9100',
                'mcp:',
                '  home:',
                '    host: home-mcp',
                '    triggers: [home]',
            ]
            yaml_config = '\n'.join(yaml_lines)
            _ = path.write_text(
                f'{yaml_config}\n',
                encoding='utf-8',
            )
            environment = {
                'ASSISTANT_CONFIG_FILE': str(path),
                'ASSISTANT_MODEL_ID': 'environment-model',
            }
            with patch.dict(os.environ, environment, clear=True):
                settings = Settings()

        self.assertEqual(settings.model_id, 'environment-model')
        self.assertEqual(settings.listen_port, 9100)
        self.assertEqual(settings.mcp['home'].host, 'home-mcp')

    def test_json_configuration_file_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'assistant.json'
            _ = path.write_text('{"llm_slots":[3],"default_timezone":"UTC"}', encoding='utf-8')
            with patch.dict(os.environ, {'ASSISTANT_CONFIG_FILE': str(path)}, clear=True):
                settings = Settings()

        self.assertEqual(settings.llm_slots, (3,))
        self.assertEqual(settings.default_timezone, 'UTC')


class RealtimeEventTest(unittest.TestCase):
    def test_unknown_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            _ = RealtimeEvent.model_validate(
                {'type': 'input_audio_buffer.commit', 'unexpected': True},
            )


class AccessLogFilterTest(unittest.TestCase):
    def test_successful_health_checks_are_suppressed_but_failures_are_retained(self) -> None:
        access_filter = SuccessfulHealthCheckFilter()
        successful_health = logging.LogRecord(
            'uvicorn.access',
            logging.INFO,
            __file__,
            1,
            '%s - "%s %s HTTP/%s" %d',
            ('client', 'GET', '/health/ready?probe=true', '1.1', 200),
            None,
        )
        failed_health = logging.LogRecord(
            'uvicorn.access',
            logging.INFO,
            __file__,
            1,
            '%s - "%s %s HTTP/%s" %d',
            ('client', 'GET', '/health/ready', '1.1', 503),
            None,
        )
        successful_request = logging.LogRecord(
            'uvicorn.access',
            logging.INFO,
            __file__,
            1,
            '%s - "%s %s HTTP/%s" %d',
            ('client', 'GET', '/v1/models', '1.1', 200),
            None,
        )

        self.assertFalse(access_filter.filter(successful_health))
        self.assertTrue(access_filter.filter(failed_health))
        self.assertTrue(access_filter.filter(successful_request))


class UpstreamHealthTest(unittest.IsolatedAsyncioTestCase):
    def test_client_expires_connections_before_upstream_idle_timeout(self) -> None:
        with patch('assistant.src.main.httpx.AsyncClient') as client_type:
            _ = AssistantRuntime(Settings())

        limits = client_type.call_args.kwargs['limits']
        self.assertEqual(limits.keepalive_expiry, 4.0)

    async def test_transport_failure_identifies_service_and_exception_type(self) -> None:
        def fail(request: httpx.Request) -> httpx.Response:
            message = 'timed out'
            raise httpx.ReadTimeout(message, request=request)

        transport = httpx.MockTransport(fail)
        async with httpx.AsyncClient(transport=transport) as client:
            with self.assertLogs('assistant.upstream', level='WARNING') as captured:
                result = await upstream_health(client, Settings())

        self.assertEqual(result, {'llm': False, 'stt': False, 'tts': False})
        llm_record = next(
            record for record in captured.records if getattr(record, 'service', None) == 'llm'
        )
        self.assertEqual(llm_record.__dict__['error_type'], 'ReadTimeout')


class LlmLoggingTest(unittest.IsolatedAsyncioTestCase):
    async def test_request_log_contains_exact_prompt_and_tool_data(self) -> None:
        submitted_payloads: list[dict[str, object]] = []

        def respond(request: httpx.Request) -> httpx.Response:
            submitted_payloads.append(json.loads(request.content))
            return httpx.Response(200, json={'content': '{"answer":"three o clock"}'})

        mcp_tool = ToolDefinition(
            name='clock__time',
            description='Return the time from the clock MCP server.',
            input_schema={
                'type': 'object',
                'properties': {'timezone': {'type': 'string'}},
            },
            triggers=frozenset({'time'}),
            executor=lambda _arguments, _context: '',
            source='mcp:clock',
        )
        prompt = build_prompt('what is the time', [mcp_tool])
        settings = Settings(llm_base_url='http://llm.test')
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            llm = LlmClient(client, settings)
            with self.assertLogs('assistant.upstream', level='DEBUG') as captured:
                _ = await llm.complete(
                    prompt,
                    7,
                    operation='tool_decision',
                    maximum_tokens=32,
                )

        request_record = next(
            record for record in captured.records if record.getMessage() == 'LLM request'
        )
        logged_payload = request_record.__dict__['payload']
        self.assertEqual(logged_payload, submitted_payloads[0])
        self.assertEqual(logged_payload['prompt'], prompt)
        self.assertIn('"name":"clock__time"', logged_payload['prompt'])
        self.assertIn('Return the time from the clock MCP server.', logged_payload['prompt'])


class TimeToolTest(unittest.TestCase):
    def test_rough_time_rounds_to_spoken_five_minute_interval(self) -> None:
        value = datetime(2026, 7, 13, 14, 44, tzinfo=UTC)

        self.assertEqual(rough_time(value), 'quarter to three')


class PromptTest(unittest.TestCase):
    def test_tool_prompt_is_deterministic(self) -> None:
        tools = discover_local_tools()

        first = build_prompt('what is the time', tools)
        second = build_prompt('what is the time', tools)

        self.assertEqual(first, second)
        self.assertIn('"name":"time"', first)

    def test_sentence_split_retains_incomplete_tail(self) -> None:
        sentences, remainder = completed_sentences('First sentence. Incomplete tail')

        self.assertEqual(sentences, ['First sentence.'])
        self.assertEqual(remainder, 'Incomplete tail')

    def test_sentence_split_returns_every_completed_sentence(self) -> None:
        sentences, remainder = completed_sentences('First sentence. Second! Third?')

        self.assertEqual(sentences, ['First sentence.', 'Second!', 'Third?'])
        self.assertEqual(remainder, '')

    def test_tool_json_is_extracted_from_model_response(self) -> None:
        response = parse_tool_response('{"tool":"time","arguments":{"mode":"rough"}}')

        self.assertEqual(response, {'tool': 'time', 'arguments': {'mode': 'rough'}})


class ToolSelectionTest(unittest.IsolatedAsyncioTestCase):
    registry: ToolRegistry

    async def asyncSetUp(self) -> None:
        self.registry = ToolRegistry(
            discover_local_tools(),
            {},
            default_timezone='local',
            maximum_result_characters=8_000,
        )

    async def test_trigger_words_use_word_boundaries(self) -> None:
        selected = await self.registry.select('Can you estimate this?')
        self.assertEqual(selected, [])

        selected = await self.registry.select('What is the date?')
        self.assertEqual([tool.name for tool in selected], ['time'])


class FakeLlm:
    def __init__(self) -> None:
        self.warms: list[tuple[str, int]] = []
        self.response = 'It is three.'

    async def warm_cache(self, prompt: str, slot: int) -> None:
        self.warms.append((prompt, slot))

    async def complete(
        self,
        prompt: str,
        slot: int,
        *,
        operation: str,
        maximum_tokens: int,
    ) -> str:
        del prompt, slot, operation, maximum_tokens
        return self.response

    async def stream(self, prompt: str, slot: int) -> AsyncIterator[str]:
        del prompt, slot
        yield self.response


class FakeTts:
    def __init__(self) -> None:
        self.requests: list[str] = []

    async def stream(self, text: str) -> AsyncIterator[tuple[bytes, AudioFormat]]:
        self.requests.append(text)
        yield (
            struct.pack('<hhhh', 1_000, 1_000, 1_000, 1_000),
            AudioFormat(
                sample_rate=24_000,
                sample_width=2,
                channels=1,
            ),
        )


class CachePipelineTest(unittest.IsolatedAsyncioTestCase):
    settings: Settings
    slots: SlotPool
    llm: FakeLlm
    tts: FakeTts
    registry: ToolRegistry
    utterance: AssistantUtterance

    async def asyncSetUp(self) -> None:
        self.settings = Settings(
            llm_slots=(7,),
            tts_sentence_pause_seconds=2 / 24_000,
            tts_sentence_crossfade_seconds=2 / 24_000,
        )
        self.slots = SlotPool(self.settings.llm_slots)
        self.llm = FakeLlm()
        self.tts = FakeTts()
        self.registry = ToolRegistry(
            discover_local_tools(),
            {},
            default_timezone='local',
            maximum_result_characters=8_000,
        )
        self.utterance = AssistantUtterance(
            self.settings,
            self.slots,
            self.llm,
            self.tts,
            self.registry,
        )

    async def asyncTearDown(self) -> None:
        await self.utterance.close()

    async def test_stable_word_updates_warm_the_same_slot(self) -> None:
        _ = await self.utterance.select_and_warm('what is the', reason='delta')
        _ = await self.utterance.select_and_warm('what is the time', reason='delta')
        _ = await self.utterance.select_and_warm('what is the time', reason='final')

        self.assertEqual(len(self.llm.warms), 2)
        self.assertEqual([slot for _, slot in self.llm.warms], [7, 7])

    async def test_generation_streams_text_and_pcm_events(self) -> None:
        events: list[dict[str, object]] = []

        async def send(event: dict[str, object]) -> None:
            events.append(event)

        with self.assertLogs('assistant.pipeline', level='DEBUG') as captured:
            await self.utterance.generate('hello', [], send)

        event_types = [str(event['type']) for event in events]
        response_record = next(
            record for record in captured.records if record.getMessage() == 'Assistant response'
        )
        self.assertEqual(response_record.__dict__['response'], 'It is three.')
        self.assertIn('response.text.delta', event_types)
        self.assertIn('response.audio.delta', event_types)
        self.assertEqual(event_types[-1], 'response.done')

    async def test_generation_splits_and_stitches_every_sentence(self) -> None:
        self.llm.response = 'First sentence. Second! Third?'
        events: list[dict[str, object]] = []

        async def send(event: dict[str, object]) -> None:
            events.append(event)

        await self.utterance.generate('hello', [], send)

        self.assertEqual(self.tts.requests, ['First sentence.', 'Second!', 'Third?'])
        pcm = b''.join(
            base64.b64decode(str(event['audio']))
            for event in events
            if event['type'] == 'response.audio.delta'
        )
        samples = struct.unpack(f'<{len(pcm) // 2}h', pcm)
        self.assertEqual(
            samples,
            (
                1_000,
                1_000,
                1_000,
                0,
                0,
                0,
                0,
                1_000,
                1_000,
                0,
                0,
                0,
                0,
                1_000,
                1_000,
                1_000,
            ),
        )


class MetricsTest(unittest.TestCase):
    def test_prometheus_metrics_include_cache_and_latency_series(self) -> None:
        response = asyncio.run(prometheus_metrics())

        self.assertIn(b'assistant_llm_cache_warm_duration_seconds', response.body)
        self.assertIn(b'assistant_pipeline_stage_duration_seconds', response.body)
        self.assertTrue(response.headers['content-type'].startswith('text/plain;'))
