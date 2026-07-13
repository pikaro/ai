from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from pydantic import ValidationError

from assistant.src.config import Settings
from assistant.src.main import RealtimeEvent, prometheus_metrics
from assistant.src.pipeline import (
    AssistantUtterance,
    build_prompt,
    completed_sentences,
    parse_tool_response,
)
from assistant.src.tooling import ToolRegistry, discover_local_tools
from assistant.src.tools.time import rough_time
from assistant.src.upstream import AudioFormat, SlotPool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class SettingsTest(unittest.TestCase):
    def test_environment_supports_nested_mcp_servers(self) -> None:
        environment = {
            'ASSISTANT_LLM_SLOTS': '[1,2]',
            'ASSISTANT_MCP__CALENDAR__HOST': 'calendar-mcp',
            'ASSISTANT_MCP__CALENDAR__TRIGGERS': '["calendar","meeting"]',
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings()

        self.assertEqual(settings.llm_slots, (1, 2))
        self.assertEqual(settings.mcp['calendar'].endpoint, 'http://calendar-mcp:8080/mcp')
        self.assertEqual(settings.mcp['calendar'].triggers, {'calendar', 'meeting'})

    def test_yaml_file_is_overridden_by_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'assistant.yaml'
            _ = path.write_text(
                'model_id: yaml-model\nmcp:\n  home:\n    host: home-mcp\n    triggers: [home]\n',
                encoding='utf-8',
            )
            environment = {
                'ASSISTANT_CONFIG_FILE': str(path),
                'ASSISTANT_MODEL_ID': 'environment-model',
            }
            with patch.dict(os.environ, environment, clear=True):
                settings = Settings()

        self.assertEqual(settings.model_id, 'environment-model')
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
        return 'It is three.'

    async def stream(self, prompt: str, slot: int) -> AsyncIterator[str]:
        del prompt, slot
        yield 'It is three.'


class FakeTts:
    async def stream(self, text: str) -> AsyncIterator[tuple[bytes, AudioFormat]]:
        del text
        yield b'\x01\x02', AudioFormat(sample_rate=24_000, sample_width=2, channels=1)


class CachePipelineTest(unittest.IsolatedAsyncioTestCase):
    settings: Settings
    slots: SlotPool
    llm: FakeLlm
    registry: ToolRegistry
    utterance: AssistantUtterance

    async def asyncSetUp(self) -> None:
        self.settings = Settings(llm_slots=(7,))
        self.slots = SlotPool(self.settings.llm_slots)
        self.llm = FakeLlm()
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
            FakeTts(),
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

        await self.utterance.generate('hello', [], send)

        event_types = [str(event['type']) for event in events]
        self.assertIn('response.text.delta', event_types)
        self.assertIn('response.audio.delta', event_types)
        self.assertEqual(event_types[-1], 'response.done')


class MetricsTest(unittest.TestCase):
    def test_prometheus_metrics_include_cache_and_latency_series(self) -> None:
        response = asyncio.run(prometheus_metrics())

        self.assertIn(b'assistant_llm_cache_warm_duration_seconds', response.body)
        self.assertIn(b'assistant_pipeline_stage_duration_seconds', response.body)
        self.assertTrue(response.headers['content-type'].startswith('text/plain;'))
