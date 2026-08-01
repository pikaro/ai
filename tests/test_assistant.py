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
from typing import TYPE_CHECKING, Self, cast
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import WebSocket, WebSocketException, status
from pydantic import SecretStr, ValidationError
from starlette.datastructures import Headers
from starlette.websockets import WebSocketState

from assistant.src.api import app as assistant_app, lifespan, prometheus_metrics
from assistant.src.config import MCPConfig, MultiVoiceConfig, Settings, VoiceCharacterConfig
from assistant.src.multi_voice import multi_voice_header
from assistant.src.pipeline import (
    BASE_SYSTEM_PROMPT,
    AssistantUtterance,
    SystemPromptFile,
    build_prompt,
    build_prompt_prefix,
    parse_tool_response,
    warm_llm_cache,
)
from assistant.src.realtime import log_request_correlation, realtime
from assistant.src.runtime import AssistantRuntime
from assistant.src.schemas import RealtimeEvent
from assistant.src.tooling import ToolDefinition, ToolRegistry, discover_local_tools
from assistant.src.tools.time import rough_time
from assistant.src.upstream import AudioFormat, LlmClient, SlotPool, TtsClient, upstream_health
from service_logging import SuccessfulHealthCheckFilter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class SettingsTest(unittest.TestCase):
    def test_system_prompt_defaults_to_requested_tmp_path(self) -> None:
        settings = Settings()

        self.assertEqual(settings.system_prompt_path, Path('/tmp/system-prompt'))  # noqa: S108
        self.assertEqual(settings.model_id, 'qwen3-4b-instruct')
        self.assertEqual(settings.default_timezone, 'Europe/Berlin')
        self.assertEqual(settings.multi_voice.default_character, 'assistant')
        self.assertEqual(settings.multi_voice.characters['assistant'].marker, '§')

    def test_multi_voice_characters_are_strict_and_unique(self) -> None:
        config = MultiVoiceConfig(
            default_character='narrator',
            characters={
                'narrator': VoiceCharacterConfig(voice='attenborough', marker='§'),
                'bandit': VoiceCharacterConfig(voice='bender', marker='¶'),
            },
        )

        self.assertEqual(config.characters['bandit'].voice, 'bender')
        with self.assertRaises(ValidationError):
            _ = MultiVoiceConfig(
                characters={
                    'first': VoiceCharacterConfig(voice='alba', marker='§'),
                    'second': VoiceCharacterConfig(voice='bender', marker='§'),
                },
            )
        with self.assertRaises(ValidationError):
            _ = VoiceCharacterConfig(voice='alba', marker='{')

    def test_multi_voice_header_rejects_an_invalid_session_voice(self) -> None:
        with self.assertRaises(ValueError):
            _ = multi_voice_header(Settings().multi_voice, selected_voice='bad>voice')

    def test_environment_supports_nested_mcp_servers(self) -> None:
        environment = {
            'LISTEN_PORT': '9003',
            'ASSISTANT_LLM_SLOTS': '[1,2]',
            'ASSISTANT_MCP__CALENDAR__HOST': 'calendar-mcp',
            'ASSISTANT_MCP__CALENDAR__TRIGGERS': '["calendar","meeting"]',
            'LOG_LEVEL': 'DEBUG',
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings()

        self.assertEqual(settings.llm_slots, (1, 2))
        self.assertEqual(settings.listen_port, 9003)
        self.assertEqual(settings.log_level, 'DEBUG')
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

    def test_session_voice_is_accepted(self) -> None:
        event = RealtimeEvent.model_validate(
            {'type': 'session.update', 'session': {'voice': 'bender'}},
        )

        self.assertIsNotNone(event.session)
        self.assertEqual(getattr(event.session, 'voice', None), 'bender')


class ConfigurationEndpointTest(unittest.IsolatedAsyncioTestCase):
    async def test_patch_rebuilds_runtime_with_ephemeral_validated_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(system_prompt_path=Path(directory) / 'system-prompt')
            original = AssistantRuntime(settings)
            assistant_app.state.runtime = original
            transport = httpx.ASGITransport(app=assistant_app)
            async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
                response = await client.patch(
                    '/config',
                    json={
                        'llm_cache_warm_min_interval_seconds': 0.75,
                        'llm_cache_warm_min_new_characters': 5,
                    },
                )

            replacement = cast('AssistantRuntime', assistant_app.state.runtime)
            try:
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()['ephemeral'])
                self.assertEqual(
                    replacement.settings.llm_cache_warm_min_interval_seconds,
                    0.75,
                )
                self.assertEqual(replacement.settings.llm_cache_warm_min_new_characters, 5)
            finally:
                await replacement.close()

    async def test_restart_only_setting_is_rejected_without_replacing_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(system_prompt_path=Path(directory) / 'system-prompt')
            runtime = AssistantRuntime(settings)
            assistant_app.state.runtime = runtime
            transport = httpx.ASGITransport(app=assistant_app)
            try:
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url='http://test',
                ) as client:
                    response = await client.patch('/config', json={'listen_port': 9000})

                self.assertEqual(response.status_code, 409)
                self.assertIs(assistant_app.state.runtime, runtime)
            finally:
                await runtime.close()


class DashboardTest(unittest.IsolatedAsyncioTestCase):
    async def test_dashboard_contains_all_three_service_editors(self) -> None:
        transport = httpx.ASGITransport(app=assistant_app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            response = await client.get('/dashboard')

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers['content-type'].startswith('text/html'))
        self.assertIn("const configServices = ['assistant', 'stt', 'tts']", response.text)
        self.assertIn('/dashboard/config/${service}', response.text)
        self.assertIn('class="json-highlight"', response.text)
        self.assertIn('function buildPatch(card)', response.text)
        self.assertIn("fetch('/system-prompt'", response.text)
        self.assertIn('id="stt-mic"', response.text)
        self.assertIn('id="assistant-mic"', response.text)
        self.assertIn('fetch(`/dashboard/stt?', response.text)
        self.assertIn("fetch('/dashboard/tts'", response.text)
        self.assertIn('id="tts-voice"', response.text)
        self.assertIn('id="tts-format"', response.text)
        self.assertIn('id="tts-pipeline"', response.text)
        self.assertIn('function schedulePcmChunk(context, playback, chunk, format)', response.text)
        self.assertIn('function collectPcmResponse(response, startedAt, context)', response.text)

    async def test_assistant_configuration_is_exposed_without_proxying(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                model_id='dashboard-model',
                system_prompt_path=Path(directory) / 'system-prompt',
            )
            runtime = AssistantRuntime(settings)
            assistant_app.state.runtime = runtime
            transport = httpx.ASGITransport(app=assistant_app)
            try:
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url='http://test',
                ) as client:
                    response = await client.get('/dashboard/config/assistant')

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['model_id'], 'dashboard-model')
            finally:
                await runtime.close()

    async def test_assistant_patch_preserves_unchanged_redacted_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                system_prompt_path=Path(directory) / 'system-prompt',
                mcp={
                    'calendar': MCPConfig(
                        host='calendar.test',
                        token=SecretStr('actual-secret'),
                    ),
                },
            )
            original = AssistantRuntime(settings)
            assistant_app.state.runtime = original
            transport = httpx.ASGITransport(app=assistant_app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url='http://test',
            ) as client:
                current = (await client.get('/dashboard/config/assistant')).json()
                current['mcp']['calendar']['enabled'] = False
                del current['mcp']['calendar']['token']
                response = await client.patch(
                    '/dashboard/config/assistant',
                    json={'mcp': current['mcp']},
                )

            replacement = cast('AssistantRuntime', assistant_app.state.runtime)
            try:
                self.assertEqual(response.status_code, 200)
                calendar = replacement.settings.mcp['calendar']
                self.assertFalse(calendar.enabled)
                token = cast('SecretStr', calendar.token)
                self.assertEqual(token.get_secret_value(), 'actual-secret')
            finally:
                await replacement.close()

    async def test_upstream_configuration_is_proxied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                stt_base_url='http://stt.test',
                system_prompt_path=Path(directory) / 'system-prompt',
            )
            runtime = AssistantRuntime(settings)
            assistant_app.state.runtime = runtime
            upstream_response = httpx.Response(
                200,
                json={'input_audio_seconds': 0.25},
                request=httpx.Request('GET', 'http://stt.test/config'),
            )
            request_configuration = AsyncMock(return_value=upstream_response)
            transport = httpx.ASGITransport(app=assistant_app)
            try:
                with patch.object(runtime.http, 'request', request_configuration):
                    async with httpx.AsyncClient(
                        transport=transport,
                        base_url='http://test',
                    ) as client:
                        response = await client.get('/dashboard/config/stt')

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {'input_audio_seconds': 0.25})
                request_configuration.assert_awaited_once_with(
                    'GET',
                    'http://stt.test/config',
                )
            finally:
                await runtime.close()

    async def test_upstream_configuration_patch_and_status_are_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                tts_base_url='http://tts.test',
                system_prompt_path=Path(directory) / 'system-prompt',
            )
            runtime = AssistantRuntime(settings)
            assistant_app.state.runtime = runtime
            upstream_response = httpx.Response(
                409,
                json={'detail': 'tts is busy'},
                headers={'Retry-After': '1'},
                request=httpx.Request('PATCH', 'http://tts.test/config'),
            )
            request_configuration = AsyncMock(return_value=upstream_response)
            transport = httpx.ASGITransport(app=assistant_app)
            patch_body = {'voice': 'alba'}
            try:
                with patch.object(runtime.http, 'request', request_configuration):
                    async with httpx.AsyncClient(
                        transport=transport,
                        base_url='http://test',
                    ) as client:
                        response = await client.patch(
                            '/dashboard/config/tts',
                            json=patch_body,
                        )

                self.assertEqual(response.status_code, 409)
                self.assertEqual(response.json(), {'detail': 'tts is busy'})
                self.assertEqual(response.headers['retry-after'], '1')
                request_configuration.assert_awaited_once_with(
                    'PATCH',
                    'http://tts.test/config',
                    json=patch_body,
                )
            finally:
                await runtime.close()

    async def test_recorded_pcm_is_forwarded_exactly_to_stt_realtime(  # noqa: C901
        self,
    ) -> None:
        class FakeSttConnection:
            def __init__(self) -> None:
                self.sent: list[str] = []
                self.responses = iter(
                    (
                        json.dumps({'type': 'session.updated'}),
                        json.dumps(
                            {
                                'type': ('conversation.item.input_audio_transcription.completed'),
                                'transcript': 'debug transcript',
                            },
                        ),
                    ),
                )

            async def __aenter__(self) -> Self:
                return self

            async def __aexit__(self, *_args: object) -> None:
                return None

            def __aiter__(self) -> FakeSttConnection:
                return self

            async def __anext__(self) -> str:
                try:
                    return next(self.responses)
                except StopIteration as error:
                    raise StopAsyncIteration from error

            async def send(self, message: str) -> None:
                self.sent.append(message)

        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                stt_base_url='http://stt.test',
                system_prompt_path=Path(directory) / 'system-prompt',
            )
            runtime = AssistantRuntime(settings)
            assistant_app.state.runtime = runtime
            stt = FakeSttConnection()
            pcm = bytes(range(256)) * 257
            transport = httpx.ASGITransport(app=assistant_app)
            try:
                with patch('assistant.src.dashboard.connect', return_value=stt) as connect_stt:
                    async with httpx.AsyncClient(
                        transport=transport,
                        base_url='http://test',
                    ) as client:
                        response = await client.post(
                            '/dashboard/stt',
                            params={'sample_rate': 48_000, 'channels': 1},
                            content=pcm,
                            headers={'Content-Type': 'application/octet-stream'},
                        )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {'transcript': 'debug transcript'})
                connect_stt.assert_called_once_with(
                    'ws://stt.test/v1/realtime',
                    open_timeout=settings.connect_timeout_seconds,
                    close_timeout=settings.connect_timeout_seconds,
                    max_size=settings.maximum_websocket_message_bytes,
                )
                events = [json.loads(message) for message in stt.sent]
                self.assertEqual(
                    [event['type'] for event in events],
                    [
                        'session.update',
                        'input_audio_buffer.append',
                        'input_audio_buffer.append',
                        'input_audio_buffer.commit',
                    ],
                )
                self.assertEqual(
                    events[0]['session'],
                    {
                        'input_audio_sample_rate': 48_000,
                        'input_audio_channels': 1,
                    },
                )
                forwarded_pcm = b''.join(base64.b64decode(event['audio']) for event in events[1:-1])
                self.assertEqual(forwarded_pcm, pcm)
            finally:
                await runtime.close()

    async def test_tts_speech_and_wav_response_are_proxied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                tts_base_url='http://tts.test',
                tts_model='debug-tts',
                system_prompt_path=Path(directory) / 'system-prompt',
            )
            runtime = AssistantRuntime(settings)
            assistant_app.state.runtime = runtime
            wav = b'RIFFdebug-wave'
            upstream_response = httpx.Response(
                200,
                content=wav,
                headers={'Content-Type': 'audio/wav'},
                request=httpx.Request('POST', 'http://tts.test/v1/audio/speech'),
            )
            request_speech = AsyncMock(return_value=upstream_response)
            transport = httpx.ASGITransport(app=assistant_app)
            try:
                with patch.object(runtime.http, 'request', request_speech):
                    async with httpx.AsyncClient(
                        transport=transport,
                        base_url='http://test',
                    ) as client:
                        response = await client.post(
                            '/dashboard/tts',
                            json={'text': 'Speak this.'},
                        )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, wav)
                self.assertEqual(response.headers['content-type'], 'audio/wav')
                request_speech.assert_awaited_once_with(
                    'POST',
                    'http://tts.test/v1/audio/speech',
                    json={
                        'model': 'debug-tts',
                        'input': 'Speak this.',
                        'response_format': 'wav',
                    },
                )
            finally:
                await runtime.close()

    async def test_tts_pipeline_pcm_voice_and_format_headers_are_streamed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                tts_base_url='http://tts.test',
                tts_model='debug-tts',
                system_prompt_path=Path(directory) / 'system-prompt',
            )
            runtime = AssistantRuntime(settings)
            assistant_app.state.runtime = runtime
            pcm = b'\x01\x02\x03\x04'
            upstream_response = httpx.Response(
                200,
                stream=httpx.ByteStream(pcm),
                headers={
                    'Content-Type': 'application/octet-stream',
                    'X-Audio-Format': 'pcm_s16le',
                    'X-Audio-Sample-Rate': '24000',
                    'X-Audio-Sample-Width': '2',
                    'X-Audio-Channels': '1',
                },
                request=httpx.Request('POST', 'http://tts.test/v1/audio/speech'),
            )
            send_speech = AsyncMock(return_value=upstream_response)
            transport = httpx.ASGITransport(app=assistant_app)
            try:
                with patch.object(runtime.http, 'send', send_speech):
                    async with httpx.AsyncClient(
                        transport=transport,
                        base_url='http://test',
                    ) as client:
                        response = await client.post(
                            '/dashboard/tts',
                            json={
                                'text': 'Stream this.',
                                'voice': 'bender',
                                'pipeline': True,
                                'response_format': 'pcm',
                            },
                        )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, pcm)
                self.assertEqual(response.headers['content-type'], 'application/octet-stream')
                self.assertEqual(response.headers['x-audio-format'], 'pcm_s16le')
                self.assertEqual(response.headers['x-audio-sample-rate'], '24000')
                send_speech.assert_awaited_once()
                await_args = send_speech.await_args_list[0]
                upstream_request = cast('httpx.Request', await_args.args[0])
                self.assertEqual(str(upstream_request.url), 'http://tts.test/v1/audio/speech')
                self.assertEqual(upstream_request.headers['x-pipeline'], 'true')
                self.assertEqual(
                    json.loads(upstream_request.content),
                    {
                        'model': 'debug-tts',
                        'input': 'Stream this.',
                        'response_format': 'pcm',
                        'voice': 'bender',
                    },
                )
                self.assertEqual(await_args.kwargs, {'stream': True})
                self.assertTrue(upstream_response.is_closed)
            finally:
                await runtime.close()


class SystemPromptEndpointTest(unittest.IsolatedAsyncioTestCase):
    async def test_get_and_put_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / 'system-prompt'
            runtime = AssistantRuntime(Settings(system_prompt_path=prompt_path))
            assistant_app.state.runtime = runtime
            original_inode = prompt_path.stat().st_ino
            transport = httpx.ASGITransport(app=assistant_app)
            try:
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url='http://test',
                ) as client:
                    initial = await client.get('/system-prompt')
                    updated = await client.put(
                        '/system-prompt',
                        content='  Updated system prompt.  \n',
                        headers={'Content-Type': 'text/plain; charset=utf-8'},
                    )
                    reloaded = await client.get('/system-prompt')

                self.assertEqual(initial.text, BASE_SYSTEM_PROMPT)
                self.assertEqual(updated.status_code, 200)
                self.assertEqual(updated.text, 'Updated system prompt.')
                self.assertEqual(reloaded.text, updated.text)
                self.assertEqual(
                    prompt_path.read_text(encoding='utf-8'),
                    'Updated system prompt.\n',
                )
                self.assertNotEqual(prompt_path.stat().st_ino, original_inode)
                self.assertFalse(runtime.operations.active)
            finally:
                await runtime.close()

    async def test_put_system_prompt_is_rejected_while_assistant_is_busy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / 'system-prompt'
            runtime = AssistantRuntime(Settings(system_prompt_path=prompt_path))
            assistant_app.state.runtime = runtime
            self.assertTrue(runtime.operations.try_acquire())
            transport = httpx.ASGITransport(app=assistant_app)
            try:
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url='http://test',
                ) as client:
                    response = await client.put('/system-prompt', content='Not applied')

                self.assertEqual(response.status_code, 409)
                self.assertEqual(runtime.system_prompt.read(), BASE_SYSTEM_PROMPT)
            finally:
                runtime.operations.release()
                await runtime.close()


class RealtimeWebSocketTest(unittest.IsolatedAsyncioTestCase):
    async def test_empty_transcript_closes_with_policy_violation(self) -> None:  # noqa: C901
        class FakeUtterance:
            def __init__(self) -> None:
                self.closed = False

            async def close(self) -> None:
                self.closed = True

        class FakeOperations:
            def __init__(self) -> None:
                self.released = False

            def try_acquire(self) -> bool:
                return True

            def release(self) -> None:
                self.released = True

        class FakeRuntime:
            def __init__(self, utterance: FakeUtterance) -> None:
                self._utterance = utterance
                self.operations = FakeOperations()

            def utterance(self) -> FakeUtterance:
                return self._utterance

        class FakeWebSocket:
            headers = Headers()
            client_state = WebSocketState.CONNECTED

            def __init__(self) -> None:
                self.accepted = False
                self.messages: list[dict[str, object]] = []
                self.close_code: int | None = None
                self.close_reason: str | None = None

            async def accept(self) -> None:
                self.accepted = True

            async def send_json(self, payload: dict[str, object]) -> None:
                self.messages.append(payload)

            async def close(self, code: int = 1000, reason: str | None = None) -> None:
                self.close_code = code
                self.close_reason = reason

        reason = 'STT produced empty transcript'
        utterance = FakeUtterance()
        runtime = FakeRuntime(utterance)
        websocket = FakeWebSocket()
        with (
            patch(
                'assistant.src.realtime.runtime_from_websocket',
                return_value=runtime,
            ),
            patch(
                'assistant.src.realtime._run_realtime_session',
                side_effect=WebSocketException(
                    code=status.WS_1008_POLICY_VIOLATION,
                    reason=reason,
                ),
            ),
        ):
            await realtime(cast('WebSocket', websocket))

        self.assertTrue(websocket.accepted)
        self.assertEqual(websocket.messages, [])
        self.assertEqual(websocket.close_code, status.WS_1008_POLICY_VIOLATION)
        self.assertEqual(websocket.close_reason, reason)
        self.assertTrue(utterance.closed)
        self.assertTrue(runtime.operations.released)

    async def test_second_session_is_closed_without_queueing(self) -> None:  # noqa: C901
        class BusyOperations:
            @staticmethod
            def try_acquire() -> bool:
                return False

        class BusyRuntime:
            operations = BusyOperations()

            @staticmethod
            def utterance() -> None:
                self.fail('a rejected session must not create an utterance')

        class FakeWebSocket:
            headers = Headers()

            def __init__(self) -> None:
                self.messages: list[dict[str, object]] = []
                self.close_code: int | None = None

            async def accept(self) -> None:
                return None

            async def send_json(self, payload: dict[str, object]) -> None:
                self.messages.append(payload)

            async def close(self, code: int = 1000) -> None:
                self.close_code = code

        websocket = FakeWebSocket()
        with patch('assistant.src.realtime.runtime_from_websocket', return_value=BusyRuntime()):
            await realtime(cast('WebSocket', websocket))

        self.assertEqual(websocket.close_code, status.WS_1013_TRY_AGAIN_LATER)
        self.assertEqual(
            websocket.messages,
            [{'type': 'error', 'message': 'another assistant session is already active'}],
        )


class RequestCorrelationLoggingTest(unittest.TestCase):
    def test_present_headers_are_logged_verbatim(self) -> None:
        headers = Headers(
            {
                'X-Request-Id': 'wake-42',
                'X-Request-Timestamp': '1784060717068',
            },
        )

        with self.assertLogs('assistant', level='INFO') as captured:
            log_request_correlation(headers)

        record = captured.records[0]
        self.assertEqual(record.__dict__['event_id'], 'ID_assistant_request_correlation_received')
        self.assertEqual(record.__dict__['request_id'], 'wake-42')
        self.assertEqual(record.__dict__['request_timestamp'], '1784060717068')

    def test_absent_headers_do_not_add_a_log_record(self) -> None:
        with patch('assistant.src.realtime.LOGGER.info') as info:
            log_request_correlation(Headers())

        info.assert_not_called()


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
        with (
            tempfile.TemporaryDirectory() as directory,
            patch('assistant.src.runtime.httpx.AsyncClient') as client_type,
        ):
            settings = Settings(system_prompt_path=Path(directory) / 'system-prompt')
            _ = AssistantRuntime(settings)

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
            record for record in captured.records if record.getMessage() == 'LLM request started'
        )
        logged_payload = request_record.__dict__['payload']
        self.assertEqual(logged_payload, submitted_payloads[0])
        self.assertEqual(logged_payload['prompt'], prompt)
        self.assertIn('"name":"clock__time"', logged_payload['prompt'])
        self.assertIn('Return the time from the clock MCP server.', logged_payload['prompt'])

    async def test_cache_warm_http_layer_does_not_duplicate_pipeline_telemetry(self) -> None:
        def respond(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={'content': ''})

        settings = Settings(llm_base_url='http://llm.test')
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            llm = LlmClient(client, settings)
            with self.assertNoLogs('assistant.upstream', level='DEBUG'):
                _ = await llm.complete(
                    'cached prompt',
                    0,
                    operation='cache_warm',
                    maximum_tokens=0,
                )

    async def test_llama_cache_ratio_uses_cached_plus_evaluated_prompt_tokens(self) -> None:
        def respond(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    'content': '',
                    'timings': {
                        'cache_n': 75,
                        'prompt_n': 25,
                        'prompt_ms': 50,
                        'predicted_n': 0,
                    },
                },
            )

        settings = Settings(llm_base_url='http://llm.test')
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            llm = LlmClient(client, settings)
            with patch(
                'assistant.src.upstream.metrics.LLM_CACHE_REUSE_RATIO.observe',
            ) as observe_reuse:
                _ = await llm.complete('prompt', 0, operation='test', maximum_tokens=0)

        observe_reuse.assert_called_once_with(0.75)


class TtsPipelineClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_incremental_text_is_sent_and_binary_pcm_is_streamed(self) -> None:  # noqa: C901
        class FakeConnection:
            def __init__(self) -> None:
                self.sent: list[str] = []
                self.incoming: list[str | bytes] = [
                    '{"type":"future.pipeline.metadata","ignored":true}',
                    (
                        '{"type":"session.ready","format":"pcm16",'
                        '"sample_rate":24000,"sample_width":2,"channels":1,'
                        '"future_field":"ignored"}'
                    ),
                    b'\x01\x02',
                    '{"type":"response.audio.done"}',
                ]

            async def __aenter__(self) -> Self:
                return self

            async def __aexit__(self, *_arguments: object) -> None:
                return

            async def send(self, message: str) -> None:
                self.sent.append(message)

            async def recv(self) -> str | bytes:
                await asyncio.sleep(0)
                return self.incoming.pop(0)

            async def close(self, code: int = 1000) -> None:
                del code

        async def text_stream() -> AsyncIterator[str]:
            yield '§First sentence.'
            yield ' Second sentence.'

        connection = FakeConnection()
        settings = Settings(tts_base_url='http://tts.test')
        async with httpx.AsyncClient() as client:
            tts = TtsClient(client, settings)
            with patch('assistant.src.upstream.connect', return_value=connection):
                chunks = [item async for item in tts.stream(text_stream(), voice='bender')]

        self.assertEqual(
            chunks,
            [(b'\x01\x02', AudioFormat(sample_rate=24_000, sample_width=2, channels=1))],
        )
        self.assertEqual(tts.pipeline_websocket_url, 'ws://tts.test/v1/audio/speech/pipeline')
        sent_events = [json.loads(message) for message in connection.sent]
        self.assertEqual(sent_events[0]['type'], 'session.start')
        self.assertNotIn('voice', sent_events[0])
        self.assertEqual(
            [event['type'] for event in sent_events[1:]],
            [
                'input_text.delta',
                'input_text.delta',
                'input_text.delta',
                'input_text.done',
            ],
        )
        self.assertEqual(
            ''.join(str(event.get('delta', '')) for event in sent_events),
            '<multi>\n<char assistant voice=bender marker=§>\n§First sentence. Second sentence.',
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
        self.assertIn('do not wrap it in JSON', first)
        self.assertIn('omit arguments when those defaults match the request', first)
        self.assertIn('"default":"rough"', first)
        self.assertIn('"default":"Europe/Berlin"', first)

    def test_custom_system_prompt_precedes_the_stable_tool_catalog(self) -> None:
        prompt = build_prompt(
            'hello',
            discover_local_tools(),
            system_prompt='Custom system prompt.',
        )

        self.assertIn('<|im_start|>system\nCustom system prompt.\nTools are available', prompt)
        self.assertLess(prompt.index('"name":"time"'), prompt.index('<|im_start|>user\nhello'))

    def test_cache_warm_prefix_excludes_generation_suffix(self) -> None:
        tools = discover_local_tools()

        prefix = build_prompt_prefix('what is', tools)
        final_prompt = build_prompt('what is the time', tools)

        self.assertTrue(prefix.endswith('<|im_start|>user\nwhat is'))
        self.assertTrue(final_prompt.startswith(prefix))
        self.assertTrue(final_prompt.endswith('<|im_start|>assistant\n'))
        self.assertNotIn('/no_think', final_prompt)
        self.assertNotIn('<think>', final_prompt)

    def test_tool_json_is_extracted_from_model_response(self) -> None:
        response = parse_tool_response('{"tool":"time","arguments":{"mode":"rough"}}')

        self.assertEqual(response, {'tool': 'time', 'arguments': {'mode': 'rough'}})


class ToolCatalogTest(unittest.IsolatedAsyncioTestCase):
    registry: ToolRegistry

    async def asyncSetUp(self) -> None:
        self.registry = ToolRegistry(
            discover_local_tools(),
            {},
            default_timezone='local',
            maximum_result_characters=8_000,
        )

    async def test_every_local_tool_is_available_without_prompt_filtering(self) -> None:
        available = await self.registry.available()

        self.assertEqual([tool.name for tool in available], ['time'])

    async def test_enabled_mcp_catalog_is_discovered_without_prompt_triggers(self) -> None:
        config = MCPConfig(host='clock-mcp', triggers=frozenset({'never-used'}))
        remote_tool = ToolDefinition(
            name='clock__time',
            description='Return remote time.',
            input_schema={'type': 'object'},
            triggers=frozenset(),
            executor=lambda _arguments, _context: '',
            source='mcp:clock',
        )
        registry = ToolRegistry(
            [],
            {'clock': config},
            default_timezone='local',
            maximum_result_characters=8_000,
        )
        discover = AsyncMock(return_value=[remote_tool])

        with patch.object(registry, '_discover_mcp_tools', new=discover):
            first = await registry.available()
            second = await registry.available()

        self.assertEqual([tool.name for tool in first], ['clock__time'])
        self.assertEqual(second, first)
        discover.assert_awaited_once_with('clock', config)


class SystemPromptFileTest(unittest.TestCase):
    def test_default_is_written_and_atomic_replacement_is_reloaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'system-prompt'
            system_prompt = SystemPromptFile(path)

            self.assertEqual(path.read_text(encoding='utf-8'), f'{BASE_SYSTEM_PROMPT}\n')
            self.assertEqual(system_prompt.read(), BASE_SYSTEM_PROMPT)

            replacement = path.with_suffix('.new')
            _ = replacement.write_text('User supplied prompt.\n', encoding='utf-8')
            _ = replacement.replace(path)

            self.assertEqual(system_prompt.read(), 'User supplied prompt.')

            _ = path.write_text('In-place edit.\n', encoding='utf-8')

            self.assertEqual(system_prompt.read(), 'In-place edit.')

    def test_existing_prompt_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'system-prompt'
            _ = path.write_text('Existing prompt.\n', encoding='utf-8')

            system_prompt = SystemPromptFile(path)

            self.assertEqual(system_prompt.read(), 'Existing prompt.')
            self.assertEqual(path.read_text(encoding='utf-8'), 'Existing prompt.\n')


class FakeLlm:
    def __init__(self) -> None:
        self.warms: list[tuple[str, int]] = []
        self.stream_requests: list[tuple[str, int | None]] = []
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

    async def stream(
        self,
        prompt: str,
        slot: int,
        *,
        operation: str = 'generation',
        maximum_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        del prompt, slot
        self.stream_requests.append((operation, maximum_tokens))
        yield self.response


class FakeTts:
    def __init__(self) -> None:
        self.requests: list[str] = []
        self.voices: list[str | None] = []

    async def stream(
        self,
        text_stream: AsyncIterator[str],
        *,
        voice: str | None = None,
    ) -> AsyncIterator[tuple[bytes, AudioFormat]]:
        self.voices.append(voice)
        self.requests.append(''.join([delta async for delta in text_stream]))
        yield (
            struct.pack('<hhhh', 1_000, 1_000, 1_000, 1_000),
            AudioFormat(
                sample_rate=24_000,
                sample_width=2,
                channels=1,
            ),
        )


class StartupWarmTest(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_warms_every_slot_with_prefix_only_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                llm_slots=(3, 5),
                system_prompt_path=Path(directory) / 'system-prompt',
            )
            runtime = AssistantRuntime(settings)
            warm_cache = AsyncMock()
            try:
                with patch.object(runtime.llm, 'warm_cache', warm_cache):
                    await runtime.start()
            finally:
                await runtime.close()

        self.assertEqual([item.args[1] for item in warm_cache.await_args_list], [3, 5])
        prompt = str(warm_cache.await_args_list[0].args[0])
        self.assertTrue(prompt.endswith('<|im_start|>user\n'))
        self.assertIn('"name":"time"', prompt)
        self.assertIn('§ means assistant using voice default', prompt)
        self.assertIn('Do not emit <multi> or <char> declarations', prompt)
        self.assertNotIn('<|im_start|>assistant', prompt)

    async def test_lifespan_waits_for_startup_warm_before_serving(self) -> None:
        runtime = AsyncMock(spec=AssistantRuntime)
        with patch('assistant.src.api.AssistantRuntime', return_value=runtime):
            async with lifespan(assistant_app):
                runtime.start.assert_awaited_once_with()
                runtime.close.assert_not_awaited()

        runtime.close.assert_awaited_once_with()


class CachePipelineTest(unittest.IsolatedAsyncioTestCase):
    settings: Settings
    slots: SlotPool
    llm: FakeLlm
    tts: FakeTts
    registry: ToolRegistry
    utterance: AssistantUtterance
    system_prompt_directory: tempfile.TemporaryDirectory[str]
    system_prompt: SystemPromptFile

    async def asyncSetUp(self) -> None:
        self.system_prompt_directory = tempfile.TemporaryDirectory()
        self.settings = Settings(
            llm_slots=(7,),
            system_prompt_path=Path(self.system_prompt_directory.name) / 'system-prompt',
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
        self.system_prompt = SystemPromptFile(self.settings.system_prompt_path)
        self.utterance = AssistantUtterance(
            self.settings,
            self.slots,
            self.llm,
            self.tts,
            self.registry,
            self.system_prompt,
        )

    async def asyncTearDown(self) -> None:
        await self.utterance.close()
        self.system_prompt_directory.cleanup()

    async def test_final_during_interval_drops_the_pending_warm(self) -> None:
        _ = self.utterance.schedule_cache_warm('what is the')
        await asyncio.sleep(0)
        _ = self.utterance.schedule_cache_warm('what is the time')
        await asyncio.sleep(0)
        _ = await self.utterance.finalize_cache_warming('what is the time')

        self.assertEqual(len(self.llm.warms), 1)
        self.assertEqual(self.llm.warms[0][1], 7)

    async def test_cache_warm_scheduler_reports_each_update_disposition(self) -> None:
        self.assertEqual(self.utterance.schedule_cache_warm('first stable words'), 'scheduled')
        self.assertEqual(
            self.utterance.schedule_cache_warm('first stable words plus more'),
            'coalesced',
        )
        self.assertEqual(
            self.utterance.schedule_cache_warm('first stable words plus more'),
            'too_small',
        )

    async def test_cache_warm_emits_one_debug_timing_event(self) -> None:
        prompt = 'static prefix plus stable transcript'

        with self.assertLogs('assistant.pipeline', level='DEBUG') as captured:
            await warm_llm_cache(self.llm, prompt, 7, reason='delta')

        self.assertEqual(len(captured.records), 1)
        record = captured.records[0]
        self.assertEqual(record.getMessage(), 'LLM cache warm completed')
        self.assertEqual(record.__dict__['reason'], 'delta')
        self.assertEqual(record.__dict__['slot'], 7)
        self.assertEqual(record.__dict__['prompt_characters'], len(prompt))
        self.assertEqual(record.__dict__['outcome'], 'success')

    async def test_warm_uses_reloaded_system_prompt_and_complete_tool_catalog(self) -> None:
        _ = self.settings.system_prompt_path.write_text(
            'Reloaded prompt.\n',
            encoding='utf-8',
        )

        _ = self.utterance.schedule_cache_warm('unrelated command')
        await asyncio.sleep(0)
        _ = await self.utterance.finalize_cache_warming('unrelated command')

        self.assertEqual(len(self.llm.warms), 1)
        self.assertIn('<|im_start|>system\nReloaded prompt.', self.llm.warms[0][0])
        self.assertIn('\nTools are available below.', self.llm.warms[0][0])
        self.assertIn('"name":"time"', self.llm.warms[0][0])
        self.assertTrue(self.llm.warms[0][0].endswith('<|im_start|>user\nunrelated command'))
        self.assertNotIn('<|im_start|>assistant', self.llm.warms[0][0])

    async def test_final_transcript_drops_pending_warms_and_drains_only_active_work(self) -> None:
        class BlockingLlm(FakeLlm):
            def __init__(self) -> None:
                super().__init__()
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def warm_cache(self, prompt: str, slot: int) -> None:
                self.warms.append((prompt, slot))
                _ = self.started.set()
                _ = await self.release.wait()

        llm = BlockingLlm()
        utterance = AssistantUtterance(
            self.settings,
            self.slots,
            llm,
            self.tts,
            self.registry,
            self.system_prompt,
        )
        try:
            _ = utterance.schedule_cache_warm('first stable words')
            _ = await llm.started.wait()
            _ = utterance.schedule_cache_warm('first stable words plus more')
            _ = utterance.schedule_cache_warm('first stable words plus newest')
            finalizing = asyncio.create_task(
                utterance.finalize_cache_warming('final corrected transcript'),
            )
            await asyncio.sleep(0)
            self.assertFalse(finalizing.done())

            _ = llm.release.set()
            _ = await finalizing

            self.assertEqual(len(llm.warms), 1)
            self.assertIn('first stable words', llm.warms[0][0])
        finally:
            await utterance.close()

    async def test_generation_streams_text_and_pcm_events(self) -> None:
        events: list[dict[str, object]] = []

        async def send(event: dict[str, object]) -> None:
            events.append(event)

        with self.assertLogs('assistant.pipeline', level='DEBUG') as captured:
            await self.utterance.generate('hello', [], send)

        event_types = [str(event['type']) for event in events]
        response_record = next(
            record
            for record in captured.records
            if record.getMessage() == 'Assistant response generated'
        )
        self.assertEqual(response_record.__dict__['response'], 'It is three.')
        self.assertIn('response.text.delta', event_types)
        self.assertIn('response.audio.delta', event_types)
        self.assertEqual(event_types[-1], 'response.done')
        self.assertEqual(self.llm.warms, [])

    async def test_generation_streams_complete_text_to_one_tts_pipeline(self) -> None:
        self.llm.response = 'First sentence. Second! Third?'
        events: list[dict[str, object]] = []

        async def send(event: dict[str, object]) -> None:
            events.append(event)

        await self.utterance.generate('hello', [], send)

        self.assertEqual(self.tts.requests, ['First sentence. Second! Third?'])
        pcm = b''.join(
            base64.b64decode(str(event['audio']))
            for event in events
            if event['type'] == 'response.audio.delta'
        )
        samples = struct.unpack(f'<{len(pcm) // 2}h', pcm)
        self.assertEqual(samples, (1_000, 1_000, 1_000, 1_000))

    async def test_selected_voice_is_forwarded_to_tts_pipeline(self) -> None:
        self.utterance.select_voice('bender')

        async def send(_event: dict[str, object]) -> None:
            return

        await self.utterance.generate('hello', [], send)

        self.assertEqual(self.tts.voices, ['bender'])

    async def test_tool_aware_answer_reaches_tts_before_llm_stream_finishes(  # noqa: C901
        self,
    ) -> None:
        first_sentence_requested = asyncio.Event()

        class GatedLlm(FakeLlm):
            async def stream(
                self,
                prompt: str,
                slot: int,
                *,
                operation: str = 'generation',
                maximum_tokens: int | None = None,
            ) -> AsyncIterator[str]:
                del prompt, slot
                self.stream_requests.append((operation, maximum_tokens))
                yield 'First sentence.'
                _ = await first_sentence_requested.wait()
                yield ' Second sentence.'

        class SignalingTts(FakeTts):
            async def stream(
                self,
                text_stream: AsyncIterator[str],
                *,
                voice: str | None = None,
            ) -> AsyncIterator[tuple[bytes, AudioFormat]]:
                self.voices.append(voice)
                parts: list[str] = []
                first_audio = True
                async for text in text_stream:
                    parts.append(text)
                    if first_audio and '.' in text:
                        first_audio = False
                        first_sentence_requested.set()
                        yield (
                            struct.pack('<hhhh', 1_000, 1_000, 1_000, 1_000),
                            AudioFormat(sample_rate=24_000, sample_width=2, channels=1),
                        )
                self.requests.append(''.join(parts))

        llm = GatedLlm()
        tts = SignalingTts()
        self.utterance.llm = llm
        self.utterance.tts = tts
        events: list[dict[str, object]] = []

        async def send(event: dict[str, object]) -> None:
            events.append(event)

        available_tools = await self.registry.available()
        await asyncio.wait_for(
            self.utterance.generate('tell me a story', available_tools, send),
            timeout=1,
        )

        self.assertEqual(tts.requests, ['First sentence. Second sentence.'])
        self.assertEqual(
            llm.stream_requests,
            [('tool_decision', self.settings.llm_tool_tokens)],
        )

    async def test_json_tool_call_is_buffered_and_only_answer_is_spoken(self) -> None:
        class ToolCallingLlm(FakeLlm):
            def __init__(self) -> None:
                super().__init__()
                self.prompts: list[str] = []
                self.responses = [
                    [
                        ' ',
                        '{"tool":"time",',
                        '"arguments":{"mode":"rough"}}',
                    ],
                    ['It is around three.', ' Done?'],
                ]

            async def stream(
                self,
                prompt: str,
                slot: int,
                *,
                operation: str = 'generation',
                maximum_tokens: int | None = None,
            ) -> AsyncIterator[str]:
                del slot
                self.prompts.append(prompt)
                self.stream_requests.append((operation, maximum_tokens))
                for token in self.responses.pop(0):
                    yield token

        llm = ToolCallingLlm()
        self.utterance.llm = llm
        self.utterance.settings = self.settings.model_copy(
            update={'maximum_tool_iterations': 1},
        )
        events: list[dict[str, object]] = []

        async def send(event: dict[str, object]) -> None:
            events.append(event)

        available_tools = await self.registry.available()
        await self.utterance.generate('what is the time', available_tools, send)

        spoken_deltas = ''.join(
            str(event['delta']) for event in events if event['type'] == 'response.text.delta'
        )
        self.assertNotIn('"tool"', spoken_deltas)
        self.assertEqual(spoken_deltas, 'It is around three. Done?')
        self.assertEqual(self.tts.requests, ['It is around three. Done?'])
        self.assertEqual(
            llm.stream_requests,
            [
                ('tool_decision', self.settings.llm_tool_tokens),
                ('tool_answer', self.settings.llm_tool_tokens),
            ],
        )
        initial_system = llm.prompts[0].split('<|im_start|>user\n', maxsplit=1)[0]
        final_system = llm.prompts[1].split('<|im_start|>user\n', maxsplit=1)[0]
        self.assertEqual(final_system, initial_system)
        self.assertIn(
            '<|im_start|>assistant\n {"tool":"time","arguments":{"mode":"rough"}}',
            llm.prompts[1],
        )
        self.assertNotIn('{"tool": "time"', llm.prompts[1])
        self.assertIn('<|im_start|>tool\n{"tool":"time","result":"', llm.prompts[1])
        self.assertIn(
            '<|im_start|>user\nNo more tools. Reply directly',
            llm.prompts[1],
        )


class MetricsTest(unittest.TestCase):
    def test_prometheus_metrics_include_cache_and_latency_series(self) -> None:
        response = asyncio.run(prometheus_metrics())

        self.assertIn(b'assistant_llm_cache_warm_duration_seconds', response.body)
        self.assertIn(b'assistant_llm_cache_warm_final_wait_duration_seconds', response.body)
        self.assertIn(b'assistant_llm_server_phase_duration_seconds', response.body)
        self.assertIn(b'assistant_pipeline_stage_duration_seconds', response.body)
        self.assertTrue(response.headers['content-type'].startswith('text/plain;'))
