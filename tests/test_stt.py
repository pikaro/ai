import asyncio
import logging
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from service_contracts.stt import parse_stt_server_event
from service_logging import SuccessfulHealthCheckFilter
from stt.src.api import app, metrics
from stt.src.config import Settings
from stt.src.runtime import AsrRuntime
from stt.src.schemas import RealtimeEvent
from stt.src.transcript import stable_word_prefix, transcript_delta


class SettingsTest(unittest.TestCase):
    def test_environment_configuration(self) -> None:
        environment = {
            'LISTEN_PORT': '9001',
            'LOG_LEVEL': 'DEBUG',
            'STT_ATTENTION_CONTEXT_SIZE': '70,0',
            'STT_LATEST_WAV_PATH': '/recordings/latest.wav',
            'STT_MAXIMUM_UPLOAD_BYTES': '1024',
            'STT_ONLINE_NORMALIZATION': 'true',
            'STT_SAVE_LATEST_WAV': 'true',
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings()

        self.assertEqual(settings.attention_context_size, (70, 0))
        self.assertEqual(settings.listen_port, 9001)
        self.assertEqual(settings.log_level, 'DEBUG')
        self.assertEqual(settings.latest_wav_path, Path('/recordings/latest.wav'))
        self.assertEqual(settings.maximum_upload_bytes, 1024)
        self.assertTrue(settings.online_normalization)
        self.assertTrue(settings.save_latest_wav)

    def test_invalid_attention_context_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            _ = Settings.model_validate({'attention_context_size': '70'})


class TranscriptTest(unittest.TestCase):
    def test_only_stable_words_are_emitted(self) -> None:
        self.assertEqual(stable_word_prefix('hello wor', 'hello world'), 'hello')

    def test_append_only_delta_is_returned(self) -> None:
        self.assertEqual(transcript_delta('hello', 'hello world'), 'world')
        self.assertEqual(transcript_delta('hello', 'yellow'), '')


class RealtimeEventTest(unittest.TestCase):
    def test_unknown_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            _ = RealtimeEvent.model_validate(
                {'type': 'input_audio_buffer.commit', 'unexpected': True},
            )

    def test_unknown_server_events_remain_forward_compatible(self) -> None:
        payload, event = parse_stt_server_event(
            '{"type":"future.stt.metadata","detail":{"version":2}}',
        )

        self.assertEqual(payload['type'], 'future.stt.metadata')
        self.assertEqual(payload['detail'], {'version': 2})
        self.assertIsNone(event)


class ConfigurationEndpointTest(unittest.IsolatedAsyncioTestCase):
    async def test_patch_updates_stream_tunables_in_memory(self) -> None:
        runtime = AsrRuntime(Settings())
        app.state.runtime = runtime
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            response = await client.patch(
                '/config',
                json={
                    'input_audio_seconds': 0.15,
                    'attention_context_size': [70, 0],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(runtime.settings.input_audio_seconds, 0.15)
        self.assertEqual(runtime.settings.attention_context_size, (70, 0))

    async def test_patch_is_rejected_while_inference_is_active(self) -> None:
        runtime = AsrRuntime(Settings())
        app.state.runtime = runtime
        self.assertTrue(runtime.operations.try_acquire())
        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
                response = await client.patch('/config', json={'input_audio_seconds': 0.2})

            self.assertEqual(response.status_code, 409)
        finally:
            runtime.operations.release()

    async def test_invalid_patch_is_rejected_without_changing_settings(self) -> None:
        runtime = AsrRuntime(Settings())
        app.state.runtime = runtime
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            response = await client.patch('/config', json={'input_audio_seconds': 0})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(runtime.settings.input_audio_seconds, 0.25)


class AccessLogFilterTest(unittest.TestCase):
    def test_successful_health_checks_are_suppressed_but_failures_are_retained(self) -> None:
        access_filter = SuccessfulHealthCheckFilter()
        successful_health = logging.LogRecord(
            'uvicorn.access',
            logging.INFO,
            __file__,
            1,
            '%s - "%s %s HTTP/%s" %d',
            ('client', 'GET', '/health/live', '1.1', 200),
            None,
        )
        failed_health = logging.LogRecord(
            'uvicorn.access',
            logging.INFO,
            __file__,
            1,
            '%s - "%s %s HTTP/%s" %d',
            ('client', 'GET', '/health/live', '1.1', 503),
            None,
        )

        self.assertFalse(access_filter.filter(successful_health))
        self.assertTrue(access_filter.filter(failed_health))


class LatestWavTest(unittest.TestCase):
    def test_latest_pcm_recording_is_saved_as_wav(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'latest.wav'
            runtime = AsrRuntime(Settings(save_latest_wav=True, latest_wav_path=destination))
            pcm = b'\x01\x02\x03\x04'

            runtime.save_latest_wav(pcm, sample_rate=16_000, channels=1)
            replacement_pcm = b'\x05\x06'
            runtime.save_latest_wav(replacement_pcm, sample_rate=16_000, channels=1)

            with wave.open(str(destination), 'rb') as wav_file:
                self.assertEqual(wav_file.getframerate(), 16_000)
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.readframes(wav_file.getnframes()), replacement_pcm)


class MetricsTest(unittest.TestCase):
    def test_prometheus_metrics_include_model_state(self) -> None:
        response = asyncio.run(metrics())

        self.assertIn(b'stt_model_ready', response.body)
        self.assertIn(b'stt_request_duration_seconds', response.body)
        self.assertIn(b'stt_stream_time_to_first_delta_seconds', response.body)
        self.assertTrue(response.headers['content-type'].startswith('text/plain;'))
