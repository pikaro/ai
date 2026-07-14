import asyncio
import logging
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from stt.src.main import (
    AsrRuntime,
    RealtimeEvent,
    Settings,
    SuccessfulHealthCheckFilter,
    metrics,
    stable_word_prefix,
    transcript_delta,
)


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
        self.assertTrue(response.headers['content-type'].startswith('text/plain;'))
