import asyncio
import io
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from fastapi import HTTPException, UploadFile

from service_logging import SuccessfulHealthCheckFilter
from tts.src.main import (
    MODEL_ID,
    Settings,
    SpeechRequest,
    TtsRuntime,
    app,
    metrics,
)


class SettingsTest(unittest.TestCase):
    def test_environment_configuration(self) -> None:
        environment = {
            'LISTEN_PORT': '9002',
            'LOG_LEVEL': 'DEBUG',
            'TTS_DATA_DIRECTORY': '/voices',
            'TTS_LANGUAGE': 'german',
            'TTS_MAXIMUM_INPUT_CHARACTERS': '120',
            'TTS_MAXIMUM_VOICE_UPLOAD_BYTES': '2048',
            'TTS_VOICE': 'juergen',
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings()

        self.assertEqual(settings.language, 'german')
        self.assertEqual(settings.listen_port, 9002)
        self.assertEqual(settings.log_level, 'DEBUG')
        self.assertEqual(settings.maximum_input_characters, 120)
        self.assertEqual(settings.maximum_voice_upload_bytes, 2048)
        self.assertEqual(settings.voice, 'juergen')
        self.assertEqual(settings.data_directory, Path('/voices'))


class VoiceStorageTest(unittest.TestCase):
    def test_uploaded_voice_is_selected_at_startup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            voice_path = Path(temporary_directory) / 'foo.safetensors'
            _ = voice_path.write_bytes(b'voice state')
            runtime = TtsRuntime(Settings(data_directory=voice_path.parent, voice='foo'))
            model = MagicMock()
            pocket_tts = MagicMock()
            pocket_tts.TTSModel.load_model.return_value = model
            torch = MagicMock()

            with patch('tts.src.main.importlib.import_module') as import_module:
                import_module.side_effect = lambda name: torch if name == 'torch' else pocket_tts
                runtime.load()

            model.get_state_for_audio_prompt.assert_called_once_with(str(voice_path))

    def test_canned_voice_is_used_without_uploaded_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = TtsRuntime(Settings(data_directory=Path(temporary_directory), voice='alba'))

            self.assertEqual(runtime.voice_source(), 'alba')

    def test_voice_upload_endpoint_writes_requested_name_without_authentication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_directory = Path(temporary_directory)
            runtime = TtsRuntime(Settings(data_directory=data_directory))
            app.state.runtime = runtime

            async def upload_voice() -> httpx.Response:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url='http://test',
                ) as client:
                    return await client.post(
                        '/v1/voices',
                        data={'name': 'foo'},
                        files={'file': ('source.safetensors', b'voice state')},
                    )

            response = asyncio.run(upload_voice())

            self.assertEqual((data_directory / 'foo.safetensors').read_bytes(), b'voice state')
            self.assertEqual(response.status_code, 201)
            self.assertEqual(
                response.json(),
                {'name': 'foo', 'filename': 'foo.safetensors', 'replaced': False},
            )

    def test_invalid_voice_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = TtsRuntime(Settings(data_directory=Path(temporary_directory)))
            upload = UploadFile(filename='source.safetensors', file=io.BytesIO(b'voice state'))

            with self.assertRaises(HTTPException):
                _ = runtime.save_voice('../foo', upload)

    def test_oversized_upload_does_not_replace_existing_voice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            voice_path = Path(temporary_directory) / 'foo.safetensors'
            _ = voice_path.write_bytes(b'original')
            runtime = TtsRuntime(
                Settings(
                    data_directory=Path(temporary_directory),
                    maximum_voice_upload_bytes=4,
                ),
            )
            upload = UploadFile(filename='source.safetensors', file=io.BytesIO(b'too large'))

            with self.assertRaises(HTTPException):
                _ = runtime.save_voice('foo', upload)

            self.assertEqual(voice_path.read_bytes(), b'original')


class RequestValidationTest(unittest.TestCase):
    runtime = TtsRuntime(Settings(maximum_input_characters=10, voice='alba'))

    def setUp(self) -> None:
        self.runtime = TtsRuntime(Settings(maximum_input_characters=10, voice='alba'))

    def test_valid_request_is_normalized(self) -> None:
        request = SpeechRequest(model=MODEL_ID, input=' hello ', voice='alba')
        self.assertEqual(self.runtime.validate_request(request), 'hello')

    def test_unknown_voice_is_rejected(self) -> None:
        request = SpeechRequest(model=MODEL_ID, input='hello', voice='unknown')
        with self.assertRaises(HTTPException):
            _ = self.runtime.validate_request(request)

    def test_unsupported_speed_is_rejected(self) -> None:
        request = SpeechRequest(model=MODEL_ID, input='hello', speed=1.5)
        with self.assertRaises(HTTPException):
            _ = self.runtime.validate_request(request)

    def test_input_limit_is_enforced(self) -> None:
        request = SpeechRequest(model=MODEL_ID, input='a' * 11)
        with self.assertRaises(HTTPException):
            _ = self.runtime.validate_request(request)


class AccessLogFilterTest(unittest.TestCase):
    def test_successful_health_checks_are_suppressed_but_failures_are_retained(self) -> None:
        access_filter = SuccessfulHealthCheckFilter()
        successful_health = logging.LogRecord(
            'uvicorn.access',
            logging.INFO,
            __file__,
            1,
            '%s - "%s %s HTTP/%s" %d',
            ('client', 'GET', '/health', '1.1', 200),
            None,
        )
        failed_health = logging.LogRecord(
            'uvicorn.access',
            logging.INFO,
            __file__,
            1,
            '%s - "%s %s HTTP/%s" %d',
            ('client', 'GET', '/health', '1.1', 503),
            None,
        )

        self.assertFalse(access_filter.filter(successful_health))
        self.assertTrue(access_filter.filter(failed_health))


class MetricsTest(unittest.TestCase):
    def test_prometheus_metrics_include_model_state(self) -> None:
        response = asyncio.run(metrics())

        self.assertIn(b'tts_model_ready', response.body)
        self.assertTrue(response.headers['content-type'].startswith('text/plain;'))
