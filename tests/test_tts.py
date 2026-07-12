import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from tts.src.main import MODEL_ID, Settings, SpeechRequest, TtsRuntime


class SettingsTest(unittest.TestCase):
    def test_environment_configuration(self) -> None:
        environment = {
            'TTS_LANGUAGE': 'german',
            'TTS_MAXIMUM_INPUT_CHARACTERS': '120',
            'TTS_VOICE': 'juergen',
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings()

        self.assertEqual(settings.language, 'german')
        self.assertEqual(settings.maximum_input_characters, 120)
        self.assertEqual(settings.voice, 'juergen')


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
