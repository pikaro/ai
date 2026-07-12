import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from stt.src.main import RealtimeEvent, Settings, stable_word_prefix, transcript_delta


class SettingsTest(unittest.TestCase):
    def test_environment_configuration(self) -> None:
        environment = {
            'ASR_ATT_CONTEXT_SIZE': '70,0',
            'ASR_MAX_UPLOAD_BYTES': '1024',
            'ASR_ONLINE_NORMALIZATION': 'true',
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings.from_environment()

        self.assertEqual(settings.attention_context_size, (70, 0))
        self.assertEqual(settings.maximum_upload_bytes, 1024)
        self.assertTrue(settings.online_normalization)

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
