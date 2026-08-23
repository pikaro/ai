import asyncio
import io
import json
import logging
import os
import struct
import tempfile
import time
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, call, patch

import httpx
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from service_contracts.tts import MODEL_ID, PipelineSessionRequest
from service_logging import SuccessfulHealthCheckFilter
from tts.src.api import app, metrics, speech_pipeline_websocket
from tts.src.config import AdditionalModelSettings, Settings
from tts.src.domain import (
    InputTooLongError,
    InvalidMultiSpeakerMarkupError,
    InvalidVoiceNameError,
    InvalidVoiceSelectorError,
    SpeechCommand,
    UndefinedSpeakerError,
    UnsupportedSpeedError,
    VoiceUploadTooLargeError,
)
from tts.src.markup import (
    MarkupSpeaker,
    MarkupText,
    MultiSpeakerMarkupParser,
    parse_multi_speaker_markup,
)
from tts.src.pipeline import (
    PcmPipeline,
    SmartChunkKnowledge,
    TextSegmenter,
)
from tts.src.runtime import TtsRuntime
from tts.src.schemas import SpeechRequest

if TYPE_CHECKING:
    from fastapi import WebSocket


class PipelineContractTest(unittest.TestCase):
    def test_session_event_type_remains_required(self) -> None:
        with self.assertRaises(ValidationError):
            _ = PipelineSessionRequest.model_validate({'model': MODEL_ID})


class SettingsTest(unittest.TestCase):
    def test_environment_configuration(self) -> None:
        environment = {
            'LISTEN_PORT': '9002',
            'LOG_LEVEL': 'DEBUG',
            'TTS_DATA_DIRECTORY': '/voices',
            'TTS_ADDITIONAL_MODELS': json.dumps(
                {
                    'kyutai/pocket-tts-german': {
                        'language': 'german',
                        'voice': 'juergen',
                    },
                },
            ),
            'TTS_LANGUAGE': 'german',
            'TTS_LATEST_WAV_PATH': '/recordings/latest.wav',
            'TTS_MAXIMUM_INPUT_CHARACTERS': '120',
            'TTS_MAXIMUM_VOICE_UPLOAD_BYTES': '2048',
            'TTS_PIPELINE_CLAUSE_PAUSE_SECONDS': '0.04',
            'TTS_PIPELINE_FIRST_SEGMENT_COMMA_DELIMITER': 'false',
            'TTS_PIPELINE_IDLE_TIMEOUT_SECONDS': '15',
            'TTS_PIPELINE_PARAGRAPH_PAUSE_SECONDS': '0.3',
            'TTS_PIPELINE_SENTENCE_CROSSFADE_SECONDS': '0.02',
            'TTS_PIPELINE_SENTENCE_PAUSE_SECONDS': '0.2',
            'TTS_PIPELINE_SENTENCE_TERMINATORS': '.!?;',
            'TTS_PIPELINE_SILENCE_CONFIRMATION_SECONDS': '0.01',
            'TTS_PIPELINE_SILENCE_THRESHOLD_DBFS': '-45',
            'TTS_PIPELINE_SMART_CHUNK_COLD_START_SPEEDUP': '5',
            'TTS_PIPELINE_SMART_CHUNK_CONFIDENCE': '0.95',
            'TTS_PIPELINE_SMART_CHUNK_ENABLED': 'false',
            'TTS_PIPELINE_SMART_CHUNK_KNOWLEDGE_FLUSH_OBSERVATIONS': '3',
            'TTS_PIPELINE_SMART_CHUNK_LLM_ID': 'qwen',
            'TTS_PIPELINE_SMART_CHUNK_SAFETY_SECONDS': '0.2',
            'TTS_PIPELINE_SPEAKER_SWITCH_PAUSE_SECONDS': '0.25',
            'TTS_PIPELINE_SPEECH_CONFIRMATION_SECONDS': '0.03',
            'TTS_PIPELINE_SPEECH_HYSTERESIS_DB': '8',
            'TTS_SAVE_LATEST_WAV': 'true',
            'TTS_VOICE': 'juergen',
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings()

        self.assertEqual(settings.language, 'german')
        self.assertEqual(
            settings.additional_models['kyutai/pocket-tts-german'].language,
            'german',
        )
        self.assertEqual(
            settings.additional_models['kyutai/pocket-tts-german'].voice,
            'juergen',
        )
        self.assertEqual(settings.listen_port, 9002)
        self.assertEqual(settings.log_level, 'DEBUG')
        self.assertEqual(settings.latest_wav_path, Path('/recordings/latest.wav'))
        self.assertEqual(settings.maximum_input_characters, 120)
        self.assertEqual(settings.maximum_voice_upload_bytes, 2048)
        self.assertEqual(settings.pipeline_clause_pause_seconds, 0.04)
        self.assertFalse(settings.pipeline_first_segment_comma_delimiter)
        self.assertEqual(settings.pipeline_idle_timeout_seconds, 15)
        self.assertEqual(settings.pipeline_paragraph_pause_seconds, 0.3)
        self.assertEqual(settings.pipeline_sentence_crossfade_seconds, 0.02)
        self.assertEqual(settings.pipeline_sentence_pause_seconds, 0.2)
        self.assertEqual(settings.pipeline_sentence_terminators, '.!?;')
        self.assertEqual(settings.pipeline_silence_confirmation_seconds, 0.01)
        self.assertEqual(settings.pipeline_silence_threshold_dbfs, -45)
        self.assertEqual(settings.pipeline_smart_chunk_cold_start_speedup, 5)
        self.assertEqual(settings.pipeline_smart_chunk_confidence, 0.95)
        self.assertFalse(settings.pipeline_smart_chunk_enabled)
        self.assertEqual(settings.pipeline_smart_chunk_knowledge_flush_observations, 3)
        self.assertEqual(settings.pipeline_smart_chunk_llm_id, 'qwen')
        self.assertEqual(settings.pipeline_smart_chunk_safety_seconds, 0.2)
        self.assertEqual(settings.pipeline_speaker_switch_pause_seconds, 0.25)
        self.assertEqual(settings.pipeline_speech_confirmation_seconds, 0.03)
        self.assertEqual(settings.pipeline_speech_hysteresis_db, 8)
        self.assertTrue(settings.save_latest_wav)
        self.assertEqual(settings.voice, 'juergen')
        self.assertEqual(settings.data_directory, Path('/voices'))

    def test_default_model_cannot_be_repeated_as_an_additional_model(self) -> None:
        with self.assertRaises(ValidationError):
            _ = Settings(
                additional_models={
                    MODEL_ID: AdditionalModelSettings(language='german', voice='juergen'),
                },
            )


class MultiModelRuntimeTest(unittest.IsolatedAsyncioTestCase):
    german_model = 'kyutai/pocket-tts-german'

    @classmethod
    def settings(cls) -> Settings:
        return Settings(
            additional_models={
                cls.german_model: AdditionalModelSettings(
                    language='german',
                    voice='juergen',
                ),
            },
        )

    def test_load_preloads_every_configured_model_and_default_voice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = self.settings().model_copy(
                update={'data_directory': Path(temporary_directory)},
            )
            runtime = TtsRuntime(settings)
            english = MagicMock(sample_rate=24_000)
            german = MagicMock(sample_rate=16_000)
            pocket_tts = MagicMock()
            pocket_tts.TTSModel.load_model.side_effect = [english, german]
            torch = MagicMock()

            with patch('tts.src.engine.importlib.import_module') as import_module:
                import_module.side_effect = lambda name: torch if name == 'torch' else pocket_tts
                runtime.load()

            self.assertEqual(runtime.model_ids, (MODEL_ID, self.german_model))
            self.assertEqual(
                pocket_tts.TTSModel.load_model.call_args_list,
                [call(language='english'), call(language='german')],
            )
            english.get_state_for_audio_prompt.assert_called_once_with('alba')
            german.get_state_for_audio_prompt.assert_called_once_with('juergen')
            runtime.close()

    def test_incremental_pipeline_remains_bound_to_selected_model(self) -> None:
        runtime = TtsRuntime(self.settings())
        runtime.engine.model = MagicMock(sample_rate=24_000)
        runtime.engine.voice_states = {'alba': object()}
        german = runtime._model(self.german_model).engine  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        german_state = object()
        german.model = MagicMock(sample_rate=16_000)
        german.voice_states = {'juergen': german_state}
        model_pcm = struct.pack('<hhhh', 1_000, 1_000, 1_000, 1_000)
        german.model.generate_audio_stream.return_value = iter([model_pcm])

        with patch.object(german, '_pcm16_bytes', side_effect=lambda chunk: chunk):
            pipeline = runtime.create_incremental_pipeline(None, model=self.german_model)
            emitted = [
                *pipeline.append_text('Guten Tag.', observed_at=time.perf_counter()),
                *pipeline.finish(observed_at=time.perf_counter()),
            ]
            pipeline.close(completed=True)

        self.assertEqual(pipeline.sample_rate, 16_000)
        self.assertEqual(b''.join(emitted), model_pcm)
        german.model.generate_audio_stream.assert_called_once_with(german_state, 'Guten Tag.')
        runtime.engine.model.generate_audio_stream.assert_not_called()

    async def test_http_request_selects_resident_model_and_its_audio_format(self) -> None:
        runtime = TtsRuntime(self.settings())
        runtime.engine.model = MagicMock(sample_rate=24_000)
        runtime.engine.voice_states = {'alba': object()}
        german = runtime._model(self.german_model).engine  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        german_state = object()
        german.model = MagicMock(sample_rate=16_000)
        german.voice_states = {'juergen': german_state}
        german.model.generate_audio_stream.return_value = iter([b'\x01\x02'])
        app.state.runtime = runtime
        transport = httpx.ASGITransport(app=app)

        with patch.object(german, '_pcm16_bytes', side_effect=lambda chunk: chunk):
            async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
                response = await client.post(
                    '/v1/audio/speech',
                    json={
                        'model': self.german_model,
                        'input': 'Guten Tag',
                        'response_format': 'pcm',
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'\x01\x02')
        self.assertEqual(response.headers['X-Audio-Sample-Rate'], '16000')
        german.model.generate_audio_stream.assert_called_once_with(german_state, 'Guten Tag')
        runtime.engine.model.generate_audio_stream.assert_not_called()

    async def test_models_endpoint_lists_every_resident_model(self) -> None:
        runtime = TtsRuntime(self.settings())
        app.state.runtime = runtime
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            response = await client.get('/v1/models')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [model['id'] for model in response.json()['data']],
            [MODEL_ID, self.german_model],
        )


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

            with patch('tts.src.engine.importlib.import_module') as import_module:
                import_module.side_effect = lambda name: torch if name == 'torch' else pocket_tts
                runtime.load()

            model.get_state_for_audio_prompt.assert_called_once_with(str(voice_path))

    def test_canned_voice_is_used_without_uploaded_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = TtsRuntime(Settings(data_directory=Path(temporary_directory), voice='alba'))

            self.assertEqual(runtime.engine.voice_source('alba'), 'alba')

    def test_voice_upload_endpoint_writes_requested_name_without_authentication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_directory = Path(temporary_directory)
            runtime = TtsRuntime(Settings(data_directory=data_directory))
            runtime.engine.voice_states['foo'] = object()
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
            self.assertNotIn('foo', runtime.engine.voice_states)

    def test_invalid_voice_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = TtsRuntime(Settings(data_directory=Path(temporary_directory)))

            with self.assertRaises(InvalidVoiceNameError):
                _ = runtime.save_voice(
                    '../foo',
                    'source.safetensors',
                    io.BytesIO(b'voice state'),
                )

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
            with self.assertRaises(VoiceUploadTooLargeError):
                _ = runtime.save_voice(
                    'foo',
                    'source.safetensors',
                    io.BytesIO(b'too large'),
                )

            self.assertEqual(voice_path.read_bytes(), b'original')


class RequestValidationTest(unittest.TestCase):
    runtime = TtsRuntime(Settings(maximum_input_characters=10, voice='alba'))

    def setUp(self) -> None:
        self.runtime = TtsRuntime(Settings(maximum_input_characters=10, voice='alba'))

    @staticmethod
    def command(request: SpeechRequest) -> SpeechCommand:
        return SpeechCommand(
            model=request.model,
            text=request.input,
            voice=request.voice,
            speed=request.speed,
        )

    def test_valid_request_is_normalized(self) -> None:
        request = SpeechRequest(model=MODEL_ID, input=' hello ', voice='alba')
        self.assertEqual(self.runtime.validate_speech(self.command(request)), 'hello')

    def test_named_voice_does_not_need_to_match_configured_default(self) -> None:
        request = SpeechRequest(model=MODEL_ID, input='hello', voice='bender')
        self.assertEqual(self.runtime.validate_speech(self.command(request)), 'hello')

    def test_unsupported_speed_is_rejected(self) -> None:
        request = SpeechRequest(model=MODEL_ID, input='hello', speed=1.5)
        with self.assertRaises(UnsupportedSpeedError):
            _ = self.runtime.validate_speech(self.command(request))

    def test_input_limit_is_enforced(self) -> None:
        request = SpeechRequest(model=MODEL_ID, input='a' * 11)
        with self.assertRaises(InputTooLongError):
            _ = self.runtime.validate_speech(self.command(request))


class PipelineTest(unittest.TestCase):
    @staticmethod
    def runtime(settings: Settings | None = None) -> TtsRuntime:
        runtime = TtsRuntime(settings or Settings())
        runtime.engine.model = MagicMock(sample_rate=24_000)
        runtime.engine.voice_states = {'alba': object()}
        runtime.engine.model.generate_audio_stream.side_effect = lambda _voice, _text: iter(
            [struct.pack('<hhhh', 1_000, 1_000, 1_000, 1_000)],
        )
        return runtime

    def test_complete_paragraph_keeps_first_sentence_immediate_and_groups_rest(
        self,
    ) -> None:
        runtime = self.runtime(
            Settings(
                pipeline_sentence_pause_seconds=2 / 24_000,
                pipeline_sentence_crossfade_seconds=2 / 24_000,
            ),
        )
        with patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk):
            pcm = b''.join(
                runtime.stream_pipeline_pcm(
                    'First sentence. Second! Third?',
                    'alba',
                ),
            )

        requested_text = [
            call.args[1]
            for call in cast('MagicMock', runtime.engine.model).generate_audio_stream.call_args_list
        ]
        self.assertEqual(requested_text, ['First sentence.', 'Second! Third?'])
        self.assertEqual(
            struct.unpack(f'<{len(pcm) // 2}h', pcm),
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
                1_000,
            ),
        )

    def test_comma_delimits_only_the_first_pipeline_segment(self) -> None:
        runtime = self.runtime()
        with patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk):
            _ = b''.join(
                runtime.stream_pipeline_pcm(
                    'First clause, rest of sentence. Second clause, remains intact.',
                    'alba',
                ),
            )

        requested_text = [
            call.args[1]
            for call in cast('MagicMock', runtime.engine.model).generate_audio_stream.call_args_list
        ]
        self.assertEqual(
            requested_text,
            ['First clause,', 'rest of sentence. Second clause, remains intact.'],
        )

    def test_clause_sentence_and_paragraph_boundaries_use_separate_pauses(self) -> None:
        runtime = self.runtime(
            Settings(
                pipeline_clause_pause_seconds=1 / 24_000,
                pipeline_paragraph_pause_seconds=3 / 24_000,
                pipeline_sentence_pause_seconds=2 / 24_000,
                pipeline_sentence_crossfade_seconds=0,
                pipeline_smart_chunk_enabled=False,
            ),
        )
        with patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk):
            pcm = b''.join(
                runtime.stream_pipeline_pcm(
                    'Yes, I can hear you. Still there.\n\nHow can I help?',
                    'alba',
                ),
            )

        self.assertEqual(
            struct.unpack(f'<{len(pcm) // 2}h', pcm),
            (
                1_000,
                1_000,
                1_000,
                1_000,
                0,
                1_000,
                1_000,
                1_000,
                1_000,
                0,
                0,
                1_000,
                1_000,
                1_000,
                1_000,
                0,
                0,
                0,
                1_000,
                1_000,
                1_000,
                1_000,
            ),
        )

    def test_late_paragraph_marker_upgrades_pending_sentence_pause(self) -> None:
        runtime = self.runtime(
            Settings(
                pipeline_paragraph_pause_seconds=3 / 24_000,
                pipeline_sentence_crossfade_seconds=0,
                pipeline_sentence_pause_seconds=1 / 24_000,
                pipeline_smart_chunk_enabled=False,
            ),
        )
        pipeline = PcmPipeline(
            runtime,
            'alba',
            capture_latest=False,
            transport='websocket',
        )
        with patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk):
            pcm = b''.join(
                (
                    *pipeline.add_segment('First.', 'sentence'),
                    *pipeline.add_segment('', 'paragraph'),
                    *pipeline.add_segment('Second.', 'input_end'),
                    *pipeline.finish(),
                ),
            )

        self.assertEqual(
            struct.unpack(f'<{len(pcm) // 2}h', pcm),
            (
                1_000,
                1_000,
                1_000,
                1_000,
                0,
                0,
                0,
                1_000,
                1_000,
                1_000,
                1_000,
            ),
        )

    def test_model_silence_counts_toward_pause_and_next_leading_silence_is_removed(
        self,
    ) -> None:
        runtime = self.runtime(
            Settings(
                pipeline_clause_pause_seconds=0.02,
                pipeline_sentence_crossfade_seconds=0,
                pipeline_silence_confirmation_seconds=0.01,
                pipeline_speech_confirmation_seconds=0.01,
            ),
        )
        cast('MagicMock', runtime.engine.model).sample_rate = 1_000

        def generate(_voice: object, text: str) -> object:
            if text == 'Yes,':
                return iter(
                    [
                        struct.pack('<20h', *([1_000] * 20)) + struct.pack('<50h', *([0] * 50)),
                    ],
                )
            return iter(
                [
                    struct.pack('<30h', *([0] * 30)) + struct.pack('<20h', *([2_000] * 20)),
                ],
            )

        cast('MagicMock', runtime.engine.model).generate_audio_stream.side_effect = generate
        with patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk):
            pcm = b''.join(runtime.stream_pipeline_pcm('Yes, Next.', 'alba'))

        self.assertEqual(
            struct.unpack(f'<{len(pcm) // 2}h', pcm),
            (*([1_000] * 20), *([0] * 20), *([2_000] * 20)),
        )

    def test_short_model_silence_is_extended_only_to_target_pause(self) -> None:
        runtime = self.runtime(
            Settings(
                pipeline_clause_pause_seconds=0.02,
                pipeline_sentence_crossfade_seconds=0,
                pipeline_silence_confirmation_seconds=0.02,
                pipeline_speech_confirmation_seconds=0.01,
            ),
        )
        cast('MagicMock', runtime.engine.model).sample_rate = 1_000

        def generate(_voice: object, text: str) -> object:
            if text == 'Yes,':
                return iter(
                    [
                        struct.pack('<20h', *([1_000] * 20)) + struct.pack('<10h', *([0] * 10)),
                    ],
                )
            return iter([struct.pack('<20h', *([2_000] * 20))])

        cast('MagicMock', runtime.engine.model).generate_audio_stream.side_effect = generate
        with patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk):
            pcm = b''.join(runtime.stream_pipeline_pcm('Yes, Next.', 'alba'))

        self.assertEqual(
            struct.unpack(f'<{len(pcm) // 2}h', pcm),
            (*([1_000] * 20), *([0] * 20), *([2_000] * 20)),
        )

    def test_zero_transitions_remove_low_level_tail_and_next_leading_noise(self) -> None:
        runtime = self.runtime(
            Settings(
                pipeline_clause_pause_seconds=0,
                pipeline_sentence_crossfade_seconds=0,
                pipeline_sentence_pause_seconds=0,
                pipeline_silence_confirmation_seconds=0.01,
                pipeline_speech_confirmation_seconds=0.02,
            ),
        )
        cast('MagicMock', runtime.engine.model).sample_rate = 1_000

        def generate(_voice: object, text: str) -> object:
            if text == 'Yes,':
                return iter(
                    [
                        struct.pack('<30h', *([1_000] * 30)) + struct.pack('<100h', *([200] * 100)),
                    ],
                )
            return iter(
                [
                    struct.pack('<80h', *([200] * 80)) + struct.pack('<30h', *([2_000] * 30)),
                ],
            )

        cast('MagicMock', runtime.engine.model).generate_audio_stream.side_effect = generate
        with (
            patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk),
            patch('tts.src.pipeline.time.perf_counter', return_value=0.0),
            self.assertNoLogs('tts', level='WARNING'),
        ):
            pcm = b''.join(runtime.stream_pipeline_pcm('Yes, Next.', 'alba'))

        self.assertEqual(
            struct.unpack(f'<{len(pcm) // 2}h', pcm),
            (*([1_000] * 30), *([2_000] * 30)),
        )

    def test_silence_threshold_is_tunable_for_a_voice_noise_floor(self) -> None:
        audio = struct.pack('<30h', *([1_000] * 30)) + struct.pack(
            '<30h',
            *([200] * 30),
        )

        def synthesize(threshold_dbfs: float) -> tuple[int, ...]:
            runtime = self.runtime(
                Settings(
                    pipeline_sentence_crossfade_seconds=0,
                    pipeline_sentence_pause_seconds=0,
                    pipeline_silence_threshold_dbfs=threshold_dbfs,
                    pipeline_speech_confirmation_seconds=0.02,
                ),
            )
            cast('MagicMock', runtime.engine.model).sample_rate = 1_000
            cast('MagicMock', runtime.engine.model).generate_audio_stream.side_effect = (
                lambda _voice, _text: iter([audio])
            )
            with patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk):
                pcm = b''.join(runtime.stream_pipeline_pcm('Yes.', 'alba'))
            return struct.unpack(f'<{len(pcm) // 2}h', pcm)

        self.assertEqual(
            synthesize(-50),
            (*([1_000] * 30), *([200] * 30)),
        )
        self.assertEqual(synthesize(-40), (*([1_000] * 30),))

    def test_speech_hysteresis_ignores_short_noise_islands(self) -> None:
        runtime = self.runtime(
            Settings(
                pipeline_sentence_crossfade_seconds=0,
                pipeline_sentence_pause_seconds=0,
                pipeline_silence_confirmation_seconds=0.01,
                pipeline_speech_confirmation_seconds=0.02,
            ),
        )
        cast('MagicMock', runtime.engine.model).sample_rate = 1_000
        cast('MagicMock', runtime.engine.model).generate_audio_stream.side_effect = (
            lambda _voice, _text: iter(
                [
                    struct.pack('<30h', *([1_000] * 30))
                    + struct.pack('<20h', *([0] * 20))
                    + struct.pack('<10h', *([1_000] * 10))
                    + struct.pack('<30h', *([0] * 30)),
                ],
            )
        )

        with (
            patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk),
            patch('tts.src.pipeline.time.perf_counter', return_value=0.0),
            self.assertNoLogs('tts', level='WARNING'),
        ):
            pcm = b''.join(runtime.stream_pipeline_pcm('Yes.', 'alba'))

        self.assertEqual(
            struct.unpack(f'<{len(pcm) // 2}h', pcm),
            (*([1_000] * 30),),
        )

    def test_speech_hysteresis_does_not_latch_short_startup_noise(self) -> None:
        runtime = self.runtime(
            Settings(
                pipeline_sentence_crossfade_seconds=0,
                pipeline_speech_confirmation_seconds=0.02,
            ),
        )
        cast('MagicMock', runtime.engine.model).sample_rate = 1_000
        cast('MagicMock', runtime.engine.model).generate_audio_stream.side_effect = (
            lambda _voice, _text: iter(
                [
                    struct.pack('<10h', *([1_000] * 10))
                    + struct.pack('<500h', *([0] * 500))
                    + struct.pack('<30h', *([2_000] * 30)),
                ],
            )
        )

        with patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk):
            pcm = b''.join(runtime.stream_pipeline_pcm('Yes.', 'alba'))

        self.assertEqual(
            struct.unpack(f'<{len(pcm) // 2}h', pcm),
            (*([2_000] * 30),),
        )

    def test_voiced_audio_after_late_clipping_logs_warning_and_error(self) -> None:
        runtime = self.runtime(
            Settings(
                pipeline_clause_pause_seconds=0,
                pipeline_sentence_crossfade_seconds=0,
                pipeline_silence_confirmation_seconds=0.01,
                pipeline_speech_confirmation_seconds=0.01,
            ),
        )
        cast('MagicMock', runtime.engine.model).sample_rate = 1_000
        clock = [0.0]

        def generate(_voice: object, text: str) -> object:
            if text == 'Foobar yes,':
                yield struct.pack('<20h', *([1_000] * 20)) + struct.pack(
                    '<30h',
                    *([0] * 30),
                )
                clock[0] = 1.0
                yield struct.pack('<20h', *([1_000] * 20))
                return
            yield struct.pack('<20h', *([2_000] * 20))

        cast('MagicMock', runtime.engine.model).generate_audio_stream.side_effect = generate
        with (
            patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk),
            patch('tts.src.pipeline.time.perf_counter', side_effect=lambda: clock[0]),
            self.assertLogs('tts', level='WARNING') as captured,
        ):
            _ = b''.join(runtime.stream_pipeline_pcm('Foobar yes, Next.', 'alba'))

        event_ids = [getattr(record, 'event_id', None) for record in captured.records]
        self.assertEqual(event_ids.count('ID_tts_pipeline_stitch_false_tail'), 1)
        sample_end_warnings = [
            record
            for record in captured.records
            if getattr(record, 'reason', None) == 'sample_end_unavailable'
        ]
        self.assertEqual(len(sample_end_warnings), 1)
        self.assertTrue(
            getattr(sample_end_warnings[0], 'speculative_tail_cut', False),
        )

    def test_internal_silence_with_lookahead_is_preserved_without_error(self) -> None:
        runtime = self.runtime(
            Settings(
                pipeline_clause_pause_seconds=0,
                pipeline_sentence_crossfade_seconds=0,
                pipeline_silence_confirmation_seconds=0.01,
                pipeline_speech_confirmation_seconds=0.01,
            ),
        )
        cast('MagicMock', runtime.engine.model).sample_rate = 1_000

        def generate(_voice: object, text: str) -> object:
            if text == 'Foobar yes,':
                return iter(
                    [
                        struct.pack('<20h', *([1_000] * 20))
                        + struct.pack('<30h', *([0] * 30))
                        + struct.pack('<20h', *([1_000] * 20)),
                    ],
                )
            return iter([struct.pack('<20h', *([2_000] * 20))])

        cast('MagicMock', runtime.engine.model).generate_audio_stream.side_effect = generate
        with (
            patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk),
            patch('tts.src.pipeline.time.perf_counter', return_value=0.0),
            self.assertNoLogs('tts', level='WARNING'),
        ):
            pcm = b''.join(runtime.stream_pipeline_pcm('Foobar yes, Next.', 'alba'))

        self.assertEqual(
            struct.unpack(f'<{len(pcm) // 2}h', pcm)[:70],
            (*([1_000] * 20), *([0] * 30), *([1_000] * 20)),
        )

    def test_long_terminal_silence_completed_with_lookahead_does_not_warn(self) -> None:
        runtime = self.runtime(
            Settings(
                pipeline_clause_pause_seconds=0,
                pipeline_sentence_crossfade_seconds=0,
                pipeline_silence_confirmation_seconds=0.01,
                pipeline_speech_confirmation_seconds=0.01,
            ),
        )
        cast('MagicMock', runtime.engine.model).sample_rate = 1_000

        def generate(_voice: object, text: str) -> object:
            if text == 'Yes,':
                return iter(
                    [
                        struct.pack('<20h', *([1_000] * 20)) + struct.pack('<600h', *([0] * 600)),
                    ],
                )
            return iter([struct.pack('<20h', *([2_000] * 20))])

        cast('MagicMock', runtime.engine.model).generate_audio_stream.side_effect = generate
        with (
            patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk),
            patch('tts.src.pipeline.time.perf_counter', return_value=0.0),
            self.assertNoLogs('tts', level='WARNING'),
        ):
            _ = b''.join(runtime.stream_pipeline_pcm('Yes, Next.', 'alba'))

    def test_slow_sample_end_logs_lookahead_warning_before_next_segment(self) -> None:
        runtime = self.runtime(
            Settings(
                pipeline_clause_pause_seconds=0,
                pipeline_sentence_crossfade_seconds=0,
                pipeline_silence_confirmation_seconds=0.01,
                pipeline_speech_confirmation_seconds=0.01,
            ),
        )
        cast('MagicMock', runtime.engine.model).sample_rate = 1_000
        clock = [0.0]

        def generate(_voice: object, text: str) -> object:
            if text == 'Yes,':
                yield struct.pack('<20h', *([1_000] * 20)) + struct.pack(
                    '<30h',
                    *([0] * 30),
                )
                clock[0] = 1.0
                return
            yield struct.pack('<20h', *([2_000] * 20))

        cast('MagicMock', runtime.engine.model).generate_audio_stream.side_effect = generate
        with (
            patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk),
            patch('tts.src.pipeline.time.perf_counter', side_effect=lambda: clock[0]),
            self.assertLogs('tts', level='WARNING') as captured,
        ):
            _ = b''.join(runtime.stream_pipeline_pcm('Yes, Next.', 'alba'))

        reasons = [getattr(record, 'reason', None) for record in captured.records]
        self.assertIn('sample_end_unavailable', reasons)

    def test_late_next_segment_logs_lookahead_warning(self) -> None:
        runtime = self.runtime(
            Settings(
                pipeline_clause_pause_seconds=0,
                pipeline_sentence_crossfade_seconds=0,
            ),
        )
        cast('MagicMock', runtime.engine.model).sample_rate = 1_000
        clock = [0.0]

        def generate(_voice: object, text: str) -> object:
            yield struct.pack('<100h', *([1_000] * 100))
            if text == 'Yes,':
                clock[0] = 1.0

        cast('MagicMock', runtime.engine.model).generate_audio_stream.side_effect = generate
        with (
            patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk),
            patch('tts.src.pipeline.time.perf_counter', side_effect=lambda: clock[0]),
            self.assertLogs('tts', level='WARNING') as captured,
        ):
            _ = b''.join(runtime.stream_pipeline_pcm('Yes, Next.', 'alba'))

        reasons = [getattr(record, 'reason', None) for record in captured.records]
        self.assertIn('next_segment_unavailable', reasons)

    def test_late_first_voice_in_next_segment_logs_lookahead_warning(self) -> None:
        runtime = self.runtime(
            Settings(
                pipeline_clause_pause_seconds=0,
                pipeline_sentence_crossfade_seconds=0,
            ),
        )
        cast('MagicMock', runtime.engine.model).sample_rate = 1_000
        clock = [0.0]

        def generate(_voice: object, text: str) -> object:
            if text != 'Yes,':
                clock[0] = 1.0
            yield struct.pack('<100h', *([1_000] * 100))

        cast('MagicMock', runtime.engine.model).generate_audio_stream.side_effect = generate
        with (
            patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk),
            patch('tts.src.pipeline.time.perf_counter', side_effect=lambda: clock[0]),
            self.assertLogs('tts', level='WARNING') as captured,
        ):
            _ = b''.join(runtime.stream_pipeline_pcm('Yes, Next.', 'alba'))

        reasons = [getattr(record, 'reason', None) for record in captured.records]
        self.assertIn('next_audio_unavailable', reasons)

    def test_first_comma_delimiter_can_be_disabled(self) -> None:
        runtime = self.runtime(Settings(pipeline_first_segment_comma_delimiter=False))
        with patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk):
            _ = b''.join(
                runtime.stream_pipeline_pcm(
                    'First clause, rest of sentence.',
                    'alba',
                ),
            )

        cast('MagicMock', runtime.engine.model).generate_audio_stream.assert_called_once_with(
            runtime.engine.voice_states['alba'],
            'First clause, rest of sentence.',
        )


class SmartChunkTest(unittest.TestCase):
    @staticmethod
    def runtime(data_directory: Path) -> TtsRuntime:
        runtime = TtsRuntime(
            Settings(
                data_directory=data_directory,
                pipeline_first_segment_comma_delimiter=False,
                pipeline_sentence_crossfade_seconds=0,
                pipeline_sentence_pause_seconds=0,
                pipeline_silence_confirmation_seconds=0.01,
                pipeline_smart_chunk_cold_start_speedup=100,
                pipeline_smart_chunk_knowledge_flush_observations=100,
                pipeline_smart_chunk_safety_seconds=0,
                pipeline_speech_confirmation_seconds=0.01,
            ),
        )
        runtime.engine.model = MagicMock(sample_rate=1_000)
        runtime.engine.voice_states = {'alba': object()}
        audio = struct.pack('<5000h', *([1_000] * 5_000))
        runtime.engine.model.generate_audio_stream.side_effect = lambda _voice, _text: iter([audio])
        return runtime

    def test_later_sentences_are_synthesized_together_when_they_fit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = self.runtime(Path(temporary_directory))
            with (
                patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk),
                patch('tts.src.pipeline.time.perf_counter', return_value=0.0),
                self.assertNoLogs('tts', level='WARNING'),
            ):
                _ = b''.join(
                    runtime.stream_pipeline_pcm(
                        'First sentence. Second sentence. Third sentence.',
                        'alba',
                    ),
                )

        model = cast('MagicMock', runtime.engine.model)
        requested_text = [call.args[1] for call in model.generate_audio_stream.call_args_list]
        self.assertEqual(
            requested_text,
            ['First sentence.', 'Second sentence. Third sentence.'],
        )

    def test_blank_line_flushes_each_paragraph_without_cross_paragraph_chunking(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = self.runtime(Path(temporary_directory))
            with (
                patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk),
                patch('tts.src.pipeline.time.perf_counter', return_value=0.0),
            ):
                _ = b''.join(
                    runtime.stream_pipeline_pcm(
                        'Opening without punctuation\n\nFirst. Second.\n\nThird. Fourth.',
                        'alba',
                    ),
                )

        model = cast('MagicMock', runtime.engine.model)
        requested_text = [call.args[1] for call in model.generate_audio_stream.call_args_list]
        self.assertEqual(
            requested_text,
            [
                'Opening without punctuation',
                'First. Second.',
                'Third. Fourth.',
            ],
        )

    def test_incremental_blank_line_emits_a_paragraph_flush(self) -> None:
        segmenter = TextSegmenter('.!?', first_segment_comma_delimiter=False)

        self.assertEqual(segmenter.append('First.'), [('First.', 'sentence')])
        self.assertEqual(segmenter.append('\n\n'), [('', 'paragraph')])

    def test_complete_input_logs_queue_and_flush_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = self.runtime(Path(temporary_directory))
            with (
                patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk),
                patch('tts.src.pipeline.time.perf_counter', return_value=0.0),
                self.assertLogs('tts', level='DEBUG') as captured,
            ):
                _ = b''.join(
                    runtime.stream_pipeline_pcm(
                        'First. Second. Third.\n\nFourth.',
                        'alba',
                    ),
                )

        decisions = [
            (getattr(record, 'decision', None), getattr(record, 'reason', None))
            for record in captured.records
            if getattr(record, 'event_id', None) == 'ID_tts_pipeline_smart_chunk_decision'
        ]
        self.assertEqual(
            decisions,
            [
                ('flush', 'first_segment_immediate'),
                ('queue', 'complete_input_available'),
                ('flush', 'paragraph_boundary'),
                ('queue', 'complete_input_available'),
                ('flush', 'input_end'),
            ],
        )

    def test_early_next_sentence_logs_conservative_misprediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = self.runtime(Path(temporary_directory))
            audio = struct.pack('<150h', *([1_000] * 150))
            cast('MagicMock', runtime.engine.model).generate_audio_stream.side_effect = (
                lambda _voice, _text: iter([audio])
            )
            pipeline = PcmPipeline(
                runtime,
                'alba',
                capture_latest=False,
                transport='websocket',
            )
            with (
                patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk),
                patch('tts.src.pipeline.time.perf_counter', return_value=0.0),
            ):
                _ = b''.join(pipeline.add_segment('First.', 'sentence'))
                _ = b''.join(pipeline.add_segment('Second.', 'sentence'))

            with (
                patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk),
                patch('tts.src.pipeline.time.perf_counter', return_value=0.01),
                self.assertLogs('tts', level='DEBUG') as captured,
            ):
                _ = b''.join(pipeline.add_segment('Third.', 'input_end'))

        records = [
            record
            for record in captured.records
            if getattr(record, 'event_id', None) == 'ID_tts_pipeline_smart_chunk_misprediction'
            and getattr(record, 'reason', None) == 'next_sentence_could_have_been_queued'
        ]
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(
            getattr(record, 'original_decision_reason', None),
            'predicted_next_sentence_misses_playback_buffer',
        )
        self.assertGreater(getattr(record, 'counterfactual_margin_seconds', 0), 0)
        self.assertEqual(getattr(record, 'paragraph_pause_seconds', None), 0.24)

    def test_misprediction_warning_contains_effective_tuning_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = self.runtime(Path(temporary_directory))
            pipeline = PcmPipeline(
                runtime,
                'alba',
                capture_latest=False,
                transport='websocket',
            )
            with (
                patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk),
                patch('tts.src.pipeline.time.perf_counter', return_value=0.0),
            ):
                _ = b''.join(pipeline.add_segment('First.', 'sentence'))
                _ = b''.join(pipeline.add_segment('Second.', 'sentence'))
                flush_deadline = cast('float', pipeline.smart_flush_deadline)

            observed_at = flush_deadline + 0.001
            with (
                patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk),
                patch('tts.src.pipeline.time.perf_counter', return_value=observed_at),
                self.assertLogs('tts', level='WARNING') as captured,
            ):
                _ = b''.join(
                    pipeline.flush_smart_queue(
                        misprediction=True,
                        observed_at=observed_at,
                    ),
                )

        self.assertEqual(len(captured.records), 1)
        record = captured.records[0]
        self.assertEqual(
            getattr(record, 'event_id', None),
            'ID_tts_pipeline_smart_chunk_misprediction',
        )
        self.assertEqual(getattr(record, 'reason', None), 'next_sentence_unavailable')
        self.assertEqual(getattr(record, 'queued_sentences', None), 1)
        self.assertEqual(getattr(record, 'confidence', None), 0.9)
        self.assertEqual(getattr(record, 'llm_id', None), 'default')
        self.assertIsInstance(getattr(record, 'llm_seconds_per_word', None), float)
        self.assertIsInstance(getattr(record, 'voice_seconds_per_word', None), float)
        self.assertIsInstance(getattr(record, 'required_buffer_seconds', None), float)

    def test_voice_and_llm_knowledge_are_persisted_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = Settings(
                data_directory=Path(temporary_directory),
                pipeline_smart_chunk_knowledge_flush_observations=1,
                pipeline_smart_chunk_llm_id='qwen',
            )
            knowledge = SmartChunkKnowledge(settings, 'alba')
            knowledge.observe_voice(
                'A measured sentence.',
                audio_seconds=2.0,
                first_audio_seconds=0.08,
            )
            knowledge.observe_llm_sentence(
                'A measured sentence.',
                arrival_seconds=0.4,
                timing_text='A measured sentence.',
            )
            knowledge.save()

            voice_path = Path(temporary_directory) / '.pipeline-knowledge' / 'voice-alba.json'
            llm_path = Path(temporary_directory) / '.pipeline-knowledge' / 'llm-qwen.json'
            voice_payload = json.loads(voice_path.read_text(encoding='utf-8'))
            llm_payload = json.loads(llm_path.read_text(encoding='utf-8'))
            reloaded = SmartChunkKnowledge(settings, 'alba')
            prediction = reloaded.prediction(settings, queued_text='Queued sentence.')

        self.assertEqual(voice_payload['voice'], 'alba')
        self.assertEqual(llm_payload['llm_id'], 'qwen')
        self.assertEqual(prediction['voice_observations'], 1)
        self.assertEqual(prediction['llm_sentence_observations'], 1)
        self.assertEqual(prediction['llm_timing_observations'], 1)


class VoiceSelectionTest(unittest.TestCase):
    def test_custom_voice_is_loaded_once_and_cached_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_directory = Path(temporary_directory)
            bender_path = data_directory / 'bender.safetensors'
            _ = bender_path.write_bytes(b'voice state')
            runtime = TtsRuntime(Settings(data_directory=data_directory, voice='alba'))
            runtime.engine.model = MagicMock()
            default_state = object()
            bender_state = object()
            runtime.engine.voice_states = {'alba': default_state}
            runtime.engine.model.get_state_for_audio_prompt.return_value = bender_state

            self.assertEqual(runtime.prepare_voice('bender'), 'bender')
            self.assertEqual(runtime.prepare_voice('bender'), 'bender')

            self.assertIs(runtime.engine.voice_states['bender'], bender_state)
            runtime.engine.model.get_state_for_audio_prompt.assert_called_once_with(
                str(bender_path)
            )

    def test_missing_or_default_selector_uses_configured_voice(self) -> None:
        runtime = TtsRuntime(Settings(voice='attenborough'))
        runtime.engine.model = MagicMock()
        default_state = object()
        runtime.engine.voice_states = {'attenborough': default_state}

        self.assertEqual(runtime.prepare_voice(None), 'attenborough')
        self.assertEqual(runtime.prepare_voice('default'), 'attenborough')
        self.assertEqual(runtime.prepare_voice(' DEFAULT '), 'attenborough')
        runtime.engine.model.get_state_for_audio_prompt.assert_not_called()

    def test_invalid_selector_is_rejected(self) -> None:
        runtime = TtsRuntime(Settings())
        runtime.engine.model = MagicMock()
        runtime.engine.voice_states = {'alba': object()}

        with self.assertRaises(InvalidVoiceSelectorError):
            _ = runtime.prepare_voice('../bender')

        runtime.engine.model.get_state_for_audio_prompt.assert_not_called()


class ConfigurationEndpointTest(unittest.IsolatedAsyncioTestCase):
    async def test_patch_updates_limits_and_reloads_selected_voice(self) -> None:
        runtime = TtsRuntime(Settings(voice='alba'))
        runtime.engine.model = MagicMock()
        replacement_voice = object()
        runtime.engine.model.get_state_for_audio_prompt.return_value = replacement_voice
        runtime.engine.voice_states = {'alba': object()}
        app.state.runtime = runtime
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            response = await client.patch(
                '/config',
                json={'maximum_input_characters': 200, 'voice': 'attenborough'},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(runtime.settings.maximum_input_characters, 200)
        self.assertEqual(runtime.settings.voice, 'attenborough')
        self.assertIs(runtime.engine.voice_states['attenborough'], replacement_voice)
        runtime.engine.model.get_state_for_audio_prompt.assert_called_once_with('attenborough')

    async def test_language_change_reports_restart_required(self) -> None:
        runtime = TtsRuntime(Settings())
        app.state.runtime = runtime
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            response = await client.patch('/config', json={'language': 'german'})

        self.assertEqual(response.status_code, 409)
        self.assertEqual(runtime.settings.language, 'english')

    async def test_additional_model_change_reports_restart_required(self) -> None:
        runtime = TtsRuntime(Settings())
        app.state.runtime = runtime
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            response = await client.patch(
                '/config',
                json={
                    'additional_models': {
                        'kyutai/pocket-tts-german': {
                            'language': 'german',
                            'voice': 'juergen',
                        },
                    },
                },
            )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(runtime.model_ids, (MODEL_ID,))

    async def test_concurrent_synthesis_is_rejected_without_waiting(self) -> None:
        runtime = TtsRuntime(Settings())
        app.state.runtime = runtime
        self.assertTrue(runtime.operations.try_acquire())
        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
                response = await client.post(
                    '/v1/audio/speech',
                    json={'model': MODEL_ID, 'input': 'hello', 'response_format': 'pcm'},
                )

            self.assertEqual(response.status_code, 409)
        finally:
            runtime.operations.release()

    async def test_completed_pcm_stream_releases_exclusive_operation(self) -> None:
        runtime = TtsRuntime(Settings())
        runtime.engine.model = MagicMock(sample_rate=24_000)
        runtime.engine.model.generate_audio_stream.return_value = iter([b'\x01\x02'])
        runtime.engine.voice_states = {'alba': object()}
        app.state.runtime = runtime
        transport = httpx.ASGITransport(app=app)
        with patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk):
            async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
                response = await client.post(
                    '/v1/audio/speech',
                    json={'model': MODEL_ID, 'input': 'hello', 'response_format': 'pcm'},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'\x01\x02')
        self.assertFalse(runtime.operations.active)

    async def test_body_voice_selects_cached_custom_voice(self) -> None:
        runtime = TtsRuntime(Settings(voice='attenborough'))
        runtime.engine.model = MagicMock(sample_rate=24_000)
        attenborough_state = object()
        bender_state = object()
        runtime.engine.voice_states = {
            'attenborough': attenborough_state,
            'bender': bender_state,
        }
        runtime.engine.model.generate_audio_stream.return_value = iter([b'\x01\x02'])
        app.state.runtime = runtime
        transport = httpx.ASGITransport(app=app)
        with patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk):
            async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
                response = await client.post(
                    '/v1/audio/speech',
                    json={
                        'model': MODEL_ID,
                        'input': 'hello',
                        'voice': 'bender',
                        'response_format': 'pcm',
                    },
                )

        self.assertEqual(response.status_code, 200)
        runtime.engine.model.generate_audio_stream.assert_called_once_with(bender_state, 'hello')

    async def test_unavailable_body_voice_is_rejected_before_streaming(self) -> None:
        runtime = TtsRuntime(Settings())
        runtime.engine.model = MagicMock()
        runtime.engine.model.get_state_for_audio_prompt.side_effect = FileNotFoundError
        runtime.engine.voice_states = {'alba': object()}
        app.state.runtime = runtime
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            response = await client.post(
                '/v1/audio/speech',
                json={
                    'model': MODEL_ID,
                    'input': 'hello',
                    'voice': 'missing',
                    'response_format': 'pcm',
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {'detail': "voice 'missing' is not available"})
        runtime.engine.model.generate_audio_stream.assert_not_called()
        self.assertFalse(runtime.operations.active)


class PipelineEndpointTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def runtime() -> TtsRuntime:
        runtime = TtsRuntime(Settings())
        runtime.engine.model = MagicMock(sample_rate=24_000)
        runtime.engine.voice_states = {'alba': object()}
        runtime.engine.model.generate_audio_stream.side_effect = lambda _voice, _text: iter(
            [b'\x01\x02\x03\x04'],
        )
        return runtime

    async def _request(
        self,
        runtime: TtsRuntime,
        path: str,
        *,
        pipeline_header: bool = False,
    ) -> httpx.Response:
        app.state.runtime = runtime
        transport = httpx.ASGITransport(app=app)
        headers = {'X-Pipeline': 'true'} if pipeline_header else None
        with patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk):
            async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
                return await client.post(
                    path,
                    headers=headers,
                    json={
                        'model': MODEL_ID,
                        'input': 'First sentence. Second sentence.',
                        'response_format': 'pcm',
                    },
                )

    async def test_default_endpoint_is_unchanged_without_pipeline_header(self) -> None:
        runtime = self.runtime()

        response = await self._request(runtime, '/v1/audio/speech')

        self.assertEqual(response.status_code, 200)
        requested_text = [
            call.args[1]
            for call in cast('MagicMock', runtime.engine.model).generate_audio_stream.call_args_list
        ]
        self.assertEqual(requested_text, ['First sentence. Second sentence.'])

    async def test_default_endpoint_pipeline_header_manages_segmentation(self) -> None:
        runtime = self.runtime()

        response = await self._request(
            runtime,
            '/v1/audio/speech',
            pipeline_header=True,
        )

        self.assertEqual(response.status_code, 200)
        requested_text = [
            call.args[1]
            for call in cast('MagicMock', runtime.engine.model).generate_audio_stream.call_args_list
        ]
        self.assertEqual(requested_text, ['First sentence.', 'Second sentence.'])

    async def test_dedicated_pipeline_endpoint_manages_segmentation(self) -> None:
        runtime = self.runtime()

        response = await self._request(runtime, '/v1/audio/speech/pipeline')

        self.assertEqual(response.status_code, 200)
        requested_text = [
            call.args[1]
            for call in cast('MagicMock', runtime.engine.model).generate_audio_stream.call_args_list
        ]
        self.assertEqual(requested_text, ['First sentence.', 'Second sentence.'])


class MultiSpeakerMarkupTest(unittest.TestCase):
    def test_parses_symbol_markers_and_named_character_tags(self) -> None:
        command = parse_multi_speaker_markup(
            '<multi>\n'
            '<char narrator voice=attenborough marker=§>\n'
            '<char bandit voice=bender marker=¶>\n'
            '§First.<bandit>Second.§Third.',
            model=MODEL_ID,
            speed=1.0,
        )

        self.assertEqual(command.speakers, {'narrator': 'attenborough', 'bandit': 'bender'})
        self.assertEqual(
            [(segment.speaker, segment.text) for segment in command.segments],
            [
                ('narrator', 'First.'),
                ('bandit', 'Second.'),
                ('narrator', 'Third.'),
            ],
        )

    def test_parses_literal_aliases_without_reserving_similar_text(self) -> None:
        command = parse_multi_speaker_markup(
            '<multi>\n'
            '<char narrator voice=attenborough alias={n}>\n'
            '<char bandit voice=bender alias={b}>\n'
            '{n}First {ordinary} text.{b}Second.',
            model=MODEL_ID,
            speed=1.0,
        )

        self.assertEqual(
            [(segment.speaker, segment.text) for segment in command.segments],
            [
                ('narrator', 'First {ordinary} text.'),
                ('bandit', 'Second.'),
            ],
        )

    def test_literal_aliases_may_span_incremental_deltas(self) -> None:
        parser = MultiSpeakerMarkupParser()
        events = parser.append(
            '<multi>\n'
            '<char narrator voice=attenborough alias={n}>\n'
            '<char bandit voice=bender alias={b}>\n',
        )
        events.extend(parser.append('{'))
        events.extend(parser.append('n}First {ordinary} text.{'))
        events.extend(parser.append('b}Second.'))
        events.extend(parser.finish())

        self.assertEqual(
            [
                (event.name if isinstance(event, MarkupSpeaker) else event.text)
                for event in events
                if isinstance(event, (MarkupSpeaker, MarkupText))
            ],
            ['narrator', 'First {ordinary} text.', 'bandit', 'Second.'],
        )

    def test_rejects_undefined_tags_and_duplicate_markers(self) -> None:
        with self.assertRaises(UndefinedSpeakerError):
            _ = parse_multi_speaker_markup(
                '<multi><char narrator voice=alba><missing>No.</missing>',
                model=MODEL_ID,
                speed=1.0,
            )

        with self.assertRaises(InvalidMultiSpeakerMarkupError):
            _ = parse_multi_speaker_markup(
                '<multi><char narrator voice="">\u00a7No.',
                model=MODEL_ID,
                speed=1.0,
            )
        with self.assertRaises(InvalidMultiSpeakerMarkupError):
            _ = parse_multi_speaker_markup(
                '<multi><char first voice=alba marker=§><char second voice=bender marker=§>§No.',
                model=MODEL_ID,
                speed=1.0,
            )


class MultiSpeakerEndpointTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def runtime(settings: Settings | None = None) -> tuple[TtsRuntime, object, object]:
        runtime = TtsRuntime(settings or Settings(voice='attenborough'))
        runtime.engine.model = MagicMock(sample_rate=24_000)
        attenborough_state = object()
        bender_state = object()
        runtime.engine.voice_states = {
            'attenborough': attenborough_state,
            'bender': bender_state,
        }

        def generate(voice_state: object, _text: str) -> object:
            value = 1_000 if voice_state is attenborough_state else 2_000
            return iter([struct.pack('<4h', *([value] * 4))])

        runtime.engine.model.generate_audio_stream.side_effect = generate
        return runtime, attenborough_state, bender_state

    @staticmethod
    def payload() -> dict[str, object]:
        return {
            'model': MODEL_ID,
            'input': {
                'speakers': {
                    'narrator': {'voice': 'attenborough'},
                    'bandit': {'voice': 'bender'},
                },
                'segments': [
                    {'speaker': 'narrator', 'text': 'First'},
                    {'speaker': 'narrator', 'text': 'continues.'},
                    {'speaker': 'bandit', 'text': 'Reply.'},
                    {'speaker': 'narrator', 'text': 'Done.'},
                ],
            },
            'response_format': 'pcm',
        }

    async def _request(
        self,
        runtime: TtsRuntime,
        payload: dict[str, object],
        path: str = '/v1/audio/speech/multi-speaker',
    ) -> httpx.Response:
        app.state.runtime = runtime
        transport = httpx.ASGITransport(app=app)
        with patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk):
            async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
                return await client.post(
                    path,
                    json=payload,
                )

    async def test_ordinary_speech_endpoint_accepts_tagged_multi_speaker_input(self) -> None:
        runtime, attenborough_state, bender_state = self.runtime(
            Settings(
                voice='attenborough',
                maximum_input_characters=40,
                pipeline_first_segment_comma_delimiter=False,
                pipeline_sentence_crossfade_seconds=0,
                pipeline_smart_chunk_enabled=False,
                pipeline_speaker_switch_pause_seconds=0,
            ),
        )
        response = await self._request(
            runtime,
            {
                'model': MODEL_ID,
                'input': (
                    '<multi>\n'
                    '<char narrator voice=attenborough marker=§>\n'
                    '<char bandit voice=bender marker=¶>\n'
                    '§First.¶Reply.§Done.'
                ),
                'response_format': 'pcm',
            },
            '/v1/audio/speech',
        )

        self.assertEqual(response.status_code, 200)
        calls = cast('MagicMock', runtime.engine.model).generate_audio_stream.call_args_list
        self.assertEqual(
            [(call.args[0], call.args[1]) for call in calls],
            [
                (attenborough_state, 'First.'),
                (bender_state, 'Reply.'),
                (attenborough_state, 'Done.'),
            ],
        )

    async def test_tagged_ordinary_request_rejects_conflicting_request_voice(self) -> None:
        runtime, _, _ = self.runtime()
        response = await self._request(
            runtime,
            {
                'model': MODEL_ID,
                'input': '<multi><char narrator voice=attenborough marker=§>§No.',
                'voice': 'bender',
            },
            '/v1/audio/speech',
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn('request voice must be omitted', response.json()['detail'])
        cast('MagicMock', runtime.engine.model).generate_audio_stream.assert_not_called()

    async def test_streams_all_voices_as_one_pcm_response_and_recording(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'latest.wav'
            runtime, attenborough_state, bender_state = self.runtime(
                Settings(
                    voice='attenborough',
                    save_latest_wav=True,
                    latest_wav_path=destination,
                    pipeline_first_segment_comma_delimiter=False,
                    pipeline_sentence_crossfade_seconds=0,
                    pipeline_smart_chunk_enabled=False,
                    pipeline_speaker_switch_pause_seconds=3 / 24_000,
                ),
            )

            response = await self._request(runtime, self.payload())

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['X-Audio-Format'], 'pcm_s16le')
            self.assertEqual(
                struct.unpack(f'<{len(response.content) // 2}h', response.content),
                (
                    *([1_000] * 4),
                    *([0] * 3),
                    *([2_000] * 4),
                    *([0] * 3),
                    *([1_000] * 4),
                ),
            )
            calls = cast('MagicMock', runtime.engine.model).generate_audio_stream.call_args_list
            self.assertEqual(
                [(call.args[0], call.args[1]) for call in calls],
                [
                    (attenborough_state, 'First continues.'),
                    (bender_state, 'Reply.'),
                    (attenborough_state, 'Done.'),
                ],
            )
            with wave.open(str(destination), 'rb') as wav_file:
                saved_pcm = wav_file.readframes(wav_file.getnframes())
            self.assertEqual(saved_pcm, response.content)
            self.assertFalse(runtime.operations.active)

    async def test_returns_all_turns_in_one_wav_response(self) -> None:
        runtime, _, _ = self.runtime(
            Settings(
                voice='attenborough',
                pipeline_first_segment_comma_delimiter=False,
                pipeline_sentence_crossfade_seconds=0,
                pipeline_smart_chunk_enabled=False,
                pipeline_speaker_switch_pause_seconds=0,
            ),
        )
        payload = self.payload()
        payload['response_format'] = 'wav'

        response = await self._request(runtime, payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['content-type'], 'audio/wav')
        with wave.open(io.BytesIO(response.content), 'rb') as wav_file:
            pcm = wav_file.readframes(wav_file.getnframes())
        self.assertEqual(
            struct.unpack(f'<{len(pcm) // 2}h', pcm),
            (*([1_000] * 4), *([2_000] * 4), *([1_000] * 4)),
        )
        self.assertFalse(runtime.operations.active)

    async def test_rejects_undefined_speaker_before_streaming(self) -> None:
        runtime, _, _ = self.runtime()
        payload = self.payload()
        cast('dict[str, object]', payload['input'])['segments'] = [
            {'speaker': 'missing', 'text': 'No voice.'},
        ]

        response = await self._request(runtime, payload)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {'detail': "speaker 'missing' is not defined"})
        cast('MagicMock', runtime.engine.model).generate_audio_stream.assert_not_called()
        self.assertFalse(runtime.operations.active)

    async def test_preloads_every_declared_voice_before_streaming(self) -> None:
        runtime = TtsRuntime(Settings(voice='attenborough'))
        runtime.engine.model = MagicMock(sample_rate=24_000)
        runtime.engine.voice_states = {'attenborough': object()}
        runtime.engine.model.get_state_for_audio_prompt.side_effect = FileNotFoundError
        payload = self.payload()
        cast('dict[str, object]', payload['input'])['segments'] = [
            {'speaker': 'narrator', 'text': 'Only the valid voice is referenced.'},
        ]

        response = await self._request(runtime, payload)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {'detail': "voice 'bender' is not available"})
        runtime.engine.model.generate_audio_stream.assert_not_called()
        self.assertFalse(runtime.operations.active)

    async def test_updates_each_selected_voice_knowledge_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_directory = Path(directory)
            runtime, _, _ = self.runtime(
                Settings(
                    voice='attenborough',
                    data_directory=data_directory,
                    pipeline_first_segment_comma_delimiter=False,
                    pipeline_sentence_crossfade_seconds=0,
                    pipeline_smart_chunk_knowledge_flush_observations=1,
                    pipeline_speaker_switch_pause_seconds=0,
                ),
            )

            response = await self._request(runtime, self.payload())

            self.assertEqual(response.status_code, 200)
            knowledge_directory = data_directory / '.pipeline-knowledge'
            for voice in ('attenborough', 'bender'):
                payload = json.loads(
                    (knowledge_directory / f'voice-{voice}.json').read_text(
                        encoding='utf-8',
                    ),
                )
                self.assertGreater(
                    payload['statistics']['audio_seconds_per_character']['count'],
                    0,
                )


class PipelineWebSocketTest(unittest.IsolatedAsyncioTestCase):
    async def test_incremental_text_produces_audio_before_input_done(self) -> None:  # noqa: C901
        runtime = TtsRuntime(
            Settings(
                pipeline_clause_pause_seconds=0,
                pipeline_sentence_crossfade_seconds=0,
                pipeline_sentence_pause_seconds=0,
            ),
        )
        runtime.engine.model = MagicMock(sample_rate=24_000)
        runtime.engine.voice_states = {'alba': object()}
        runtime.engine.model.generate_audio_stream.side_effect = lambda _voice, _text: iter(
            [b'\x01\x02\x03\x04'],
        )

        class FakeWebSocket:
            def __init__(self) -> None:
                self.app = SimpleNamespace(state=SimpleNamespace(runtime=runtime))
                self.client_state = WebSocketState.CONNECTED
                self.received = 0
                self.incoming = [
                    '{"type":"session.start","model":"kyutai/pocket-tts"}',
                    '{"type":"input_text.delta","delta":"First clause,"}',
                    '{"type":"input_text.delta","delta":" rest of sentence."}',
                    '{"type":"input_text.done"}',
                ]
                self.json_events: list[dict[str, object]] = []
                self.audio: list[bytes] = []
                self.first_audio_after_received: int | None = None
                self.close_code: int | None = None

            async def accept(self) -> None:
                return

            async def receive_text(self) -> str:
                message = self.incoming.pop(0)
                self.received += 1
                return message

            async def send_json(self, payload: dict[str, object]) -> None:
                self.json_events.append(payload)

            async def send_bytes(self, payload: bytes) -> None:
                if self.first_audio_after_received is None:
                    self.first_audio_after_received = self.received
                self.audio.append(payload)

            async def close(self, code: int, reason: str = '') -> None:
                del reason
                self.close_code = code
                self.client_state = WebSocketState.DISCONNECTED

        websocket = FakeWebSocket()
        with patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk):
            await speech_pipeline_websocket(cast('WebSocket', websocket))

        self.assertLess(cast('int', websocket.first_audio_after_received), 4)
        self.assertEqual(b''.join(websocket.audio), b'\x01\x02\x03\x04\x01\x02\x03\x04')
        self.assertEqual(
            [call.args[1] for call in runtime.engine.model.generate_audio_stream.call_args_list],
            ['First clause,', 'rest of sentence.'],
        )
        self.assertEqual(websocket.json_events[0]['type'], 'session.ready')
        self.assertEqual(websocket.json_events[-1]['type'], 'response.audio.done')
        self.assertEqual(websocket.close_code, 1000)
        self.assertFalse(runtime.operations.active)

    async def test_tagged_text_switches_preloaded_voices_before_input_done(self) -> None:  # noqa: C901
        runtime = TtsRuntime(
            Settings(
                voice='attenborough',
                maximum_input_characters=40,
                pipeline_first_segment_comma_delimiter=False,
                pipeline_sentence_crossfade_seconds=0,
                pipeline_sentence_pause_seconds=0,
                pipeline_smart_chunk_enabled=False,
                pipeline_speaker_switch_pause_seconds=0,
            ),
        )
        runtime.engine.model = MagicMock(sample_rate=24_000)
        attenborough_state = object()
        bender_state = object()
        runtime.engine.voice_states = {
            'attenborough': attenborough_state,
            'bender': bender_state,
        }
        runtime.engine.model.generate_audio_stream.side_effect = lambda _voice, _text: iter(
            [b'\x01\x02\x03\x04'],
        )

        class FakeWebSocket:
            def __init__(self) -> None:
                self.app = SimpleNamespace(state=SimpleNamespace(runtime=runtime))
                self.client_state = WebSocketState.CONNECTED
                self.received = 0
                self.incoming = [
                    '{"type":"session.start","model":"kyutai/pocket-tts"}',
                    json.dumps(
                        {
                            'type': 'input_text.delta',
                            'delta': (
                                '<multi>\n'
                                '<char narrator voice=attenborough marker=§>\n'
                                '<char bandit voice=bender marker=¶>\n'
                            ),
                        },
                    ),
                    json.dumps({'type': 'input_text.delta', 'delta': '§First sentence.'}),
                    json.dumps({'type': 'input_text.delta', 'delta': '¶Second sentence.'}),
                    '{"type":"input_text.done"}',
                ]
                self.json_events: list[dict[str, object]] = []
                self.audio: list[bytes] = []
                self.first_audio_after_received: int | None = None
                self.close_code: int | None = None

            async def accept(self) -> None:
                return

            async def receive_text(self) -> str:
                message = self.incoming.pop(0)
                self.received += 1
                return message

            async def send_json(self, payload: dict[str, object]) -> None:
                self.json_events.append(payload)

            async def send_bytes(self, payload: bytes) -> None:
                if self.first_audio_after_received is None:
                    self.first_audio_after_received = self.received
                self.audio.append(payload)

            async def close(self, code: int, reason: str = '') -> None:
                del reason
                self.close_code = code
                self.client_state = WebSocketState.DISCONNECTED

        websocket = FakeWebSocket()
        with patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk):
            await speech_pipeline_websocket(cast('WebSocket', websocket))

        self.assertLess(cast('int', websocket.first_audio_after_received), 5)
        calls = runtime.engine.model.generate_audio_stream.call_args_list
        self.assertEqual(
            [(call.args[0], call.args[1]) for call in calls],
            [
                (attenborough_state, 'First sentence.'),
                (bender_state, 'Second sentence.'),
            ],
        )
        self.assertEqual(websocket.json_events[-1]['type'], 'response.audio.done')
        self.assertEqual(websocket.close_code, 1000)
        self.assertFalse(runtime.operations.active)


class LatestWavTest(unittest.TestCase):
    def test_wav_response_is_saved_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'latest.wav'
            runtime = TtsRuntime(Settings(save_latest_wav=True, latest_wav_path=destination))
            runtime.engine.model = MagicMock(sample_rate=24_000)
            runtime.engine.voice_states = {'alba': object()}
            response_body = b'exact WAV response bytes'

            with patch.object(runtime.engine, '_wav_bytes', return_value=response_body):
                generated = runtime.generate_wav('hello', 'alba')

            self.assertEqual(generated, response_body)
            self.assertEqual(destination.read_bytes(), generated)

    def test_streamed_pcm_is_saved_from_the_exact_emitted_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'latest.wav'
            runtime = TtsRuntime(Settings(save_latest_wav=True, latest_wav_path=destination))
            runtime.engine.model = MagicMock(sample_rate=24_000)
            runtime.engine.voice_states = {'alba': object()}
            pcm_chunks = [b'\x01\x02', b'\x03\x04\x05\x06']
            runtime.engine.model.generate_audio_stream.return_value = iter(pcm_chunks)

            with patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk):
                emitted = list(runtime.stream_pcm('hello', 'alba'))

            self.assertEqual(emitted, pcm_chunks)
            with wave.open(str(destination), 'rb') as wav_file:
                self.assertEqual(wav_file.getframerate(), 24_000)
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                saved_pcm = wav_file.readframes(wav_file.getnframes())
            self.assertEqual(saved_pcm, b''.join(emitted))

    def test_interrupted_stream_does_not_replace_previous_recording(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'latest.wav'
            previous_recording = b'previous recording'
            _ = destination.write_bytes(previous_recording)
            runtime = TtsRuntime(Settings(save_latest_wav=True, latest_wav_path=destination))
            runtime.engine.model = MagicMock(sample_rate=24_000)
            runtime.engine.voice_states = {'alba': object()}
            runtime.engine.model.generate_audio_stream.return_value = iter(
                [b'\x01\x02', b'\x03\x04']
            )

            with patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk):
                stream = runtime.stream_pcm('hello', 'alba')
                self.assertEqual(next(stream), b'\x01\x02')
                stream.close()

            self.assertEqual(destination.read_bytes(), previous_recording)

    def test_pipeline_saves_the_exact_stitched_pcm_stream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'latest.wav'
            runtime = TtsRuntime(
                Settings(
                    save_latest_wav=True,
                    latest_wav_path=destination,
                    pipeline_sentence_crossfade_seconds=0,
                    pipeline_sentence_pause_seconds=1 / 24_000,
                ),
            )
            runtime.engine.model = MagicMock(sample_rate=24_000)
            runtime.engine.voice_states = {'alba': object()}
            runtime.engine.model.generate_audio_stream.side_effect = lambda _voice, _text: iter(
                [b'\x01\x02\x03\x04'],
            )

            with patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk):
                emitted = b''.join(
                    runtime.stream_pipeline_pcm('First sentence. Second sentence.', 'alba'),
                )

            with wave.open(str(destination), 'rb') as wav_file:
                saved_pcm = wav_file.readframes(wav_file.getnframes())
            self.assertEqual(saved_pcm, emitted)
            self.assertEqual(emitted, b'\x01\x02\x03\x04\0\0\x01\x02\x03\x04')

    def test_interrupted_pipeline_does_not_replace_previous_recording(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'latest.wav'
            previous_recording = b'previous recording'
            _ = destination.write_bytes(previous_recording)
            runtime = TtsRuntime(
                Settings(
                    save_latest_wav=True,
                    latest_wav_path=destination,
                    pipeline_sentence_crossfade_seconds=0,
                    pipeline_speech_confirmation_seconds=1 / 24_000,
                ),
            )
            runtime.engine.model = MagicMock(sample_rate=24_000)
            runtime.engine.voice_states = {'alba': object()}
            runtime.engine.model.generate_audio_stream.side_effect = lambda _voice, _text: iter(
                [b'\x01\x02', b'\x03\x04'],
            )

            with patch.object(runtime.engine, '_pcm16_bytes', side_effect=lambda chunk: chunk):
                stream = runtime.stream_pipeline_pcm('First sentence. Second sentence.', 'alba')
                self.assertEqual(next(stream), b'\x01\x02')
                stream.close()

            self.assertEqual(destination.read_bytes(), previous_recording)


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
        self.assertIn(b'tts_request_duration_seconds', response.body)
        self.assertIn(b'tts_time_to_first_audio_seconds', response.body)
        self.assertIn(b'tts_pipeline_requests_total', response.body)
        self.assertIn(b'tts_pipeline_segments_total', response.body)
        self.assertTrue(response.headers['content-type'].startswith('text/plain;'))
