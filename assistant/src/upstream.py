from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import ValidationError
from websockets.asyncio.client import ClientConnection, connect

from assistant.src import metrics
from service_contracts.tts import (
    PIPELINE_OUTPUT_ADAPTER,
    PipelineError,
    PipelineSessionReady,
    PipelineSessionRequest,
    PipelineTextDelta,
    PipelineTextDone,
)

LOGGER = logging.getLogger('assistant.upstream')

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from assistant.src.config import Settings


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_rate: int
    sample_width: int
    channels: int


def _require_audio_format(audio_format: AudioFormat | None) -> AudioFormat:
    if audio_format is not None:
        return audio_format
    message = 'TTS pipeline sent audio before session metadata'
    raise RuntimeError(message)


def _untyped_pipeline_event(payload: dict[object, object]) -> tuple[str | None, AudioFormat | None]:
    event_type = payload.get('type')
    if event_type == 'session.ready':
        return (
            'session.ready',
            AudioFormat(
                sample_rate=_integer_payload_field(payload, 'sample_rate'),
                sample_width=_integer_payload_field(payload, 'sample_width'),
                channels=_integer_payload_field(payload, 'channels'),
            ),
        )
    if event_type == 'error':
        error_message = str(payload.get('message') or 'TTS pipeline failed')
        raise RuntimeError(error_message)
    return (event_type if isinstance(event_type, str) else None), None


def _integer_payload_field(payload: dict[object, object], name: str) -> int:
    value = payload[name]
    if not isinstance(value, (str, int, float)):
        message = f'pipeline event field {name!r} is not an integer'
        raise TypeError(message)
    return int(value)


def _pipeline_event(message: str) -> tuple[str | None, AudioFormat | None]:
    payload = json.loads(message)
    if not isinstance(payload, dict):
        error_message = 'TTS pipeline returned an invalid event'
        raise TypeError(error_message)
    try:
        event = PIPELINE_OUTPUT_ADAPTER.validate_python(payload)
    except ValidationError:
        return _untyped_pipeline_event(payload)
    if isinstance(event, PipelineSessionReady):
        return (
            event.type,
            AudioFormat(
                sample_rate=event.sample_rate,
                sample_width=event.sample_width,
                channels=event.channels,
            ),
        )
    if isinstance(event, PipelineError):
        raise RuntimeError(event.message or 'TTS pipeline failed')  # noqa: TRY004
    return event.type, None


class SlotPool:
    """Lease configured llama.cpp slots for the lifetime of an LLM utterance."""

    def __init__(self, slots: tuple[int, ...]) -> None:
        self._available = deque(slots)
        self._condition = asyncio.Condition()

    async def acquire(self) -> int:
        started = time.perf_counter()
        metrics.SLOT_WAITERS.inc()
        try:
            async with self._condition:
                _ = await self._condition.wait_for(lambda: bool(self._available))
                slot = self._available.popleft()
        finally:
            metrics.SLOT_WAITERS.dec()
        metrics.SLOTS_IN_USE.inc()
        metrics.SLOT_WAIT_SECONDS.observe(time.perf_counter() - started)
        return slot

    async def release(self, slot: int) -> None:
        async with self._condition:
            self._available.append(slot)
            self._condition.notify(1)
        metrics.SLOTS_IN_USE.dec()


def _parse_sse_json(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith('data:'):
        stripped = stripped[5:].strip()
    if stripped == '[DONE]':
        return None
    try:
        value = json.loads(stripped)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


class LlmClient:
    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    async def warm_cache(self, prompt: str, slot: int) -> None:
        try:
            _ = await self.complete(
                prompt,
                slot,
                operation='cache_warm',
                maximum_tokens=self.settings.llm_cache_warm_tokens,
            )
        except httpx.HTTPStatusError:
            if self.settings.llm_cache_warm_tokens != 0:
                raise
            _ = await self.complete(
                prompt,
                slot,
                operation='cache_warm_fallback',
                maximum_tokens=self.settings.llm_cache_warm_fallback_tokens,
            )

    async def complete(
        self,
        prompt: str,
        slot: int,
        *,
        operation: str,
        maximum_tokens: int,
    ) -> str:
        started = time.perf_counter()
        outcome = 'success'
        error_type: str | None = None
        url = f'{self.settings.llm_base_url.rstrip("/")}/completion'
        payload = self._payload(prompt, slot, maximum_tokens=maximum_tokens, stream=False)
        LOGGER.info(
            'LLM request started',
            extra={
                'event_id': 'ID_assistant_llm_request_started',
                'operation': operation,
                'slot': slot,
                'stream': False,
                'prompt_characters': len(prompt),
                'maximum_tokens': maximum_tokens,
            },
        )
        LOGGER.debug(
            'LLM request',
            extra={
                'event_id': 'ID_assistant_llm_request',
                'operation': operation,
                'url': url,
                'payload': payload,
            },
        )
        try:
            response = await self.client.post(url, json=payload)
            _ = response.raise_for_status()
            body = response.json()
            self._observe_timings(body)
            LOGGER.debug(
                'LLM response',
                extra={
                    'event_id': 'ID_assistant_llm_response',
                    'operation': operation,
                    'slot': slot,
                    'response': body,
                },
            )
            content = body.get('content', '')
            return content if isinstance(content, str) else ''
        except Exception as error:
            outcome = 'error'
            error_type = type(error).__name__
            raise
        finally:
            duration_seconds = time.perf_counter() - started
            metrics.LLM_REQUESTS.labels(operation=operation, outcome=outcome).inc()
            metrics.LLM_REQUEST_SECONDS.labels(operation=operation).observe(duration_seconds)
            LOGGER.info(
                'LLM request completed',
                extra={
                    'event_id': 'ID_assistant_llm_request_completed',
                    'operation': operation,
                    'slot': slot,
                    'stream': False,
                    'outcome': outcome,
                    'duration_seconds': duration_seconds,
                    'error_type': error_type,
                },
            )

    async def stream(  # noqa: C901
        self,
        prompt: str,
        slot: int,
        *,
        operation: str = 'generation',
        maximum_tokens: int | None = None,
    ) -> AsyncGenerator[str]:
        started = time.perf_counter()
        outcome = 'success'
        error_type: str | None = None
        final_data: dict[str, Any] = {}
        first_token = True
        token_limit = self.settings.llm_max_tokens if maximum_tokens is None else maximum_tokens
        url = f'{self.settings.llm_base_url.rstrip("/")}/completion'
        payload = self._payload(
            prompt,
            slot,
            maximum_tokens=token_limit,
            stream=True,
        )
        LOGGER.info(
            'LLM request started',
            extra={
                'event_id': 'ID_assistant_llm_request_started',
                'operation': operation,
                'slot': slot,
                'stream': True,
                'prompt_characters': len(prompt),
                'maximum_tokens': token_limit,
            },
        )
        LOGGER.debug(
            'LLM request',
            extra={
                'event_id': 'ID_assistant_llm_request',
                'operation': operation,
                'url': url,
                'payload': payload,
            },
        )
        try:
            async with self.client.stream(
                'POST',
                url,
                json=payload,
            ) as response:
                _ = response.raise_for_status()
                async for line in response.aiter_lines():
                    data = _parse_sse_json(line)
                    if data is None:
                        continue
                    final_data = data
                    content = data.get('content')
                    if isinstance(content, str) and content:
                        if first_token:
                            first_token = False
                            LOGGER.info(
                                'LLM first token received',
                                extra={
                                    'event_id': 'ID_assistant_llm_first_token_received',
                                    'operation': operation,
                                    'slot': slot,
                                    'duration_seconds': time.perf_counter() - started,
                                },
                            )
                        yield content
        except Exception as error:
            outcome = 'error'
            error_type = type(error).__name__
            raise
        finally:
            duration_seconds = time.perf_counter() - started
            self._observe_timings(final_data)
            LOGGER.debug(
                'LLM response',
                extra={
                    'event_id': 'ID_assistant_llm_response',
                    'operation': operation,
                    'slot': slot,
                    'response': final_data,
                },
            )
            metrics.LLM_REQUESTS.labels(operation=operation, outcome=outcome).inc()
            metrics.LLM_REQUEST_SECONDS.labels(operation=operation).observe(duration_seconds)
            LOGGER.info(
                'LLM request completed',
                extra={
                    'event_id': 'ID_assistant_llm_request_completed',
                    'operation': operation,
                    'slot': slot,
                    'stream': True,
                    'outcome': outcome,
                    'duration_seconds': duration_seconds,
                    'error_type': error_type,
                },
            )

    def _payload(
        self,
        prompt: str,
        slot: int,
        *,
        maximum_tokens: int,
        stream: bool,
    ) -> dict[str, object]:
        return {
            'prompt': prompt,
            'n_predict': maximum_tokens,
            'id_slot': slot,
            'cache_prompt': True,
            'temperature': self.settings.llm_temperature,
            'top_p': 1.0,
            'stream': stream,
            'timings_per_token': True,
            'stop': ['<|im_end|>', '<|im_start|>', '<|endoftext|>', '</s>'],
            'repeat_penalty': 1.1,
        }

    @staticmethod
    def _observe_timings(data: dict[str, Any]) -> None:  # noqa: C901
        timings = data.get('timings')
        if not isinstance(timings, dict):
            timings = {}
        evaluated_tokens = timings.get('prompt_n')
        if not isinstance(evaluated_tokens, (int, float)):
            evaluated_tokens = data.get('tokens_evaluated')
        cached_tokens = timings.get('cache_n')
        if not isinstance(cached_tokens, (int, float)):
            cached_tokens = data.get('tokens_cached')
        generated_tokens = timings.get('predicted_n')
        if not isinstance(generated_tokens, (int, float)):
            generated_tokens = data.get('tokens_predicted')
        prompt_tokens = evaluated_tokens
        if isinstance(evaluated_tokens, (int, float)) and isinstance(
            cached_tokens,
            (int, float),
        ):
            prompt_tokens = evaluated_tokens + cached_tokens
        values = {
            'prompt': prompt_tokens,
            'evaluated': evaluated_tokens,
            'cached': cached_tokens,
            'generated': generated_tokens,
        }
        for kind, value in values.items():
            if isinstance(value, (int, float)) and value >= 0:
                metrics.LLM_TOKENS.labels(kind=kind).observe(value)
        for phase, key in (('prompt', 'prompt_per_second'), ('decode', 'predicted_per_second')):
            value = timings.get(key)
            if isinstance(value, (int, float)) and value >= 0:
                metrics.LLM_TOKENS_PER_SECOND.labels(phase=phase).observe(value)
        for phase, key in (('prompt', 'prompt_ms'), ('decode', 'predicted_ms')):
            value = timings.get(key)
            if isinstance(value, (int, float)) and value >= 0:
                metrics.LLM_SERVER_SECONDS.labels(phase=phase).observe(value / 1_000)
        if (
            isinstance(prompt_tokens, (int, float))
            and prompt_tokens > 0
            and isinstance(cached_tokens, (int, float))
            and cached_tokens >= 0
        ):
            metrics.LLM_CACHE_REUSE_RATIO.observe(min(1.0, cached_tokens / prompt_tokens))


class TtsClient:
    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    async def stream(  # noqa: C901
        self,
        text_stream: AsyncIterator[str],
        *,
        voice: str | None = None,
    ) -> AsyncGenerator[tuple[bytes, AudioFormat]]:
        started = time.perf_counter()
        outcome = 'success'
        output_bytes = 0
        audio_format: AudioFormat | None = None
        sender: asyncio.Task[None] | None = None
        response_done = False
        try:
            async with connect(
                self.pipeline_websocket_url,
                open_timeout=self.settings.connect_timeout_seconds,
                close_timeout=self.settings.connect_timeout_seconds,
                max_size=self.settings.maximum_websocket_message_bytes,
                max_queue=2,
            ) as websocket:
                await websocket.send(
                    PipelineSessionRequest(
                        type='session.start',
                        model=self.settings.tts_model,
                        voice=voice,
                    ).model_dump_json(exclude_none=True),
                )
                sender = asyncio.create_task(self._send_text(websocket, text_stream))
                while not response_done:
                    message = await self._receive_or_sender(websocket, sender)
                    if isinstance(message, bytes):
                        audio_format = _require_audio_format(audio_format)
                        if message:
                            output_bytes += len(message)
                            yield message, audio_format
                        continue

                    event_type, ready_format = _pipeline_event(message)
                    if event_type == 'session.ready':
                        audio_format = ready_format
                    elif event_type == 'response.audio.done':
                        response_done = True

                await sender
        except asyncio.CancelledError:
            outcome = 'cancelled'
            raise
        except Exception:
            outcome = 'error'
            raise
        finally:
            if sender is not None:
                if not sender.done():
                    _ = sender.cancel()
                _ = await asyncio.gather(sender, return_exceptions=True)
            wall_seconds = time.perf_counter() - started
            if audio_format is not None:
                bytes_per_second = (
                    audio_format.sample_rate * audio_format.sample_width * audio_format.channels
                )
                audio_seconds = output_bytes / bytes_per_second if bytes_per_second else 0
                metrics.TTS_AUDIO_SECONDS.observe(audio_seconds)
                if audio_seconds > 0:
                    metrics.TTS_REALTIME_FACTOR.observe(wall_seconds / audio_seconds)
            metrics.TTS_REQUESTS.labels(outcome=outcome).inc()
            metrics.TTS_REQUEST_SECONDS.observe(wall_seconds)

    @property
    def pipeline_websocket_url(self) -> str:
        parsed = urlsplit(self.settings.tts_base_url.rstrip('/'))
        scheme = 'wss' if parsed.scheme == 'https' else 'ws'
        return urlunsplit(
            (scheme, parsed.netloc, '/v1/audio/speech/pipeline', '', ''),
        )

    @staticmethod
    async def _send_text(
        websocket: ClientConnection,
        text_stream: AsyncIterator[str],
    ) -> None:
        try:
            async for delta in text_stream:
                if delta:
                    await websocket.send(
                        PipelineTextDelta(
                            type='input_text.delta',
                            delta=delta,
                        ).model_dump_json(),
                    )
            await websocket.send(
                PipelineTextDone(type='input_text.done').model_dump_json(),
            )
        except BaseException:
            with suppress(Exception):
                await websocket.close(code=1011)
            raise

    @staticmethod
    async def _receive_or_sender(
        websocket: ClientConnection,
        sender: asyncio.Task[None],
    ) -> str | bytes:
        receive = asyncio.create_task(websocket.recv())
        done, _ = await asyncio.wait((receive, sender), return_when=asyncio.FIRST_COMPLETED)
        if sender in done:
            try:
                await sender
            except BaseException:
                _ = receive.cancel()
                _ = await asyncio.gather(receive, return_exceptions=True)
                raise
        return await receive


async def upstream_health(
    client: httpx.AsyncClient,
    settings: Settings,
) -> dict[str, bool]:
    endpoints = {
        'llm': f'{settings.llm_base_url.rstrip("/")}/health',
        'stt': f'{settings.stt_base_url.rstrip("/")}/health/ready',
        'tts': f'{settings.tts_base_url.rstrip("/")}/health/ready',
    }

    async def check(name: str, url: str) -> tuple[str, bool]:
        try:
            response = await client.get(url, timeout=settings.health_timeout_seconds)
            ready = response.status_code == 200  # noqa: PLR2004
            if not ready:
                LOGGER.warning(
                    'Upstream health check failed',
                    extra={
                        'event_id': 'ID_assistant_upstream_health_check_failed',
                        'service': name,
                        'status_code': response.status_code,
                    },
                )
        except httpx.HTTPError as error:
            ready = False
            LOGGER.warning(
                'Upstream health check failed',
                extra={
                    'event_id': 'ID_assistant_upstream_health_check_failed',
                    'service': name,
                    'error_type': type(error).__name__,
                },
            )
        metrics.UPSTREAM_READY.labels(service=name).set(int(ready))
        return name, ready

    return dict(await asyncio.gather(*(check(name, url) for name, url in endpoints.items())))
