from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from assistant.src import metrics

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from assistant.src.config import Settings


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_rate: int
    sample_width: int
    channels: int


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
        try:
            response = await self.client.post(
                f'{self.settings.llm_base_url.rstrip("/")}/completion',
                json=self._payload(prompt, slot, maximum_tokens=maximum_tokens, stream=False),
            )
            _ = response.raise_for_status()
            body = response.json()
            self._observe_timings(body)
            content = body.get('content', '')
            return content if isinstance(content, str) else ''
        except Exception:
            outcome = 'error'
            raise
        finally:
            metrics.LLM_REQUESTS.labels(operation=operation, outcome=outcome).inc()
            metrics.LLM_REQUEST_SECONDS.labels(operation=operation).observe(
                time.perf_counter() - started,
            )

    async def stream(self, prompt: str, slot: int) -> AsyncGenerator[str]:
        started = time.perf_counter()
        outcome = 'success'
        final_data: dict[str, Any] = {}
        try:
            async with self.client.stream(
                'POST',
                f'{self.settings.llm_base_url.rstrip("/")}/completion',
                json=self._payload(
                    prompt,
                    slot,
                    maximum_tokens=self.settings.llm_max_tokens,
                    stream=True,
                ),
            ) as response:
                _ = response.raise_for_status()
                async for line in response.aiter_lines():
                    data = _parse_sse_json(line)
                    if data is None:
                        continue
                    final_data = data
                    content = data.get('content')
                    if isinstance(content, str) and content:
                        yield content
        except Exception:
            outcome = 'error'
            raise
        finally:
            self._observe_timings(final_data)
            metrics.LLM_REQUESTS.labels(operation='generation', outcome=outcome).inc()
            metrics.LLM_REQUEST_SECONDS.labels(operation='generation').observe(
                time.perf_counter() - started,
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
        values = {
            'prompt': timings.get('prompt_n') or data.get('tokens_evaluated'),
            'cached': data.get('tokens_cached'),
            'generated': timings.get('predicted_n') or data.get('tokens_predicted'),
        }
        for kind, value in values.items():
            if isinstance(value, (int, float)) and value >= 0:
                metrics.LLM_TOKENS.labels(kind=kind).observe(value)
        for phase, key in (('prompt', 'prompt_per_second'), ('decode', 'predicted_per_second')):
            value = timings.get(key)
            if isinstance(value, (int, float)) and value >= 0:
                metrics.LLM_TOKENS_PER_SECOND.labels(phase=phase).observe(value)
        prompt_tokens = values['prompt']
        cached_tokens = values['cached']
        if (
            isinstance(prompt_tokens, (int, float))
            and prompt_tokens > 0
            and isinstance(cached_tokens, (int, float))
        ):
            metrics.LLM_CACHE_REUSE_RATIO.observe(min(1.0, cached_tokens / prompt_tokens))


class TtsClient:
    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    async def stream(  # noqa: C901
        self,
        text: str,
    ) -> AsyncGenerator[tuple[bytes, AudioFormat]]:
        started = time.perf_counter()
        outcome = 'success'
        output_bytes = 0
        audio_format: AudioFormat | None = None
        try:
            async with self.client.stream(
                'POST',
                f'{self.settings.tts_base_url.rstrip("/")}/v1/audio/speech',
                json={
                    'model': self.settings.tts_model,
                    'input': text,
                    'response_format': 'pcm',
                },
            ) as response:
                _ = response.raise_for_status()
                audio_format = AudioFormat(
                    sample_rate=int(response.headers.get('X-Audio-Sample-Rate', '24000')),
                    sample_width=int(response.headers.get('X-Audio-Sample-Width', '2')),
                    channels=int(response.headers.get('X-Audio-Channels', '1')),
                )
                async for chunk in response.aiter_bytes():
                    if chunk:
                        output_bytes += len(chunk)
                        yield chunk, audio_format
        except Exception:
            outcome = 'error'
            raise
        finally:
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
        except httpx.HTTPError:
            ready = False
        metrics.UPSTREAM_READY.labels(service=name).set(int(ready))
        return name, ready

    return dict(await asyncio.gather(*(check(name, url) for name, url in endpoints.items())))
