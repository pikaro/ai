from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol

from assistant.src import metrics

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from assistant.src.config import Settings
    from assistant.src.tooling import ToolDefinition, ToolRegistry
    from assistant.src.upstream import AudioFormat, SlotPool

LOGGER = logging.getLogger('assistant.pipeline')
QWEN_ASSISTANT_PREFILL = '<think>\n\n</think>\n'
BASE_SYSTEM_PROMPT = (
    'You are a concise local voice assistant. Reply directly to the user in one or two short '
    'spoken sentences. Do not expose hidden reasoning or implementation details.'
)
EventSender = Callable[[dict[str, object]], Awaitable[None]]


class LlmProtocol(Protocol):
    async def warm_cache(self, prompt: str, slot: int) -> None: ...

    async def complete(
        self,
        prompt: str,
        slot: int,
        *,
        operation: str,
        maximum_tokens: int,
    ) -> str: ...

    def stream(self, prompt: str, slot: int) -> AsyncIterator[str]: ...


class TtsProtocol(Protocol):
    def stream(self, text: str) -> AsyncIterator[tuple[bytes, AudioFormat]]: ...


def build_prompt(
    transcript: str,
    tools: list[ToolDefinition],
    history: list[tuple[str, str]] | None = None,
    *,
    force_answer: bool = False,
) -> str:
    system_prompt = BASE_SYSTEM_PROMPT
    if tools:
        tool_data = [
            tool.prompt_description() for tool in sorted(tools, key=lambda item: item.name)
        ]
        system_prompt += (
            '\nTools are available below. If a tool is needed, respond with only a JSON object '
            'of the form {"tool":"name","arguments":{}}. If no tool is needed or a tool result '
            'already answers the question, respond with {"answer":"short spoken answer"}. '
            'Never invent a tool name or tool result.\nAvailable tools: '
            f'{json.dumps(tool_data, ensure_ascii=False, separators=(",", ":"))}'
        )
    if force_answer:
        system_prompt += '\nTool use is complete. Return only {"answer":"short spoken answer"}.'

    parts = [
        f'<|im_start|>system\n{system_prompt}\n<|im_end|>\n',
        f'<|im_start|>user\n{transcript.strip()}\n/no_think\n<|im_end|>\n',
    ]
    for role, content in history or []:
        parts.append(f'<|im_start|>{role}\n{content}\n<|im_end|>\n')
    parts.append(f'<|im_start|>assistant\n{QWEN_ASSISTANT_PREFILL}')
    return ''.join(parts)


def clean_response(text: str) -> str:
    cleaned = text.removeprefix(QWEN_ASSISTANT_PREFILL)
    for marker in ('<|im_end|>', '<|im_start|>', '<|endoftext|>', '</s>'):
        cleaned = cleaned.replace(marker, '')
    return cleaned.strip()


def completed_sentences(text: str) -> tuple[list[str], str]:
    sentences: list[str] = []
    start = 0
    for match in re.finditer(r'[.!?](?=\s|$)', text):
        sentence = text[start : match.end()].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    return sentences, text[start:].lstrip()


def parse_tool_response(text: str) -> dict[str, Any] | None:
    cleaned = clean_response(text).strip('` \n')
    match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
    if match is None:
        return None
    try:
        value = json.loads(match.group(0))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


class AssistantUtterance:
    def __init__(
        self,
        settings: Settings,
        slots: SlotPool,
        llm: LlmProtocol,
        tts: TtsProtocol,
        tools: ToolRegistry,
    ) -> None:
        self.settings = settings
        self.slots = slots
        self.llm = llm
        self.tts = tts
        self.tools = tools
        self.started_at = time.perf_counter()
        self.first_audio_at: float | None = None
        self.first_delta_at: float | None = None
        self.slot: int | None = None
        self.last_warmed_prompt: str | None = None
        self.last_tool_names: tuple[str, ...] = ()
        self.cache_warm_count = 0

    def note_audio(self, byte_count: int) -> None:
        if self.first_audio_at is None:
            self.first_audio_at = time.perf_counter()
        metrics.AUDIO_INPUT_BYTES.inc(byte_count)

    async def select_and_warm(
        self,
        transcript: str,
        *,
        reason: str,
    ) -> list[ToolDefinition]:
        selected_tools = await self.tools.select(transcript)
        tool_names = tuple(sorted(tool.name for tool in selected_tools))
        if self.last_warmed_prompt is not None and tool_names != self.last_tool_names:
            metrics.CACHE_TOOLSET_CHANGES.inc()
        self.last_tool_names = tool_names
        prompt = build_prompt(transcript, selected_tools)
        await self._warm(prompt, reason=reason)
        return selected_tools

    async def _warm(self, prompt: str, *, reason: str) -> None:
        if prompt == self.last_warmed_prompt:
            return
        slot = await self._ensure_slot()
        started = time.perf_counter()
        outcome = 'success'
        try:
            await self.llm.warm_cache(prompt, slot)
            self.last_warmed_prompt = prompt
            self.cache_warm_count += 1
            metrics.CACHE_PROMPT_CHARACTERS.observe(len(prompt))
        except Exception:
            outcome = 'error'
            raise
        finally:
            metrics.CACHE_WARMS.labels(reason=reason, outcome=outcome).inc()
            metrics.CACHE_WARM_SECONDS.labels(reason=reason).observe(
                time.perf_counter() - started,
            )

    async def generate(
        self,
        transcript: str,
        selected_tools: list[ToolDefinition],
        send: EventSender,
    ) -> None:
        final_at = time.perf_counter()
        metrics.TRANSCRIPT_CHARACTERS.observe(len(transcript))
        if self.first_audio_at is not None:
            metrics.PIPELINE_SECONDS.labels(stage='stt_final').observe(
                final_at - self.first_audio_at,
            )
        prompt = build_prompt(transcript, selected_tools)
        await self._warm(prompt, reason='final')
        await send(
            {
                'type': 'response.created',
                'response': {'model': self.settings.model_id, 'audio_format': 'pcm16'},
            },
        )

        if selected_tools:
            response_text = await self._resolve_tools(transcript, selected_tools)
            metrics.LLM_TIME_TO_FIRST_TOKEN.observe(time.perf_counter() - final_at)
            await self._speak_text(response_text, final_at, send)
        else:
            await self._stream_and_speak(prompt, final_at, send)

        metrics.PIPELINE_SECONDS.labels(stage='response_after_transcript').observe(
            time.perf_counter() - final_at,
        )
        metrics.PIPELINE_SECONDS.labels(stage='session').observe(
            time.perf_counter() - self.started_at,
        )
        await send({'type': 'response.done'})

    async def _resolve_tools(  # noqa: C901
        self,
        transcript: str,
        selected_tools: list[ToolDefinition],
    ) -> str:
        slot = await self._ensure_slot()
        tools_by_name = {tool.name: tool for tool in selected_tools}
        history: list[tuple[str, str]] = []
        for _ in range(self.settings.maximum_tool_iterations):
            prompt = build_prompt(transcript, selected_tools, history)
            raw_response = await self.llm.complete(
                prompt,
                slot,
                operation='tool_decision',
                maximum_tokens=self.settings.llm_tool_tokens,
            )
            decision = parse_tool_response(raw_response)
            if decision is None:
                return clean_response(raw_response)
            answer = decision.get('answer')
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
            tool_name = decision.get('tool') or decision.get('name')
            arguments = decision.get('arguments', {})
            if not isinstance(tool_name, str) or tool_name not in tools_by_name:
                return clean_response(raw_response)
            if not isinstance(arguments, dict):
                arguments = {}
            tool = tools_by_name[tool_name]
            try:
                result = await self.tools.call(tool, arguments)
            except Exception:
                LOGGER.exception('assistant tool call failed', extra={'tool': tool.name})
                result = json.dumps({'error': 'tool execution failed'})
            history.extend(
                (
                    ('assistant', f'{QWEN_ASSISTANT_PREFILL}{json.dumps(decision)}'),
                    ('tool', json.dumps({'tool': tool.name, 'result': result})),
                ),
            )

        final_prompt = build_prompt(
            transcript,
            selected_tools,
            history,
            force_answer=True,
        )
        raw_response = await self.llm.complete(
            final_prompt,
            slot,
            operation='tool_answer',
            maximum_tokens=self.settings.llm_tool_tokens,
        )
        decision = parse_tool_response(raw_response)
        if decision is not None and isinstance(decision.get('answer'), str):
            return str(decision['answer']).strip()
        return clean_response(raw_response)

    async def _stream_and_speak(  # noqa: C901
        self,
        prompt: str,
        final_at: float,
        send: EventSender,
    ) -> None:
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=2)
        tts_task = asyncio.create_task(self._tts_worker(queue, final_at, send))
        parts: list[str] = []
        pending_text = ''
        first_token = True
        try:
            slot = await self._ensure_slot()
            async for token in self.llm.stream(prompt, slot):
                if first_token:
                    metrics.LLM_TIME_TO_FIRST_TOKEN.observe(time.perf_counter() - final_at)
                    first_token = False
                parts.append(token)
                pending_text += token
                await send({'type': 'response.text.delta', 'delta': token})
                sentences, pending_text = completed_sentences(pending_text)
                for sentence in sentences:
                    await self._queue_or_raise(queue, sentence, tts_task)
            response_text = clean_response(''.join(parts))
            if not response_text:
                self._raise_empty_response()
            if pending_text.strip():
                await self._queue_or_raise(queue, pending_text.strip(), tts_task)
            await send({'type': 'response.text.done', 'text': response_text})
            metrics.LLM_OUTPUT_CHARACTERS.inc(len(response_text))
            await self._queue_or_raise(queue, None, tts_task)
            await self.release_slot()
            await tts_task
        except BaseException:
            _ = tts_task.cancel()
            _ = await asyncio.gather(tts_task, return_exceptions=True)
            raise

    async def _speak_text(  # noqa: C901
        self,
        response_text: str,
        final_at: float,
        send: EventSender,
    ) -> None:
        cleaned = clean_response(response_text)
        if not cleaned:
            self._raise_empty_response()
        await send({'type': 'response.text.delta', 'delta': cleaned})
        await send({'type': 'response.text.done', 'text': cleaned})
        metrics.LLM_OUTPUT_CHARACTERS.inc(len(cleaned))
        await self.release_slot()
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=2)
        tts_task = asyncio.create_task(self._tts_worker(queue, final_at, send))
        try:
            sentences, remainder = completed_sentences(cleaned)
            for sentence in sentences:
                await self._queue_or_raise(queue, sentence, tts_task)
            if remainder:
                await self._queue_or_raise(queue, remainder, tts_task)
            if not sentences and not remainder:
                await self._queue_or_raise(queue, cleaned, tts_task)
            await self._queue_or_raise(queue, None, tts_task)
            await tts_task
        except BaseException:
            _ = tts_task.cancel()
            _ = await asyncio.gather(tts_task, return_exceptions=True)
            raise

    async def _tts_worker(  # noqa: C901
        self,
        queue: asyncio.Queue[str | None],
        final_at: float,
        send: EventSender,
    ) -> None:
        expected_format: AudioFormat | None = None
        first_audio = True
        while (text := await queue.get()) is not None:
            async for chunk, audio_format in self.tts.stream(text):
                if expected_format is None:
                    expected_format = audio_format
                elif audio_format != expected_format:
                    message = 'TTS audio format changed during the response'
                    raise RuntimeError(message)
                if first_audio:
                    first_audio = False
                    metrics.TTS_TIME_TO_FIRST_AUDIO.observe(time.perf_counter() - final_at)
                    await send(
                        {
                            'type': 'response.audio.started',
                            'format': 'pcm16',
                            'sample_rate': audio_format.sample_rate,
                            'sample_width': audio_format.sample_width,
                            'channels': audio_format.channels,
                        },
                    )
                metrics.AUDIO_OUTPUT_BYTES.inc(len(chunk))
                await send(
                    {
                        'type': 'response.audio.delta',
                        'audio': base64.b64encode(chunk).decode('ascii'),
                    },
                )
        await send({'type': 'response.audio.done'})

    @staticmethod
    def _raise_empty_response() -> None:
        message = 'LLM produced an empty response'
        raise RuntimeError(message)

    @staticmethod
    async def _queue_or_raise(
        queue: asyncio.Queue[str | None],
        item: str | None,
        consumer: asyncio.Task[None],
    ) -> None:
        if consumer.done():
            await consumer
        put_task = asyncio.create_task(queue.put(item))
        done, _ = await asyncio.wait((put_task, consumer), return_when=asyncio.FIRST_COMPLETED)
        if consumer in done:
            _ = put_task.cancel()
            _ = await asyncio.gather(put_task, return_exceptions=True)
            await consumer
        await put_task

    async def _ensure_slot(self) -> int:
        if self.slot is None:
            self.slot = await self.slots.acquire()
        return self.slot

    async def release_slot(self) -> None:
        if self.slot is None:
            return
        slot = self.slot
        self.slot = None
        await self.slots.release(slot)

    async def close(self) -> None:
        metrics.CACHE_WARMS_PER_SESSION.observe(self.cache_warm_count)
        await self.release_slot()
