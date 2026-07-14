from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import sys
import time
from array import array
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


def _log_generated_response(response_text: str) -> None:
    LOGGER.info(
        'Assistant response generated',
        extra={'event_id': 'ID_assistant_response_generated', 'characters': len(response_text)},
    )
    LOGGER.debug(
        'Assistant response',
        extra={'event_id': 'ID_assistant_response', 'response': response_text},
    )


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


def _audio_frame_width(audio_format: AudioFormat) -> int:
    if audio_format.sample_width < 1 or audio_format.channels < 1:
        message = 'TTS returned an invalid audio format'
        raise RuntimeError(message)
    return audio_format.sample_width * audio_format.channels


def _duration_frames(duration_seconds: float, sample_rate: int) -> int:
    return int(duration_seconds * sample_rate + 0.5)


def _linear_fade_pcm16(  # noqa: C901
    pcm: bytes,
    audio_format: AudioFormat,
    *,
    start_frame: int,
    fade_frames: int,
    fade_in: bool,
) -> bytes:
    if not pcm or fade_frames <= 1 or start_frame >= fade_frames:
        return pcm
    if audio_format.sample_width != 2:  # noqa: PLR2004
        message = 'TTS sentence crossfade requires 16-bit PCM'
        raise RuntimeError(message)

    frame_width = _audio_frame_width(audio_format)
    if len(pcm) % frame_width:
        message = 'TTS returned a partial PCM frame'
        raise RuntimeError(message)

    samples = array('h')
    samples.frombytes(pcm)
    if sys.byteorder == 'big':
        samples.byteswap()
    frames_to_fade = min(len(pcm) // frame_width, fade_frames - start_frame)
    denominator = fade_frames - 1
    for frame_offset in range(frames_to_fade):
        fade_frame = start_frame + frame_offset
        numerator = fade_frame if fade_in else denominator - fade_frame
        sample_offset = frame_offset * audio_format.channels
        for channel in range(audio_format.channels):
            index = sample_offset + channel
            samples[index] = round(samples[index] * numerator / denominator)
    if sys.byteorder == 'big':
        samples.byteswap()
    return samples.tobytes()


class _SentencePcmBuffer:
    """Stream a sentence while retaining only the tail needed for its boundary fade."""

    def __init__(
        self,
        audio_format: AudioFormat,
        crossfade_seconds: float,
        *,
        fade_in: bool,
    ) -> None:
        self.audio_format = audio_format
        self.frame_width = _audio_frame_width(audio_format)
        self.crossfade_frames = _duration_frames(
            crossfade_seconds,
            audio_format.sample_rate,
        )
        self.fade_in = fade_in
        self.frames_emitted = 0
        self.buffer = bytearray()

    def append(self, pcm: bytes) -> bytes:
        self.buffer.extend(pcm)
        held_bytes = self.crossfade_frames * self.frame_width
        emit_bytes = max(0, len(self.buffer) - held_bytes)
        emit_bytes -= emit_bytes % self.frame_width
        if not emit_bytes:
            return b''
        output = bytes(self.buffer[:emit_bytes])
        del self.buffer[:emit_bytes]
        return self._apply_fade_in(output)

    def finish(self, *, fade_out: bool) -> bytes:
        if len(self.buffer) % self.frame_width:
            message = 'TTS returned a partial PCM frame'
            raise RuntimeError(message)
        output = self._apply_fade_in(bytes(self.buffer))
        self.buffer.clear()
        if not fade_out:
            return output
        return _linear_fade_pcm16(
            output,
            self.audio_format,
            start_frame=0,
            fade_frames=len(output) // self.frame_width,
            fade_in=False,
        )

    def _apply_fade_in(self, pcm: bytes) -> bytes:
        output = pcm
        if self.fade_in:
            output = _linear_fade_pcm16(
                pcm,
                self.audio_format,
                start_frame=self.frames_emitted,
                fade_frames=self.crossfade_frames,
                fade_in=True,
            )
        self.frames_emitted += len(pcm) // self.frame_width
        return output


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
            duration_seconds = time.perf_counter() - started
            metrics.CACHE_WARMS.labels(reason=reason, outcome=outcome).inc()
            metrics.CACHE_WARM_SECONDS.labels(reason=reason).observe(duration_seconds)
            LOGGER.info(
                'LLM cache warm completed',
                extra={
                    'event_id': 'ID_assistant_llm_cache_warm_completed',
                    'reason': reason,
                    'slot': slot,
                    'outcome': outcome,
                    'duration_seconds': duration_seconds,
                    'prompt_characters': len(prompt),
                },
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
        LOGGER.info(
            'Tool resolution started',
            extra={
                'event_id': 'ID_assistant_tool_resolution_started',
                'slot': slot,
                'tools': sorted(tools_by_name),
            },
        )
        for iteration in range(1, self.settings.maximum_tool_iterations + 1):
            prompt = build_prompt(transcript, selected_tools, history)
            raw_response = await self.llm.complete(
                prompt,
                slot,
                operation='tool_decision',
                maximum_tokens=self.settings.llm_tool_tokens,
            )
            decision = parse_tool_response(raw_response)
            LOGGER.debug(
                'Tool decision',
                extra={
                    'event_id': 'ID_assistant_tool_decision',
                    'iteration': iteration,
                    'slot': slot,
                    'decision': decision,
                    'raw_response': raw_response,
                },
            )
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
                LOGGER.exception(
                    'Assistant tool call failed',
                    extra={
                        'event_id': 'ID_assistant_tool_call_failed',
                        'tool': tool.name,
                        'source': tool.source,
                        'arguments': arguments,
                    },
                )
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
            _log_generated_response(response_text)
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
        _log_generated_response(cleaned)
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
        previous_segment_had_audio = False
        text = await queue.get()
        while text is not None:
            LOGGER.info(
                'TTS segment requested',
                extra={'event_id': 'ID_assistant_tts_segment_requested', 'characters': len(text)},
            )
            LOGGER.debug(
                'TTS segment',
                extra={'event_id': 'ID_assistant_tts_segment', 'text': text},
            )
            segment: _SentencePcmBuffer | None = None
            async for chunk, audio_format in self.tts.stream(text):
                if expected_format is None:
                    expected_format = audio_format
                elif audio_format != expected_format:
                    message = 'TTS audio format changed during the response'
                    raise RuntimeError(message)
                if not chunk:
                    continue
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
                if segment is None:
                    segment = _SentencePcmBuffer(
                        audio_format,
                        self.settings.tts_sentence_crossfade_seconds,
                        fade_in=previous_segment_had_audio,
                    )
                    if previous_segment_had_audio:
                        pause_frames = _duration_frames(
                            self.settings.tts_sentence_pause_seconds,
                            audio_format.sample_rate,
                        )
                        await self._send_audio_delta(
                            b'\0' * (pause_frames * segment.frame_width),
                            send,
                        )
                await self._send_audio_delta(segment.append(chunk), send)

            next_text = await queue.get()
            if segment is not None:
                await self._send_audio_delta(
                    segment.finish(fade_out=next_text is not None),
                    send,
                )
                previous_segment_had_audio = True
            text = next_text
        await send({'type': 'response.audio.done'})

    @staticmethod
    async def _send_audio_delta(pcm: bytes, send: EventSender) -> None:
        if not pcm:
            return
        metrics.AUDIO_OUTPUT_BYTES.inc(len(pcm))
        await send(
            {
                'type': 'response.audio.delta',
                'audio': base64.b64encode(pcm).decode('ascii'),
            },
        )

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
