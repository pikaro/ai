from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from assistant.src import metrics
from assistant.src.multi_voice import system_prompt_with_multi_voice

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from assistant.src.config import Settings
    from assistant.src.tooling import ToolDefinition, ToolRegistry
    from assistant.src.upstream import AudioFormat, SlotPool

LOGGER = logging.getLogger('assistant.pipeline')
BASE_SYSTEM_PROMPT = """
You are a helpful voice assistant.

Reply directly in natural spoken language. Never use Emoji. Write for listening rather than reading.
Use conversational sentences and avoid numbered lists unless they make the answer easier to follow
aloud. Do not refer to formatting, markdown, bullet points, links, or text on the screen unless
the user explicitly asks about them.

Prefer to be concise, and usually answer in one or two short sentences. If the task demands it,
such as for a longer explanation or prose, you may respond more freely.

Begin your response with a short sentence or subclause so speech can start as quickly as possible.

If the user's request is ambiguous or missing critical information, ask one brief clarifying
question instead of guessing. If there is an obvious next step that would help the user, briefly
suggest it in one sentence.

If you are unsure, say so briefly. Do not invent facts or pretend certainty.
""".strip()
EventSender = Callable[[dict[str, object]], Awaitable[None]]


class SystemPromptFile:
    """Keep the system prompt in a user-editable file and reload stable revisions."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._signature: tuple[int, int, int, int, int] | None = None
        self._prompt: str | None = None
        self._write_default_if_missing()
        _ = self.read()

    def read(self) -> str:
        try:
            revision = self._read_revision()
        except (OSError, UnicodeError):
            if self._prompt is None:
                raise
            LOGGER.exception(
                'System prompt reload failed',
                extra={
                    'event_id': 'ID_assistant_system_prompt_reload_failed',
                    'path': str(self.path),
                },
            )
            return self._prompt

        if revision is None:
            return self._defer_reload()

        prompt, signature = revision
        if signature != self._signature or self._prompt is None:
            self._prompt = prompt
            self._signature = signature
            LOGGER.info(
                'System prompt loaded',
                extra={
                    'event_id': 'ID_assistant_system_prompt_loaded',
                    'path': str(self.path),
                    'characters': len(prompt),
                },
            )
        return self._prompt

    def write(self, prompt: str) -> str:
        """Atomically replace the prompt file and load the resulting revision."""
        temporary_path: Path | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=self.path.parent,
                prefix=f'.{self.path.name}.',
                suffix='.tmp',
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                _ = temporary.write(f'{prompt.strip()}\n')
                temporary.flush()
                os.fsync(temporary.fileno())
            _ = temporary_path.replace(self.path)
        except (OSError, UnicodeError):
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)
            raise
        return self.read()

    def _read_revision(
        self,
    ) -> tuple[str, tuple[int, int, int, int, int]] | None:
        signature = self._current_signature()
        if signature == self._signature and self._prompt is not None:
            return self._prompt, signature
        return self._read_stable_revision()

    def _current_signature(self) -> tuple[int, int, int, int, int]:
        try:
            status = self.path.stat()
        except FileNotFoundError:
            self._write_default_if_missing()
            status = self.path.stat()
        return self._file_signature(status)

    def _defer_reload(self) -> str:
        if self._prompt is None:
            message = f'system prompt changed while being read: {self.path}'
            raise RuntimeError(message)
        LOGGER.warning(
            'System prompt changed while being read; retaining previous revision',
            extra={
                'event_id': 'ID_assistant_system_prompt_reload_deferred',
                'path': str(self.path),
            },
        )
        return self._prompt

    def _read_stable_revision(
        self,
    ) -> tuple[str, tuple[int, int, int, int, int]] | None:
        for _attempt in range(2):
            try:
                before = self.path.stat()
                prompt = self.path.read_text(encoding='utf-8').strip()
                after = self.path.stat()
            except FileNotFoundError:
                self._write_default_if_missing()
                continue
            before_signature = self._file_signature(before)
            signature = self._file_signature(after)
            if before_signature == signature:
                return prompt, signature
        return None

    def _write_default_if_missing(self) -> None:
        try:
            with self.path.open('x', encoding='utf-8') as prompt_file:
                _ = prompt_file.write(f'{BASE_SYSTEM_PROMPT}\n')
        except FileExistsError:
            return

    @staticmethod
    def _file_signature(status: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            status.st_dev,
            status.st_ino,
            status.st_size,
            status.st_mtime_ns,
            status.st_ctime_ns,
        )


def _log_generated_response(response_text: str) -> None:
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            'Assistant response generated',
            extra={
                'event_id': 'ID_assistant_response_generated',
                'characters': len(response_text),
                'response': response_text,
            },
        )
    else:
        LOGGER.info(
            'Assistant response generated',
            extra={
                'event_id': 'ID_assistant_response_generated',
                'characters': len(response_text),
            },
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

    def stream(
        self,
        prompt: str,
        slot: int,
        *,
        operation: str = 'generation',
        maximum_tokens: int | None = None,
    ) -> AsyncIterator[str]: ...


class TtsProtocol(Protocol):
    def stream(
        self,
        text_stream: AsyncIterator[str],
        *,
        voice: str | None = None,
        multi_voice: bool = True,
    ) -> AsyncIterator[tuple[bytes, AudioFormat]]: ...


async def warm_llm_cache(
    llm: LlmProtocol,
    prompt: str,
    slot: int,
    *,
    reason: str,
) -> None:
    """Warm one llama.cpp slot and record the shared cache-warm telemetry."""
    started = time.perf_counter()
    outcome = 'success'
    try:
        await llm.warm_cache(prompt, slot)
        metrics.CACHE_PROMPT_CHARACTERS.observe(len(prompt))
    except Exception:
        outcome = 'error'
        raise
    finally:
        duration_seconds = time.perf_counter() - started
        metrics.CACHE_WARMS.labels(reason=reason, outcome=outcome).inc()
        metrics.CACHE_WARM_SECONDS.labels(reason=reason).observe(duration_seconds)
        LOGGER.debug(
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


def build_prompt_prefix(
    transcript: str,
    tools: list[ToolDefinition],
    *,
    system_prompt: str = BASE_SYSTEM_PROMPT,
) -> str:
    """Build only the appendable prompt prefix suitable for cache warming."""
    complete_system_prompt = system_prompt
    if tools:
        tool_data = [
            tool.prompt_description() for tool in sorted(tools, key=lambda item: item.name)
        ]
        complete_system_prompt += (
            '\nTools are available below. If a tool is needed, respond with only a JSON object '
            'of the form {"tool":"name","arguments":{}}. Otherwise, reply directly with the '
            'short spoken answer and do not wrap it in JSON. Never invent a tool name or tool '
            'result. Tool schemas may list default values; omit arguments when those defaults '
            'match the request and include only arguments needed to override them.\n'
            'Available tools: '
            f'{json.dumps(tool_data, ensure_ascii=False, separators=(",", ":"))}'
        )
    return ''.join(
        (
            f'<|im_start|>system\n{complete_system_prompt}\n<|im_end|>\n',
            f'<|im_start|>user\n{transcript.strip()}',
        )
    )


def build_prompt(
    transcript: str,
    tools: list[ToolDefinition],
    history: list[tuple[str, str]] | None = None,
    *,
    system_prompt: str = BASE_SYSTEM_PROMPT,
) -> str:
    parts = [f'{build_prompt_prefix(transcript, tools, system_prompt=system_prompt)}\n<|im_end|>\n']
    for role, content in history or []:
        parts.append(f'<|im_start|>{role}\n{content}\n<|im_end|>\n')
    parts.append('<|im_start|>assistant\n')
    return ''.join(parts)


def clean_response(text: str) -> str:
    cleaned = text
    for marker in ('<|im_end|>', '<|im_start|>', '<|endoftext|>', '</s>'):
        cleaned = cleaned.replace(marker, '')
    return cleaned.strip()


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
    def __init__(  # noqa: PLR0913
        self,
        settings: Settings,
        slots: SlotPool,
        llm: LlmProtocol,
        tts: TtsProtocol,
        tools: ToolRegistry,
        system_prompt: SystemPromptFile,
    ) -> None:
        self.settings = settings
        self.slots = slots
        self.llm = llm
        self.tts = tts
        self.tools = tools
        self.system_prompt = system_prompt
        self.started_at = time.perf_counter()
        self.first_audio_at: float | None = None
        self.first_delta_at: float | None = None
        self.final_transcript_at: float | None = None
        self.voice: str | None = None
        self.multi_voice_enabled = True
        self.slot: int | None = None
        self.last_warmed_prompt: str | None = None
        self.last_tool_names: tuple[str, ...] = ()
        self.cache_warm_count = 0
        self._pending_warm_transcript: str | None = None
        self._last_scheduled_transcript = ''
        self._last_warm_started_at: float | None = None
        self._warm_worker: asyncio.Task[None] | None = None
        self._warm_worker_state = 'idle'
        self._cache_finalizing = False

    def select_voice(self, voice: str) -> None:
        self.voice = voice
        LOGGER.info(
            'Assistant voice selected',
            extra={
                'event_id': 'ID_assistant_voice_selected',
                'voice': voice,
            },
        )

    def set_multi_voice(self, *, enabled: bool) -> None:
        self.multi_voice_enabled = enabled
        LOGGER.info(
            'Assistant multi-voice mode selected',
            extra={
                'event_id': 'ID_assistant_multi_voice_selected',
                'enabled': enabled,
            },
        )

    def note_audio(self, byte_count: int) -> None:
        if self.first_audio_at is None:
            self.first_audio_at = time.perf_counter()
        metrics.AUDIO_INPUT_BYTES.inc(byte_count)

    def note_final_transcript(self, received_at: float | None = None) -> None:
        if self.final_transcript_at is not None:
            return
        self.final_transcript_at = received_at if received_at is not None else time.perf_counter()
        if self.first_audio_at is not None:
            metrics.PIPELINE_SECONDS.labels(stage='stt_final').observe(
                self.final_transcript_at - self.first_audio_at,
            )

    def _active_system_prompt(self) -> str:
        if not self.multi_voice_enabled:
            return self.system_prompt.read()
        return system_prompt_with_multi_voice(
            self.system_prompt.read(),
            self.settings.multi_voice,
        )

    def schedule_cache_warm(self, transcript: str) -> str:
        """Schedule a stable transcript without queueing every intermediate revision."""
        if not self.settings.llm_cache_warm_enabled:
            disposition = 'disabled'
            metrics.CACHE_WARM_UPDATES.labels(disposition=disposition).inc()
            return disposition
        if self._cache_finalizing:
            disposition = 'finalizing'
            metrics.CACHE_WARM_UPDATES.labels(disposition=disposition).inc()
            return disposition
        if (
            self._last_scheduled_transcript
            and transcript.startswith(self._last_scheduled_transcript)
            and len(transcript) - len(self._last_scheduled_transcript)
            < self.settings.llm_cache_warm_min_new_characters
        ):
            disposition = 'too_small'
            metrics.CACHE_WARM_UPDATES.labels(disposition=disposition).inc()
            return disposition

        disposition = 'coalesced' if self._pending_warm_transcript is not None else 'scheduled'
        self._pending_warm_transcript = transcript
        self._last_scheduled_transcript = transcript
        metrics.CACHE_WARM_UPDATES.labels(disposition=disposition).inc()
        if self._warm_worker is None:
            self._warm_worker = asyncio.create_task(self._run_cache_warms())
        return disposition

    async def finalize_cache_warming(self, _transcript: str) -> list[ToolDefinition]:
        """Discard pending revisions, drain one active warm, and load available tools."""
        started = time.perf_counter()
        worker_state = self._warm_worker_state
        pending_update = self._pending_warm_transcript is not None
        await self._stop_cache_warming()
        duration_seconds = time.perf_counter() - started
        metrics.CACHE_WARM_FINAL_WAIT_SECONDS.observe(duration_seconds)
        LOGGER.debug(
            'Final transcript cache barrier completed',
            extra={
                'event_id': 'ID_assistant_llm_cache_final_barrier_completed',
                'duration_seconds': duration_seconds,
                'waited_for_active_warm': worker_state == 'active',
                'cancelled_interval_wait': worker_state == 'waiting',
                'dropped_pending_update': pending_update,
                'cache_warm_count': self.cache_warm_count,
            },
        )
        available_tools = await self.tools.available()
        self._note_tool_catalog(available_tools)
        return available_tools

    async def _run_cache_warms(self) -> None:  # noqa: C901
        try:
            while not self._cache_finalizing:
                transcript = self._pending_warm_transcript
                self._pending_warm_transcript = None
                if transcript is None:
                    return

                if self._last_warm_started_at is not None:
                    wait_seconds = (
                        self._last_warm_started_at
                        + self.settings.llm_cache_warm_min_interval_seconds
                        - time.perf_counter()
                    )
                    if wait_seconds > 0:
                        self._warm_worker_state = 'waiting'
                        await asyncio.sleep(wait_seconds)
                        if self._cache_finalizing:
                            return
                        if self._pending_warm_transcript is not None:
                            transcript = self._pending_warm_transcript
                            self._pending_warm_transcript = None

                self._warm_worker_state = 'active'
                self._last_warm_started_at = time.perf_counter()
                try:
                    await self._catalog_and_warm(transcript)
                except Exception:
                    LOGGER.exception(
                        'Incremental LLM cache warm failed',
                        extra={'event_id': 'ID_assistant_llm_cache_warm_failed'},
                    )
                finally:
                    self._warm_worker_state = 'idle'
        finally:
            self._warm_worker_state = 'idle'
            self._warm_worker = None

    async def _catalog_and_warm(self, transcript: str) -> None:
        available_tools = await self.tools.available()
        self._note_tool_catalog(available_tools)
        prompt = build_prompt_prefix(
            transcript,
            available_tools,
            system_prompt=self._active_system_prompt(),
        )
        await self._warm(prompt, reason='delta')

    def _note_tool_catalog(self, available_tools: list[ToolDefinition]) -> None:
        tool_names = tuple(sorted(tool.name for tool in available_tools))
        if self.last_warmed_prompt is not None and tool_names != self.last_tool_names:
            metrics.CACHE_TOOLSET_CHANGES.inc()
        self.last_tool_names = tool_names

    async def _warm(self, prompt: str, *, reason: str) -> None:
        if prompt == self.last_warmed_prompt:
            return
        slot = await self._ensure_slot()
        await warm_llm_cache(self.llm, prompt, slot, reason=reason)
        self.last_warmed_prompt = prompt
        self.cache_warm_count += 1

    async def generate(
        self,
        transcript: str,
        available_tools: list[ToolDefinition],
        send: EventSender,
    ) -> None:
        final_at = self.final_transcript_at or time.perf_counter()
        metrics.TRANSCRIPT_CHARACTERS.observe(len(transcript))
        prompt = build_prompt(
            transcript,
            available_tools,
            system_prompt=self._active_system_prompt(),
        )
        await send(
            {
                'type': 'response.created',
                'response': {'model': self.settings.model_id, 'audio_format': 'pcm16'},
            },
        )
        metrics.PIPELINE_SECONDS.labels(stage='llm_request').observe(
            time.perf_counter() - final_at,
        )

        if available_tools:
            await self._resolve_tools_and_speak(
                transcript,
                available_tools,
                final_at,
                send,
            )
        else:
            await self._stream_and_speak(prompt, final_at, send)

        metrics.PIPELINE_SECONDS.labels(stage='response_after_transcript').observe(
            time.perf_counter() - final_at,
        )
        metrics.PIPELINE_SECONDS.labels(stage='session').observe(
            time.perf_counter() - self.started_at,
        )
        await send({'type': 'response.done'})

    async def _resolve_tools_and_speak(  # noqa: C901
        self,
        transcript: str,
        available_tools: list[ToolDefinition],
        final_at: float,
        send: EventSender,
    ) -> None:
        slot = await self._ensure_slot()
        tools_by_name = {tool.name: tool for tool in available_tools}
        history: list[tuple[str, str]] = []
        observe_first_token = True
        for iteration in range(1, self.settings.maximum_tool_iterations + 1):
            prompt = build_prompt(
                transcript,
                available_tools,
                history,
                system_prompt=self._active_system_prompt(),
            )
            response_stream = self.llm.stream(
                prompt,
                slot,
                operation='tool_decision',
                maximum_tokens=self.settings.llm_tool_tokens,
            )
            raw_response = await self._stream_spoken_or_buffer_json(
                response_stream,
                final_at,
                send,
                observe_first_token=observe_first_token,
            )
            observe_first_token = False
            if raw_response is None:
                return
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
                await self._speak_text(raw_response, final_at, send)
                return
            answer = decision.get('answer')
            if isinstance(answer, str) and answer.strip():
                await self._speak_text(answer, final_at, send)
                return
            tool_name = decision.get('tool') or decision.get('name')
            arguments = decision.get('arguments', {})
            if not isinstance(tool_name, str) or tool_name not in tools_by_name:
                await self._speak_text(raw_response, final_at, send)
                return
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
                    },
                )
                result = json.dumps({'error': 'tool execution failed'})
            history.extend(
                (
                    ('assistant', raw_response),
                    (
                        'tool',
                        json.dumps(
                            {'tool': tool.name, 'result': result},
                            ensure_ascii=False,
                            separators=(',', ':'),
                        ),
                    ),
                ),
            )

        final_history = [
            *history,
            (
                'user',
                'No more tools. Reply directly with a short spoken answer, not JSON.',
            ),
        ]
        final_prompt = build_prompt(
            transcript,
            available_tools,
            final_history,
            system_prompt=self._active_system_prompt(),
        )
        response_stream = self.llm.stream(
            final_prompt,
            slot,
            operation='tool_answer',
            maximum_tokens=self.settings.llm_tool_tokens,
        )
        raw_response = await self._stream_spoken_or_buffer_json(
            response_stream,
            final_at,
            send,
            observe_first_token=observe_first_token,
        )
        if raw_response is None:
            return
        decision = parse_tool_response(raw_response)
        if decision is not None and isinstance(decision.get('answer'), str):
            await self._speak_text(str(decision['answer']), final_at, send)
            return
        await self._speak_text(raw_response, final_at, send)

    async def _stream_spoken_or_buffer_json(
        self,
        response_stream: AsyncIterator[str],
        final_at: float,
        send: EventSender,
        *,
        observe_first_token: bool,
    ) -> str | None:
        buffered: list[str] = []
        first_token_pending = observe_first_token
        async for token in response_stream:
            if first_token_pending:
                self._observe_llm_first_token(final_at)
                first_token_pending = False
            buffered.append(token)
            response_prefix = ''.join(buffered).lstrip()
            if not response_prefix:
                continue
            if response_prefix.startswith('{'):
                buffered.extend(
                    [remaining_token async for remaining_token in response_stream],
                )
                return ''.join(buffered)
            await self._speak_token_stream(
                self._prepend_tokens(buffered, response_stream),
                final_at,
                send,
                observe_first_token=False,
            )
            return None
        self._raise_empty_response()
        return None

    @staticmethod
    async def _prepend_tokens(
        buffered: list[str],
        response_stream: AsyncIterator[str],
    ) -> AsyncIterator[str]:
        for token in buffered:
            yield token
        async for token in response_stream:
            yield token

    async def _stream_and_speak(
        self,
        prompt: str,
        final_at: float,
        send: EventSender,
    ) -> None:
        slot = await self._ensure_slot()
        await self._speak_token_stream(
            self.llm.stream(prompt, slot),
            final_at,
            send,
            observe_first_token=True,
        )

    async def _speak_token_stream(  # noqa: C901
        self,
        response_stream: AsyncIterator[str],
        final_at: float,
        send: EventSender,
        *,
        observe_first_token: bool,
    ) -> None:
        async def text_stream() -> AsyncIterator[str]:  # noqa: C901
            parts: list[str] = []
            first_token_pending = observe_first_token
            first_tts_text = True
            async for token in response_stream:
                if first_token_pending:
                    self._observe_llm_first_token(final_at)
                    first_token_pending = False
                parts.append(token)
                await send({'type': 'response.text.delta', 'delta': token})
                if token:
                    if first_tts_text:
                        metrics.PIPELINE_SECONDS.labels(stage='tts_first_request').observe(
                            time.perf_counter() - final_at,
                        )
                        first_tts_text = False
                    yield token
            response_text = clean_response(''.join(parts))
            metrics.PIPELINE_SECONDS.labels(stage='llm_complete').observe(
                time.perf_counter() - final_at,
            )
            if not response_text:
                self._raise_empty_response()
            _log_generated_response(response_text)
            await send({'type': 'response.text.done', 'text': response_text})
            metrics.LLM_OUTPUT_CHARACTERS.inc(len(response_text))
            await self.release_slot()

        await self._stream_tts_audio(text_stream(), final_at, send)

    @staticmethod
    def _observe_llm_first_token(final_at: float) -> None:
        first_token_at = time.perf_counter()
        metrics.LLM_TIME_TO_FIRST_TOKEN.observe(first_token_at - final_at)
        metrics.PIPELINE_SECONDS.labels(stage='llm_first_token').observe(
            first_token_at - final_at,
        )

    async def _speak_text(
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

        async def text_stream() -> AsyncIterator[str]:
            yield cleaned

        metrics.PIPELINE_SECONDS.labels(stage='tts_first_request').observe(
            time.perf_counter() - final_at,
        )
        await self._stream_tts_audio(text_stream(), final_at, send)

    async def _stream_tts_audio(  # noqa: C901
        self,
        text_stream: AsyncIterator[str],
        final_at: float,
        send: EventSender,
    ) -> None:
        expected_format: AudioFormat | None = None
        first_audio = True
        async for chunk, audio_format in self.tts.stream(
            text_stream,
            voice=self.voice,
            multi_voice=self.multi_voice_enabled,
        ):
            if expected_format is None:
                expected_format = audio_format
            elif audio_format != expected_format:
                message = 'TTS audio format changed during the response'
                raise RuntimeError(message)
            if not chunk:
                continue
            if first_audio:
                first_audio = False
                first_audio_at = time.perf_counter()
                metrics.TTS_TIME_TO_FIRST_AUDIO.observe(first_audio_at - final_at)
                metrics.PIPELINE_SECONDS.labels(stage='tts_first_audio').observe(
                    first_audio_at - final_at,
                )
                await send(
                    {
                        'type': 'response.audio.started',
                        'format': 'pcm16',
                        'sample_rate': audio_format.sample_rate,
                        'sample_width': audio_format.sample_width,
                        'channels': audio_format.channels,
                    },
                )
            await send(
                {
                    'type': 'response.audio.delta',
                    'audio': base64.b64encode(chunk).decode('ascii'),
                },
            )
            metrics.AUDIO_OUTPUT_BYTES.inc(len(chunk))

        metrics.PIPELINE_SECONDS.labels(stage='tts_complete').observe(
            time.perf_counter() - final_at,
        )
        await send({'type': 'response.audio.done'})

    @staticmethod
    def _raise_empty_response() -> None:
        message = 'LLM produced an empty response'
        raise RuntimeError(message)

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

    async def _stop_cache_warming(self) -> None:
        self._cache_finalizing = True
        if self._pending_warm_transcript is not None:
            self._pending_warm_transcript = None
            metrics.CACHE_WARM_UPDATES.labels(disposition='dropped_on_final').inc()
        worker = self._warm_worker
        if worker is None:
            return
        if self._warm_worker_state == 'waiting':
            _ = worker.cancel()
        _ = await asyncio.gather(worker, return_exceptions=True)

    async def close(self) -> None:
        await self._stop_cache_warming()
        metrics.CACHE_WARMS_PER_SESSION.observe(self.cache_warm_count)
        await self.release_slot()
