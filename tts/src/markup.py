from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Never

from tts.src.domain import (
    InvalidMultiSpeakerMarkupError,
    MultiSpeakerCommand,
    SpeakerSegment,
    UndefinedSpeakerError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

MULTI_SPEAKER_MARKER = '<multi>'
_NAME_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}')
_RESERVED_NAMES = frozenset({'char', 'multi'})
_MINIMUM_QUOTED_VALUE_LENGTH = 2


@dataclass(frozen=True, slots=True)
class MarkupCharacter:
    name: str
    voice: str
    marker: str | None
    alias: str | None


@dataclass(frozen=True, slots=True)
class MarkupHeader:
    characters: Mapping[str, MarkupCharacter]


@dataclass(frozen=True, slots=True)
class MarkupSpeaker:
    name: str


@dataclass(frozen=True, slots=True)
class MarkupText:
    text: str


MarkupEvent = MarkupHeader | MarkupSpeaker | MarkupText


def _raise_invalid(message: str) -> Never:
    raise InvalidMultiSpeakerMarkupError(message)


def _parse_character_tag(tag: str) -> MarkupCharacter:  # noqa: C901
    content = tag.removeprefix('<char').removesuffix('>').strip()
    parts = content.split()
    if not parts or _NAME_PATTERN.fullmatch(parts[0]) is None:
        _raise_invalid('character declarations require a valid character name')
    name = parts[0]
    if name in _RESERVED_NAMES:
        _raise_invalid(f'character name {name!r} is reserved')

    attributes: dict[str, str] = {}
    for part in parts[1:]:
        key, separator, value = part.partition('=')
        if not separator or not key or not value:
            _raise_invalid(f'invalid attribute in character {name!r}')
        if key in attributes:
            _raise_invalid(f'duplicate attribute {key!r} in character {name!r}')
        if (
            len(value) >= _MINIMUM_QUOTED_VALUE_LENGTH
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
        ):
            value = value[1:-1]
        attributes[key] = value

    unknown = sorted(set(attributes) - {'alias', 'marker', 'voice'})
    if unknown:
        _raise_invalid(f'unknown character attribute {unknown[0]!r}')
    voice = attributes.get('voice')
    if voice is None:
        _raise_invalid(f'character {name!r} requires a voice')
    if _NAME_PATTERN.fullmatch(voice) is None:
        _raise_invalid(f'voice for character {name!r} must be a valid voice name')
    marker = attributes.get('marker')
    if marker is not None and (
        len(marker) != 1 or marker.isspace() or marker.isalnum() or marker in {'<', '>', '"', "'"}
    ):
        _raise_invalid(
            f'marker for character {name!r} must be one non-alphanumeric symbol',
        )
    return MarkupCharacter(
        name=name,
        voice=voice,
        marker=marker,
        alias=attributes.get('alias'),
    )


@dataclass(frozen=True, slots=True)
class _BodyBoundary:
    index: int
    kind: Literal['selector', 'tag', 'pending']
    selector: str | None = None


class MultiSpeakerMarkupParser:
    """Incrementally parse strict flat multi-speaker markup into transport-neutral events."""

    def __init__(self) -> None:
        self._buffer = ''
        self._phase: Literal['start', 'header', 'body'] = 'start'
        self._characters: dict[str, MarkupCharacter] = {}
        self._selectors: dict[str, str] = {}
        self._current_speaker: str | None = None
        self._turn_has_text = False

    def append(self, text: str) -> list[MarkupEvent]:  # noqa: C901, PLR0912
        self._buffer += text
        events: list[MarkupEvent] = []
        while True:
            if self._phase == 'start':
                if MULTI_SPEAKER_MARKER.startswith(self._buffer):
                    if self._buffer != MULTI_SPEAKER_MARKER:
                        break
                elif not self._buffer.startswith(MULTI_SPEAKER_MARKER):
                    _raise_invalid(f'input must start with {MULTI_SPEAKER_MARKER}')
                self._buffer = self._buffer[len(MULTI_SPEAKER_MARKER) :]
                self._phase = 'header'
                continue

            if self._phase == 'header':
                self._buffer = self._buffer.lstrip()
                if not self._buffer:
                    break
                if '<char '.startswith(self._buffer):
                    break
                if self._buffer.startswith('<char '):
                    tag_end = self._buffer.find('>')
                    if tag_end < 0:
                        break
                    character = _parse_character_tag(self._buffer[: tag_end + 1])
                    if character.name in self._characters:
                        _raise_invalid(f'character {character.name!r} is declared more than once')
                    if character.marker is not None:
                        self._register_selector(character.marker, character.name)
                    if character.alias:
                        self._register_selector(character.alias, character.name)
                    self._characters[character.name] = character
                    self._buffer = self._buffer[tag_end + 1 :]
                    continue
                if self._buffer.startswith('<char'):
                    tag_end = self._buffer.find('>')
                    if tag_end < 0:
                        break
                    _raise_invalid('malformed character declaration')
                if not self._characters:
                    _raise_invalid('header must declare at least one character')
                events.append(MarkupHeader(dict(self._characters)))
                self._phase = 'body'
                continue

            if not self._drain_body(events):
                break
        return events

    def finish(self) -> list[MarkupEvent]:  # noqa: C901
        events = self.append('')
        if self._phase == 'start':
            _raise_invalid(f'input must start with {MULTI_SPEAKER_MARKER}')
        if self._phase == 'header':
            if self._buffer:
                _raise_invalid('incomplete character declaration')
            _raise_invalid('body must start with a character selector')
        while self._drain_body(events, final=True):
            pass
        if self._buffer:
            if '<' in self._buffer:
                _raise_invalid('incomplete character tag')
            self._emit_text(events, self._buffer)
            self._buffer = ''
        if self._current_speaker is None:
            _raise_invalid('body must start with a character selector')
        if not self._turn_has_text:
            _raise_invalid(f'character {self._current_speaker!r} has no text')
        return events

    def _drain_body(  # noqa: C901, PLR0911
        self,
        events: list[MarkupEvent],
        *,
        final: bool = False,
    ) -> bool:
        if self._current_speaker is None:
            self._buffer = self._buffer.lstrip()
            if not self._buffer:
                return False
            selector_speaker, pending = self._selector_at_start(final=final)
            if pending:
                return False
            if selector_speaker is not None:
                selector, speaker = selector_speaker
                self._buffer = self._buffer[len(selector) :]
                self._switch_speaker(events, speaker)
                return True
            if self._buffer.startswith('<'):
                tag_end = self._buffer.find('>')
                if tag_end < 0:
                    return False
                tag = self._buffer[: tag_end + 1]
                self._buffer = self._buffer[tag_end + 1 :]
                self._switch_speaker(events, self._speaker_from_tag(tag))
                return True
            _raise_invalid('body must start with a character selector')

        boundary = self._next_body_boundary(final=final)
        if boundary is None:
            self._emit_text(events, self._buffer)
            self._buffer = ''
            return False
        if boundary.index:
            self._emit_text(events, self._buffer[: boundary.index])
            self._buffer = self._buffer[boundary.index :]
            return True
        if boundary.kind == 'pending':
            return False
        if boundary.kind == 'tag':
            tag_end = self._buffer.find('>')
            if tag_end < 0:
                return False
            tag = self._buffer[: tag_end + 1]
            self._buffer = self._buffer[tag_end + 1 :]
            self._switch_speaker(events, self._speaker_from_tag(tag))
            return True

        selector = boundary.selector or ''
        speaker = self._selectors[selector]
        self._buffer = self._buffer[len(selector) :]
        self._switch_speaker(events, speaker)
        return True

    def _next_body_boundary(self, *, final: bool) -> _BodyBoundary | None:  # noqa: C901
        candidates: list[_BodyBoundary] = []
        tag_index = self._buffer.find('<')
        if tag_index >= 0:
            candidates.append(_BodyBoundary(tag_index, 'tag'))
        for selector in self._selectors:
            selector_index = self._buffer.find(selector)
            if selector_index >= 0:
                candidates.append(_BodyBoundary(selector_index, 'selector', selector))
        boundary = min(
            candidates,
            default=None,
            key=lambda item: (
                item.index,
                item.kind == 'tag',
                -(len(item.selector) if item.selector is not None else 0),
            ),
        )
        if final:
            return boundary
        pending_index = self._pending_selector_index()
        if pending_index is not None and (boundary is None or pending_index <= boundary.index):
            return _BodyBoundary(pending_index, 'pending')
        return boundary

    def _selector_at_start(self, *, final: bool) -> tuple[tuple[str, str] | None, bool]:
        matches = [
            (selector, speaker)
            for selector, speaker in self._selectors.items()
            if self._buffer.startswith(selector)
        ]
        pending = any(
            len(self._buffer) < len(selector) and selector.startswith(self._buffer)
            for selector in self._selectors
        )
        if pending and not final:
            return None, True
        return max(matches, default=None, key=lambda item: len(item[0])), False

    def _pending_selector_index(self) -> int | None:
        indexes: list[int] = []
        for selector in self._selectors:
            maximum_prefix = min(len(self._buffer), len(selector) - 1)
            for length in range(maximum_prefix, 0, -1):
                if self._buffer.endswith(selector[:length]):
                    indexes.append(len(self._buffer) - length)
                    break
        return min(indexes, default=None)

    def _register_selector(self, selector: str, speaker: str) -> None:
        existing = self._selectors.get(selector)
        if existing is not None and existing != speaker:
            message = f'selector {selector!r} is shared by {existing!r} and {speaker!r}'
            _raise_invalid(message)
        self._selectors[selector] = speaker

    def _speaker_from_tag(self, tag: str) -> str:
        match = re.fullmatch(r'<([A-Za-z0-9][A-Za-z0-9_-]{0,63})>', tag)
        if match is None:
            _raise_invalid(f'invalid character tag {tag!r}')
        speaker = match.group(1)
        if speaker not in self._characters:
            raise UndefinedSpeakerError(speaker)
        return speaker

    def _switch_speaker(self, events: list[MarkupEvent], speaker: str) -> None:
        if self._current_speaker is not None and not self._turn_has_text:
            _raise_invalid(f'character {self._current_speaker!r} has no text')
        if speaker == self._current_speaker:
            self._emit_text(events, ' ')
            return
        self._current_speaker = speaker
        self._turn_has_text = False
        events.append(MarkupSpeaker(speaker))

    def _emit_text(self, events: list[MarkupEvent], text: str) -> None:
        if not text:
            return
        self._turn_has_text = self._turn_has_text or bool(text.strip())
        events.append(MarkupText(text))


def parse_multi_speaker_markup(  # noqa: C901
    markup: str,
    *,
    model: str,
    speed: float,
) -> MultiSpeakerCommand:
    """Parse one complete tagged request into the existing structured command."""
    parser = MultiSpeakerMarkupParser()
    events = [*parser.append(markup), *parser.finish()]
    speakers: dict[str, str] = {}
    segments: list[SpeakerSegment] = []
    text_parts: list[str] | None = None
    current_speaker: str | None = None
    for event in events:
        if isinstance(event, MarkupHeader):
            speakers = {name: character.voice for name, character in event.characters.items()}
        elif isinstance(event, MarkupSpeaker):
            if current_speaker is not None and text_parts is not None:
                segments.append(SpeakerSegment(current_speaker, ''.join(text_parts)))
            current_speaker = event.name
            text_parts = []
        elif text_parts is not None:
            text_parts.append(event.text)
    if current_speaker is not None and text_parts is not None:
        segments.append(SpeakerSegment(current_speaker, ''.join(text_parts)))
    return MultiSpeakerCommand(
        model=model,
        speakers=speakers,
        segments=tuple(segments),
        speed=speed,
    )
