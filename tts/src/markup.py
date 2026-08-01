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

    unknown = sorted(set(attributes) - {'marker', 'voice'})
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
    return MarkupCharacter(name=name, voice=voice, marker=marker)


class MultiSpeakerMarkupParser:
    """Incrementally parse strict flat multi-speaker markup into transport-neutral events."""

    def __init__(self) -> None:
        self._buffer = ''
        self._phase: Literal['start', 'header', 'body'] = 'start'
        self._characters: dict[str, MarkupCharacter] = {}
        self._markers: dict[str, str] = {}
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
                        existing = self._markers.get(character.marker)
                        if existing is not None:
                            message = ''.join(
                                (
                                    f'marker {character.marker!r} is shared by {existing!r} and ',
                                    f'{character.name!r}',
                                ),
                            )
                            _raise_invalid(message)
                        self._markers[character.marker] = character.name
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
            _raise_invalid('body must start with a character marker')
        if self._buffer:
            if '<' in self._buffer:
                _raise_invalid('incomplete character tag')
            self._emit_text(events, self._buffer)
            self._buffer = ''
        if self._current_speaker is None:
            _raise_invalid('body must start with a character marker')
        if not self._turn_has_text:
            _raise_invalid(f'character {self._current_speaker!r} has no text')
        return events

    def _drain_body(self, events: list[MarkupEvent]) -> bool:  # noqa: C901, PLR0911
        if self._current_speaker is None:
            self._buffer = self._buffer.lstrip()
            if not self._buffer:
                return False
            marker_speaker = self._marker_at_start()
            if marker_speaker is not None:
                marker, speaker = marker_speaker
                self._buffer = self._buffer[len(marker) :]
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
            _raise_invalid('body must start with a character marker')

        boundary = self._next_body_boundary()
        if boundary is None:
            self._emit_text(events, self._buffer)
            self._buffer = ''
            return False
        boundary_index, boundary_kind = boundary
        if boundary_index:
            self._emit_text(events, self._buffer[:boundary_index])
            self._buffer = self._buffer[boundary_index:]
            return True
        if boundary_kind == 'tag':
            tag_end = self._buffer.find('>')
            if tag_end < 0:
                return False
            tag = self._buffer[: tag_end + 1]
            self._buffer = self._buffer[tag_end + 1 :]
            self._switch_speaker(events, self._speaker_from_tag(tag))
            return True

        speaker = self._markers[boundary_kind]
        self._buffer = self._buffer[len(boundary_kind) :]
        self._switch_speaker(events, speaker)
        return True

    def _next_body_boundary(self) -> tuple[int, str] | None:
        candidates: list[tuple[int, str]] = []
        tag_index = self._buffer.find('<')
        if tag_index >= 0:
            candidates.append((tag_index, 'tag'))
        for marker in self._markers:
            marker_index = self._buffer.find(marker)
            if marker_index >= 0:
                candidates.append((marker_index, marker))
        return min(candidates, default=None, key=lambda item: item[0])

    def _marker_at_start(self) -> tuple[str, str] | None:
        return next(
            (
                (marker, speaker)
                for marker, speaker in self._markers.items()
                if self._buffer.startswith(marker)
            ),
            None,
        )

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
