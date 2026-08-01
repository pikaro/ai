from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from assistant.src.config import MultiVoiceConfig

_VOICE_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}')


def multi_voice_header(
    config: MultiVoiceConfig,
    *,
    selected_voice: str | None = None,
) -> str:
    """Build the complete header prepended to one assistant TTS text stream."""
    lines = ['<multi>']
    for name, character in config.characters.items():
        voice = character.voice
        if name == config.default_character and selected_voice is not None:
            voice = selected_voice.strip() or 'default'
        if _VOICE_PATTERN.fullmatch(voice) is None:
            message = 'voice must be "default" or a valid voice name'
            raise ValueError(message)
        lines.append(f'<char {name} voice={voice} marker={character.marker}>')
    return f'{"\n".join(lines)}\n'


def system_prompt_with_multi_voice(base_prompt: str, config: MultiVoiceConfig) -> str:
    """Append stable marker instructions while leaving the editable prompt untouched."""
    character_lines = [
        (
            f'- {character.marker} means {name} using voice {character.voice}'
            + (' and is the default character' if name == config.default_character else '')
        )
        for name, character in config.characters.items()
    ]
    default_marker = config.characters[config.default_character].marker
    usage = ''.join(
        (
            f'Start every spoken answer with {default_marker}. Insert another marker exactly ',
            'when that character starts speaking. Write markers directly, without angle brackets. ',
            'Do not emit <multi> or <char> declarations; the assistant adds them automatically. ',
            'A JSON tool request is not a spoken answer and must not contain a voice marker.',
        ),
    )
    instructions = '\n'.join(
        (
            'For every spoken answer, use the following voice-character markers:',
            *character_lines,
            usage,
        ),
    )
    return f'{base_prompt.rstrip()}\n\n{instructions}'
