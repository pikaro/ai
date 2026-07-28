from __future__ import annotations

import logging

LOGGER = logging.getLogger('stt')


def log_transcription(stage: str, text: str) -> None:
    LOGGER.info(
        'Transcription produced',
        extra={
            'event_id': 'ID_stt_transcription_produced',
            'stage': stage,
            'characters': len(text),
        },
    )
    LOGGER.debug(
        'Transcription',
        extra={'event_id': 'ID_stt_transcription', 'stage': stage, 'transcript': text},
    )


def extract_text(result: object) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        value = result.get('text') or result.get('transcript')
        return str(value or '').strip()
    value = getattr(result, 'text', None)
    return str(value if value is not None else result or '').strip()


def extract_first_text(result: object) -> str:
    if isinstance(result, (list, tuple)):
        return '' if not result else extract_text(result[0])
    return extract_text(result)


def transcript_delta(previous: str, current: str) -> str:
    if not previous:
        return current.strip()
    return current[len(previous) :].strip() if current.startswith(previous) else ''


def stable_word_prefix(previous: str, current: str) -> str:
    common_length = 0
    for previous_character, current_character in zip(previous, current, strict=False):
        if previous_character != current_character:
            break
        common_length += 1
    if common_length == 0:
        return ''

    prefix = current[:common_length]
    if common_length < len(current) and not current[common_length].isspace():
        boundary = prefix.rstrip().rfind(' ')
        prefix = '' if boundary < 0 else prefix[:boundary]
    return prefix.strip()
