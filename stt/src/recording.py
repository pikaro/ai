from __future__ import annotations

import logging
import os
import tempfile
import wave
from contextlib import suppress
from pathlib import Path

LOGGER = logging.getLogger('stt')


def save_wav_atomic(destination: Path, pcm: bytes, sample_rate: int, channels: int) -> None:
    """Atomically replace a diagnostic WAV with the supplied raw client PCM."""
    temporary_path: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f'.{destination.name}.',
            suffix='.tmp',
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with wave.open(temporary, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm)
            temporary.flush()
            os.fsync(temporary.fileno())
        _ = temporary_path.replace(destination)
    except (OSError, wave.Error):
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
        LOGGER.exception(
            'Failed to save latest input recording',
            extra={'event_id': 'ID_stt_latest_recording_save_failed', 'path': str(destination)},
        )
        return
    LOGGER.debug(
        'Saved latest input recording',
        extra={
            'event_id': 'ID_stt_latest_recording_saved',
            'pcm_bytes': len(pcm),
            'sample_rate': sample_rate,
            'channels': channels,
            'path': str(destination),
        },
    )
