from __future__ import annotations

import os
import tempfile
import wave
from contextlib import suppress
from pathlib import Path
from typing import Any


class AtomicWavWriter:
    """Build a WAV incrementally and publish it only after successful completion."""

    def __init__(self, destination: Path, sample_rate: int) -> None:
        self.destination = destination
        self._temporary_path: Path | None = None
        self._temporary: Any | None = None
        self._wav_file: wave.Wave_write | None = None
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = tempfile.NamedTemporaryFile(  # noqa: SIM115
                dir=destination.parent,
                prefix=f'.{destination.name}.',
                suffix='.tmp',
                delete=False,
            )
            self._temporary = temporary
            self._temporary_path = Path(temporary.name)
            self._wav_file = wave.open(temporary, 'wb')  # noqa: SIM115
            self._wav_file.setnchannels(1)
            self._wav_file.setsampwidth(2)
            self._wav_file.setframerate(sample_rate)
        except (OSError, wave.Error):
            self.abort()
            raise

    def write(self, pcm: bytes) -> None:
        if self._wav_file is None:
            message = 'WAV capture is not open'
            raise RuntimeError(message)
        self._wav_file.writeframesraw(pcm)

    def commit(self) -> None:
        temporary = self._temporary
        temporary_path = self._temporary_path
        wav_file = self._wav_file
        if temporary is None or temporary_path is None or wav_file is None:
            message = 'WAV capture is not open'
            raise RuntimeError(message)

        wav_file.close()
        self._wav_file = None
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary.close()
        self._temporary = None
        _ = temporary_path.replace(self.destination)
        self._temporary_path = None

    def abort(self) -> None:
        wav_file = self._wav_file
        self._wav_file = None
        if wav_file is not None:
            with suppress(OSError, wave.Error):
                wav_file.close()

        temporary = self._temporary
        self._temporary = None
        if temporary is not None:
            with suppress(OSError):
                temporary.close()

        temporary_path = self._temporary_path
        self._temporary_path = None
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
