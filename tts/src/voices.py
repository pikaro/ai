from __future__ import annotations

import os
import re
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tts.src.domain import (
    EmptyVoiceUploadError,
    InvalidVoiceNameError,
    UnsupportedVoiceFileError,
    VoiceUploadTooLargeError,
)

if TYPE_CHECKING:
    from typing import BinaryIO

READ_CHUNK_BYTES = 1024 * 1024
VOICE_NAME_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}')


def is_voice_name(value: str) -> bool:
    return VOICE_NAME_PATTERN.fullmatch(value) is not None


@dataclass(frozen=True, slots=True)
class StoredVoice:
    name: str
    filename: str
    replaced: bool


class VoiceRepository:
    """Persist custom voice states independently of an HTTP upload transport."""

    def __init__(self, directory: Path, maximum_bytes: int) -> None:
        self._directory = directory
        self._maximum_bytes = maximum_bytes

    def store(
        self,
        name: str,
        source_filename: str | None,
        source: BinaryIO,
    ) -> StoredVoice:
        if not is_voice_name(name):
            raise InvalidVoiceNameError
        if Path(source_filename or '').suffix.casefold() != '.safetensors':
            raise UnsupportedVoiceFileError

        destination = self._directory / f'{name}.safetensors'
        replaced = destination.exists()
        self._store_atomic(source, destination)
        return StoredVoice(name=name, filename=destination.name, replaced=replaced)

    def list(self) -> list[StoredVoice]:
        voices: list[StoredVoice] = []
        for path in self._directory.glob('*.safetensors'):
            name = path.stem
            if is_voice_name(name):
                voices.append(StoredVoice(name=name, filename=path.name, replaced=False))
        return voices

    def _store_atomic(self, source: BinaryIO, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        total_bytes = 0
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f'.{destination.name}.',
                suffix='.tmp',
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                while chunk := source.read(READ_CHUNK_BYTES):
                    total_bytes += len(chunk)
                    _check_voice_upload_size(total_bytes, self._maximum_bytes)
                    _ = temporary.write(chunk)
                _check_voice_upload_not_empty(total_bytes)
                temporary.flush()
                os.fsync(temporary.fileno())
            _ = temporary_path.replace(destination)
        except BaseException:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)
            raise


def _check_voice_upload_size(total_bytes: int, maximum_bytes: int) -> None:
    if total_bytes > maximum_bytes:
        raise VoiceUploadTooLargeError


def _check_voice_upload_not_empty(total_bytes: int) -> None:
    if total_bytes == 0:
        raise EmptyVoiceUploadError
