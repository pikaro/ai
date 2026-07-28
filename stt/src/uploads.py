from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Final

from stt.src.domain import AudioUploadTooLargeError, UnsupportedAudioExtensionError

if TYPE_CHECKING:
    from typing import BinaryIO

READ_CHUNK_BYTES: Final = 1024 * 1024
SUPPORTED_UPLOAD_SUFFIXES: Final = frozenset(
    {'.flac', '.m4a', '.mp3', '.mp4', '.mpeg', '.mpga', '.ogg', '.wav', '.webm'},
)


def save_temporary_upload(
    source_filename: str | None,
    source: BinaryIO,
    maximum_bytes: int,
) -> Path:
    suffix = Path(source_filename or 'audio.wav').suffix.casefold() or '.wav'
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise UnsupportedAudioExtensionError(suffix)

    total_bytes = 0
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        path = Path(temporary.name)
        try:
            while chunk := source.read(READ_CHUNK_BYTES):
                total_bytes += len(chunk)
                _check_upload_size(total_bytes, maximum_bytes)
                _ = temporary.write(chunk)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    return path


def _check_upload_size(total_bytes: int, maximum_bytes: int) -> None:
    if total_bytes > maximum_bytes:
        raise AudioUploadTooLargeError
