"""FileStorage adapter: original uploads on the local disk.

A cloud adapter (S3-compatible: Cloudflare R2, AWS S3, Backblaze B2, MinIO)
can replace this later behind the same port.
"""

import logging
import os
import re
import tempfile
from pathlib import Path

from app.application.ports import FileStorage
from app.domain.errors import StorageUnavailable

logger = logging.getLogger(__name__)

_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
FAILED = "The file store is unavailable."


class LocalFileStorage(FileStorage):
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, key: str, data: bytes) -> None:
        """Write to a temporary file, then rename: a crash never leaves half a file."""
        path = self._path(key)
        try:
            fd, tmp = tempfile.mkstemp(dir=self._root, prefix=".tmp-")
            with os.fdopen(fd, "wb") as out:
                out.write(data)
            os.replace(tmp, path)
        except OSError as exc:
            logger.exception("file_storage.save_failed", extra={"key": key})
            raise StorageUnavailable(FAILED) from exc

    def delete(self, key: str) -> None:
        try:
            self._path(key).unlink(missing_ok=True)
        except OSError as exc:
            logger.exception("file_storage.delete_failed", extra={"key": key})
            raise StorageUnavailable(FAILED) from exc

    def _path(self, key: str) -> Path:
        if not _SAFE_KEY.match(key) or ".." in key:
            raise ValueError(f"Unsafe file key: {key!r}")
        return self._root / key
