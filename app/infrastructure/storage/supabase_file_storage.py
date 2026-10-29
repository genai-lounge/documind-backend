"""FileStorage adapter: original uploads in a private Supabase Storage bucket.

Calls Supabase's Storage REST API directly with httpx2 (no SDK). Use a secret
key (sb_secret_...) or the legacy service_role key: both bypass row-level
security, so they belong only in the backend's environment.
"""

import logging
import random
import re
import time
from collections.abc import Callable
from typing import Any

import httpx2

from app.application.ports import FileStorage
from app.domain.errors import StorageUnavailable

logger = logging.getLogger(__name__)

FAILED = "The file store is unavailable."
MAX_BACKOFF_SECONDS = 5.0

_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
_SAFE_BUCKET = re.compile(r"^[a-z0-9][a-z0-9._-]{1,62}$")
_CONTENT_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain; charset=utf-8",
}


class SupabaseFileStorage(FileStorage):
    def __init__(
        self,
        url: str,
        secret_key: str,
        bucket: str,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        transport: httpx2.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not secret_key:
            raise ValueError("A Supabase secret key is required.")
        if not _SAFE_BUCKET.match(bucket):
            raise ValueError(f"Invalid bucket name: {bucket!r}")
        headers = {"apikey": secret_key}
        if secret_key.startswith("eyJ"):
            headers["Authorization"] = f"Bearer {secret_key}"
        self.bucket = bucket
        self._max_retries = max_retries
        self._sleep = sleep
        self._client = httpx2.Client(
            base_url=url.rstrip("/") + "/storage/v1",
            headers=headers,
            timeout=httpx2.Timeout(timeout_seconds, connect=5.0),
            transport=transport,
        )

    def ensure_bucket(self) -> bool:
        """Create the bucket (private) if it doesn't exist. Returns True if created."""
        response = self._send("GET", f"/bucket/{self.bucket}")
        if response.status_code == 200:
            return False
        if not _is_not_found(response):
            raise self._failure("ensure_bucket", response)

        created = self._send(
            "POST",
            "/bucket",
            json={"id": self.bucket, "name": self.bucket, "public": False},
        )
        if created.status_code == 200:
            logger.info("file_storage.bucket_created", extra={"bucket": self.bucket})
            return True
        if _mentions(created, "already exists"):
            return False
        raise self._failure("create_bucket", created)

    def save(self, key: str, data: bytes) -> None:
        response = self._send(
            "POST",
            f"/object/{self.bucket}/{_checked(key)}",
            content=data,
            headers={
                "Content-Type": _content_type(key),
                "x-upsert": "true",
            },
        )
        if response.status_code != 200:
            raise self._failure("save", response)

    def delete(self, key: str) -> None:
        """Batch delete: succeeds even when the object is already gone."""
        response = self._send(
            "DELETE", f"/object/{self.bucket}", json={"prefixes": [_checked(key)]}
        )
        if response.status_code != 200:
            raise self._failure("delete", response)

    def close(self) -> None:
        self._client.close()

    def _send(self, method: str, path: str, **kwargs: Any) -> httpx2.Response:
        """Send, retrying network errors, 429, and 5xx with exponential backoff."""
        attempt = 0
        while True:
            try:
                response = self._client.request(method, path, **kwargs)
                retryable = response.status_code == 429 or response.status_code >= 500
                if not retryable or attempt >= self._max_retries:
                    return response
                reason = f"http_{response.status_code}"
            except httpx2.TransportError as exc:
                if attempt >= self._max_retries:
                    logger.warning(
                        "file_storage.unreachable",
                        extra={"provider": "supabase", "reason": type(exc).__name__},
                    )
                    raise StorageUnavailable(FAILED) from exc
                reason = type(exc).__name__
            wait = min(MAX_BACKOFF_SECONDS, 2.0**attempt + random.uniform(0, 0.5))
            logger.warning(
                "file_storage.retrying",
                extra={
                    "provider": "supabase",
                    "reason": reason,
                    "attempt": attempt + 1,
                },
            )
            self._sleep(wait)
            attempt += 1

    def _failure(self, operation: str, response: httpx2.Response) -> StorageUnavailable:
        logger.error(
            "file_storage.failed",
            extra={
                "provider": "supabase",
                "operation": operation,
                "status": response.status_code,
                "detail": response.text[:200],
            },
        )
        return StorageUnavailable(FAILED)


def _checked(key: str) -> str:
    if not _SAFE_KEY.match(key) or ".." in key:
        raise ValueError(f"Unsafe file key: {key!r}")
    return key


def _content_type(key: str) -> str:
    return _CONTENT_TYPES.get(
        key.rsplit(".", 1)[-1].lower(), "application/octet-stream"
    )


def _is_not_found(response: httpx2.Response) -> bool:
    """Supabase reports a missing bucket as 404, or as 400 with "not found" in the body."""
    return response.status_code == 404 or (
        response.status_code == 400 and _mentions(response, "not found")
    )


def _mentions(response: httpx2.Response, phrase: str) -> bool:
    return phrase in response.text.lower()
