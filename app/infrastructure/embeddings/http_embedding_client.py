"""Shared HTTPS plumbing for hosted embedding APIs (Voyage AI, OpenAI, ...).

Owns everything vendor-neutral: authentication header, timeouts, retries with
exponential backoff, and turning HTTP failures into domain errors. Each vendor
adapter only builds its request body and reads its response.
"""

import logging
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import batched
from typing import Any

import httpx2

from app.domain.errors import EmbeddingRateLimited, EmbeddingUnavailable

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 10.0

RATE_LIMITED = (
    "The embedding service is busy right now (rate limit reached). "
    "Please try again in a minute."
)
UNAVAILABLE = (
    "The embedding service is temporarily unavailable. Please try again shortly."
)
MISCONFIGURED = "The embedding service is not configured correctly."
REJECTED = "The embedding service rejected the request."
UNEXPECTED = "The embedding service returned an unexpected response."


@dataclass(frozen=True)
class _Failure:
    error: EmbeddingUnavailable
    retryable: bool
    reason: str
    retry_after: float | None = None


class HttpEmbeddingClient:
    def __init__(
        self,
        provider: str,
        base_url: str,
        api_key: str,
        *,
        timeout_seconds: float,
        max_retries: int,
        rate_limit_hint: str = "",
        auth_hint: str = "Check the API key.",
        transport: httpx2.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError(f"An API key is required for {provider}.")
        self.provider = provider
        self._max_retries = max_retries
        self._rate_limit_hint = rate_limit_hint
        self._auth_hint = auth_hint
        self._sleep = sleep
        self._client = httpx2.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx2.Timeout(timeout_seconds, connect=5.0),
            transport=transport,
        )

    def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        """POST and return the parsed JSON body, retrying transient failures."""
        attempt = 0
        while True:
            try:
                response = self._client.post(path, json=payload)
            except httpx2.TransportError as exc:
                failure = _Failure(
                    EmbeddingUnavailable(UNAVAILABLE), True, type(exc).__name__
                )
            else:
                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise EmbeddingUnavailable(UNEXPECTED) from exc
                failure = self._failure_for(response)

            wait = _backoff_seconds(attempt, failure.retry_after)
            if not failure.retryable or attempt >= self._max_retries or wait is None:
                logger.warning(
                    "embedding.failed",
                    extra={
                        "provider": self.provider,
                        "reason": failure.reason,
                        "attempts": attempt + 1,
                    },
                )
                raise failure.error

            logger.warning(
                "embedding.retrying",
                extra={
                    "provider": self.provider,
                    "reason": failure.reason,
                    "attempt": attempt + 1,
                    "wait_seconds": round(wait, 2),
                },
            )
            self._sleep(wait)
            attempt += 1

    def close(self) -> None:
        self._client.close()

    def _failure_for(self, response: httpx2.Response) -> _Failure:
        status = response.status_code
        reason = f"http_{status}"
        detail = _error_detail(response)

        if status == 429:
            retry_after = _retry_after_seconds(response)
            logger.warning(
                "embedding.rate_limited",
                extra={
                    "provider": self.provider,
                    "detail": detail,
                    "hint": self._rate_limit_hint,
                },
            )
            return _Failure(
                EmbeddingRateLimited(RATE_LIMITED, retry_after),
                True,
                reason,
                retry_after,
            )
        if status >= 500:
            return _Failure(EmbeddingUnavailable(UNAVAILABLE), True, reason)
        if status in (401, 403):
            logger.error(
                "embedding.unauthorized",
                extra={
                    "provider": self.provider,
                    "detail": detail,
                    "hint": self._auth_hint,
                },
            )
            return _Failure(EmbeddingUnavailable(MISCONFIGURED), False, reason)
        logger.error(
            "embedding.rejected", extra={"provider": self.provider, "detail": detail}
        )
        return _Failure(EmbeddingUnavailable(REJECTED), False, reason)


def parse_embedding_data(body: Any, expected: int) -> list[tuple[float, ...]]:
    """Read the `data: [{index, embedding}]` list both Voyage and OpenAI return."""
    try:
        items = sorted(body["data"], key=lambda item: item["index"])
        vectors = [tuple(float(x) for x in item["embedding"]) for item in items]
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise EmbeddingUnavailable(UNEXPECTED) from exc
    if len(vectors) != expected:
        raise EmbeddingUnavailable(UNEXPECTED)
    return vectors


def total_tokens(body: Any) -> int:
    try:
        return int(body.get("usage", {}).get("total_tokens", 0))
    except (AttributeError, TypeError, ValueError):
        return 0


def _backoff_seconds(attempt: int, retry_after: float | None) -> float | None:
    """1s, 2s, 4s, ... plus jitter, capped. None if the server asks us to wait too long."""
    if retry_after is not None:
        return retry_after if retry_after <= MAX_BACKOFF_SECONDS else None
    return min(MAX_BACKOFF_SECONDS, 2.0**attempt + random.uniform(0, 1))


def _retry_after_seconds(response: httpx2.Response) -> float | None:
    try:
        return max(0.0, float(response.headers["retry-after"]))
    except (KeyError, ValueError):
        return None


def _error_detail(response: httpx2.Response) -> str:
    try:
        body = response.json()
        detail = body.get("detail") or body.get("error", {}).get("message", "")
    except (ValueError, AttributeError):
        detail = response.text
    return str(detail)[:200]


def embed_in_batches(
    texts: Sequence[str],
    batch_size: int,
    request_batch: Callable[[list[str]], tuple[list[tuple[float, ...]], int]],
    *,
    provider: str,
    model: str,
    input_type: str,
) -> list[tuple[float, ...]]:
    """Embed `texts` in batches and log one `embedding.completed` line with usage."""
    if not texts:
        return []
    started = time.perf_counter()
    vectors: list[tuple[float, ...]] = []
    tokens = 0
    batches = 0
    for batch in batched(texts, batch_size):
        batch_vectors, batch_tokens = request_batch(list(batch))
        vectors.extend(batch_vectors)
        tokens += batch_tokens
        batches += 1
    logger.info(
        "embedding.completed",
        extra={
            "provider": provider,
            "model": model,
            "input_type": input_type,
            "texts": len(texts),
            "batches": batches,
            "total_tokens": tokens,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )
    return vectors
