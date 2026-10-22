"""Simulated Voyage AI / OpenAI embeddings endpoints, served in-process via httpx2.MockTransport.

Both vendors accept `{"input": [...], "model": ...}` and return
`{"data": [{"index", "embedding"}], "usage": {"total_tokens"}}`.

Tests script failures (status codes, timeouts, raw responses) that are
returned first, in order; after that every request succeeds.
"""

import hashlib
import json
from typing import Any

import httpx2

from app.infrastructure.embeddings.openai_embedding_provider import (
    OpenAIEmbeddingProvider,
)
from app.infrastructure.embeddings.voyage_embedding_provider import (
    VoyageEmbeddingProvider,
)

API_KEY = "test-voyage-key"
TIMEOUT = "timeout"

ScriptItem = int | str | httpx2.Response


def fake_vector(text: str, dimensions: int) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [(digest[i % len(digest)] - 127.5) / 127.5 for i in range(dimensions)]


class FakeVoyageServer:
    def __init__(
        self,
        script: list[ScriptItem] | None = None,
        reverse: bool = False,
        default_dimensions: int = 1536,
    ) -> None:
        self.script = list(script or [])
        self.default_dimensions = default_dimensions
        self.reverse = reverse
        self.requests: list[dict[str, Any]] = []
        self.auth_headers: list[str | None] = []
        self.paths: list[str] = []
        self.sleeps: list[float] = []

    def provider(self, dimensions: int = 8, **kwargs: Any) -> VoyageEmbeddingProvider:
        return VoyageEmbeddingProvider(
            api_key=API_KEY,
            model_name="voyage-4-lite",
            dimensions=dimensions,
            transport=httpx2.MockTransport(self._handle),
            sleep=self.sleeps.append,
            **kwargs,
        )

    def openai_provider(self, **kwargs: Any) -> OpenAIEmbeddingProvider:
        return OpenAIEmbeddingProvider(
            api_key=API_KEY,
            transport=httpx2.MockTransport(self._handle),
            sleep=self.sleeps.append,
            **kwargs,
        )

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        self.auth_headers.append(request.headers.get("authorization"))
        self.paths.append(request.url.path)

        if self.script:
            item = self.script.pop(0)
            if item == TIMEOUT:
                raise httpx2.ReadTimeout("timed out", request=request)
            if isinstance(item, httpx2.Response):
                return item
            return httpx2.Response(
                int(item), json={"detail": f"simulated error {item}"}
            )

        data = [
            {
                "object": "embedding",
                "embedding": fake_vector(text, self._dimensions(body)),
                "index": i,
            }
            for i, text in enumerate(body["input"])
        ]
        if self.reverse:
            data.reverse()
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "data": data,
                "model": body["model"],
                "usage": {"total_tokens": sum(len(t) // 4 + 1 for t in body["input"])},
            },
        )

    def _dimensions(self, body: dict[str, Any]) -> int:
        requested = body.get("output_dimension") or body.get("dimensions")
        return int(requested or self.default_dimensions)
