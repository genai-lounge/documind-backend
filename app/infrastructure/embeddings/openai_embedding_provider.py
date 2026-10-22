"""EmbeddingProvider adapter for OpenAI's embeddings API, over plain HTTPS.

OpenAI embeds documents and queries the same way (there is no input type).
Vectors are normalised to length 1. The `dimensions` parameter shortens
text-embedding-3 vectors and is only sent when explicitly configured.
"""

import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx2

from app.application.ports import EmbeddingProvider
from app.domain.models import EmbeddingModel, Vector
from app.infrastructure.embeddings.http_embedding_client import (
    HttpEmbeddingClient,
    embed_in_batches,
    parse_embedding_data,
    total_tokens,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "text-embedding-3-small"
NATIVE_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
SHORTENABLE_MODELS = frozenset({"text-embedding-3-small", "text-embedding-3-large"})


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_MODEL,
        dimensions: int | None = None,
        *,
        batch_size: int = 128,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        base_url: str = DEFAULT_BASE_URL,
        transport: httpx2.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        resolved = dimensions or NATIVE_DIMENSIONS.get(model_name)
        if resolved is None:
            raise ValueError(
                f"Unknown dimensions for OpenAI model '{model_name}'; set them explicitly."
            )
        self.model = EmbeddingModel(name=model_name, dimensions=resolved)
        self._send_dimensions = dimensions is not None
        self._batch_size = batch_size
        self._client = HttpEmbeddingClient(
            provider="openai",
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            rate_limit_hint="Check your OpenAI usage tier and rate limits.",
            auth_hint="Check OPENAI_API_KEY.",
            transport=transport,
            sleep=sleep,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> Vector:
        return self._embed([text], "query")[0]

    def close(self) -> None:
        self._client.close()

    def _embed(self, texts: Sequence[str], input_type: str) -> list[Vector]:
        def request_batch(batch: list[str]) -> tuple[list[Vector], int]:
            payload: dict[str, Any] = {
                "input": batch,
                "model": self.model.name,
                "encoding_format": "float",
            }
            if self._send_dimensions:
                payload["dimensions"] = self.model.dimensions
            body = self._client.post_json("/embeddings", payload)
            return parse_embedding_data(body, len(batch)), total_tokens(body)

        return embed_in_batches(
            texts,
            self._batch_size,
            request_batch,
            provider="openai",
            model=self.model.name,
            input_type=input_type,
        )
