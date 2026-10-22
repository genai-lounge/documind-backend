"""EmbeddingProvider adapter for Voyage AI's embeddings API, over plain HTTPS.

Calls the REST endpoint directly (no SDK) through the shared HttpEmbeddingClient.
See docs/adr/0009-voyage-embeddings-over-http.md.
"""

import time
from collections.abc import Callable, Sequence
from typing import Literal

import httpx2

from app.application.ports import EmbeddingProvider
from app.domain.models import EmbeddingModel, Vector
from app.infrastructure.embeddings.http_embedding_client import (
    HttpEmbeddingClient,
    embed_in_batches,
    parse_embedding_data,
    total_tokens,
)

DEFAULT_BASE_URL = "https://api.voyageai.com/v1"
DEFAULT_MODEL = "voyage-4-lite"
SUPPORTED_DIMENSIONS = (256, 512, 1024, 2048)

InputType = Literal["document", "query"]


class VoyageEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_MODEL,
        dimensions: int = 1024,
        *,
        batch_size: int = 128,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        base_url: str = DEFAULT_BASE_URL,
        transport: httpx2.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = EmbeddingModel(name=model_name, dimensions=dimensions)
        self._batch_size = batch_size
        self._client = HttpEmbeddingClient(
            provider="voyage",
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            rate_limit_hint="Accounts without a payment method have very low limits "
            "(about 3 requests/minute); adding one raises them.",
            auth_hint="Check VOYAGE_API_KEY.",
            transport=transport,
            sleep=sleep,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> Vector:
        return self._embed([text], "query")[0]

    def close(self) -> None:
        self._client.close()

    def _embed(self, texts: Sequence[str], input_type: InputType) -> list[Vector]:
        def request_batch(batch: list[str]) -> tuple[list[Vector], int]:
            body = self._client.post_json(
                "/embeddings",
                {
                    "input": batch,
                    "model": self.model.name,
                    "input_type": input_type,
                    "output_dimension": self.model.dimensions,
                    "truncation": True,
                },
            )
            return parse_embedding_data(body, len(batch)), total_tokens(body)

        return embed_in_batches(
            texts,
            self._batch_size,
            request_batch,
            provider="voyage",
            model=self.model.name,
            input_type=input_type,
        )
