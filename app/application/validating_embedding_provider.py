"""A decorator that checks every embedding provider's output.

It implements the same EmbeddingProvider port and wraps any other provider,
so the checks apply to every vendor without changing their adapters
(Open/Closed Principle). Bad output becomes EmbeddingUnavailable instead of
corrupting similarity scores or, later, the vector store.
"""

import math
from collections.abc import Sequence

from app.application.ports import EmbeddingProvider
from app.domain.errors import EmbeddingUnavailable
from app.domain.models import Vector


class ValidatingEmbeddingProvider(EmbeddingProvider):
    def __init__(self, inner: EmbeddingProvider) -> None:
        self._inner = inner
        self.model = inner.model

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        vectors = self._inner.embed_documents(texts)
        if len(vectors) != len(texts):
            raise EmbeddingUnavailable(
                f"The embedding service returned {len(vectors)} vectors "
                f"for {len(texts)} texts."
            )
        for vector in vectors:
            self._check(vector)
        return vectors

    def embed_query(self, text: str) -> Vector:
        vector = self._inner.embed_query(text)
        self._check(vector)
        return vector

    def _check(self, vector: Vector) -> None:
        if len(vector) != self.model.dimensions:
            raise EmbeddingUnavailable(
                f"The embedding service returned a vector of length {len(vector)}; "
                f"expected {self.model.dimensions}."
            )
        if not all(math.isfinite(x) for x in vector):
            raise EmbeddingUnavailable(
                "The embedding service returned invalid numbers."
            )
        if not any(vector):
            raise EmbeddingUnavailable(
                "The embedding service returned an empty vector."
            )
