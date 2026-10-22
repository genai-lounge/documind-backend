"""Use case: how similar does the embedding model think two texts are?

Powers the similarity playground, a way to build intuition for what
embeddings capture before they are used for search.
"""

from dataclasses import dataclass

from app.application.ports import EmbeddingProvider
from app.domain.models import EmbeddingModel
from app.domain.similarity import cosine_similarity


@dataclass(frozen=True)
class SimilarityResult:
    similarity: float
    model: EmbeddingModel


class CompareTexts:
    def __init__(self, embedder: EmbeddingProvider) -> None:
        self._embedder = embedder

    def execute(self, text_a: str, text_b: str) -> SimilarityResult:
        """Embed both texts in one request (rate limits count requests) and compare."""
        vector_a, vector_b = self._embedder.embed_documents([text_a, text_b])
        return SimilarityResult(
            similarity=cosine_similarity(vector_a, vector_b),
            model=self._embedder.model,
        )
