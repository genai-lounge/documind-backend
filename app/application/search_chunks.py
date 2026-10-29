"""Use case: find the stored chunks closest in meaning to a question."""

import logging
import time
from dataclasses import dataclass

from app.application.ports import EmbeddingProvider, VectorSearcher
from app.domain.errors import InvalidInput
from app.domain.models import EmbeddingModel, SearchHit

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 2000


@dataclass(frozen=True)
class SearchResult:
    hits: list[SearchHit]
    model: EmbeddingModel


class SearchChunks:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        searcher: VectorSearcher,
        default_top_k: int,
        max_top_k: int,
    ) -> None:
        self._embedder = embedder
        self._searcher = searcher
        self._default_top_k = default_top_k
        self._max_top_k = max_top_k

    def execute(self, query: str, top_k: int | None = None) -> SearchResult:
        query = query.strip()
        if not query:
            raise InvalidInput("Enter a question or some text to search for.")
        if len(query) > MAX_QUERY_CHARS:
            raise InvalidInput(
                f"The query is longer than {MAX_QUERY_CHARS} characters."
            )
        k = self._default_top_k if top_k is None else top_k
        if not 1 <= k <= self._max_top_k:
            raise InvalidInput(f"top_k must be between 1 and {self._max_top_k}.")

        started = time.perf_counter()
        vector = self._embedder.embed_query(query)
        hits = self._searcher.search(self._embedder.model, vector, k)
        logger.info(
            "retrieval.searched",
            extra={
                "top_k": k,
                "hits": len(hits),
                "best_score": round(hits[0].score, 4) if hits else None,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return SearchResult(hits=hits, model=self._embedder.model)
