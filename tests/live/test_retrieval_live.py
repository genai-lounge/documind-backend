"""Live retrieval check with the configured real embedding provider.

Run with `make test-live`. Skipped unless the provider chosen by
EMBEDDING_PROVIDER is configured. Two small requests with made-up text.
"""

import pytest

from app.application.search_chunks import SearchChunks
from app.core.config import ConfigurationError, Settings
from app.domain.models import Chunk, EmbeddedChunk
from app.embedding_providers import build_embedding_provider
from tests.fakes import InMemoryVectorStore
from tests.retrieval_dataset import DOCUMENTS, QUESTIONS

pytestmark = pytest.mark.live


def test_known_question_retrieves_the_expected_document() -> None:
    try:
        embedder = build_embedding_provider(Settings())
    except ConfigurationError as exc:
        pytest.skip(str(exc))

    texts = list(DOCUMENTS.values())
    vectors = InMemoryVectorStore()
    for (name, text), vector in zip(
        DOCUMENTS.items(), embedder.embed_documents(texts), strict=True
    ):
        vectors.upsert(
            embedder.model,
            name,
            name,
            [EmbeddedChunk(Chunk(0, 1, text, 0, len(text)), vector)],
        )

    question, expected = QUESTIONS[1]
    hits = SearchChunks(embedder, vectors, 3, 3).execute(question).hits

    assert hits[0].filename == expected
