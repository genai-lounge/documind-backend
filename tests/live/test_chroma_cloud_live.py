"""Live test against Chroma Cloud. Run with `make test-live`.

Skipped unless CHROMA_MODE=cloud and CHROMA_API_KEY are set. Uses a throwaway
collection with made-up vectors and deletes it afterwards (a few small
reads and writes against your Chroma Cloud credits).
"""

import uuid

import pytest

from app.core.config import Settings
from app.domain.models import Chunk, EmbeddedChunk, EmbeddingModel
from app.infrastructure.vectorstores.chroma_vector_store import collection_name
from app.vector_stores import build_vector_store

pytestmark = pytest.mark.live


def test_upsert_search_delete_round_trip() -> None:
    settings = Settings()
    if settings.chroma_mode != "cloud" or settings.chroma_api_key is None:
        pytest.skip("CHROMA_MODE=cloud with CHROMA_API_KEY is not configured")

    store = build_vector_store(settings)
    model = EmbeddingModel(f"documind-live-test-{uuid.uuid4().hex[:8]}", 3)
    try:
        store.upsert(
            model,
            "doc",
            "a.txt",
            [
                EmbeddedChunk(Chunk(0, 1, "revenue", 0, 7), (1.0, 0.0, 0.0)),
                EmbeddedChunk(Chunk(1, 2, "cat", 0, 3), (0.0, 1.0, 0.0)),
            ],
        )

        hits = store.search(model, (0.9, 0.1, 0.0), 2)
        assert [h.chunk.text for h in hits] == ["revenue", "cat"]
        assert hits[0].score == pytest.approx(0.9939, abs=1e-3)

        store.delete_document(model, "doc")
        assert store.document_chunks(model, "doc") == []
    finally:
        store._client.delete_collection(collection_name(model))
