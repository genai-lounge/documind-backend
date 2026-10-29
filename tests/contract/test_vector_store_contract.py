"""Contract tests: rules every vector store (writer + searcher + reader) must follow.

Runs against the in-memory fake and the real Chroma in a temporary folder;
Qdrant (Phase 9) will be added to STORES and must pass the same tests.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import pytest

from app.domain.models import (
    Chunk,
    EmbeddedChunk,
    EmbeddingModel,
    SearchHit,
    Vector,
)
from app.infrastructure.vectorstores.chroma_vector_store import ChromaVectorStore
from tests.fakes import InMemoryVectorStore

MODEL = EmbeddingModel("model-a", 3)
OTHER = EmbeddingModel("model-b", 3)


class Store(Protocol):
    """Any store implementing all three vector ports (matched by shape)."""

    def upsert(
        self,
        model: EmbeddingModel,
        document_id: str,
        filename: str,
        chunks: Sequence[EmbeddedChunk],
    ) -> None: ...

    def delete_document(self, model: EmbeddingModel, document_id: str) -> None: ...

    def search(
        self, model: EmbeddingModel, vector: Vector, top_k: int
    ) -> list[SearchHit]: ...

    def document_chunks(
        self, model: EmbeddingModel, document_id: str
    ) -> list[Chunk]: ...


STORES: dict[str, Callable[[Path], Store]] = {
    "InMemoryVectorStore": lambda _: InMemoryVectorStore(),
    "ChromaVectorStore": lambda path: ChromaVectorStore.local(path / "chroma"),
}


@pytest.fixture(params=list(STORES))
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Store:
    return STORES[request.param](tmp_path)


def chunk(index: int, vector: tuple[float, ...], page: int | None = 1) -> EmbeddedChunk:
    return EmbeddedChunk(
        Chunk(index, page, f"text {index}", index * 10, index * 10 + 7), vector
    )


def test_search_returns_the_closest_chunks_best_first(store: Store) -> None:
    store.upsert(
        MODEL,
        "doc",
        "a.txt",
        [
            chunk(0, (1.0, 0.0, 0.0)),
            chunk(1, (0.0, 1.0, 0.0)),
            chunk(2, (0.7, 0.7, 0.0)),
        ],
    )

    hits = store.search(MODEL, (1.0, 0.0, 0.0), top_k=2)

    assert [h.chunk.index for h in hits] == [0, 2]
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert hits[0].score > hits[1].score
    assert all(-1.0 <= h.score <= 1.0 for h in hits)


def test_hits_carry_source_details(store: Store) -> None:
    store.upsert(MODEL, "doc", "report.pdf", [chunk(4, (1.0, 0.0, 0.0), page=7)])

    [hit] = store.search(MODEL, (1.0, 0.0, 0.0), top_k=5)

    assert (hit.document_id, hit.filename) == ("doc", "report.pdf")
    assert hit.chunk == Chunk(4, 7, "text 4", 40, 47)


def test_page_number_none_survives_the_round_trip(store: Store) -> None:
    store.upsert(MODEL, "doc", "a.docx", [chunk(0, (1.0, 0.0, 0.0), page=None)])

    assert store.document_chunks(MODEL, "doc")[0].page_number is None


def test_models_are_kept_apart(store: Store) -> None:
    store.upsert(MODEL, "doc-a", "a.txt", [chunk(0, (1.0, 0.0, 0.0))])
    store.upsert(OTHER, "doc-b", "b.txt", [chunk(0, (1.0, 0.0, 0.0))])

    assert [h.document_id for h in store.search(MODEL, (1.0, 0.0, 0.0), 10)] == [
        "doc-a"
    ]
    assert [h.document_id for h in store.search(OTHER, (1.0, 0.0, 0.0), 10)] == [
        "doc-b"
    ]


def test_upsert_replaces_chunks_with_the_same_index(store: Store) -> None:
    store.upsert(MODEL, "doc", "a.txt", [chunk(0, (1.0, 0.0, 0.0))])
    store.upsert(MODEL, "doc", "a.txt", [chunk(0, (0.0, 1.0, 0.0))])

    hits = store.search(MODEL, (0.0, 1.0, 0.0), 10)

    assert len(hits) == 1
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)


def test_delete_removes_only_that_document(store: Store) -> None:
    store.upsert(
        MODEL, "gone", "a.txt", [chunk(0, (1.0, 0.0, 0.0)), chunk(1, (0.0, 1.0, 0.0))]
    )
    store.upsert(MODEL, "kept", "b.txt", [chunk(0, (1.0, 0.0, 0.0))])

    store.delete_document(MODEL, "gone")

    assert store.document_chunks(MODEL, "gone") == []
    assert [h.document_id for h in store.search(MODEL, (1.0, 0.0, 0.0), 10)] == ["kept"]


def test_deleting_an_unknown_document_is_a_no_op(store: Store) -> None:
    store.delete_document(MODEL, "never-existed")


def test_document_chunks_are_ordered_by_index(store: Store) -> None:
    store.upsert(
        MODEL,
        "doc",
        "a.txt",
        [
            chunk(2, (0.0, 0.0, 1.0)),
            chunk(0, (1.0, 0.0, 0.0)),
            chunk(1, (0.0, 1.0, 0.0)),
        ],
    )

    assert [c.index for c in store.document_chunks(MODEL, "doc")] == [0, 1, 2]


def test_searching_an_empty_store_finds_nothing(store: Store) -> None:
    assert store.search(MODEL, (1.0, 0.0, 0.0), 5) == []
