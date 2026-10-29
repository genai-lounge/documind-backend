from datetime import UTC, datetime, timedelta

import pytest

from app.application.manage_documents import DeleteDocument, GetDocument, ListDocuments
from app.domain.errors import DocumentNotFound
from app.domain.models import (
    Chunk,
    DocumentRecord,
    DocumentStatus,
    EmbeddedChunk,
    EmbeddingModel,
)
from tests.fakes import (
    FakeEmbeddingProvider,
    InMemoryDocumentRepository,
    InMemoryFileStorage,
    InMemoryVectorStore,
)

EMBEDDER = FakeEmbeddingProvider(dimensions=4)
OTHER_MODEL = EmbeddingModel("older-model", 4)
T0 = datetime(2026, 10, 1, tzinfo=UTC)


def record(
    id_: str, minutes: int = 0, model: EmbeddingModel | None = None
) -> DocumentRecord:
    return DocumentRecord(
        id=id_,
        filename=f"{id_}.txt",
        file_type="txt",
        size_bytes=10,
        status=DocumentStatus.INDEXED,
        created_at=T0 + timedelta(minutes=minutes),
        chunk_count=2,
        embedding_model=model or EMBEDDER.model,
    )


def stored(vectors: InMemoryVectorStore, rec: DocumentRecord) -> None:
    assert rec.embedding_model is not None
    vectors.upsert(
        rec.embedding_model,
        rec.id,
        rec.filename,
        [
            EmbeddedChunk(Chunk(i, 1, f"text {i}", 0, 6), (1.0, 0.0, 0.0, float(i)))
            for i in (1, 0)
        ],
    )


def test_lists_newest_first_and_flags_other_models() -> None:
    repo = InMemoryDocumentRepository()
    repo.add(record("old", 0))
    repo.add(record("new", 5))
    repo.add(record("legacy", 3, OTHER_MODEL))

    overviews = ListDocuments(repo, EMBEDDER).execute()

    assert [o.record.id for o in overviews] == ["new", "legacy", "old"]
    assert [o.searchable for o in overviews] == [True, False, True]


def test_get_returns_stored_chunks_in_order() -> None:
    repo, vectors = InMemoryDocumentRepository(), InMemoryVectorStore()
    rec = record("doc")
    repo.add(rec)
    stored(vectors, rec)

    detail = GetDocument(repo, vectors, EMBEDDER).execute("doc")

    assert detail.overview.record == rec
    assert [c.index for c in detail.chunks] == [0, 1]


def test_get_unknown_document() -> None:
    with pytest.raises(DocumentNotFound):
        GetDocument(
            InMemoryDocumentRepository(), InMemoryVectorStore(), EMBEDDER
        ).execute("nope")


def test_delete_removes_vectors_file_and_record() -> None:
    repo, files, vectors = (
        InMemoryDocumentRepository(),
        InMemoryFileStorage(),
        InMemoryVectorStore(),
    )
    rec = record("doc")
    repo.add(rec)
    files.save(rec.file_key, b"data")
    stored(vectors, rec)
    stored(vectors, record("keep"))

    DeleteDocument(repo, files, vectors).execute("doc")

    assert repo.get("doc") is None
    assert files.files == {}
    assert vectors.document_chunks(EMBEDDER.model, "doc") == []
    assert len(vectors.document_chunks(EMBEDDER.model, "keep")) == 2


def test_delete_uses_the_documents_own_model() -> None:
    repo, files, vectors = (
        InMemoryDocumentRepository(),
        InMemoryFileStorage(),
        InMemoryVectorStore(),
    )
    rec = record("legacy", model=OTHER_MODEL)
    repo.add(rec)
    stored(vectors, rec)

    DeleteDocument(repo, files, vectors).execute("legacy")

    assert vectors.count(OTHER_MODEL) == 0


def test_delete_unknown_document() -> None:
    with pytest.raises(DocumentNotFound):
        DeleteDocument(
            InMemoryDocumentRepository(), InMemoryFileStorage(), InMemoryVectorStore()
        ).execute("nope")
