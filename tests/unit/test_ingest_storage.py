"""IngestDocument's storage behaviour: order of steps, statuses, and failure cleanup."""

import io
from datetime import UTC, datetime

import pytest

from app.application.ingest_document import IngestDocument
from app.application.parser_registry import ParserRegistry
from app.domain.errors import (
    DocumentParseError,
    EmbeddingUnavailable,
    StorageUnavailable,
    UnsupportedFileType,
)
from app.domain.models import DocumentStatus, ParsedContent, Section
from tests.fakes import (
    FakeChunker,
    FakeEmbeddingProvider,
    FakeParser,
    InMemoryDocumentRepository,
    InMemoryFileStorage,
    InMemoryVectorStore,
)

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)


class Setup:
    def __init__(
        self,
        parser: FakeParser | None = None,
        embedder: FakeEmbeddingProvider | None = None,
        vectors: InMemoryVectorStore | None = None,
    ) -> None:
        self.files = InMemoryFileStorage()
        self.documents = InMemoryDocumentRepository()
        self.vectors = vectors or InMemoryVectorStore()
        self.embedder = embedder or FakeEmbeddingProvider()
        self.use_case = IngestDocument(
            parsers=ParserRegistry({".txt": parser or FakeParser()}),
            chunker=FakeChunker(),
            embedder=self.embedder,
            files=self.files,
            documents=self.documents,
            vectors=self.vectors,
            max_upload_bytes=1000,
            new_id=lambda: "doc1",
            now=lambda: NOW,
        )

    def ingest(self, data: bytes = b"hello") -> None:
        self.use_case.execute("notes.txt", io.BytesIO(data))


def test_successful_upload_stores_record_file_and_vectors() -> None:
    s = Setup()

    s.ingest(b"hello")

    record = s.documents.get("doc1")
    assert record is not None
    assert record.status is DocumentStatus.INDEXED
    assert (record.filename, record.file_type, record.size_bytes) == (
        "notes.txt",
        "txt",
        5,
    )
    assert record.created_at == NOW
    assert record.chunk_count == 1
    assert record.embedding_model == s.embedder.model
    assert s.files.files == {"doc1.txt": b"hello"}
    assert s.vectors.count(s.embedder.model) == 1


def test_result_carries_the_record() -> None:
    s = Setup()

    result = s.use_case.execute("notes.txt", io.BytesIO(b"hi"))

    assert result.record.status is DocumentStatus.INDEXED
    assert result.document.id == result.record.id == "doc1"


def test_document_without_text_is_stored_as_no_text() -> None:
    s = Setup(parser=FakeParser(ParsedContent(sections=(Section(1, "  "),))))

    s.ingest()

    record = s.documents.get("doc1")
    assert record is not None
    assert record.status is DocumentStatus.NO_TEXT
    assert record.embedding_model is None
    assert s.files.files
    assert s.embedder.calls == []


@pytest.mark.parametrize(
    ("parser", "filename", "error"),
    [
        (FakeParser(error=DocumentParseError("broken")), "a.txt", DocumentParseError),
        (FakeParser(), "a.png", UnsupportedFileType),
    ],
)
def test_rejected_uploads_leave_no_trace(
    parser: FakeParser, filename: str, error: type[Exception]
) -> None:
    s = Setup(parser=parser)

    with pytest.raises(error):
        s.use_case.execute(filename, io.BytesIO(b"abc"))

    assert s.documents.list() == []
    assert s.files.files == {}


def test_embedding_failure_marks_the_document_failed_and_keeps_the_file() -> None:
    s = Setup(embedder=FakeEmbeddingProvider(error=EmbeddingUnavailable("down")))

    with pytest.raises(EmbeddingUnavailable):
        s.ingest()

    record = s.documents.get("doc1")
    assert record is not None
    assert record.status is DocumentStatus.FAILED
    assert record.error_code == "embedding_unavailable"
    assert "doc1.txt" in s.files.files


def test_index_failure_removes_partial_vectors() -> None:
    vectors = InMemoryVectorStore(fail_on_upsert=StorageUnavailable("disk full"))
    s = Setup(vectors=vectors)

    with pytest.raises(StorageUnavailable):
        s.ingest()

    record = s.documents.get("doc1")
    assert record is not None
    assert record.status is DocumentStatus.FAILED
    assert record.error_code == "storage_unavailable"
    assert vectors.count(s.embedder.model) == 0


def test_unexpected_errors_are_recorded_as_internal_errors() -> None:
    s = Setup(embedder=FakeEmbeddingProvider(error=RuntimeError("bug")))

    with pytest.raises(RuntimeError):
        s.ingest()

    record = s.documents.get("doc1")
    assert record is not None
    assert record.error_code == "internal_error"
