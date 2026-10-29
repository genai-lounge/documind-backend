"""Use cases for stored documents: list them, inspect one, delete one."""

import logging
from dataclasses import dataclass

from app.application.ports import (
    DocumentRepository,
    EmbeddingProvider,
    FileStorage,
    VectorIndexReader,
    VectorIndexWriter,
)
from app.domain.errors import DocumentNotFound
from app.domain.models import Chunk, DocumentRecord, EmbeddingModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentOverview:
    record: DocumentRecord
    # False when the document was indexed with a different embedding model than
    # the one configured now: search can't see it until it is re-indexed.
    searchable: bool


@dataclass(frozen=True)
class DocumentDetail:
    overview: DocumentOverview
    chunks: list[Chunk]


def _overview(record: DocumentRecord, current: EmbeddingModel) -> DocumentOverview:
    return DocumentOverview(record=record, searchable=record.embedding_model == current)


class ListDocuments:
    def __init__(
        self, documents: DocumentRepository, embedder: EmbeddingProvider
    ) -> None:
        self._documents = documents
        self._embedder = embedder

    def execute(self) -> list[DocumentOverview]:
        return [_overview(r, self._embedder.model) for r in self._documents.list()]


class GetDocument:
    def __init__(
        self,
        documents: DocumentRepository,
        vectors: VectorIndexReader,
        embedder: EmbeddingProvider,
    ) -> None:
        self._documents = documents
        self._vectors = vectors
        self._embedder = embedder

    def execute(self, document_id: str) -> DocumentDetail:
        record = self._documents.get(document_id)
        if record is None:
            raise DocumentNotFound("No document with this ID exists.")
        chunks = (
            self._vectors.document_chunks(record.embedding_model, record.id)
            if record.embedding_model is not None
            else []
        )
        return DocumentDetail(_overview(record, self._embedder.model), chunks)


class DeleteDocument:
    """Removes vectors, then the file, then the record.

    The record goes last so an interrupted delete can simply be retried; every
    step is a no-op when its data is already gone.
    """

    def __init__(
        self,
        documents: DocumentRepository,
        files: FileStorage,
        vectors: VectorIndexWriter,
    ) -> None:
        self._documents = documents
        self._files = files
        self._vectors = vectors

    def execute(self, document_id: str) -> None:
        record = self._documents.get(document_id)
        if record is None:
            raise DocumentNotFound("No document with this ID exists.")
        if record.embedding_model is not None:
            self._vectors.delete_document(record.embedding_model, record.id)
        self._files.delete(record.file_key)
        self._documents.delete(record.id)
        logger.info("document.deleted", extra={"document_id": record.id})
