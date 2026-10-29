"""Use case: accept an upload, then extract, chunk, embed, store, and index it.

A use case describes one thing the application does, step by step, using only
ports and domain objects. It does not know about HTTP, FastAPI, or which
library parses PDFs, so it can be tested with fakes and reused from other
entry points (a CLI, a background worker).

Three stores (files, records, vectors) can't share one transaction, so the
order is chosen so the app never claims success it didn't achieve:

    validate, parse, chunk          nothing stored yet; a bad file leaves no trace
    record "processing" -> save file -> embed -> index vectors
    record "indexed" (or "no_text")
    on any failure after the record exists: remove vectors, record "failed"

The original file is kept on failure so the document can be retried later.
"""

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import PurePath
from typing import BinaryIO

from app.application.parser_registry import ParserRegistry
from app.application.ports import (
    Chunker,
    DocumentRepository,
    EmbeddingProvider,
    FileStorage,
    VectorIndexWriter,
)
from app.domain.errors import (
    DocuMindError,
    EmptyFile,
    FileTooLarge,
    FileTypeMismatch,
    MissingFilename,
    UnsupportedFileType,
)
from app.domain.models import (
    Chunk,
    Document,
    DocumentRecord,
    DocumentStatus,
    EmbeddedChunk,
    ExtractionStatus,
    ParsedContent,
)

logger = logging.getLogger(__name__)

NO_TEXT_WARNING = "No text was found in this file."


def _new_document_id() -> str:
    return uuid.uuid4().hex


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class IngestionResult:
    document: Document
    record: DocumentRecord


class IngestDocument:
    def __init__(
        self,
        parsers: ParserRegistry,
        chunker: Chunker,
        embedder: EmbeddingProvider,
        files: FileStorage,
        documents: DocumentRepository,
        vectors: VectorIndexWriter,
        max_upload_bytes: int,
        new_id: Callable[[], str] = _new_document_id,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._parsers = parsers
        self._chunker = chunker
        self._embedder = embedder
        self._files = files
        self._documents = documents
        self._vectors = vectors
        self._max_upload_bytes = max_upload_bytes
        self._new_id = new_id
        self._now = now

    def execute(self, filename: str, stream: BinaryIO) -> IngestionResult:
        """Checks run cheapest first: a bad request fails before any parsing,
        and parsing problems surface before anything is stored or paid for."""
        started = time.perf_counter()

        name = PurePath(filename).name
        if not name:
            raise MissingFilename("The uploaded file has no name.")

        extension = PurePath(name).suffix.lower()
        parser = self._parsers.get(extension)
        if parser is None:
            raise UnsupportedFileType(
                f"'{extension or name}' files are not supported. "
                f"Upload a {self._supported_formats()} file."
            )

        data = stream.read(self._max_upload_bytes + 1)
        if len(data) > self._max_upload_bytes:
            raise FileTooLarge(
                f"The file is larger than the {_format_size(self._max_upload_bytes)} limit."
            )
        if not data:
            raise EmptyFile("The file is empty.")

        if not parser.matches_content(data):
            raise FileTypeMismatch(
                f"The file has a {extension} name but is not a {parser.format_name}."
            )

        content = _with_no_text_warning(parser.parse(data))
        chunks = self._chunker.chunk(content.sections)

        record = DocumentRecord(
            id=self._new_id(),
            filename=name,
            file_type=extension.removeprefix("."),
            size_bytes=len(data),
            status=DocumentStatus.PROCESSING,
            created_at=self._now(),
            page_count=content.page_count,
            chunk_count=len(chunks),
            embedding_model=self._embedder.model if chunks else None,
        )
        self._documents.add(record)

        try:
            embedded = self._store_and_index(record, data, chunks)
            record = replace(
                record,
                status=DocumentStatus.INDEXED if chunks else DocumentStatus.NO_TEXT,
            )
            self._documents.update(record)
        except Exception as exc:
            self._mark_failed(record, exc)
            raise

        logger.info(
            "document.ingested",
            extra={
                "document_id": record.id,
                "file_type": record.file_type,
                "size_bytes": record.size_bytes,
                "sections": len(content.sections),
                "chunks": len(chunks),
                "status": record.status.value,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        document = Document(
            id=record.id,
            filename=record.filename,
            file_type=record.file_type,
            size_bytes=record.size_bytes,
            status=ExtractionStatus.EXTRACTED
            if content.has_text
            else ExtractionStatus.NO_TEXT,
            content=content,
            chunks=embedded,
            embedding_model=record.embedding_model,
        )
        return IngestionResult(document=document, record=record)

    def _store_and_index(
        self, record: DocumentRecord, data: bytes, chunks: list[Chunk]
    ) -> tuple[EmbeddedChunk, ...]:
        self._files.save(record.file_key, data)
        if not chunks or record.embedding_model is None:
            return ()
        vectors = self._embedder.embed_documents([c.text for c in chunks])
        embedded = tuple(
            EmbeddedChunk(chunk=c, vector=v)
            for c, v in zip(chunks, vectors, strict=True)
        )
        self._vectors.upsert(
            record.embedding_model, record.id, record.filename, embedded
        )
        return embedded

    def _mark_failed(self, record: DocumentRecord, error: Exception) -> None:
        """Best effort: a failure here must not hide the original error."""
        code = error.code if isinstance(error, DocuMindError) else "internal_error"
        if record.embedding_model is not None:
            try:
                self._vectors.delete_document(record.embedding_model, record.id)
            except Exception:
                logger.exception(
                    "document.cleanup_failed", extra={"document_id": record.id}
                )
        try:
            self._documents.update(
                replace(record, status=DocumentStatus.FAILED, error_code=code)
            )
        except Exception:
            logger.exception(
                "document.mark_failed_failed", extra={"document_id": record.id}
            )
        logger.warning(
            "document.failed", extra={"document_id": record.id, "error_code": code}
        )

    def _supported_formats(self) -> str:
        """E.g. (".pdf", ".docx", ".txt") -> "PDF, DOCX, or TXT"."""
        names = [
            ext.removeprefix(".").upper() for ext in self._parsers.supported_extensions
        ]
        if len(names) <= 2:
            return " or ".join(names)
        return ", ".join(names[:-1]) + ", or " + names[-1]


def _with_no_text_warning(content: ParsedContent) -> ParsedContent:
    """If nothing was extracted and the parser gave no reason, add a generic warning."""
    if content.has_text or content.warnings:
        return content
    return replace(content, warnings=(NO_TEXT_WARNING,))


def _format_size(size_bytes: int) -> str:
    megabyte = 1024 * 1024
    if size_bytes % megabyte == 0:
        return f"{size_bytes // megabyte} MB"
    return f"{size_bytes} bytes"
