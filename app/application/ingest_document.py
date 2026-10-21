"""Use case: accept an uploaded file and extract its text.

A use case describes one thing the application does, step by step, using only
ports and domain objects. It does not know about HTTP, FastAPI, or which
library parses PDFs, so it can be tested with fakes and reused from other
entry points (a CLI, a background worker).

Later phases extend this flow: chunking and embeddings (Phase 2), then file
storage and indexing (Phase 3).
"""

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import PurePath
from typing import BinaryIO

from app.application.parser_registry import ParserRegistry
from app.domain.errors import (
    EmptyFile,
    FileTooLarge,
    FileTypeMismatch,
    MissingFilename,
    UnsupportedFileType,
)
from app.domain.models import Document, ExtractionStatus, ParsedContent

logger = logging.getLogger(__name__)

NO_TEXT_WARNING = "No text was found in this file."


def _new_document_id() -> str:
    return uuid.uuid4().hex


class IngestDocument:
    def __init__(
        self,
        parsers: ParserRegistry,
        max_upload_bytes: int,
        new_id: Callable[[], str] = _new_document_id,
    ) -> None:
        self._parsers = parsers
        self._max_upload_bytes = max_upload_bytes
        self._new_id = new_id

    def execute(self, filename: str, stream: BinaryIO) -> Document:
        """Validate the upload, parse it, and return the extracted document.

        Checks run cheapest first, so a bad request fails before any parsing.
        """
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
        document = Document(
            id=self._new_id(),
            filename=name,
            file_type=extension.removeprefix("."),
            size_bytes=len(data),
            status=ExtractionStatus.EXTRACTED
            if content.has_text
            else ExtractionStatus.NO_TEXT,
            content=content,
        )

        logger.info(
            "document.ingested",
            extra={
                "document_id": document.id,
                "file_type": document.file_type,
                "size_bytes": document.size_bytes,
                "sections": len(content.sections),
                "status": document.status.value,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return document

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
