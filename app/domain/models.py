"""Core concepts of DocuMind, as plain Python objects.

The domain layer is the centre of the architecture: it imports nothing from
the rest of the app and no frameworks or vendor libraries. Everything else
depends on it, never the other way round.

`frozen=True` makes instances immutable, so a value cannot be changed by
accident after it is created.
"""

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class Section:
    """A piece of a document's text: one PDF page, or a whole DOCX/TXT file."""

    page_number: int | None
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


@dataclass(frozen=True)
class DocumentMetadata:
    title: str | None = None
    author: str | None = None


@dataclass(frozen=True)
class ParsedContent:
    """What a parser extracts from a file, before the app adds its own data."""

    sections: tuple[Section, ...]
    page_count: int | None = None
    metadata: DocumentMetadata = DocumentMetadata()
    warnings: tuple[str, ...] = ()

    @property
    def has_text(self) -> bool:
        return any(section.has_text for section in self.sections)


class ExtractionStatus(StrEnum):
    EXTRACTED = "extracted"
    NO_TEXT = "no_text"


Vector = tuple[float, ...]


@dataclass(frozen=True)
class EmbeddingModel:
    """Identifies the vector space: vectors are only comparable within one model."""

    name: str
    dimensions: int


@dataclass(frozen=True)
class Chunk:
    """A piece of one section's text, small enough to retrieve and cite on its own.

    `start_char` and `end_char` locate the chunk within its section's text.
    """

    index: int
    page_number: int | None
    text: str
    start_char: int
    end_char: int

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: Chunk
    vector: Vector


@dataclass(frozen=True)
class Document:
    """An uploaded document after text extraction, chunking, and embedding.

    `embedding_model` is None when there was no text to embed.
    """

    id: str
    filename: str
    file_type: str
    size_bytes: int
    status: ExtractionStatus
    content: ParsedContent
    chunks: tuple[EmbeddedChunk, ...] = ()
    embedding_model: EmbeddingModel | None = None
