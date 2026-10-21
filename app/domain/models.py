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


@dataclass(frozen=True)
class Document:
    """An uploaded document after text extraction."""

    id: str
    filename: str
    file_type: str
    size_bytes: int
    status: ExtractionStatus
    content: ParsedContent
