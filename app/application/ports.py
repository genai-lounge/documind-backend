"""Ports: the interfaces the application needs from the outside world.

A port says WHAT the app needs ("turn a PDF's bytes into text") without
saying HOW or WITH WHICH LIBRARY. Concrete implementations (adapters) live in
app/infrastructure/ and are plugged in by the composition root
(app/container.py). This is the Dependency Inversion Principle: the use cases
depend on these abstractions, and the vendor code depends on them too.

Ports are kept small (Interface Segregation): a caller only depends on the
methods it actually uses. Later phases add ports such as FileStorage,
VectorStore, and LLMProvider.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.domain.models import Chunk, EmbeddingModel, ParsedContent, Section, Vector


class DocumentParser(ABC):
    """Extracts text from the bytes of one file format.

    Contract every adapter must follow (checked by tests/contract/):
    - `parse` raises DocumentParseError for any problem with the content and
      never lets a library-specific exception escape.
    - `parse` returns at least one Section.
    """

    format_name: str

    @abstractmethod
    def matches_content(self, data: bytes) -> bool:
        """Cheaply check that the bytes really are this format.

        Used to reject renamed files (e.g. an .exe called report.pdf) before
        parsing. Return True for formats that have no reliable signature.
        """

    @abstractmethod
    def parse(self, data: bytes) -> ParsedContent:
        """Extract the text, page structure, metadata, and warnings."""


class Chunker(ABC):
    """Splits a document's sections into chunks.

    Contract every adapter must follow (checked by tests/contract/):
    - chunks are numbered 0, 1, 2, ... across the whole document;
    - each chunk lies within one section and keeps that section's page number;
    - `chunk.text == section.text[chunk.start_char:chunk.end_char]`;
    - no chunk is empty, and sections without text produce no chunks.
    """

    @abstractmethod
    def chunk(self, sections: Sequence[Section]) -> list[Chunk]: ...


class EmbeddingProvider(ABC):
    """Turns text into vectors that capture its meaning.

    Documents and queries are embedded separately because some models encode
    them slightly differently, so a question lands near the passages that
    answer it. Vectors from different models must never be compared.

    Contract every adapter must follow (checked by tests/contract/):
    - one vector per input text, in the same order;
    - every vector has exactly `model.dimensions` finite numbers;
    - service failures raise EmbeddingUnavailable (or EmbeddingRateLimited),
      never a library-specific exception.
    """

    model: EmbeddingModel

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...

    @abstractmethod
    def embed_query(self, text: str) -> Vector: ...
