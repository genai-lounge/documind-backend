"""Ports: the interfaces the application needs from the outside world.

A port says WHAT the app needs ("turn a PDF's bytes into text") without
saying HOW or WITH WHICH LIBRARY. Concrete implementations (adapters) live in
app/infrastructure/ and are plugged in by the composition root
(app/container.py). This is the Dependency Inversion Principle: the use cases
depend on these abstractions, and the vendor code depends on them too.

Ports are kept small (Interface Segregation): a caller only depends on the
methods it actually uses. Later phases add ports such as Chunker,
EmbeddingProvider, FileStorage, VectorStore, and LLMProvider.
"""

from abc import ABC, abstractmethod

from app.domain.models import ParsedContent


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
