"""Fake adapters for unit tests.

A fake implements a port with simple, predictable behaviour and no real
library behind it. Use-case tests run against fakes, so they are fast and
test only the use case's own logic.
"""

import hashlib
from collections.abc import Sequence

from app.application.ports import Chunker, DocumentParser, EmbeddingProvider
from app.domain.models import Chunk, EmbeddingModel, ParsedContent, Section, Vector


class FakeParser(DocumentParser):
    format_name = "fake document"

    def __init__(
        self,
        content: ParsedContent | None = None,
        signature_matches: bool = True,
        error: Exception | None = None,
    ) -> None:
        self.content = content or ParsedContent(sections=(Section(1, "some text"),))
        self.signature_matches = signature_matches
        self.error = error
        self.parsed: list[bytes] = []

    def matches_content(self, data: bytes) -> bool:
        return self.signature_matches

    def parse(self, data: bytes) -> ParsedContent:
        self.parsed.append(data)
        if self.error:
            raise self.error
        return self.content


class FakeChunker(Chunker):
    """One chunk per section that has text."""

    def chunk(self, sections: Sequence[Section]) -> list[Chunk]:
        with_text = [s for s in sections if s.has_text]
        return [
            Chunk(
                index=i,
                page_number=s.page_number,
                text=s.text,
                start_char=0,
                end_char=len(s.text),
            )
            for i, s in enumerate(with_text)
        ]


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic vectors derived from a hash of the text: same text, same vector.

    The vectors carry no meaning, but they let tests check counts, order,
    shapes, and error handling without a network or an API key.
    """

    def __init__(self, dimensions: int = 8, error: Exception | None = None) -> None:
        self.model = EmbeddingModel(name="fake-embedder", dimensions=dimensions)
        self.error = error
        self.calls: list[tuple[str, list[str]]] = []

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        self.calls.append(("document", list(texts)))
        if self.error:
            raise self.error
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        self.calls.append(("query", [text]))
        if self.error:
            raise self.error
        return self._vector(text)

    def _vector(self, text: str) -> Vector:
        digest = hashlib.sha256(text.encode()).digest()
        return tuple(
            (digest[i % len(digest)] - 127.5) / 127.5
            for i in range(self.model.dimensions)
        )
