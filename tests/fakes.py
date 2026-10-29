"""Fake adapters for unit tests.

A fake implements a port with simple, predictable behaviour and no real
library behind it. Use-case tests run against fakes, so they are fast and
test only the use case's own logic.
"""

import hashlib
import math
import re
from collections.abc import Sequence

from app.application.ports import (
    Chunker,
    DocumentParser,
    DocumentRepository,
    EmbeddingProvider,
    FileStorage,
    VectorIndexReader,
    VectorIndexWriter,
    VectorSearcher,
)
from app.domain.models import (
    Chunk,
    DocumentRecord,
    EmbeddedChunk,
    EmbeddingModel,
    ParsedContent,
    SearchHit,
    Section,
    Vector,
)
from app.domain.similarity import cosine_similarity


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


class WordOverlapEmbeddingProvider(EmbeddingProvider):
    """Texts that share words get similar vectors (a hashed bag of words).

    Not a real language model, but deterministic and meaningful enough for
    retrieval tests: a question about "revenue growth" lands near a passage
    containing those words, with no network or API key.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.model = EmbeddingModel(name="word-overlap", dimensions=dimensions)

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> Vector:
        return self._vector(text)

    def _vector(self, text: str) -> Vector:
        counts = [0.0] * self.model.dimensions
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if len(word) > 2:
                index = int(hashlib.sha256(word.encode()).hexdigest(), 16)
                counts[index % self.model.dimensions] += 1.0
        norm = math.sqrt(sum(c * c for c in counts)) or 1.0
        if not any(counts):
            counts[0] = 1.0
        return tuple(c / norm for c in counts)


class InMemoryFileStorage(FileStorage):
    def __init__(self, fail_on_save: Exception | None = None) -> None:
        self.files: dict[str, bytes] = {}
        self.fail_on_save = fail_on_save

    def save(self, key: str, data: bytes) -> None:
        if self.fail_on_save:
            raise self.fail_on_save
        self.files[key] = data

    def delete(self, key: str) -> None:
        self.files.pop(key, None)


class InMemoryDocumentRepository(DocumentRepository):
    def __init__(self) -> None:
        self.records: dict[str, DocumentRecord] = {}

    def add(self, record: DocumentRecord) -> None:
        if record.id in self.records:
            raise ValueError(f"Duplicate id {record.id}")
        self.records[record.id] = record

    def update(self, record: DocumentRecord) -> None:
        self.records[record.id] = record

    def get(self, document_id: str) -> DocumentRecord | None:
        return self.records.get(document_id)

    def list(self) -> list[DocumentRecord]:
        return sorted(
            self.records.values(), key=lambda r: (r.created_at, r.id), reverse=True
        )

    def delete(self, document_id: str) -> None:
        self.records.pop(document_id, None)


class InMemoryVectorStore(VectorIndexWriter, VectorSearcher, VectorIndexReader):
    """Brute-force cosine search over a dict, one 'collection' per model."""

    def __init__(self, fail_on_upsert: Exception | None = None) -> None:
        self.collections: dict[
            EmbeddingModel, dict[str, tuple[str, str, EmbeddedChunk]]
        ] = {}
        self.fail_on_upsert = fail_on_upsert

    def upsert(
        self,
        model: EmbeddingModel,
        document_id: str,
        filename: str,
        chunks: Sequence[EmbeddedChunk],
    ) -> None:
        collection = self.collections.setdefault(model, {})
        for c in chunks:
            collection[f"{document_id}:{c.chunk.index}"] = (document_id, filename, c)
        if self.fail_on_upsert:
            raise self.fail_on_upsert

    def delete_document(self, model: EmbeddingModel, document_id: str) -> None:
        collection = self.collections.get(model, {})
        for key in [k for k, v in collection.items() if v[0] == document_id]:
            del collection[key]

    def search(
        self, model: EmbeddingModel, vector: Vector, top_k: int
    ) -> list[SearchHit]:
        scored = sorted(
            (
                (cosine_similarity(vector, c.vector), doc_id, filename, c.chunk)
                for doc_id, filename, c in self.collections.get(model, {}).values()
            ),
            key=lambda item: item[0],
            reverse=True,
        )[:top_k]
        return [
            SearchHit(rank=i, score=score, document_id=d, filename=f, chunk=chunk)
            for i, (score, d, f, chunk) in enumerate(scored, start=1)
        ]

    def document_chunks(self, model: EmbeddingModel, document_id: str) -> list[Chunk]:
        return sorted(
            (
                c.chunk
                for d, _, c in self.collections.get(model, {}).values()
                if d == document_id
            ),
            key=lambda chunk: chunk.index,
        )

    def count(self, model: EmbeddingModel) -> int:
        return len(self.collections.get(model, {}))
