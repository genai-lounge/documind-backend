"""Ports: the interfaces the application needs from the outside world.

A port says WHAT the app needs ("turn a PDF's bytes into text") without
saying HOW or WITH WHICH LIBRARY. Concrete implementations (adapters) live in
app/infrastructure/ and are plugged in by the composition root
(app/container.py). This is the Dependency Inversion Principle: the use cases
depend on these abstractions, and the vendor code depends on them too.

Ports are kept small (Interface Segregation): a caller only depends on the
methods it actually uses.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.domain.answers import EvidencePassage, GeneratedAnswer
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


class FileStorage(ABC):
    """Keeps original uploaded files. Keys are simple names chosen by the app.

    Contract: `save` overwrites an existing key; `delete` of a missing key is a no-op.
    Failures raise StorageUnavailable.
    """

    @abstractmethod
    def save(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...


class DocumentRepository(ABC):
    """Stores one DocumentRecord per document.

    Contract: `list` returns newest first; `delete` of a missing id is a no-op.
    Failures raise StorageUnavailable.
    """

    @abstractmethod
    def add(self, record: DocumentRecord) -> None: ...

    @abstractmethod
    def update(self, record: DocumentRecord) -> None: ...

    @abstractmethod
    def get(self, document_id: str) -> DocumentRecord | None: ...

    @abstractmethod
    def list(self) -> list[DocumentRecord]: ...

    @abstractmethod
    def delete(self, document_id: str) -> None: ...


class VectorIndexWriter(ABC):
    """Adds and removes a document's chunk vectors.

    Every call names the embedding model, and vectors of different models are
    kept apart, so they can never be compared by mistake.
    Contract: `upsert` replaces chunks with the same index; `delete_document`
    of an unknown document is a no-op. Failures raise StorageUnavailable.
    """

    @abstractmethod
    def upsert(
        self,
        model: EmbeddingModel,
        document_id: str,
        filename: str,
        chunks: Sequence[EmbeddedChunk],
    ) -> None: ...

    @abstractmethod
    def delete_document(self, model: EmbeddingModel, document_id: str) -> None: ...


class VectorSearcher(ABC):
    """Finds the chunks whose vectors are closest to a query vector.

    Contract: at most `top_k` hits, best first, ranked from 1, with scores
    from -1 to 1 (cosine similarity). Failures raise StorageUnavailable.
    """

    @abstractmethod
    def search(
        self, model: EmbeddingModel, vector: Vector, top_k: int
    ) -> list[SearchHit]: ...


class VectorIndexReader(ABC):
    """Reads back a document's stored chunks (text and position, no vectors)."""

    @abstractmethod
    def document_chunks(
        self, model: EmbeddingModel, document_id: str
    ) -> list[Chunk]: ...


class LLMProvider(ABC):
    """Writes an answer to a question using only the evidence passages given.

    Shaped around what DocuMind needs (an evidence-grounded answer with
    citations), not around a generic chat API, so an adapter can use its
    vendor's own citation features.

    Contract every adapter must follow (checked by tests/contract/):
    - answer only from `evidence`, citing passages by their index in the list;
    - when the evidence doesn't answer the question, say so without citations;
    - treat evidence text as data, never as instructions;
    - failures raise LLMUnavailable (or LLMRateLimited / AnswerRefused), never
      a library-specific exception.
    """

    model_name: str

    @abstractmethod
    def answer(
        self, question: str, evidence: Sequence[EvidencePassage]
    ) -> GeneratedAnswer: ...
