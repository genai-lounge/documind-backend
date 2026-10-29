"""The API contract: the exact JSON shapes clients send and receive.

These Pydantic models are separate from the domain models on purpose. The
domain can evolve freely, while this contract only changes deliberately,
because the frontend (frontend/src/features/documents/types.ts) depends on it.
FastAPI also uses these models to generate the docs at /docs.
"""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from app.application.compare_texts import SimilarityResult
from app.application.manage_documents import DocumentDetail, DocumentOverview
from app.application.search_chunks import MAX_QUERY_CHARS, SearchResult
from app.domain.models import (
    Chunk,
    Document,
    DocumentRecord,
    DocumentStatus,
    EmbeddedChunk,
    EmbeddingModel,
    ExtractionStatus,
    SearchHit,
)

EMBEDDING_PREVIEW_LENGTH = 8
COMPARE_TEXT_MAX_CHARS = 2000


class DocumentMetadataOut(BaseModel):
    title: str | None = None
    author: str | None = None


class DocumentSectionOut(BaseModel):
    page_number: int | None
    text: str
    char_count: int


class EmbeddingModelOut(BaseModel):
    model: str = Field(examples=["voyage-4-lite"])
    dimensions: int

    @classmethod
    def from_domain(cls, model: EmbeddingModel) -> "EmbeddingModelOut":
        return cls(model=model.name, dimensions=model.dimensions)


class ChunkOut(BaseModel):
    chunk_index: int
    page_number: int | None
    text: str
    start_char: int = Field(description="Position of the chunk within its section.")
    end_char: int
    char_count: int
    embedding_preview: list[float] = Field(
        description=f"The first {EMBEDDING_PREVIEW_LENGTH} numbers of the chunk's vector."
    )

    @classmethod
    def from_domain(cls, embedded: EmbeddedChunk) -> "ChunkOut":
        chunk = embedded.chunk
        return cls(
            chunk_index=chunk.index,
            page_number=chunk.page_number,
            text=chunk.text,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            char_count=chunk.char_count,
            embedding_preview=list(embedded.vector[:EMBEDDING_PREVIEW_LENGTH]),
        )


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str = Field(examples=["pdf", "docx", "txt"])
    size_bytes: int
    status: ExtractionStatus
    page_count: int | None
    metadata: DocumentMetadataOut
    sections: list[DocumentSectionOut]
    warnings: list[str]
    embedding: EmbeddingModelOut | None = Field(
        description="The model that embedded the chunks; null when there was no text."
    )
    chunks: list[ChunkOut]
    index_status: DocumentStatus = Field(
        description="Whether the document was stored and indexed for search."
    )
    created_at: datetime

    @classmethod
    def from_domain(
        cls, document: Document, record: DocumentRecord
    ) -> "DocumentResponse":
        content = document.content
        return cls(
            index_status=record.status,
            created_at=record.created_at,
            id=document.id,
            filename=document.filename,
            file_type=document.file_type,
            size_bytes=document.size_bytes,
            status=document.status,
            page_count=content.page_count,
            metadata=DocumentMetadataOut(
                title=content.metadata.title, author=content.metadata.author
            ),
            sections=[
                DocumentSectionOut(
                    page_number=s.page_number, text=s.text, char_count=s.char_count
                )
                for s in content.sections
            ],
            warnings=list(content.warnings),
            embedding=EmbeddingModelOut.from_domain(document.embedding_model)
            if document.embedding_model
            else None,
            chunks=[ChunkOut.from_domain(c) for c in document.chunks],
        )


CompareText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=COMPARE_TEXT_MAX_CHARS
    ),
]


class SimilarityRequest(BaseModel):
    text_a: CompareText
    text_b: CompareText


class SimilarityResponse(BaseModel):
    similarity: float = Field(
        description="Cosine similarity: 1 same meaning, 0 unrelated, -1 opposite."
    )
    model: str
    dimensions: int

    @classmethod
    def from_domain(cls, result: SimilarityResult) -> "SimilarityResponse":
        return cls(
            similarity=result.similarity,
            model=result.model.name,
            dimensions=result.model.dimensions,
        )


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    """Shape of every error the API returns."""

    detail: ErrorDetail


class StoredChunkOut(BaseModel):
    chunk_index: int
    page_number: int | None
    text: str
    start_char: int
    end_char: int
    char_count: int

    @classmethod
    def from_domain(cls, chunk: Chunk) -> "StoredChunkOut":
        return cls(
            chunk_index=chunk.index,
            page_number=chunk.page_number,
            text=chunk.text,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            char_count=chunk.char_count,
        )


class DocumentSummaryOut(BaseModel):
    id: str
    filename: str
    file_type: str
    size_bytes: int
    status: DocumentStatus
    created_at: datetime
    page_count: int | None
    chunk_count: int
    embedding: EmbeddingModelOut | None
    error_code: str | None = Field(
        description="Why indexing failed, when status is failed."
    )
    searchable: bool = Field(
        description="False when indexed with a different embedding model than the "
        "current one; such documents need re-indexing before search can find them."
    )

    @classmethod
    def from_domain(cls, overview: DocumentOverview) -> "DocumentSummaryOut":
        r = overview.record
        return cls(
            id=r.id,
            filename=r.filename,
            file_type=r.file_type,
            size_bytes=r.size_bytes,
            status=r.status,
            created_at=r.created_at,
            page_count=r.page_count,
            chunk_count=r.chunk_count,
            embedding=EmbeddingModelOut.from_domain(r.embedding_model)
            if r.embedding_model
            else None,
            error_code=r.error_code,
            searchable=overview.searchable,
        )


class DocumentDetailOut(DocumentSummaryOut):
    chunks: list[StoredChunkOut]

    @classmethod
    def from_detail(cls, detail: DocumentDetail) -> "DocumentDetailOut":
        summary = DocumentSummaryOut.from_domain(detail.overview)
        return cls(
            **summary.model_dump(),
            chunks=[StoredChunkOut.from_domain(c) for c in detail.chunks],
        )


class SearchRequest(BaseModel):
    query: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True, min_length=1, max_length=MAX_QUERY_CHARS
        ),
    ]
    top_k: int | None = Field(
        default=None, description="Defaults to SEARCH_DEFAULT_TOP_K."
    )


class SearchHitOut(BaseModel):
    rank: int
    score: float = Field(
        description="Cosine similarity, -1 to 1. It ranks results; it is not a "
        "probability that the passage is relevant."
    )
    document_id: str
    filename: str
    page_number: int | None
    chunk_index: int
    text: str
    start_char: int
    end_char: int

    @classmethod
    def from_domain(cls, hit: SearchHit) -> "SearchHitOut":
        return cls(
            rank=hit.rank,
            score=hit.score,
            document_id=hit.document_id,
            filename=hit.filename,
            page_number=hit.chunk.page_number,
            chunk_index=hit.chunk.index,
            text=hit.chunk.text,
            start_char=hit.chunk.start_char,
            end_char=hit.chunk.end_char,
        )


class SearchResponse(BaseModel):
    embedding: EmbeddingModelOut
    hits: list[SearchHitOut]

    @classmethod
    def from_domain(cls, result: SearchResult) -> "SearchResponse":
        return cls(
            embedding=EmbeddingModelOut.from_domain(result.model),
            hits=[SearchHitOut.from_domain(h) for h in result.hits],
        )
