"""The API contract: the exact JSON shapes clients send and receive.

These Pydantic models are separate from the domain models on purpose. The
domain can evolve freely, while this contract only changes deliberately,
because the frontend (frontend/src/features/documents/types.ts) depends on it.
FastAPI also uses these models to generate the docs at /docs.
"""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from app.application.compare_texts import SimilarityResult
from app.domain.models import Document, EmbeddedChunk, EmbeddingModel, ExtractionStatus

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

    @classmethod
    def from_domain(cls, document: Document) -> "DocumentResponse":
        content = document.content
        return cls(
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
