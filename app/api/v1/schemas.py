"""The API contract: the exact JSON shapes clients send and receive.

These Pydantic models are separate from the domain models on purpose. The
domain can evolve freely, while this contract only changes deliberately,
because the frontend (frontend/src/features/documents/types.ts) depends on it.
FastAPI also uses these models to generate the docs at /docs.
"""

from pydantic import BaseModel, Field

from app.domain.models import Document, ExtractionStatus


class DocumentMetadataOut(BaseModel):
    title: str | None = None
    author: str | None = None


class DocumentSectionOut(BaseModel):
    page_number: int | None
    text: str
    char_count: int


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
        )


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    """Shape of every error the API returns."""

    detail: ErrorDetail
