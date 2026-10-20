"""Pydantic models for API requests and responses."""

from typing import Literal

from pydantic import BaseModel

FileType = Literal["pdf", "docx", "txt"]

DocumentStatus = Literal["extracted", "no_text"]

class DocumentMetadata(BaseModel):
    title: str | None = None
    author: str | None = None

class DocumentSection(BaseModel):
    page_number: int | None
    text: str
    char_count: int

class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: FileType
    size_bytes: int
    status: DocumentStatus
    page_count: int | None
    metadata: DocumentMetadata
    sections: list[DocumentSection]
    warnings: list[str]

class ErrorDetail(BaseModel):
    code: str
    message: str

class ErrorResponse(BaseModel):
    """Shape of every error the API returns: {"detail": {"code": ..., "message": ...}}."""

    detail: ErrorDetail
