"""HTTP endpoints for documents.

A route's only job is HTTP: take the request apart, call a use case, and turn
the result into a response. All rules (size limits, supported formats) live in
the use case; all error-to-status mapping lives in app/api/errors.py.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile

from app.api.dependencies import get_ingest_document
from app.api.v1.schemas import DocumentResponse, ErrorResponse
from app.application.ingest_document import IngestDocument

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Empty file or missing filename"},
        413: {
            "model": ErrorResponse,
            "description": "File larger than the upload limit",
        },
        415: {
            "model": ErrorResponse,
            "description": "Unsupported or mismatched file type",
        },
        422: {"model": ErrorResponse, "description": "File could not be parsed"},
    },
)

def upload_document(
    file: UploadFile,
    ingest: Annotated[IngestDocument, Depends(get_ingest_document)],
) -> DocumentResponse:
    document = ingest.execute(filename=file.filename or "", stream=file.file)
    return DocumentResponse.from_domain(document)
