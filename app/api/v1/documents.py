"""HTTP endpoints for documents.

A route's only job is HTTP: take the request apart, call a use case, and turn
the result into a response. All rules (size limits, supported formats) live in
the use case; all error-to-status mapping lives in app/api/errors.py.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response, UploadFile

from app.api.dependencies import (
    get_delete_document,
    get_get_document,
    get_ingest_document,
    get_list_documents,
)
from app.api.v1.schemas import (
    DocumentDetailOut,
    DocumentResponse,
    DocumentSummaryOut,
    ErrorResponse,
)
from app.application.ingest_document import IngestDocument
from app.application.manage_documents import DeleteDocument, GetDocument, ListDocuments

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
        503: {
            "model": ErrorResponse,
            "description": "Embedding service or storage unavailable",
        },
    },
)
def upload_document(
    file: UploadFile,
    ingest: Annotated[IngestDocument, Depends(get_ingest_document)],
) -> DocumentResponse:
    result = ingest.execute(filename=file.filename or "", stream=file.file)
    return DocumentResponse.from_domain(result.document, result.record)


NOT_FOUND: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "No such document"}
}


@router.get("", response_model=list[DocumentSummaryOut])
def list_documents(
    use_case: Annotated[ListDocuments, Depends(get_list_documents)],
) -> list[DocumentSummaryOut]:
    return [DocumentSummaryOut.from_domain(o) for o in use_case.execute()]


@router.get("/{document_id}", response_model=DocumentDetailOut, responses=NOT_FOUND)
def get_document(
    document_id: str,
    use_case: Annotated[GetDocument, Depends(get_get_document)],
) -> DocumentDetailOut:
    return DocumentDetailOut.from_detail(use_case.execute(document_id))


@router.delete("/{document_id}", status_code=204, responses=NOT_FOUND)
def delete_document(
    document_id: str,
    use_case: Annotated[DeleteDocument, Depends(get_delete_document)],
) -> Response:
    use_case.execute(document_id)
    return Response(status_code=204)
