"""Document upload endpoint: validate the file, extract its text, return the result.

Phase 1 stores nothing. The file is read in memory and discarded after the response.
"""

import uuid
from collections.abc import Callable
from pathlib import PurePath

from fastapi import APIRouter, HTTPException, UploadFile

from app import config
from app.document_parser import (
    DocumentParseError,
    ExtractedDocument,
    extract_docx,
    extract_pdf,
    extract_txt,
)
from app.schemas import (
    DocumentMetadata,
    DocumentResponse,
    DocumentSection,
    ErrorResponse,
    FileType,
)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

EXTRACTORS: dict[FileType, Callable[[bytes], ExtractedDocument]] = {
    "pdf": extract_pdf,
    "docx": extract_docx,
    "txt": extract_txt,
}

def api_error(status_code: int, code: str, message: str) -> HTTPException:
    """Build an error with the API's standard {"code", "message"} detail."""
    return HTTPException(
        status_code=status_code, detail={"code": code, "message": message}
    )

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

def upload_document(file: UploadFile) -> DocumentResponse:
    filename = PurePath(file.filename or "").name
    if not filename:
        raise api_error(400, "missing_filename", "The uploaded file has no name.")

    file_type = _file_type_from_name(filename)

    data = file.file.read(config.MAX_UPLOAD_SIZE_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_SIZE_BYTES:
        raise api_error(
            413,
            "file_too_large",
            f"The file is larger than the {config.MAX_UPLOAD_SIZE_MB} MB limit.",
        )
    if not data:
        raise api_error(400, "empty_file", "The file is empty.")

    _check_content_matches_type(data, file_type)

    try:
        extracted = EXTRACTORS[file_type](data)
    except DocumentParseError as exc:
        raise api_error(422, "malformed_document", str(exc)) from exc

    return _build_response(filename, file_type, len(data), extracted)

def _file_type_from_name(filename: str) -> FileType:
    suffix = PurePath(filename).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "docx"
    if suffix == ".txt":
        return "txt"
    raise api_error(
        415,
        "unsupported_file_type",
        f"'{suffix or filename}' files are not supported. Upload a PDF, DOCX, or TXT file.",
    )

def _check_content_matches_type(data: bytes, file_type: FileType) -> None:
    """Reject files whose content does not match their extension (e.g. a renamed .exe).

    PDFs carry a "%PDF-" marker near the start; DOCX files are ZIP archives,
    which start with "PK\\x03\\x04". TXT has no marker; extract_txt checks it instead.
    """
    if file_type == "pdf" and b"%PDF-" not in data[:1024]:
        raise api_error(
            415, "file_type_mismatch", "The file has a .pdf name but is not a PDF."
        )
    if file_type == "docx" and not data.startswith(b"PK\x03\x04"):
        raise api_error(
            415,
            "file_type_mismatch",
            "The file has a .docx name but is not a Word document.",
        )

def _build_response(
    filename: str, file_type: FileType, size_bytes: int, extracted: ExtractedDocument
) -> DocumentResponse:
    sections = [
        DocumentSection(page_number=s.page_number, text=s.text, char_count=len(s.text))
        for s in extracted.sections
    ]
    has_text = any(s.text.strip() for s in sections)

    warnings: list[str] = []
    if not has_text:
        if file_type == "pdf":
            warnings.append(
                "No text was found in this PDF. It may be a scanned image; "
                "OCR is not supported."
            )
        else:
            warnings.append("No text was found in this file.")
    elif file_type == "pdf":
        for s in sections:
            if not s.text.strip():
                warnings.append(
                    f"Page {s.page_number} has no extractable text "
                    "(it may be a scanned image or blank)."
                )

    return DocumentResponse(
        id=uuid.uuid4().hex,
        filename=filename,
        file_type=file_type,
        size_bytes=size_bytes,
        status="extracted" if has_text else "no_text",
        page_count=extracted.page_count,
        metadata=DocumentMetadata(title=extracted.title, author=extracted.author),
        sections=sections,
        warnings=warnings,
    )
