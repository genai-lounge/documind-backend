"""Turns errors into HTTP responses: the only place that knows about status codes.

Every error the API returns has the same shape:

    {"detail": {"code": "file_too_large", "message": "The file is larger than ..."}}

Use cases raise domain errors (app/domain/errors.py); this module decides which
HTTP status each one becomes. Unexpected errors are handled by the request
middleware (app/api/middleware.py), which logs them and returns a generic 500.
"""

import logging
import math

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.domain.errors import (
    AnswerRefused,
    DocumentNotFound,
    DocumentParseError,
    DocuMindError,
    EmbeddingRateLimited,
    EmbeddingUnavailable,
    EmptyFile,
    FileTooLarge,
    FileTypeMismatch,
    InvalidInput,
    LLMRateLimited,
    LLMUnavailable,
    MissingFilename,
    StorageUnavailable,
    UnsupportedFileType,
)

logger = logging.getLogger(__name__)

STATUS_BY_ERROR: dict[type[DocuMindError], int] = {
    MissingFilename: 400,
    EmptyFile: 400,
    FileTooLarge: 413,
    UnsupportedFileType: 415,
    FileTypeMismatch: 415,
    DocumentParseError: 422,
    EmbeddingUnavailable: 503,
    InvalidInput: 422,
    DocumentNotFound: 404,
    StorageUnavailable: 503,
    LLMUnavailable: 503,
    AnswerRefused: 422,
}

CODE_BY_HTTP_STATUS = {404: "not_found", 405: "method_not_allowed"}


def error_body(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"detail": {"code": code, "message": message}}


def status_for(error: DocuMindError) -> int:
    """Look up the status for the error's class, or its closest parent class."""
    for cls in type(error).__mro__:
        if cls in STATUS_BY_ERROR:
            return STATUS_BY_ERROR[cls]
    return 500


async def _handle_domain_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, DocuMindError)
    status = status_for(exc)
    logger.info("request.rejected", extra={"code": exc.code, "status": status})
    headers = None
    if (
        isinstance(exc, EmbeddingRateLimited | LLMRateLimited)
        and exc.retry_after_seconds is not None
    ):
        headers = {"Retry-After": str(math.ceil(exc.retry_after_seconds))}
    return JSONResponse(
        status_code=status,
        content=error_body(exc.code, exc.message),
        headers=headers,
    )


async def _handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    """FastAPI's own request validation (e.g. the `file` field is missing)."""
    return JSONResponse(
        status_code=422,
        content=error_body(
            "invalid_request",
            "The request is missing a required field or has an invalid value.",
        ),
    )


async def _handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
    """Framework-level errors such as 404 Not Found, in our standard shape."""
    assert isinstance(exc, StarletteHTTPException)
    code = CODE_BY_HTTP_STATUS.get(exc.status_code, "http_error")
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(code, str(exc.detail)),
        headers=exc.headers,
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(DocuMindError, _handle_domain_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
