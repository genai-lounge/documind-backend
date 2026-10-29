"""Errors that describe what went wrong in DocuMind's own terms.

Each error has a stable `code` (for programs, e.g. the frontend) and a
message safe to show to users. They know nothing about HTTP: the API layer
decides which status code each one becomes (see app/api/errors.py).
"""


class DocuMindError(Exception):
    """Base class for every expected, user-facing error in the app."""

    code = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UploadRejected(DocuMindError):
    """The uploaded file failed validation."""


class MissingFilename(UploadRejected):
    code = "missing_filename"


class EmptyFile(UploadRejected):
    code = "empty_file"


class FileTooLarge(UploadRejected):
    code = "file_too_large"


class UnsupportedFileType(UploadRejected):
    code = "unsupported_file_type"


class FileTypeMismatch(UploadRejected):
    """The file's content does not match its extension (e.g. a renamed .exe)."""

    code = "file_type_mismatch"


class DocumentParseError(DocuMindError):
    """The file is damaged, password-protected, wrongly encoded, etc."""

    code = "malformed_document"


class EmbeddingUnavailable(DocuMindError):
    """The embedding service failed, timed out, or returned something unusable."""

    code = "embedding_unavailable"


class EmbeddingRateLimited(EmbeddingUnavailable):
    """The embedding service is refusing requests because a usage limit was reached."""

    code = "embedding_rate_limited"

    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class InvalidInput(DocuMindError):
    """A request value breaks a business rule (e.g. top_k above the maximum)."""

    code = "invalid_input"


class DocumentNotFound(DocuMindError):
    code = "document_not_found"


class StorageUnavailable(DocuMindError):
    """A store (files, records, or vectors) failed. Details are only in the logs."""

    code = "storage_unavailable"
