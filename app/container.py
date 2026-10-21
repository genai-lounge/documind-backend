"""Composition root: builds the application's object graph from Settings.

This is the only module that knows which concrete adapters the app uses. To
swap a vendor (a different PDF library today; Qdrant instead of Chroma, or a
different embedding model, in later phases), change the wiring here. Use
cases and routes stay untouched.

Everything is built once, eagerly, at startup: a misconfigured adapter fails
immediately when the server starts, not on the first user request.
"""

from app.application.ingest_document import IngestDocument
from app.application.parser_registry import ParserRegistry
from app.core.config import Settings
from app.infrastructure.parsing.plain_text_parser import PlainTextParser
from app.infrastructure.parsing.pymupdf_parser import PyMuPdfParser
from app.infrastructure.parsing.python_docx_parser import PythonDocxParser


class Container:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.parser_registry = ParserRegistry(
            {
                ".pdf": PyMuPdfParser(),
                ".docx": PythonDocxParser(),
                ".txt": PlainTextParser(),
            }
        )

        self.ingest_document = IngestDocument(
            parsers=self.parser_registry,
            max_upload_bytes=settings.max_upload_size_bytes,
        )
