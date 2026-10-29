"""Composition root: builds the application's object graph from Settings.

This is the only module that knows which concrete adapters the app uses. To
swap a vendor, change the wiring here (embedding vendors are chosen in
app/embedding_providers.py); use cases and routes stay untouched.

Everything is built once, eagerly, at startup: a misconfigured adapter fails
immediately when the server starts, not on the first user request.
"""

from typing import Protocol, runtime_checkable

from app.application.compare_texts import CompareTexts
from app.application.ingest_document import IngestDocument
from app.application.manage_documents import DeleteDocument, GetDocument, ListDocuments
from app.application.parser_registry import ParserRegistry
from app.application.ports import EmbeddingProvider
from app.application.search_chunks import SearchChunks
from app.application.validating_embedding_provider import ValidatingEmbeddingProvider
from app.core.config import Settings
from app.embedding_providers import build_embedding_provider
from app.file_storages import build_file_storage
from app.infrastructure.chunking.recursive_character_chunker import (
    RecursiveCharacterChunker,
)
from app.infrastructure.parsing.plain_text_parser import PlainTextParser
from app.infrastructure.parsing.pymupdf_parser import PyMuPdfParser
from app.infrastructure.parsing.python_docx_parser import PythonDocxParser
from app.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)
from app.vector_stores import build_vector_store


@runtime_checkable
class _Closeable(Protocol):
    def close(self) -> None: ...


class Container:
    def __init__(
        self, settings: Settings, embedding_provider: EmbeddingProvider | None = None
    ) -> None:
        """`embedding_provider` replaces the configured vendor, e.g. with a fake in tests."""
        self.settings = settings
        self._vendor_provider = embedding_provider or build_embedding_provider(settings)

        self.parser_registry = ParserRegistry(
            {
                ".pdf": PyMuPdfParser(),
                ".docx": PythonDocxParser(),
                ".txt": PlainTextParser(),
            }
        )
        self.chunker = RecursiveCharacterChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.embedding_provider = ValidatingEmbeddingProvider(self._vendor_provider)

        self.files = build_file_storage(settings)
        self.documents = SqliteDocumentRepository(settings.database_path)
        self.vector_store = build_vector_store(settings)

        self.ingest_document = IngestDocument(
            parsers=self.parser_registry,
            chunker=self.chunker,
            embedder=self.embedding_provider,
            files=self.files,
            documents=self.documents,
            vectors=self.vector_store,
            max_upload_bytes=settings.max_upload_size_bytes,
        )
        self.list_documents = ListDocuments(self.documents, self.embedding_provider)
        self.get_document = GetDocument(
            self.documents, self.vector_store, self.embedding_provider
        )
        self.delete_document = DeleteDocument(
            self.documents, self.files, self.vector_store
        )
        self.search_chunks = SearchChunks(
            embedder=self.embedding_provider,
            searcher=self.vector_store,
            default_top_k=settings.search_default_top_k,
            max_top_k=settings.search_max_top_k,
        )
        self.compare_texts = CompareTexts(embedder=self.embedding_provider)

    def close(self) -> None:
        """Release network connections at shutdown."""
        for resource in (self._vendor_provider, self.files):
            if isinstance(resource, _Closeable):
                resource.close()
