"""DocumentRepository adapter: document records in a SQLite file.

Uses Python's built-in sqlite3 module (no ORM). Each operation opens its own
short-lived connection, which is safe when FastAPI runs requests in several
threads. Postgres could replace this later behind the same port.
"""

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path

from app.application.ports import DocumentRepository
from app.domain.errors import StorageUnavailable
from app.domain.models import DocumentRecord, DocumentStatus, EmbeddingModel

logger = logging.getLogger(__name__)

FAILED = "The document store is unavailable."
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id                   TEXT PRIMARY KEY,
    filename             TEXT NOT NULL,
    file_type            TEXT NOT NULL,
    size_bytes           INTEGER NOT NULL,
    status               TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    page_count           INTEGER,
    chunk_count          INTEGER NOT NULL DEFAULT 0,
    embedding_model      TEXT,
    embedding_dimensions INTEGER,
    error_code           TEXT
);
CREATE INDEX IF NOT EXISTS documents_created_at ON documents (created_at);
"""

_COLUMNS = (
    "id, filename, file_type, size_bytes, status, created_at, page_count, "
    "chunk_count, embedding_model, embedding_dimensions, error_code"
)


class SqliteDocumentRepository(DocumentRepository):
    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.executescript(_SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def add(self, record: DocumentRecord) -> None:
        with self._connection() as conn:
            conn.execute(
                f"INSERT INTO documents ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                _to_row(record),
            )

    def update(self, record: DocumentRecord) -> None:
        with self._connection() as conn:
            conn.execute(
                f"REPLACE INTO documents ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                _to_row(record),
            )

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._connection() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return _from_row(row) if row else None

    def list(self) -> list[DocumentRecord]:
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM documents ORDER BY created_at DESC, id"
            ).fetchall()
        return [_from_row(row) for row in rows]

    def delete(self, document_id: str) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """A connection that commits on success, rolls back on error, and always closes."""
        try:
            with closing(sqlite3.connect(self._path, timeout=10)) as conn, conn:
                yield conn
        except sqlite3.Error as exc:
            logger.exception("document_store.failed")
            raise StorageUnavailable(FAILED) from exc


def _to_row(r: DocumentRecord) -> tuple[object, ...]:
    return (
        r.id,
        r.filename,
        r.file_type,
        r.size_bytes,
        r.status.value,
        r.created_at.isoformat(),
        r.page_count,
        r.chunk_count,
        r.embedding_model.name if r.embedding_model else None,
        r.embedding_model.dimensions if r.embedding_model else None,
        r.error_code,
    )


def _from_row(row: tuple[object, ...]) -> DocumentRecord:
    (
        id_,
        filename,
        file_type,
        size,
        status,
        created,
        pages,
        chunks,
        model,
        dims,
        error,
    ) = row
    return DocumentRecord(
        id=str(id_),
        filename=str(filename),
        file_type=str(file_type),
        size_bytes=int(str(size)),
        status=DocumentStatus(str(status)),
        created_at=datetime.fromisoformat(str(created)),
        page_count=int(str(pages)) if pages is not None else None,
        chunk_count=int(str(chunks)),
        embedding_model=EmbeddingModel(str(model), int(str(dims)))
        if model is not None and dims is not None
        else None,
        error_code=str(error) if error is not None else None,
    )
