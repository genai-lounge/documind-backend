"""DocumentRepository adapter: document records in PostgreSQL (e.g. Supabase).

Uses psycopg 3 with a small connection pool. Prepared statements are turned
off, so it also works through transaction-mode poolers such as Supabase's
(port 6543), which serverless hosts like Vercel should use.

The table lives in its own `documind` schema: Supabase publishes the `public`
schema through its REST API, and these records must only be reachable through
this backend.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg_pool import ConnectionPool, PoolTimeout

from app.application.ports import DocumentRepository
from app.domain.errors import StorageUnavailable
from app.domain.models import DocumentRecord, DocumentStatus, EmbeddingModel

logger = logging.getLogger(__name__)

FAILED = "The document store is unavailable."

_SCHEMA = """
CREATE SCHEMA IF NOT EXISTS documind;
CREATE TABLE IF NOT EXISTS documind.documents (
    id                   TEXT PRIMARY KEY,
    filename             TEXT NOT NULL,
    file_type            TEXT NOT NULL,
    size_bytes           BIGINT NOT NULL,
    status               TEXT NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL,
    page_count           INTEGER,
    chunk_count          INTEGER NOT NULL DEFAULT 0,
    embedding_model      TEXT,
    embedding_dimensions INTEGER,
    error_code           TEXT
);
CREATE INDEX IF NOT EXISTS documents_created_at ON documind.documents (created_at);
"""

_COLUMNS = (
    "id, filename, file_type, size_bytes, status, created_at, page_count, "
    "chunk_count, embedding_model, embedding_dimensions, error_code"
)
_PLACEHOLDERS = ", ".join(["%s"] * 11)
_UPDATE_ALL_BUT_ID = """
    filename = EXCLUDED.filename, file_type = EXCLUDED.file_type,
    size_bytes = EXCLUDED.size_bytes, status = EXCLUDED.status,
    created_at = EXCLUDED.created_at, page_count = EXCLUDED.page_count,
    chunk_count = EXCLUDED.chunk_count, embedding_model = EXCLUDED.embedding_model,
    embedding_dimensions = EXCLUDED.embedding_dimensions, error_code = EXCLUDED.error_code
"""


class PostgresDocumentRepository(DocumentRepository):
    def __init__(
        self, url: str, *, max_connections: int = 4, timeout_seconds: float = 10
    ) -> None:
        """Connects and creates the table if needed; raises StorageUnavailable if it can't."""
        self._pool = ConnectionPool(
            url,
            min_size=1,
            max_size=max_connections,
            timeout=timeout_seconds,
            max_idle=60,
            check=ConnectionPool.check_connection,
            kwargs={"prepare_threshold": None, "connect_timeout": int(timeout_seconds)},
            open=False,
        )
        try:
            self._pool.open(wait=True, timeout=timeout_seconds)
        except PoolTimeout as exc:
            self._pool.close()
            logger.error("document_store.unreachable", extra={"provider": "postgres"})
            raise StorageUnavailable(FAILED) from exc
        with self._connection() as conn:
            conn.execute(_SCHEMA)

    def add(self, record: DocumentRecord) -> None:
        with self._connection() as conn:
            conn.execute(
                f"INSERT INTO documind.documents ({_COLUMNS}) VALUES ({_PLACEHOLDERS})",
                _to_row(record),
            )

    def update(self, record: DocumentRecord) -> None:
        with self._connection() as conn:
            conn.execute(
                f"INSERT INTO documind.documents ({_COLUMNS}) VALUES ({_PLACEHOLDERS}) "
                f"ON CONFLICT (id) DO UPDATE SET {_UPDATE_ALL_BUT_ID}",
                _to_row(record),
            )

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._connection() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM documind.documents WHERE id = %s",
                (document_id,),
            ).fetchone()
        return _from_row(row) if row else None

    def list(self) -> list[DocumentRecord]:
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM documind.documents ORDER BY created_at DESC, id"
            ).fetchall()
        return [_from_row(row) for row in rows]

    def delete(self, document_id: str) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM documind.documents WHERE id = %s", (document_id,))

    def close(self) -> None:
        self._pool.close()

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection[Any]]:
        """A pooled connection that commits on success and rolls back on error."""
        try:
            with self._pool.connection() as conn:
                yield conn
        except (psycopg.Error, PoolTimeout) as exc:
            logger.exception("document_store.failed", extra={"provider": "postgres"})
            raise StorageUnavailable(FAILED) from exc


def _to_row(r: DocumentRecord) -> tuple[object, ...]:
    return (
        r.id,
        r.filename,
        r.file_type,
        r.size_bytes,
        r.status.value,
        r.created_at,
        r.page_count,
        r.chunk_count,
        r.embedding_model.name if r.embedding_model else None,
        r.embedding_model.dimensions if r.embedding_model else None,
        r.error_code,
    )


def _from_row(row: tuple[Any, ...]) -> DocumentRecord:
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
        id=id_,
        filename=filename,
        file_type=file_type,
        size_bytes=size,
        status=DocumentStatus(status),
        created_at=created,
        page_count=pages,
        chunk_count=chunks,
        embedding_model=EmbeddingModel(model, dims)
        if model is not None and dims is not None
        else None,
        error_code=error,
    )
