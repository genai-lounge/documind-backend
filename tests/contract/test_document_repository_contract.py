"""Contract tests: rules every DocumentRepository must follow (fake and SQLite)."""

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from app.application.ports import DocumentRepository
from app.domain.models import DocumentRecord, DocumentStatus, EmbeddingModel
from app.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)
from tests.fakes import InMemoryDocumentRepository

REPOSITORIES: dict[str, Callable[[Path], DocumentRepository]] = {
    "InMemoryDocumentRepository": lambda _: InMemoryDocumentRepository(),
    "SqliteDocumentRepository": lambda path: SqliteDocumentRepository(
        path / "db.sqlite"
    ),
}
T0 = datetime(2026, 10, 27, 9, 30, 15, 123456, tzinfo=UTC)


@pytest.fixture(params=list(REPOSITORIES))
def repo(request: pytest.FixtureRequest, tmp_path: Path) -> DocumentRepository:
    return REPOSITORIES[request.param](tmp_path)


BASE = DocumentRecord(
    id="doc",
    filename="report.pdf",
    file_type="pdf",
    size_bytes=2048,
    status=DocumentStatus.INDEXED,
    created_at=T0,
    page_count=3,
    chunk_count=7,
    embedding_model=EmbeddingModel("text-embedding-3-small", 1536),
)


def record(id_: str = "doc", minutes: int = 0, **changes: Any) -> DocumentRecord:
    return replace(BASE, id=id_, created_at=T0 + timedelta(minutes=minutes), **changes)


def test_round_trips_every_field(repo: DocumentRepository) -> None:
    original = record()
    repo.add(original)

    assert repo.get("doc") == original


def test_round_trips_empty_optional_fields(repo: DocumentRepository) -> None:
    original = record(
        page_count=None,
        chunk_count=0,
        embedding_model=None,
        status=DocumentStatus.FAILED,
        error_code="embedding_unavailable",
    )
    repo.add(original)

    assert repo.get("doc") == original


def test_unknown_id_returns_none(repo: DocumentRepository) -> None:
    assert repo.get("nope") is None


def test_update_replaces_the_record(repo: DocumentRepository) -> None:
    repo.add(record(status=DocumentStatus.PROCESSING))
    repo.update(record(status=DocumentStatus.INDEXED))

    stored = repo.get("doc")
    assert stored is not None and stored.status is DocumentStatus.INDEXED


def test_list_is_newest_first(repo: DocumentRepository) -> None:
    repo.add(record("a", minutes=0))
    repo.add(record("c", minutes=10))
    repo.add(record("b", minutes=5))

    assert [r.id for r in repo.list()] == ["c", "b", "a"]


def test_delete_is_idempotent(repo: DocumentRepository) -> None:
    repo.add(record())

    repo.delete("doc")
    repo.delete("doc")

    assert repo.get("doc") is None
    assert repo.list() == []
