"""Choosing SQLite or Postgres for document records from Settings."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from app import document_stores
from app.core.config import ConfigurationError, Settings
from app.domain.errors import StorageUnavailable
from app.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)


def settings(tmp_path: Path, **values: Any) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path, **values)


def test_sqlite_is_the_default(tmp_path: Path) -> None:
    store = document_stores.build_document_store(settings(tmp_path))

    assert isinstance(store, SqliteDocumentRepository)
    assert (tmp_path / "documind.db").exists()


@pytest.mark.parametrize("url", [None, "", "   ", "mysql://host/db", "db.supabase.co"])
def test_postgres_needs_a_postgres_url(tmp_path: Path, url: str | None) -> None:
    with pytest.raises(ConfigurationError, match="DATABASE_URL"):
        document_stores.build_document_store(
            settings(
                tmp_path,
                document_store="postgres",
                database_url=SecretStr(url) if url is not None else None,
            )
        )


def test_an_unreachable_database_stops_startup_without_showing_the_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreachable(url: str) -> None:
        raise StorageUnavailable("down")

    monkeypatch.setattr(document_stores, "PostgresDocumentRepository", unreachable)

    with pytest.raises(ConfigurationError, match="Could not connect") as raised:
        document_stores.build_document_store(
            settings(
                tmp_path,
                document_store="postgres",
                database_url=SecretStr("postgresql://user:hunter2@host:6543/postgres"),
            )
        )

    assert "hunter2" not in str(raised.value)
