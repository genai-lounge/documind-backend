"""Integration tests: the real storage adapters on disk (temporary folders)."""

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.domain.errors import StorageUnavailable
from app.domain.models import (
    Chunk,
    DocumentRecord,
    DocumentStatus,
    EmbeddedChunk,
    EmbeddingModel,
)
from app.domain.similarity import cosine_similarity
from app.infrastructure.storage.local_file_storage import LocalFileStorage
from app.infrastructure.storage.sqlite_document_repository import (
    SCHEMA_VERSION,
    SqliteDocumentRepository,
)
from app.infrastructure.vectorstores.chroma_vector_store import (
    ChromaVectorStore,
    collection_name,
)

MODEL = EmbeddingModel("BAAI/bge-small-en-v1.5", 3)

# --- Chroma ---


def test_chroma_data_survives_a_restart(tmp_path: Path) -> None:
    ChromaVectorStore.local(tmp_path).upsert(
        MODEL,
        "doc",
        "a.txt",
        [EmbeddedChunk(Chunk(0, 1, "hello", 0, 5), (1.0, 0.0, 0.0))],
    )

    reopened = ChromaVectorStore.local(tmp_path)

    assert [h.chunk.text for h in reopened.search(MODEL, (1.0, 0.0, 0.0), 5)] == [
        "hello"
    ]


def test_collection_names_are_valid_and_distinct() -> None:
    a = collection_name(EmbeddingModel("BAAI/bge-small-en-v1.5", 384))
    b = collection_name(EmbeddingModel("BAAI-bge-small-en-v1.5", 384))
    c = collection_name(EmbeddingModel("BAAI/bge-small-en-v1.5", 512))

    assert a.startswith("chunks-BAAI-bge-small-en-v1.5-384-")
    assert len({a, b, c}) == 3
    assert all(
        set(n)
        <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        for n in (a, b, c)
    )


def test_chroma_failures_become_storage_unavailable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = ChromaVectorStore.local(tmp_path)
    store.upsert(
        MODEL, "doc", "a.txt", [EmbeddedChunk(Chunk(0, 1, "x", 0, 1), (1.0, 0.0, 0.0))]
    )

    with caplog.at_level(logging.ERROR), pytest.raises(StorageUnavailable):
        store.search(MODEL, (1.0, 0.0), 5)


# --- SQLite ---


def record() -> DocumentRecord:
    return DocumentRecord(
        id="doc",
        filename="a.txt",
        file_type="txt",
        size_bytes=1,
        status=DocumentStatus.INDEXED,
        created_at=datetime(2026, 10, 27, tzinfo=UTC),
    )


def test_sqlite_data_survives_a_restart(tmp_path: Path) -> None:
    SqliteDocumentRepository(tmp_path / "db.sqlite").add(record())

    assert SqliteDocumentRepository(tmp_path / "db.sqlite").get("doc") == record()


def test_sqlite_records_its_schema_version(tmp_path: Path) -> None:
    SqliteDocumentRepository(tmp_path / "db.sqlite")
    with sqlite3.connect(tmp_path / "db.sqlite") as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_sqlite_errors_become_storage_unavailable(tmp_path: Path) -> None:
    repo = SqliteDocumentRepository(tmp_path / "db.sqlite")
    repo.add(record())

    with pytest.raises(StorageUnavailable):
        repo.add(record())


# --- Local files ---


def test_files_are_saved_overwritten_and_deleted(tmp_path: Path) -> None:
    files = LocalFileStorage(tmp_path)

    files.save("doc.txt", b"one")
    files.save("doc.txt", b"two")
    assert (tmp_path / "doc.txt").read_bytes() == b"two"

    files.delete("doc.txt")
    files.delete("doc.txt")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("key", ["../escape.txt", "a/b.txt", "", ".hidden", "x" * 300])
def test_unsafe_file_keys_are_refused(tmp_path: Path, key: str) -> None:
    with pytest.raises(ValueError):
        LocalFileStorage(tmp_path).save(key, b"data")


@pytest.mark.parametrize("space", ["l2", "ip", "cosine"])
def test_scores_are_cosine_whatever_distance_the_server_uses(
    tmp_path: Path, space: Literal["l2", "ip", "cosine"]
) -> None:
    client = chromadb.PersistentClient(
        path=str(tmp_path), settings=ChromaSettings(anonymized_telemetry=False)
    )
    store = ChromaVectorStore(client, {"hnsw": {"space": space}})
    vectors = {0: (3.0, 0.0, 0.0), 1: (0.0, 10.0, 0.0), 2: (2.0, 2.0, 0.0)}
    store.upsert(
        MODEL,
        "doc",
        "a.txt",
        [EmbeddedChunk(Chunk(i, 1, f"t{i}", 0, 2), v) for i, v in vectors.items()],
    )
    query = (5.0, 1.0, 0.0)

    hits = store.search(MODEL, query, 3)

    expected = sorted(
        ((cosine_similarity(query, v), i) for i, v in vectors.items()), reverse=True
    )
    assert [h.chunk.index for h in hits] == [i for _, i in expected]
    assert [h.score for h in hits] == pytest.approx([s for s, _ in expected])
