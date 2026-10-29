"""Choosing local or cloud Chroma from Settings."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from app import vector_stores
from app.core.config import ConfigurationError, Settings
from app.infrastructure.vectorstores.chroma_vector_store import ChromaVectorStore


def settings(tmp_path: Path, **values: Any) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path, **values)


def test_local_is_the_default(tmp_path: Path) -> None:
    store = vector_stores.build_vector_store(settings(tmp_path))

    assert isinstance(store, ChromaVectorStore)
    assert (tmp_path / "chroma").is_dir()


@pytest.mark.parametrize("key", [None, "", "  "])
def test_cloud_needs_an_api_key(tmp_path: Path, key: str | None) -> None:
    with pytest.raises(ConfigurationError, match="CHROMA_API_KEY"):
        vector_stores.build_vector_store(
            settings(
                tmp_path,
                chroma_mode="cloud",
                chroma_api_key=SecretStr(key) if key is not None else None,
            )
        )


def test_cloud_passes_the_configured_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def record(**kw: Any) -> None:
        calls.append(kw)

    monkeypatch.setattr(ChromaVectorStore, "cloud", record)

    vector_stores.build_vector_store(
        settings(
            tmp_path,
            chroma_mode="cloud",
            chroma_api_key=SecretStr(" ck-123 "),
            chroma_tenant="tenant-1",
            chroma_database="documind",
        )
    )

    assert calls == [
        {
            "api_key": "ck-123",
            "tenant": "tenant-1",
            "database": "documind",
            "host": "api.trychroma.com",
        }
    ]


def test_cloud_connection_problems_fail_at_startup_without_leaking_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(**kw: Any) -> None:
        raise PermissionError("401 Unauthorized")

    monkeypatch.setattr(ChromaVectorStore, "cloud", refuse)

    with pytest.raises(
        ConfigurationError, match="Could not connect to Chroma Cloud"
    ) as error:
        vector_stores.build_vector_store(
            settings(
                tmp_path, chroma_mode="cloud", chroma_api_key=SecretStr("ck-secret")
            )
        )

    assert "ck-secret" not in str(error.value)
