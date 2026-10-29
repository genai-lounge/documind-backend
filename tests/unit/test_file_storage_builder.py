"""Choosing local or Supabase file storage from Settings."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from app import file_storages
from app.core.config import ConfigurationError, Settings
from app.infrastructure.storage.local_file_storage import LocalFileStorage
from app.infrastructure.storage.supabase_file_storage import SupabaseFileStorage


def settings(tmp_path: Path, **values: Any) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path, **values)


def supabase(tmp_path: Path, **values: Any) -> Settings:
    defaults: dict[str, Any] = {
        "file_storage": "supabase",
        "supabase_url": "https://project.supabase.co",
        "supabase_secret_key": SecretStr("sb_secret_abc"),
    }
    return settings(tmp_path, **{**defaults, **values})


def test_local_is_the_default(tmp_path: Path) -> None:
    assert isinstance(
        file_storages.build_file_storage(settings(tmp_path)), LocalFileStorage
    )


@pytest.mark.parametrize(
    "url", [None, "", "project.supabase.co", "http://insecure.example"]
)
def test_supabase_needs_an_https_url(tmp_path: Path, url: str | None) -> None:
    with pytest.raises(ConfigurationError, match="SUPABASE_URL"):
        file_storages.build_file_storage(supabase(tmp_path, supabase_url=url))


@pytest.mark.parametrize("key", [None, "", "   "])
def test_supabase_needs_a_secret_key(tmp_path: Path, key: str | None) -> None:
    with pytest.raises(ConfigurationError, match="SUPABASE_SECRET_KEY"):
        file_storages.build_file_storage(
            supabase(
                tmp_path,
                supabase_secret_key=SecretStr(key) if key is not None else None,
            )
        )


def test_supabase_bucket_is_ensured_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ensured: list[str] = []

    def ensure(self: SupabaseFileStorage) -> bool:
        ensured.append(self.bucket)
        return True

    monkeypatch.setattr(SupabaseFileStorage, "ensure_bucket", ensure)

    storage = file_storages.build_file_storage(
        supabase(tmp_path, supabase_bucket="uploads")
    )

    assert isinstance(storage, SupabaseFileStorage)
    assert ensured == ["uploads"]


def test_startup_problems_are_reported_without_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(self: SupabaseFileStorage) -> bool:
        raise RuntimeError("401 for key sb_secret_abc")

    monkeypatch.setattr(SupabaseFileStorage, "ensure_bucket", refuse)

    with pytest.raises(
        ConfigurationError, match="Could not use Supabase Storage"
    ) as error:
        file_storages.build_file_storage(supabase(tmp_path))

    assert "sb_secret_abc" not in str(error.value)
