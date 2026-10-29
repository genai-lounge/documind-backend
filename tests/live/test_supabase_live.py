"""Live test against your Supabase Storage bucket. Run with `make test-live`.

Skipped unless FILE_STORAGE=supabase is configured. Creates the bucket if it
is missing, then uploads, overwrites, and deletes one tiny made-up file.
"""

import uuid

import pytest

from app.core.config import Settings
from app.file_storages import build_file_storage

pytestmark = pytest.mark.live


def test_save_overwrite_delete_round_trip() -> None:
    settings = Settings()
    if settings.file_storage != "supabase":
        pytest.skip("FILE_STORAGE=supabase is not configured")

    storage = build_file_storage(settings)
    key = f"documind-live-test-{uuid.uuid4().hex[:8]}.txt"
    try:
        storage.save(key, b"first version")
        storage.save(key, b"second version")
    finally:
        storage.delete(key)
        storage.delete(key)
