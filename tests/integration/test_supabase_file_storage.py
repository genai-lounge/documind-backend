"""Integration tests: the Supabase Storage adapter against a simulated Supabase API."""

import logging

import pytest

from app.domain.errors import StorageUnavailable
from app.infrastructure.storage.supabase_file_storage import SupabaseFileStorage
from tests.fake_supabase import SECRET_KEY, TIMEOUT, URL, FakeSupabase

# --- Authentication ---


def test_secret_keys_go_only_in_the_apikey_header() -> None:
    supabase = FakeSupabase(buckets={"documents"})

    supabase.storage().save("doc.txt", b"hi")

    request = supabase.requests[0]
    assert request.headers["apikey"] == SECRET_KEY
    assert "authorization" not in request.headers


def test_legacy_service_role_keys_also_send_a_bearer_token() -> None:
    supabase = FakeSupabase(buckets={"documents"})

    supabase.storage(key="eyJlegacy.jwt.key").save("doc.txt", b"hi")

    assert supabase.requests[0].headers["authorization"] == "Bearer eyJlegacy.jwt.key"


# --- Bucket ---


def test_creates_a_private_bucket_when_missing() -> None:
    supabase = FakeSupabase()

    assert supabase.storage().ensure_bucket() is True
    assert supabase.buckets == {"documents": False}


def test_existing_bucket_is_left_alone() -> None:
    supabase = FakeSupabase(buckets={"documents"})

    assert supabase.storage().ensure_bucket() is False
    assert [r.method for r in supabase.requests] == ["GET"]


def test_bad_credentials_while_checking_the_bucket() -> None:
    with pytest.raises(StorageUnavailable):
        FakeSupabase(script=[401]).storage().ensure_bucket()


# --- Files ---


def test_save_uploads_to_the_bucket_with_a_content_type() -> None:
    supabase = FakeSupabase(buckets={"documents"})

    supabase.storage().save("abc.pdf", b"%PDF-1.7 data")

    assert supabase.objects[("documents", "abc.pdf")] == (
        b"%PDF-1.7 data",
        "application/pdf",
    )
    request = supabase.requests[0]
    assert request.url == f"{URL}/storage/v1/object/documents/abc.pdf"
    assert request.headers["x-upsert"] == "true"


def test_save_overwrites() -> None:
    supabase = FakeSupabase(buckets={"documents"})
    storage = supabase.storage()

    storage.save("doc.txt", b"one")
    storage.save("doc.txt", b"two")

    assert supabase.objects[("documents", "doc.txt")][0] == b"two"


def test_delete_removes_the_file_and_is_idempotent() -> None:
    supabase = FakeSupabase(buckets={"documents"})
    storage = supabase.storage()
    storage.save("doc.txt", b"hi")

    storage.delete("doc.txt")
    storage.delete("doc.txt")

    assert supabase.objects == {}


@pytest.mark.parametrize("key", ["../escape.txt", "a/b.txt", "", ".hidden"])
def test_unsafe_keys_are_refused_before_any_request(key: str) -> None:
    supabase = FakeSupabase(buckets={"documents"})

    with pytest.raises(ValueError):
        supabase.storage().save(key, b"x")

    assert supabase.requests == []


def test_invalid_bucket_names_are_refused() -> None:
    with pytest.raises(ValueError):
        SupabaseFileStorage(URL, SECRET_KEY, "Bad Bucket!")


# --- Failures and retries ---


@pytest.mark.parametrize("failure", [503, 429, TIMEOUT])
def test_transient_failures_are_retried(failure: int | str) -> None:
    supabase = FakeSupabase(buckets={"documents"}, script=[failure])

    supabase.storage().save("doc.txt", b"hi")

    assert ("documents", "doc.txt") in supabase.objects
    assert len(supabase.sleeps) == 1


@pytest.mark.parametrize("failure", [503, TIMEOUT])
def test_persistent_outage_becomes_storage_unavailable(failure: int | str) -> None:
    supabase = FakeSupabase(buckets={"documents"}, script=[failure] * 3)

    with pytest.raises(StorageUnavailable):
        supabase.storage(max_retries=2).save("doc.txt", b"hi")

    assert len(supabase.sleeps) == 2


@pytest.mark.parametrize("status", [400, 401, 403])
def test_client_errors_are_not_retried(status: int) -> None:
    supabase = FakeSupabase(buckets={"documents"}, script=[status])

    with pytest.raises(StorageUnavailable):
        supabase.storage().save("doc.txt", b"hi")

    assert supabase.sleeps == []


def test_the_key_is_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    supabase = FakeSupabase(buckets={"documents"}, script=[503, 401])

    with caplog.at_level(logging.INFO), pytest.raises(StorageUnavailable):
        supabase.storage().save("doc.txt", b"hi")

    assert SECRET_KEY not in " ".join(str(vars(r)) for r in caplog.records)
