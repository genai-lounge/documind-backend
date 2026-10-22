import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_defaults(settings: Settings) -> None:
    assert settings.max_upload_size_mb == 10
    assert settings.max_upload_size_bytes == 10 * 1024 * 1024
    assert settings.cors_allowed_origins == ["http://localhost:3008"]
    assert settings.log_format == "text"


def test_reads_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "5")
    monkeypatch.setenv("LOG_FORMAT", "json")

    settings = Settings(_env_file=None)

    assert settings.max_upload_size_bytes == 5 * 1024 * 1024
    assert settings.log_format == "json"


def test_cors_origins_are_comma_separated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://a.example, http://b.example")

    assert Settings(_env_file=None).cors_allowed_origins == [
        "http://a.example",
        "http://b.example",
    ]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MAX_UPLOAD_SIZE_MB", "0"),
        ("MAX_UPLOAD_SIZE_MB", "lots"),
        ("LOG_FORMAT", "xml"),
        ("LOG_LEVEL", "LOUD"),
    ],
)
def test_invalid_values_fail_at_startup(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_chunking_and_embedding_defaults(settings: Settings) -> None:
    assert (settings.chunk_size, settings.chunk_overlap) == (800, 150)
    assert settings.embedding_provider == "openai"
    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.openai_embedding_dimensions is None
    assert (settings.voyage_model, settings.voyage_dimensions) == (
        "voyage-4-lite",
        1024,
    )
    assert settings.fastembed_model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_batch_size == 128


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CHUNK_SIZE", "50"),
        ("CHUNK_OVERLAP", "-1"),
        ("OPENAI_EMBEDDING_DIMENSIONS", "0"),
        ("EMBEDDING_BATCH_SIZE", "0"),
    ],
)
def test_invalid_chunking_and_embedding_values(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_empty_values_count_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSIONS", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    settings = Settings(_env_file=None)

    assert settings.openai_embedding_dimensions is None
    assert settings.openai_api_key is None


def test_overlap_must_be_smaller_than_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "200")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")

    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(_env_file=None)


def test_dimensions_are_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSIONS", "512")

    assert Settings(_env_file=None).openai_embedding_dimensions == 512


def test_api_key_is_hidden_when_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-super-secret")

    settings = Settings(_env_file=None)

    assert "pa-super-secret" not in repr(settings)
    assert settings.voyage_api_key is not None
    assert settings.voyage_api_key.get_secret_value() == "pa-super-secret"
