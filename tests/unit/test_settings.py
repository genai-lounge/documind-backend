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
