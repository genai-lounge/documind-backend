"""Application settings, loaded from environment variables and backend/.env.

pydantic-settings reads each field from the environment variable with the same
name (case-insensitive) and validates it. Invalid configuration (for example
MAX_UPLOAD_SIZE_MB=abc) stops the app at startup with a clear message, instead
of failing later in the middle of a request.

Only the composition root (app/container.py) and app/main.py read Settings.
Everything else receives plain values through its constructor, so no business
code depends on where configuration comes from.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BACKEND_DIR / ".env"


class ConfigurationError(RuntimeError):
    """The app can't start because a required setting is missing or invalid."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE, extra="ignore", env_ignore_empty=True
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = "text"

    cors_allowed_origins: Annotated[list[str], NoDecode] = ["http://localhost:3008"]

    max_upload_size_mb: int = Field(default=10, gt=0, le=1024)

    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=150, ge=0)

    embedding_provider: str = "openai"
    embedding_batch_size: int = Field(default=128, ge=1, le=1000)
    embedding_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    embedding_max_retries: int = Field(default=3, ge=0, le=10)

    openai_api_key: SecretStr | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int | None = Field(default=None, ge=1)

    voyage_api_key: SecretStr | None = None
    voyage_model: str = "voyage-4-lite"
    voyage_dimensions: int = 1024

    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    fastembed_cache_dir: Path = BACKEND_DIR / "data" / "models"

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024
