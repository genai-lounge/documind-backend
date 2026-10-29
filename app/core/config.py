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
VERCEL_MAX_UPLOAD_MB = 4


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

    data_dir: Path = BACKEND_DIR / "data"

    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-haiku-5-5"
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    anthropic_max_tokens: int = Field(default=16000, ge=1024, le=128000)
    anthropic_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    anthropic_max_retries: int = Field(default=2, ge=0, le=10)
    answer_top_k: int = Field(default=5, ge=1, le=20)

    document_store: Literal["sqlite", "postgres"] = "sqlite"
    database_url: SecretStr | None = None

    vercel: bool = False

    file_storage: Literal["local", "supabase"] = "local"
    supabase_url: str | None = None
    supabase_secret_key: SecretStr | None = None
    supabase_bucket: str = "documents"

    chroma_mode: Literal["local", "cloud"] = "local"
    chroma_api_key: SecretStr | None = None
    chroma_tenant: str | None = None
    chroma_database: str | None = None
    chroma_cloud_host: str = "api.trychroma.com"

    search_default_top_k: int = Field(default=5, ge=1, le=100)
    search_max_top_k: int = Field(default=20, ge=1, le=100)

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
            value = value.split(",")
        if isinstance(value, list):
            return [str(o).strip().rstrip("/") for o in value if str(o).strip()]
        return value

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @model_validator(mode="after")
    def _default_top_k_within_max(self) -> "Settings":
        if self.search_default_top_k > self.search_max_top_k:
            raise ValueError("SEARCH_DEFAULT_TOP_K must not exceed SEARCH_MAX_TOP_K")
        return self

    @model_validator(mode="after")
    def _serverless_needs_online_storage(self) -> "Settings":
        """Vercel sets VERCEL=1. Its disk is temporary, and request bodies are capped at 4.5 MB."""
        if not self.vercel:
            return self
        problems = []
        if self.document_store != "postgres":
            problems.append("DOCUMENT_STORE=postgres")
        if self.file_storage != "supabase":
            problems.append("FILE_STORAGE=supabase")
        if self.chroma_mode != "cloud":
            problems.append("CHROMA_MODE=cloud")
        if self.max_upload_size_mb > VERCEL_MAX_UPLOAD_MB:
            problems.append(f"MAX_UPLOAD_SIZE_MB={VERCEL_MAX_UPLOAD_MB} or less")
        if problems:
            raise ValueError(
                "On Vercel, files on disk are lost between requests and uploads are "
                "limited to 4.5 MB. Set " + ", ".join(problems) + "."
            )
        return self

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "documind.db"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024
