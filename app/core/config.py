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

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = "text"

    cors_allowed_origins: Annotated[list[str], NoDecode] = ["http://localhost:3008"]

    max_upload_size_mb: int = Field(default=10, gt=0, le=1024)

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024
