"""Chooses where document records are kept: DOCUMENT_STORE=sqlite or postgres.

Part of the composition root. Configuration and connection problems become
ConfigurationError, so the app refuses to start with a clear message.
"""

import logging

from app.application.ports import DocumentRepository
from app.core.config import ConfigurationError, Settings
from app.domain.errors import StorageUnavailable
from app.infrastructure.storage.postgres_document_repository import (
    PostgresDocumentRepository,
)
from app.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)

logger = logging.getLogger(__name__)


def build_document_store(settings: Settings) -> DocumentRepository:
    if settings.document_store == "sqlite":
        return SqliteDocumentRepository(settings.database_path)

    url = (
        settings.database_url.get_secret_value().strip()
        if settings.database_url
        else ""
    )
    if not url.startswith(("postgresql://", "postgres://")):
        raise ConfigurationError(
            "DATABASE_URL is not set (postgresql://...). In Supabase, open Connect → "
            "Transaction pooler and copy the URI, or set DOCUMENT_STORE=sqlite."
        )
    try:
        store = PostgresDocumentRepository(url)
    except StorageUnavailable as exc:
        raise ConfigurationError(
            "Could not connect to the Postgres database. Check DATABASE_URL (including "
            "the password) and that the Supabase project isn't paused."
        ) from exc
    logger.info("document_store.connected", extra={"provider": "postgres"})
    return store
