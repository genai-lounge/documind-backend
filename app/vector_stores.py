"""Chooses the vector store from Settings: CHROMA_MODE=local or cloud.

Part of the composition root. Configuration and connection problems become
ConfigurationError, so the app refuses to start with a clear message.
"""

import logging

from app.core.config import ConfigurationError, Settings
from app.infrastructure.vectorstores.chroma_vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


def build_vector_store(settings: Settings) -> ChromaVectorStore:
    if settings.chroma_mode == "local":
        return ChromaVectorStore.local(settings.chroma_dir)

    key = settings.chroma_api_key
    if key is None or not key.get_secret_value().strip():
        raise ConfigurationError(
            "CHROMA_API_KEY is not set. Create an API key in your Chroma Cloud "
            "dashboard (https://www.trychroma.com) and add CHROMA_API_KEY=... to "
            "backend/.env, or set CHROMA_MODE=local."
        )
    try:
        store = ChromaVectorStore.cloud(
            api_key=key.get_secret_value().strip(),
            tenant=settings.chroma_tenant,
            database=settings.chroma_database,
            host=settings.chroma_cloud_host,
        )
    except Exception as exc:
        raise ConfigurationError(
            "Could not connect to Chroma Cloud. Check CHROMA_API_KEY (and "
            "CHROMA_TENANT / CHROMA_DATABASE if your key covers more than one "
            f"database). Details: {type(exc).__name__}: {exc}"
        ) from exc
    logger.info(
        "vector_store.connected",
        extra={"mode": "cloud", "host": settings.chroma_cloud_host},
    )
    return store
