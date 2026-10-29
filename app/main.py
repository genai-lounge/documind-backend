"""Application entry point.

`create_app()` is an app factory: it builds a fully wired FastAPI app from
Settings. Uvicorn calls it at startup (`uvicorn app.main:create_app --factory`),
and tests call it with their own Settings and a fake embedding provider.
Nothing is built at import time, so importing this module has no side effects.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_error_handlers
from app.api.middleware import REQUEST_ID_HEADER, RequestContextMiddleware
from app.api.v1.router import router as v1_router
from app.application.ports import EmbeddingProvider
from app.container import Container
from app.core.config import Settings
from app.core.logging_config import configure_logging


def create_app(
    settings: Settings | None = None,
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level, settings.log_format)
    container = Container(settings, embedding_provider=embedding_provider)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        container.close()

    app = FastAPI(title="DocuMind API", version="0.1.0", lifespan=lifespan)
    app.state.ingest_document = container.ingest_document
    app.state.compare_texts = container.compare_texts
    app.state.list_documents = container.list_documents
    app.state.get_document = container.get_document
    app.state.delete_document = container.delete_document
    app.state.search_chunks = container.search_chunks

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER],
    )

    register_error_handlers(app)
    app.include_router(v1_router)
    return app
