"""Application entry point.

`create_app()` is an app factory: it builds a fully wired FastAPI app from
Settings. Tests call it with their own Settings; uvicorn uses the module-level
`app` below (`uv run uvicorn app.main:app`).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_error_handlers
from app.api.middleware import REQUEST_ID_HEADER, RequestContextMiddleware
from app.api.v1.router import router as v1_router
from app.container import Container
from app.core.config import Settings
from app.core.logging_config import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level, settings.log_format)
    container = Container(settings)

    app = FastAPI(title="DocuMind API", version="0.1.0")
    app.state.ingest_document = container.ingest_document

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER],
    )

    register_error_handlers(app)
    app.include_router(v1_router)
    return app


app = create_app()
