"""Version 1 of the API. Every v1 route is mounted under /api/v1.

A future breaking change would go in app/api/v2/ while v1 keeps working.
"""

from fastapi import APIRouter

from app.api.v1 import chat, documents, embeddings, health, retrieval

router = APIRouter(prefix="/api/v1")
router.include_router(health.router)
router.include_router(documents.router)
router.include_router(embeddings.router)
router.include_router(retrieval.router)
router.include_router(chat.router)
