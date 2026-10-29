"""HTTP endpoints for inspecting retrieval: which stored chunks match a query."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_search_chunks
from app.api.v1.schemas import ErrorResponse, SearchRequest, SearchResponse
from app.application.search_chunks import SearchChunks

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Empty query or top_k out of range",
        },
        503: {
            "model": ErrorResponse,
            "description": "Embedding or vector store unavailable",
        },
    },
)
def search(
    body: SearchRequest,
    search_chunks: Annotated[SearchChunks, Depends(get_search_chunks)],
) -> SearchResponse:
    return SearchResponse.from_domain(search_chunks.execute(body.query, body.top_k))
