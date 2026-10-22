"""HTTP endpoints for exploring embeddings."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_compare_texts
from app.api.v1.schemas import ErrorResponse, SimilarityRequest, SimilarityResponse
from app.application.compare_texts import CompareTexts

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post(
    "/similarity",
    response_model=SimilarityResponse,
    responses={
        422: {"model": ErrorResponse, "description": "A text is empty or too long"},
        503: {"model": ErrorResponse, "description": "Embedding service unavailable"},
    },
)
def compare_texts(
    body: SimilarityRequest,
    compare: Annotated[CompareTexts, Depends(get_compare_texts)],
) -> SimilarityResponse:
    return SimilarityResponse.from_domain(compare.execute(body.text_a, body.text_b))
