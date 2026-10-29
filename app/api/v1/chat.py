"""HTTP endpoint for asking questions about the stored documents."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_answer_question
from app.api.v1.schemas import AnswerResponse, AskRequest, ErrorResponse
from app.application.answer_question import AnswerQuestion

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "",
    response_model=AnswerResponse,
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Invalid question, or the model declined",
        },
        503: {
            "model": ErrorResponse,
            "description": "Language model, embeddings, or storage unavailable",
        },
    },
)
def ask(
    body: AskRequest,
    answer_question: Annotated[AnswerQuestion, Depends(get_answer_question)],
) -> AnswerResponse:
    return AnswerResponse.from_domain(answer_question.execute(body.question))
