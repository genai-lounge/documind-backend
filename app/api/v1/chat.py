"""HTTP endpoints for asking questions about the stored documents.

`POST /chat` returns the whole answer at once. `POST /chat/stream` returns
server-sent events as the answer is written:

    sources   every passage given to the model (sent before the answer starts)
    delta     the next piece of answer text (including "[n]" markers)
    citation  a checked citation, sent when it first appears
    done      whether the evidence was insufficient, and the model used
    error     {code, message} if something fails after streaming has started

Problems found before the first byte (invalid question, search or model
connection failures) are normal HTTP errors, like every other endpoint.
"""

import json
import logging
from collections.abc import Iterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response
from fastapi.sse import EventSourceResponse, format_sse_event
from starlette.types import Receive, Scope, Send

from app.api.dependencies import get_answer_question
from app.api.errors import error_body
from app.api.v1.schemas import (
    AnswerResponse,
    AskRequest,
    CitationOut,
    ErrorResponse,
    SourceOut,
    StreamDoneOut,
)
from app.application.answer_question import AnswerQuestion, AnswerSession
from app.domain.answers import AnswerCitationAdded, AnswerDone, AnswerTextDelta
from app.domain.errors import DocuMindError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

ERRORS: dict[int | str, dict[str, object]] = {
    422: {
        "model": ErrorResponse,
        "description": "Invalid question, or the model declined",
    },
    503: {
        "model": ErrorResponse,
        "description": "Language model, embeddings, or storage unavailable",
    },
}


@router.post("", response_model=AnswerResponse, responses=ERRORS)
def ask(
    body: AskRequest,
    answer_question: Annotated[AnswerQuestion, Depends(get_answer_question)],
) -> AnswerResponse:
    answer = answer_question.execute(body.question, body.history_turns())
    return AnswerResponse.from_domain(answer)


@router.post(
    "/stream",
    response_class=Response,
    responses={
        200: {
            "description": "Server-sent events",
            "content": {"text/event-stream": {}},
        },
        **ERRORS,
    },
)
def ask_streaming(
    body: AskRequest,
    answer_question: Annotated[AnswerQuestion, Depends(get_answer_question)],
) -> EventSourceResponse:
    session = answer_question.open(body.question, body.history_turns())
    return _AnswerEventStream(session)


class _AnswerEventStream(EventSourceResponse):
    """Closes the answer session when the response ends for any reason.

    If the client disconnects (the Stop button), Starlette stops reading the
    event generator without closing it, so the model stream would stay open
    until garbage collection. Closing here stops it straight away.
    """

    def __init__(self, session: AnswerSession) -> None:
        super().__init__(
            _events(session),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        self._session = session

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._session.close()


def _sse(event: str, data: Any) -> bytes:
    return format_sse_event(event=event, data_str=json.dumps(data))


def _events(session: AnswerSession) -> Iterator[bytes]:
    """Runs while the client reads; closing early (Stop) closes the model stream."""
    try:
        yield _sse(
            "sources", [SourceOut.from_hit(h).model_dump() for h in session.retrieved]
        )
        for event in session.events():
            if isinstance(event, AnswerTextDelta):
                yield _sse("delta", {"text": event.text})
            elif isinstance(event, AnswerCitationAdded):
                yield _sse(
                    "citation", CitationOut.from_domain(event.citation).model_dump()
                )
            elif isinstance(event, AnswerDone):
                done = StreamDoneOut(
                    insufficient_evidence=event.insufficient_evidence, model=event.model
                )
                yield _sse("done", done.model_dump())
    except DocuMindError as exc:
        logger.info("chat.stream_failed", extra={"code": exc.code})
        yield _sse("error", error_body(exc.code, exc.message)["detail"])
    except Exception:
        logger.exception("chat.stream_failed")
        yield _sse(
            "error",
            error_body("internal_error", "An unexpected error occurred.")["detail"],
        )
    finally:
        session.close()
