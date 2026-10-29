"""Use case: answer a question from the stored documents, with checked citations.

    1. retrieve the closest chunks (SearchChunks)
    2. nothing retrieved -> "insufficient evidence", without calling the model
    3. ask the LLMProvider, giving it only those chunks as evidence
    4. keep only citations that point at a passage actually sent and quote it;
       number them [1], [2], ... in order of first appearance
    5. no valid citation -> "insufficient evidence"

Neither the question nor the answer text is ever logged.
"""

import logging
import time

from app.application.ports import LLMProvider
from app.application.search_chunks import MAX_QUERY_CHARS, SearchChunks
from app.domain.answers import (
    Answer,
    AnswerSegment,
    Citation,
    CitedSpan,
    EvidencePassage,
)
from app.domain.errors import InvalidInput
from app.domain.models import SearchHit, chunk_id

logger = logging.getLogger(__name__)

NO_DOCUMENTS_ANSWER = (
    "There are no indexed documents to answer from yet. Upload documents first."
)


class AnswerQuestion:
    def __init__(self, search: SearchChunks, llm: LLMProvider, top_k: int) -> None:
        self._search = search
        self._llm = llm
        self._top_k = top_k

    def execute(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            raise InvalidInput("Enter a question.")
        if len(question) > MAX_QUERY_CHARS:
            raise InvalidInput(
                f"The question is longer than {MAX_QUERY_CHARS} characters."
            )

        started = time.perf_counter()
        hits = tuple(self._search.execute(question, self._top_k).hits)
        if not hits:
            return Answer(
                text=NO_DOCUMENTS_ANSWER,
                citations=(),
                insufficient_evidence=True,
                model=None,
                retrieved=(),
            )

        passages = [_passage(hit) for hit in hits]
        generated = self._llm.answer(question, passages)

        numbers: dict[int, int] = {}
        citations: list[Citation] = []
        parts: list[str] = []
        dropped = 0
        for segment in generated.segments:
            markers: list[int] = []
            for span in segment.cited:
                if not _is_valid(span, passages):
                    dropped += 1
                    continue
                if span.passage_index not in numbers:
                    numbers[span.passage_index] = len(numbers) + 1
                    citations.append(
                        Citation(
                            number=numbers[span.passage_index],
                            hit=hits[span.passage_index],
                            quote=span.quoted_text.strip(),
                        )
                    )
                if numbers[span.passage_index] not in markers:
                    markers.append(numbers[span.passage_index])
            parts.append(_with_markers(segment, markers))

        logger.info(
            "answer.generated",
            extra={
                "model": generated.model,
                "passages": len(passages),
                "citations": len(citations),
                "dropped_citations": dropped,
                "insufficient_evidence": not citations,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return Answer(
            text="".join(parts).strip(),
            citations=tuple(citations),
            insufficient_evidence=not citations,
            model=generated.model,
            retrieved=hits,
        )


def _passage(hit: SearchHit) -> EvidencePassage:
    page = (
        f" · page {hit.chunk.page_number}" if hit.chunk.page_number is not None else ""
    )
    return EvidencePassage(
        source_id=chunk_id(hit.document_id, hit.chunk.index),
        title=f"{hit.filename}{page}",
        text=hit.chunk.text,
    )


def _is_valid(span: CitedSpan, passages: list[EvidencePassage]) -> bool:
    """The cited passage must exist, and the quoted text must really be in it."""
    if not 0 <= span.passage_index < len(passages):
        return False
    quote = " ".join(span.quoted_text.split())
    text = " ".join(passages[span.passage_index].text.split())
    return bool(quote) and quote in text


def _with_markers(segment: AnswerSegment, markers: list[int]) -> str:
    """Put [n] markers after the segment's text, before any trailing whitespace."""
    if not markers:
        return segment.text
    body = segment.text.rstrip()
    trailing = segment.text[len(body) :]
    return body + "".join(f"[{n}]" for n in markers) + trailing
