"""Use case: answer a question from the stored documents, with checked citations.

    1. retrieve the closest chunks; for a follow-up, search with the previous
       question plus the new one, so "how do I request them?" still finds
       the passages about vacation days
    2. nothing retrieved -> "insufficient evidence", without calling the model
    3. stream the model's answer, giving it only those chunks as evidence and
       the last few turns as conversation context
    4. pass each citation on only if it points at a passage actually sent and
       quotes it; number them [1], [2], ... in order of first appearance, and
       add the markers when the cited run of text ends
    5. no valid citation -> "insufficient evidence"

`open()` does everything that can fail before any text exists (validation,
search, connecting to the model), so callers can report those errors normally;
the returned session then streams. Neither questions nor answers are logged.
"""

import logging
import time
from collections.abc import Iterator, Sequence

from app.application.ports import LLMAnswerStream, LLMProvider
from app.application.search_chunks import MAX_QUERY_CHARS, SearchChunks
from app.domain.answers import (
    Answer,
    AnswerCitationAdded,
    AnswerDone,
    AnswerEvent,
    AnswerTextDelta,
    BlockEnd,
    ChatTurn,
    Citation,
    CitationClaim,
    CitedSpan,
    EvidencePassage,
    TextChunk,
)
from app.domain.errors import InvalidInput
from app.domain.models import SearchHit, chunk_id

logger = logging.getLogger(__name__)

NO_DOCUMENTS_ANSWER = (
    "There are no indexed documents to answer from yet. Upload documents first."
)
MAX_HISTORY_TURNS = 3


class AnswerSession:
    """One answer being produced. Iterate `events()` once; always `close()`."""

    def __init__(
        self,
        retrieved: tuple[SearchHit, ...],
        passages: list[EvidencePassage],
        stream: LLMAnswerStream | None,
    ) -> None:
        self.retrieved = retrieved
        self._passages = passages
        self._stream = stream

    def events(self) -> Iterator[AnswerEvent]:
        if self._stream is None:
            yield AnswerTextDelta(NO_DOCUMENTS_ANSWER)
            yield AnswerDone(insufficient_evidence=True, model=None)
            return

        started = time.perf_counter()
        numbers: dict[int, int] = {}
        pending: list[int] = []
        dropped = 0
        try:
            for event in self._stream:
                if isinstance(event, TextChunk):
                    if event.text:
                        yield AnswerTextDelta(event.text)
                elif isinstance(event, CitationClaim):
                    span = event.span
                    if not _is_valid(span, self._passages):
                        dropped += 1
                        continue
                    if span.passage_index not in numbers:
                        numbers[span.passage_index] = len(numbers) + 1
                        yield AnswerCitationAdded(
                            Citation(
                                number=numbers[span.passage_index],
                                hit=self.retrieved[span.passage_index],
                                quote=span.quoted_text.strip(),
                            )
                        )
                    if numbers[span.passage_index] not in pending:
                        pending.append(numbers[span.passage_index])
                elif isinstance(event, BlockEnd) and pending:
                    yield AnswerTextDelta("".join(f"[{n}]" for n in pending))
                    pending = []
        finally:
            self.close()

        if pending:
            yield AnswerTextDelta("".join(f"[{n}]" for n in pending))
        logger.info(
            "answer.generated",
            extra={
                "model": self._stream.model,
                "passages": len(self._passages),
                "citations": len(numbers),
                "dropped_citations": dropped,
                "insufficient_evidence": not numbers,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        yield AnswerDone(insufficient_evidence=not numbers, model=self._stream.model)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()


class AnswerQuestion:
    def __init__(self, search: SearchChunks, llm: LLMProvider, top_k: int) -> None:
        self._search = search
        self._llm = llm
        self._top_k = top_k

    def open(self, question: str, history: Sequence[ChatTurn] = ()) -> AnswerSession:
        question = question.strip()
        if not question:
            raise InvalidInput("Enter a question.")
        if len(question) > MAX_QUERY_CHARS:
            raise InvalidInput(
                f"The question is longer than {MAX_QUERY_CHARS} characters."
            )
        recent = list(history)[-MAX_HISTORY_TURNS:]

        query = question
        if recent:
            query = f"{recent[-1].question.strip()}\n{question}"[-MAX_QUERY_CHARS:]
        hits = tuple(self._search.execute(query, self._top_k).hits)
        if not hits:
            return AnswerSession(retrieved=(), passages=[], stream=None)

        passages = [_passage(hit) for hit in hits]
        stream = self._llm.stream_answer(question, passages, recent)
        return AnswerSession(retrieved=hits, passages=passages, stream=stream)

    def execute(self, question: str, history: Sequence[ChatTurn] = ()) -> Answer:
        """The whole answer at once, collected from the same checked event stream."""
        session = self.open(question, history)
        text: list[str] = []
        citations: list[Citation] = []
        done = AnswerDone(insufficient_evidence=True, model=None)
        for event in session.events():
            if isinstance(event, AnswerTextDelta):
                text.append(event.text)
            elif isinstance(event, AnswerCitationAdded):
                citations.append(event.citation)
            else:
                done = event
        return Answer(
            text="".join(text).strip(),
            citations=tuple(citations),
            insufficient_evidence=done.insufficient_evidence,
            model=done.model,
            retrieved=session.retrieved,
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
