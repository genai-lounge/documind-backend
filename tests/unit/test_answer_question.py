"""AnswerQuestion: retrieval -> model -> citation checking, with fakes."""

import logging

import pytest

from app.application.answer_question import NO_DOCUMENTS_ANSWER, AnswerQuestion
from app.application.search_chunks import SearchChunks
from app.domain.answers import AnswerSegment, CitedSpan
from app.domain.errors import AnswerRefused, InvalidInput, LLMUnavailable
from app.domain.models import Chunk, EmbeddedChunk
from tests.fakes import (
    FakeLLMProvider,
    InMemoryVectorStore,
    WordOverlapEmbeddingProvider,
)
from tests.retrieval_dataset import DOCUMENTS

EMBEDDER = WordOverlapEmbeddingProvider()


def make(
    llm: FakeLLMProvider | None = None, empty: bool = False
) -> tuple[AnswerQuestion, FakeLLMProvider]:
    vectors = InMemoryVectorStore()
    if not empty:
        for name, text in DOCUMENTS.items():
            vectors.upsert(
                EMBEDDER.model,
                name.removesuffix(".txt"),
                name,
                [
                    EmbeddedChunk(
                        Chunk(0, 1, text, 0, len(text)), EMBEDDER.embed_query(text)
                    )
                ],
            )
    llm = llm or FakeLLMProvider()
    search = SearchChunks(EMBEDDER, vectors, default_top_k=3, max_top_k=10)
    return AnswerQuestion(search, llm, top_k=3), llm


def test_answers_with_a_checked_numbered_citation() -> None:
    use_case, _ = make()

    answer = use_case.execute("What is the name of the office cat?")

    assert not answer.insufficient_evidence
    assert answer.text.endswith("[1]")
    [citation] = answer.citations
    assert citation.number == 1
    assert citation.hit.filename == "office.txt"
    assert "Biscuit" in citation.quote
    assert answer.model == "fake-llm"
    assert len(answer.retrieved) == 3


def test_model_receives_only_the_retrieved_passages_with_titles() -> None:
    use_case, llm = make()

    use_case.execute("  What is the name of the office cat?  ")

    question, evidence = llm.calls[0]
    assert question == "What is the name of the office cat?"
    assert len(evidence) == 3
    assert evidence[0].title == "office.txt · page 1"
    assert evidence[0].source_id == "office:0"


def test_no_documents_means_insufficient_evidence_without_calling_the_model() -> None:
    use_case, llm = make(empty=True)

    answer = use_case.execute("Anything?")

    assert answer.insufficient_evidence
    assert answer.text == NO_DOCUMENTS_ANSWER
    assert llm.calls == []


def test_an_answer_without_citations_is_insufficient_evidence() -> None:
    llm = FakeLLMProvider(
        [AnswerSegment("The documents don't say how tall the cat is.")]
    )
    use_case, _ = make(llm)

    answer = use_case.execute("How tall is the cat?")

    assert answer.insufficient_evidence
    assert answer.citations == ()
    assert answer.text == "The documents don't say how tall the cat is."


@pytest.mark.parametrize(
    "span",
    [
        CitedSpan(7, "Biscuit"),
        CitedSpan(-1, "Biscuit"),
        CitedSpan(0, "A sentence that appears in no passage."),
        CitedSpan(0, "   "),
    ],
)
def test_citations_that_dont_match_the_evidence_are_dropped(
    span: CitedSpan, caplog: pytest.LogCaptureFixture
) -> None:
    llm = FakeLLMProvider([AnswerSegment("The cat is Biscuit.", (span,))])
    use_case, _ = make(llm)

    with caplog.at_level(logging.INFO):
        answer = use_case.execute("What is the name of the office cat?")

    assert answer.citations == ()
    assert answer.insufficient_evidence
    assert "[" not in answer.text
    record = next(r for r in caplog.records if r.message == "answer.generated")
    assert record.__dict__["dropped_citations"] == 1


def test_quotes_are_matched_ignoring_whitespace_differences() -> None:
    use_case, llm = make()
    office_text = DOCUMENTS["office.txt"]
    llm.segments = [
        AnswerSegment(
            "Biscuit.", (CitedSpan(0, "  " + office_text.replace(" ", "\n", 2)),)
        )
    ]

    answer = use_case.execute("What is the name of the office cat?")

    assert len(answer.citations) == 1


def test_citations_are_numbered_in_order_of_first_appearance() -> None:
    use_case, llm = make()
    use_case.execute("office cat Biscuit vacation days revenue")
    _, evidence = llm.calls[0]
    llm.segments = [
        AnswerSegment("First claim.", (CitedSpan(2, evidence[2].text),)),
        AnswerSegment(
            " Second claim.",
            (CitedSpan(0, evidence[0].text), CitedSpan(2, evidence[2].text)),
        ),
        AnswerSegment(" Third claim.", (CitedSpan(0, evidence[0].text),)),
    ]

    answer = use_case.execute("office cat Biscuit vacation days revenue")

    assert answer.text == "First claim.[1] Second claim.[2][1] Third claim.[2]"
    assert [(c.number, c.hit.filename) for c in answer.citations] == [
        (1, evidence[2].title.split(" · ")[0]),
        (2, evidence[0].title.split(" · ")[0]),
    ]


@pytest.mark.parametrize("error", [LLMUnavailable("down"), AnswerRefused("no")])
def test_model_errors_propagate(error: Exception) -> None:
    use_case, _ = make(FakeLLMProvider(error=error))

    with pytest.raises(type(error)):
        use_case.execute("What is the name of the office cat?")


@pytest.mark.parametrize("question", ["", "   ", "x" * 2001])
def test_rejects_invalid_questions(question: str) -> None:
    use_case, llm = make()

    with pytest.raises(InvalidInput):
        use_case.execute(question)

    assert llm.calls == []


def test_never_logs_the_question_or_the_answer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    use_case, _ = make()

    with caplog.at_level(logging.INFO):
        answer = use_case.execute("What is the secret name of the office cat?")

    logged = " ".join(str(vars(r)) for r in caplog.records)
    assert "secret name" not in logged
    assert answer.text not in logged
