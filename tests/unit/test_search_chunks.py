import logging

import pytest

from app.application.search_chunks import SearchChunks
from app.domain.errors import InvalidInput
from app.domain.models import Chunk, EmbeddedChunk
from tests.fakes import InMemoryVectorStore, WordOverlapEmbeddingProvider

EMBEDDER = WordOverlapEmbeddingProvider()
PASSAGES = [
    "Quarterly revenue grew twelve percent thanks to strong demand.",
    "The office cat is named Biscuit and sleeps by the window.",
    "Employees receive twenty five vacation days every year.",
]


def make_search(default_top_k: int = 2, max_top_k: int = 5) -> SearchChunks:
    vectors = InMemoryVectorStore()
    vectors.upsert(
        EMBEDDER.model,
        "doc",
        "handbook.txt",
        [
            EmbeddedChunk(Chunk(i, i + 1, text, 0, len(text)), v)
            for i, (text, v) in enumerate(
                zip(PASSAGES, EMBEDDER.embed_documents(PASSAGES), strict=True)
            )
        ],
    )
    return SearchChunks(EMBEDDER, vectors, default_top_k, max_top_k)


def test_finds_the_passage_that_answers_the_question() -> None:
    result = make_search().execute("How many vacation days do employees get?")

    best = result.hits[0]
    assert best.rank == 1
    assert "vacation" in best.chunk.text
    assert best.filename == "handbook.txt"
    assert best.chunk.page_number == 3
    assert result.model == EMBEDDER.model


def test_hits_are_ranked_best_first() -> None:
    hits = make_search(default_top_k=3).execute("cat named Biscuit").hits

    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_uses_the_default_and_explicit_top_k() -> None:
    search = make_search(default_top_k=2)

    assert len(search.execute("revenue").hits) == 2
    assert len(search.execute("revenue", top_k=1).hits) == 1


@pytest.mark.parametrize(
    ("query", "top_k", "message"),
    [
        ("   ", None, "Enter a question"),
        ("x" * 2001, None, "longer than"),
        ("revenue", 0, "between 1 and 5"),
        ("revenue", 6, "between 1 and 5"),
    ],
)
def test_rejects_invalid_input(query: str, top_k: int | None, message: str) -> None:
    with pytest.raises(InvalidInput, match=message):
        make_search().execute(query, top_k)


def test_logs_the_search_but_never_the_query(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        make_search().execute("secret merger plans")

    record = next(r for r in caplog.records if r.message == "retrieval.searched")
    assert record.__dict__["hits"] == 2
    assert "secret merger plans" not in " ".join(str(vars(r)) for r in caplog.records)
