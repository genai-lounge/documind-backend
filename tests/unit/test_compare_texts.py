import pytest

from app.application.compare_texts import CompareTexts
from app.domain.errors import EmbeddingUnavailable
from tests.fakes import FakeEmbeddingProvider


def test_identical_texts_are_fully_similar() -> None:
    result = CompareTexts(FakeEmbeddingProvider()).execute("same", "same")

    assert result.similarity == pytest.approx(1.0)


def test_reports_the_model_used() -> None:
    embedder = FakeEmbeddingProvider(dimensions=16)

    result = CompareTexts(embedder).execute("a", "b")

    assert result.model == embedder.model
    assert -1.0 <= result.similarity <= 1.0


def test_embeds_both_texts_in_a_single_call() -> None:
    embedder = FakeEmbeddingProvider()

    CompareTexts(embedder).execute("car", "automobile")

    assert embedder.calls == [("document", ["car", "automobile"])]


def test_provider_errors_propagate() -> None:
    embedder = FakeEmbeddingProvider(error=EmbeddingUnavailable("down"))

    with pytest.raises(EmbeddingUnavailable):
        CompareTexts(embedder).execute("a", "b")
