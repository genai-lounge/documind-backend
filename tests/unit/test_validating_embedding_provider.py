from collections.abc import Sequence

import pytest

from app.application.ports import EmbeddingProvider
from app.application.validating_embedding_provider import ValidatingEmbeddingProvider
from app.domain.errors import EmbeddingUnavailable
from app.domain.models import EmbeddingModel, Vector
from tests.fakes import FakeEmbeddingProvider


class ScriptedProvider(EmbeddingProvider):
    """Returns exactly the vectors it is given, to simulate a misbehaving vendor."""

    def __init__(self, vectors: list[Vector], dimensions: int = 3) -> None:
        self.model = EmbeddingModel("scripted", dimensions)
        self.vectors = vectors
        self.called = False

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        self.called = True
        return self.vectors

    def embed_query(self, text: str) -> Vector:
        return self.vectors[0]


def test_passes_good_vectors_through() -> None:
    inner = FakeEmbeddingProvider(dimensions=4)
    provider = ValidatingEmbeddingProvider(inner)

    assert provider.embed_documents(["a", "b"]) == inner.embed_documents(["a", "b"])
    assert provider.model == inner.model


def test_empty_input_never_calls_the_service() -> None:
    inner = ScriptedProvider([])

    assert ValidatingEmbeddingProvider(inner).embed_documents([]) == []
    assert not inner.called


def test_wrong_number_of_vectors() -> None:
    provider = ValidatingEmbeddingProvider(ScriptedProvider([(1.0, 2.0, 3.0)]))

    with pytest.raises(EmbeddingUnavailable, match="1 vectors for 2 texts"):
        provider.embed_documents(["a", "b"])


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ((1.0, 2.0), "length 2; expected 3"),
        ((1.0, float("nan"), 3.0), "invalid numbers"),
        ((1.0, float("inf"), 3.0), "invalid numbers"),
        ((0.0, 0.0, 0.0), "empty vector"),
    ],
)
def test_rejects_bad_vectors(vector: Vector, message: str) -> None:
    provider = ValidatingEmbeddingProvider(ScriptedProvider([vector]))

    with pytest.raises(EmbeddingUnavailable, match=message):
        provider.embed_documents(["a"])
    with pytest.raises(EmbeddingUnavailable, match=message):
        provider.embed_query("a")
