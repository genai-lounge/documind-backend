"""The FastEmbed adapter, with a stand-in model so no model is downloaded.

The real model is exercised by tests/live/test_fastembed_live.py.
"""

from collections.abc import Iterable
from typing import Any

import pytest

from app.domain.errors import EmbeddingUnavailable
from app.infrastructure.embeddings.fastembed_provider import FastEmbedProvider
from tests.fake_voyage import fake_vector


class StandInModel:
    """Mimics fastembed.TextEmbedding: lazily yields one vector per text."""

    def __init__(self, dimensions: int = 4, fail: bool = False) -> None:
        self.dimensions = dimensions
        self.fail = fail
        self.calls: list[tuple[str, list[str], dict[str, Any]]] = []

    def passage_embed(self, texts: Iterable[str], **kwargs: Any) -> Iterable[Any]:
        texts = list(texts)
        self.calls.append(("passage", texts, kwargs))
        return self._vectors(texts)

    def query_embed(self, query: str | Iterable[str], **kwargs: Any) -> Iterable[Any]:
        texts = [query] if isinstance(query, str) else list(query)
        self.calls.append(("query", texts, kwargs))
        return self._vectors(texts)

    def _vectors(self, texts: list[str]) -> Iterable[Any]:
        for text in texts:
            if self.fail:
                raise RuntimeError("onnx exploded")
            yield fake_vector(text, self.dimensions)


def make(model: StandInModel) -> FastEmbedProvider:
    return FastEmbedProvider("BAAI/bge-small-en-v1.5", model=model, dimensions=4)


def test_documents_use_passage_embedding_with_the_batch_size() -> None:
    model = StandInModel()

    vectors = make(model).embed_documents(["a", "b"])

    assert vectors == [tuple(fake_vector(t, 4)) for t in ["a", "b"]]
    assert model.calls == [("passage", ["a", "b"], {"batch_size": 128})]


def test_queries_use_query_embedding() -> None:
    model = StandInModel()

    make(model).embed_query("what?")

    assert model.calls[0][:2] == ("query", ["what?"])


def test_empty_input_never_runs_the_model() -> None:
    model = StandInModel()

    assert make(model).embed_documents([]) == []
    assert model.calls == []


def test_model_failures_become_embedding_unavailable() -> None:
    with pytest.raises(EmbeddingUnavailable):
        make(StandInModel(fail=True)).embed_documents(["a"])


def test_reports_its_model() -> None:
    provider = make(StandInModel())

    assert (provider.model.name, provider.model.dimensions) == (
        "BAAI/bge-small-en-v1.5",
        4,
    )
