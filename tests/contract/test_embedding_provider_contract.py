"""Contract tests: rules EVERY EmbeddingProvider must follow.

Hosted vendors run against a simulated API and FastEmbed against a stand-in
model; tests against the real services live in tests/live/ (opt-in).
"""

import math
from collections.abc import Callable

import pytest

from app.application.ports import EmbeddingProvider
from app.application.validating_embedding_provider import ValidatingEmbeddingProvider
from app.infrastructure.embeddings.fastembed_provider import FastEmbedProvider
from tests.fake_voyage import FakeVoyageServer
from tests.fakes import FakeEmbeddingProvider
from tests.integration.test_fastembed_provider import StandInModel

PROVIDERS: dict[str, Callable[[], EmbeddingProvider]] = {
    "FakeEmbeddingProvider": lambda: FakeEmbeddingProvider(dimensions=16),
    "VoyageEmbeddingProvider": lambda: FakeVoyageServer().provider(dimensions=16),
    "OpenAIEmbeddingProvider": lambda: FakeVoyageServer().openai_provider(
        dimensions=16
    ),
    "FastEmbedProvider": lambda: FastEmbedProvider(
        model=StandInModel(dimensions=16), dimensions=16
    ),
    "ValidatingEmbeddingProvider": lambda: ValidatingEmbeddingProvider(
        FakeEmbeddingProvider(dimensions=16)
    ),
}


@pytest.fixture(params=list(PROVIDERS))
def provider(request: pytest.FixtureRequest) -> EmbeddingProvider:
    return PROVIDERS[request.param]()


def test_describes_its_model(provider: EmbeddingProvider) -> None:
    assert provider.model.name
    assert provider.model.dimensions > 0


def test_one_vector_per_text_with_the_model_dimensions(
    provider: EmbeddingProvider,
) -> None:
    vectors = provider.embed_documents(["first", "second", "third"])

    assert len(vectors) == 3
    for vector in vectors:
        assert len(vector) == provider.model.dimensions
        assert all(math.isfinite(x) for x in vector)


def test_preserves_input_order(provider: EmbeddingProvider) -> None:
    together = provider.embed_documents(["x", "y"])

    assert together == [
        provider.embed_documents(["x"])[0],
        provider.embed_documents(["y"])[0],
    ]


def test_empty_input_gives_no_vectors(provider: EmbeddingProvider) -> None:
    assert provider.embed_documents([]) == []


def test_embeds_queries(provider: EmbeddingProvider) -> None:
    assert len(provider.embed_query("a question?")) == provider.model.dimensions
