"""Live tests against the real Voyage AI API. Run with `make test-live`.

Skipped unless VOYAGE_API_KEY is set (in the environment or backend/.env).
They make three small requests, which fits even the free tier's limit of
3 requests per minute, and use only made-up sentences.
"""

from collections.abc import Iterator

import pytest

from app.application.validating_embedding_provider import ValidatingEmbeddingProvider
from app.core.config import Settings
from app.domain.similarity import cosine_similarity
from app.infrastructure.embeddings.voyage_embedding_provider import (
    VoyageEmbeddingProvider,
)

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def voyage() -> Iterator[ValidatingEmbeddingProvider]:
    settings = Settings()
    if (
        settings.voyage_api_key is None
        or not settings.voyage_api_key.get_secret_value()
    ):
        pytest.skip("VOYAGE_API_KEY is not set")
    provider = VoyageEmbeddingProvider(
        api_key=settings.voyage_api_key.get_secret_value(),
        model_name=settings.voyage_model,
        dimensions=settings.voyage_dimensions,
    )
    yield ValidatingEmbeddingProvider(provider)
    provider.close()


def test_documents_get_vectors_of_the_configured_size(
    voyage: ValidatingEmbeddingProvider,
) -> None:
    vectors = voyage.embed_documents(["The cat sat on the mat.", "Revenue grew 12%."])

    assert len(vectors) == 2
    assert all(len(v) == voyage.model.dimensions for v in vectors)


def test_similar_meanings_score_higher_than_unrelated_ones(
    voyage: ValidatingEmbeddingProvider,
) -> None:
    car, automobile, banana = voyage.embed_documents(["car", "automobile", "banana"])

    assert cosine_similarity(car, automobile) > cosine_similarity(car, banana)


def test_a_question_lands_near_its_answer(voyage: ValidatingEmbeddingProvider) -> None:
    question = voyage.embed_query("How much did revenue grow last quarter?")
    answer, unrelated = voyage.embed_documents(
        ["In Q3, revenue increased by 12% to $4M.", "The office cat is named Biscuit."]
    )

    assert cosine_similarity(question, answer) > cosine_similarity(question, unrelated)
