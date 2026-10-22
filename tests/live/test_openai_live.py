"""Live tests against the real OpenAI API. Run with `make test-live`.

Skipped unless OPENAI_API_KEY is set. Two small requests with made-up text.
"""

from collections.abc import Iterator

import pytest

from app.application.validating_embedding_provider import ValidatingEmbeddingProvider
from app.core.config import Settings
from app.domain.similarity import cosine_similarity
from app.infrastructure.embeddings.openai_embedding_provider import (
    OpenAIEmbeddingProvider,
)

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def openai() -> Iterator[ValidatingEmbeddingProvider]:
    settings = Settings()
    if settings.openai_api_key is None:
        pytest.skip("OPENAI_API_KEY is not set")
    provider = OpenAIEmbeddingProvider(
        api_key=settings.openai_api_key.get_secret_value(),
        model_name=settings.openai_embedding_model,
        dimensions=settings.openai_embedding_dimensions,
    )
    yield ValidatingEmbeddingProvider(provider)
    provider.close()


def test_similar_meanings_score_higher_than_unrelated_ones(
    openai: ValidatingEmbeddingProvider,
) -> None:
    car, automobile, banana = openai.embed_documents(["car", "automobile", "banana"])

    assert cosine_similarity(car, automobile) > cosine_similarity(car, banana)


def test_a_question_lands_near_its_answer(openai: ValidatingEmbeddingProvider) -> None:
    question = openai.embed_query("How much did revenue grow last quarter?")
    answer, unrelated = openai.embed_documents(
        ["In Q3, revenue increased by 12% to $4M.", "The office cat is named Biscuit."]
    )

    assert cosine_similarity(question, answer) > cosine_similarity(question, unrelated)
