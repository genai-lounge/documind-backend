"""Live test of the real FastEmbed model. Run with `make test-live`.

Skipped unless the extra is installed (`uv sync --extra fastembed`). The first
run downloads the model (about 67 MB) to backend/data/models/.
"""

import pytest

from app.core.config import Settings
from app.domain.similarity import cosine_similarity

pytestmark = pytest.mark.live


def test_similar_meanings_score_higher_than_unrelated_ones() -> None:
    pytest.importorskip("fastembed")
    from app.infrastructure.embeddings.fastembed_provider import FastEmbedProvider

    settings = Settings()
    provider = FastEmbedProvider(
        model_name=settings.fastembed_model,
        cache_dir=str(settings.fastembed_cache_dir),
    )

    car, automobile, banana = provider.embed_documents(["car", "automobile", "banana"])

    assert len(car) == provider.model.dimensions == 384
    assert cosine_similarity(car, automobile) > cosine_similarity(car, banana)
