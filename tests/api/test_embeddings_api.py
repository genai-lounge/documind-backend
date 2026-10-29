"""The similarity playground endpoint, and how embedding failures reach clients."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.domain.errors import EmbeddingRateLimited, EmbeddingUnavailable
from app.main import create_app
from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider

URL = "/api/v1/embeddings/similarity"


def make_client(embedder: FakeEmbeddingProvider | None = None) -> TestClient:
    return TestClient(
        create_app(
            Settings(_env_file=None),
            embedding_provider=embedder or FakeEmbeddingProvider(),
            llm_provider=FakeLLMProvider(),
        )
    )


def test_compares_two_texts() -> None:
    response = make_client().post(URL, json={"text_a": "car", "text_b": "automobile"})

    assert response.status_code == 200
    body = response.json()
    assert -1.0 <= body["similarity"] <= 1.0
    assert body["model"] == "fake-embedder"
    assert body["dimensions"] == 8


def test_identical_texts_score_one() -> None:
    body = make_client().post(URL, json={"text_a": "same", "text_b": "same"}).json()

    assert body["similarity"] == pytest.approx(1.0)


def test_surrounding_whitespace_is_ignored() -> None:
    embedder = FakeEmbeddingProvider()

    make_client(embedder).post(URL, json={"text_a": "  car ", "text_b": "bus\n"})

    assert embedder.calls == [("document", ["car", "bus"])]


@pytest.mark.parametrize(
    "payload",
    [
        {"text_a": "", "text_b": "x"},
        {"text_a": "   ", "text_b": "x"},
        {"text_a": "x" * 2001, "text_b": "x"},
        {"text_a": "only one"},
        {},
    ],
)
def test_invalid_input_is_rejected(payload: dict[str, str]) -> None:
    response = make_client().post(URL, json=payload)

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_request"


def test_embedding_outage_returns_503() -> None:
    embedder = FakeEmbeddingProvider(error=EmbeddingUnavailable("Service down."))

    response = make_client(embedder).post(URL, json={"text_a": "a", "text_b": "b"})

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "embedding_unavailable",
        "message": "Service down.",
    }


def test_rate_limit_returns_503_with_retry_after() -> None:
    embedder = FakeEmbeddingProvider(error=EmbeddingRateLimited("Busy.", 29.2))

    response = make_client(embedder).post(URL, json={"text_a": "a", "text_b": "b"})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "embedding_rate_limited"
    assert response.headers["Retry-After"] == "30"


def test_upload_fails_with_503_when_embedding_is_down() -> None:
    embedder = FakeEmbeddingProvider(error=EmbeddingUnavailable("Service down."))

    response = make_client(embedder).post(
        "/api/v1/documents", files={"file": ("a.txt", b"some text")}
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "embedding_unavailable"
