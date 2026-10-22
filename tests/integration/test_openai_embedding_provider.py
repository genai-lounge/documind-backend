"""Integration tests: the OpenAI adapter against a simulated OpenAI API.

Retries, backoff, and error mapping live in the shared HTTP client and are
covered in depth by the Voyage tests; these focus on OpenAI's request format.
"""

import pytest

from app.domain.errors import EmbeddingRateLimited, EmbeddingUnavailable
from app.infrastructure.embeddings.http_embedding_client import MISCONFIGURED
from app.infrastructure.embeddings.openai_embedding_provider import (
    OpenAIEmbeddingProvider,
)
from tests.fake_voyage import API_KEY, FakeVoyageServer, fake_vector


def test_sends_the_documented_request() -> None:
    server = FakeVoyageServer()

    server.openai_provider().embed_documents(["hello"])

    assert server.paths == ["/v1/embeddings"]
    assert server.auth_headers == [f"Bearer {API_KEY}"]
    assert server.requests == [
        {
            "input": ["hello"],
            "model": "text-embedding-3-small",
            "encoding_format": "float",
        }
    ]


def test_uses_native_dimensions_by_default() -> None:
    provider = FakeVoyageServer().openai_provider()

    assert provider.model.dimensions == 1536
    assert len(provider.embed_documents(["a"])[0]) == 1536


def test_sends_dimensions_only_when_configured() -> None:
    server = FakeVoyageServer()

    vector = server.openai_provider(dimensions=256).embed_query("a question")

    assert server.requests[0]["dimensions"] == 256
    assert len(vector) == 256


def test_queries_are_embedded_like_documents() -> None:
    server = FakeVoyageServer()

    server.openai_provider().embed_query("q")

    assert "input_type" not in server.requests[0]


def test_batches_and_keeps_order() -> None:
    server = FakeVoyageServer(reverse=True)
    texts = [f"t{i}" for i in range(5)]

    vectors = server.openai_provider(batch_size=2).embed_documents(texts)

    assert [len(r["input"]) for r in server.requests] == [2, 2, 1]
    assert vectors == [tuple(fake_vector(t, 1536)) for t in texts]


def test_retries_then_reports_rate_limits() -> None:
    server = FakeVoyageServer(script=[429, 429])

    with pytest.raises(EmbeddingRateLimited):
        server.openai_provider(max_retries=1).embed_documents(["a"])

    assert len(server.requests) == 2


def test_bad_key_gives_a_safe_message() -> None:
    server = FakeVoyageServer(script=[401])

    with pytest.raises(EmbeddingUnavailable) as error:
        server.openai_provider().embed_documents(["a"])

    assert error.value.message == MISCONFIGURED


def test_unknown_model_needs_explicit_dimensions() -> None:
    with pytest.raises(ValueError, match="Unknown dimensions"):
        OpenAIEmbeddingProvider(api_key="sk-test", model_name="my-custom-model")
