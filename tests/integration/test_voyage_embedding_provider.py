"""Integration tests: the Voyage adapter against a simulated Voyage API."""

import logging

import httpx2
import pytest

from app.domain.errors import EmbeddingRateLimited, EmbeddingUnavailable
from app.infrastructure.embeddings.http_embedding_client import MISCONFIGURED
from app.infrastructure.embeddings.voyage_embedding_provider import (
    VoyageEmbeddingProvider,
)
from tests.fake_voyage import API_KEY, TIMEOUT, FakeVoyageServer, fake_vector

# --- Requests ---


def test_sends_the_documented_request() -> None:
    server = FakeVoyageServer()

    server.provider(dimensions=8).embed_documents(["hello"])

    assert server.paths == ["/v1/embeddings"]
    assert server.auth_headers == [f"Bearer {API_KEY}"]
    assert server.requests == [
        {
            "input": ["hello"],
            "model": "voyage-4-lite",
            "input_type": "document",
            "output_dimension": 8,
            "truncation": True,
        }
    ]


def test_queries_use_the_query_input_type() -> None:
    server = FakeVoyageServer()

    vector = server.provider().embed_query("what is revenue?")

    assert server.requests[0]["input_type"] == "query"
    assert vector == tuple(fake_vector("what is revenue?", 8))


def test_returns_vectors_in_input_order_even_if_the_server_reorders() -> None:
    server = FakeVoyageServer(reverse=True)

    vectors = server.provider().embed_documents(["a", "b", "c"])

    assert vectors == [tuple(fake_vector(t, 8)) for t in ["a", "b", "c"]]


def test_splits_large_inputs_into_batches() -> None:
    server = FakeVoyageServer()
    texts = [f"text {i}" for i in range(300)]

    vectors = server.provider(batch_size=128).embed_documents(texts)

    assert [len(r["input"]) for r in server.requests] == [128, 128, 44]
    assert vectors == [tuple(fake_vector(t, 8)) for t in texts]


def test_empty_input_makes_no_request() -> None:
    server = FakeVoyageServer()

    assert server.provider().embed_documents([]) == []
    assert server.requests == []


# --- Retries ---


def test_retries_rate_limits_then_succeeds() -> None:
    server = FakeVoyageServer(script=[429])

    vectors = server.provider().embed_documents(["a"])

    assert len(vectors) == 1
    assert len(server.requests) == 2
    assert len(server.sleeps) == 1
    assert 1.0 <= server.sleeps[0] <= 2.0


def test_backoff_grows_exponentially() -> None:
    server = FakeVoyageServer(script=[500, 500, 500])

    server.provider(max_retries=3).embed_documents(["a"])

    first, second, third = server.sleeps
    assert 1 <= first <= 2 and 2 <= second <= 3 and 4 <= third <= 5


def test_gives_up_after_max_retries() -> None:
    server = FakeVoyageServer(script=[429, 429, 429, 429])

    with pytest.raises(EmbeddingRateLimited):
        server.provider(max_retries=3).embed_documents(["a"])

    assert len(server.requests) == 4
    assert len(server.sleeps) == 3


def test_honours_retry_after() -> None:
    server = FakeVoyageServer(
        script=[httpx2.Response(429, headers={"retry-after": "2"})]
    )

    server.provider().embed_documents(["a"])

    assert server.sleeps == [2.0]


def test_does_not_wait_when_retry_after_is_too_long() -> None:
    server = FakeVoyageServer(
        script=[httpx2.Response(429, headers={"retry-after": "60"})]
    )

    with pytest.raises(EmbeddingRateLimited) as error:
        server.provider().embed_documents(["a"])

    assert error.value.retry_after_seconds == 60.0
    assert server.sleeps == []


def test_retries_server_errors_and_timeouts() -> None:
    server = FakeVoyageServer(script=[503, TIMEOUT])

    assert len(server.provider().embed_documents(["a"])) == 1
    assert len(server.requests) == 3


@pytest.mark.parametrize("failure", [503, TIMEOUT])
def test_persistent_outage_is_unavailable_not_rate_limited(failure: int | str) -> None:
    server = FakeVoyageServer(script=[failure] * 4)

    with pytest.raises(EmbeddingUnavailable) as error:
        server.provider(max_retries=3).embed_documents(["a"])

    assert not isinstance(error.value, EmbeddingRateLimited)


@pytest.mark.parametrize("status", [400, 401, 403, 422])
def test_client_errors_are_not_retried(status: int) -> None:
    server = FakeVoyageServer(script=[status])

    with pytest.raises(EmbeddingUnavailable):
        server.provider().embed_documents(["a"])

    assert len(server.requests) == 1
    assert server.sleeps == []


def test_bad_api_key_gives_a_safe_message() -> None:
    server = FakeVoyageServer(script=[401])

    with pytest.raises(EmbeddingUnavailable) as error:
        server.provider().embed_documents(["a"])

    assert error.value.message == MISCONFIGURED
    assert API_KEY not in error.value.message


# --- Malformed responses ---


@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(200, text="not json"),
        httpx2.Response(200, json={"unexpected": True}),
        httpx2.Response(200, json={"data": []}),
        httpx2.Response(200, json={"data": [{"index": 0}]}),
        httpx2.Response(200, json={"data": [{"index": 0, "embedding": ["x"]}]}),
    ],
)
def test_malformed_responses_are_unavailable(response: httpx2.Response) -> None:
    server = FakeVoyageServer(script=[response])

    with pytest.raises(EmbeddingUnavailable):
        server.provider().embed_documents(["a"])


# --- Observability and safety ---


def test_logs_usage_but_never_the_key_or_the_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = FakeVoyageServer(script=[429])

    with caplog.at_level(logging.INFO):
        server.provider().embed_documents(["confidential sentence"])

    completed = next(r for r in caplog.records if r.message == "embedding.completed")
    assert completed.__dict__["total_tokens"] > 0
    assert completed.__dict__["model"] == "voyage-4-lite"
    assert any(r.message == "embedding.retrying" for r in caplog.records)
    everything_logged = " ".join(str(vars(r)) for r in caplog.records)
    assert API_KEY not in everything_logged
    assert "confidential sentence" not in everything_logged


def test_requires_an_api_key() -> None:
    with pytest.raises(ValueError):
        VoyageEmbeddingProvider(api_key="", model_name="voyage-4-lite", dimensions=8)
