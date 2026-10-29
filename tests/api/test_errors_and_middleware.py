"""Cross-cutting HTTP behaviour: error shape, request IDs, unexpected errors."""

import logging
import re

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_ingest_document
from app.core.config import Settings
from app.main import create_app
from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider

URL = "/api/v1/documents"
FRONTEND = "http://localhost:3008"


@pytest.fixture
def client() -> TestClient:
    return TestClient(
        create_app(
            Settings(_env_file=None),
            embedding_provider=FakeEmbeddingProvider(),
            llm_provider=FakeLLMProvider(),
        )
    )


class ExplodingUseCase:
    def execute(self, filename: str, stream: object) -> None:
        raise RuntimeError("database password=hunter2 leaked in a bug")


@pytest.fixture
def broken_client() -> TestClient:
    app = create_app(
        Settings(_env_file=None),
        embedding_provider=FakeEmbeddingProvider(),
        llm_provider=FakeLLMProvider(),
    )
    app.dependency_overrides[get_ingest_document] = ExplodingUseCase
    return TestClient(app)


def test_unknown_route_uses_standard_error_shape(client: TestClient) -> None:
    response = client.get("/api/v1/nope")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "not_found"


def test_wrong_method_uses_standard_error_shape(client: TestClient) -> None:
    response = client.put(URL)

    assert response.status_code == 405
    assert response.json()["detail"]["code"] == "method_not_allowed"


def test_every_response_has_a_request_id(client: TestClient) -> None:
    response = client.get("/api/v1/health")

    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["X-Request-ID"])


def test_safe_incoming_request_id_is_reused(client: TestClient) -> None:
    response = client.get("/api/v1/health", headers={"X-Request-ID": "trace-abc.1"})

    assert response.headers["X-Request-ID"] == "trace-abc.1"


def test_unsafe_incoming_request_id_is_replaced(client: TestClient) -> None:
    response = client.get(
        "/api/v1/health", headers={"X-Request-ID": "bad id\nINJECTED"}
    )

    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["X-Request-ID"])


def test_browser_can_read_the_request_id(client: TestClient) -> None:
    response = client.get("/api/v1/health", headers={"Origin": FRONTEND})

    assert "x-request-id" in response.headers["access-control-expose-headers"].lower()


def test_unexpected_error_returns_generic_500(broken_client: TestClient) -> None:
    response = broken_client.post(
        URL, files={"file": ("a.txt", b"hi")}, headers={"Origin": FRONTEND}
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": {"code": "internal_error", "message": "An unexpected error occurred."}
    }
    assert "hunter2" not in response.text
    assert response.headers["access-control-allow-origin"] == FRONTEND
    assert response.headers["X-Request-ID"]


def test_unexpected_error_is_logged_with_stack_trace_and_request_id(
    broken_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        response = broken_client.post(URL, files={"file": ("a.txt", b"hi")})

    failure = next(r for r in caplog.records if r.message == "request.failed")
    assert failure.exc_info is not None
    assert failure.__dict__["request_id"] == response.headers["X-Request-ID"]


def test_each_request_is_logged_once(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        client.get("/api/v1/health")

    completed = [r for r in caplog.records if r.message == "request.completed"]
    assert len(completed) == 1
    assert completed[0].__dict__["status"] == 200
    assert completed[0].__dict__["path"] == "/api/v1/health"
