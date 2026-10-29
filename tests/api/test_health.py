from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider

client = TestClient(
    create_app(
        Settings(_env_file=None),
        embedding_provider=FakeEmbeddingProvider(),
        llm_provider=FakeLLMProvider(),
    )
)


def test_health_returns_ok() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
