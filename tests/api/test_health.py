from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.fakes import FakeEmbeddingProvider

client = TestClient(
    create_app(Settings(_env_file=None), embedding_provider=FakeEmbeddingProvider())
)


def test_health_returns_ok() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
