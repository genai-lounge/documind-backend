"""End to end over HTTP: upload, list, inspect, search, delete, and restart.

Uses the real SQLite, Chroma, and file storage in a temporary folder, with a
word-overlap embedder so results are deterministic and need no network.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.application.ports import EmbeddingProvider
from app.core.config import Settings
from app.domain.errors import EmbeddingUnavailable
from app.main import create_app
from tests.fakes import (
    FakeEmbeddingProvider,
    FakeLLMProvider,
    WordOverlapEmbeddingProvider,
)
from tests.retrieval_dataset import DOCUMENTS, QUESTIONS

DOCS = "/api/v1/documents"
SEARCH = "/api/v1/retrieval/search"


def make_client(
    data_dir: Path, embedder: EmbeddingProvider | None = None
) -> TestClient:
    settings = Settings(_env_file=None, data_dir=data_dir)
    provider = embedder or WordOverlapEmbeddingProvider()
    return TestClient(
        create_app(
            settings, embedding_provider=provider, llm_provider=FakeLLMProvider()
        )
    )


def upload(client: TestClient, name: str, text: str) -> str:
    response = client.post(DOCS, files={"file": (name, text.encode())})
    assert response.status_code == 200, response.text
    document_id: str = response.json()["id"]
    return document_id


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    client = make_client(tmp_path)
    for name, text in DOCUMENTS.items():
        upload(client, name, text)
    return client


def test_upload_reports_indexing(tmp_path: Path) -> None:
    body = (
        make_client(tmp_path)
        .post(DOCS, files={"file": ("a.txt", b"hello there")})
        .json()
    )

    assert body["index_status"] == "indexed"
    assert body["created_at"]


def test_library_lists_every_document(client: TestClient) -> None:
    documents = client.get(DOCS).json()

    assert sorted(d["filename"] for d in documents) == sorted(DOCUMENTS)
    assert all(d["status"] == "indexed" for d in documents)
    assert all(d["searchable"] for d in documents)
    assert all(d["chunk_count"] >= 1 for d in documents)
    assert all(d["embedding"]["model"] == "word-overlap" for d in documents)


def test_document_detail_includes_stored_chunks(client: TestClient) -> None:
    document = next(d for d in client.get(DOCS).json() if d["filename"] == "office.txt")

    detail = client.get(f"{DOCS}/{document['id']}").json()

    assert detail["filename"] == "office.txt"
    assert "Biscuit" in detail["chunks"][0]["text"]
    assert detail["chunks"][0]["chunk_index"] == 0


@pytest.mark.parametrize(("question", "expected"), QUESTIONS)
def test_known_questions_retrieve_the_expected_document(
    client: TestClient, question: str, expected: str
) -> None:
    body = client.post(SEARCH, json={"query": question, "top_k": 3}).json()

    assert body["hits"][0]["filename"] == expected
    assert body["hits"][0]["rank"] == 1
    assert body["embedding"]["model"] == "word-overlap"
    assert [h["rank"] for h in body["hits"]] == list(range(1, len(body["hits"]) + 1))


def test_delete_removes_the_document_everywhere(
    client: TestClient, tmp_path: Path
) -> None:
    document = next(d for d in client.get(DOCS).json() if d["filename"] == "office.txt")

    assert client.delete(f"{DOCS}/{document['id']}").status_code == 204

    assert client.get(f"{DOCS}/{document['id']}").status_code == 404
    assert "office.txt" not in [d["filename"] for d in client.get(DOCS).json()]
    hits = client.post(
        SEARCH, json={"query": "office cat Biscuit", "top_k": 10}
    ).json()["hits"]
    assert "office.txt" not in [h["filename"] for h in hits]
    assert not (tmp_path / "uploads" / f"{document['id']}.txt").exists()


def test_deleting_twice_is_not_found(client: TestClient) -> None:
    document_id = client.get(DOCS).json()[0]["id"]
    client.delete(f"{DOCS}/{document_id}")

    response = client.delete(f"{DOCS}/{document_id}")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "document_not_found"


def test_everything_survives_a_restart(client: TestClient, tmp_path: Path) -> None:
    restarted = make_client(tmp_path)

    assert len(restarted.get(DOCS).json()) == len(DOCUMENTS)
    question, expected = QUESTIONS[0]
    hits = restarted.post(SEARCH, json={"query": question}).json()["hits"]
    assert hits[0]["filename"] == expected


def test_failed_upload_is_listed_as_failed(tmp_path: Path) -> None:
    broken = make_client(
        tmp_path, FakeEmbeddingProvider(error=EmbeddingUnavailable("down"))
    )

    response = broken.post(DOCS, files={"file": ("a.txt", b"some text")})

    assert response.status_code == 503
    [document] = broken.get(DOCS).json()
    assert document["status"] == "failed"
    assert document["error_code"] == "embedding_unavailable"


def test_documents_from_another_model_are_flagged(
    client: TestClient, tmp_path: Path
) -> None:
    other_model = make_client(tmp_path, FakeEmbeddingProvider())

    assert not any(d["searchable"] for d in other_model.get(DOCS).json())
    assert other_model.post(SEARCH, json={"query": "revenue"}).json()["hits"] == []


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"query": "  "}, "invalid_request"),
        ({"query": "x", "top_k": 0}, "invalid_input"),
        ({"query": "x", "top_k": 21}, "invalid_input"),
        ({}, "invalid_request"),
    ],
)
def test_invalid_searches_are_rejected(
    client: TestClient, payload: dict[str, object], code: str
) -> None:
    response = client.post(SEARCH, json=payload)

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == code


def test_frontend_may_delete(client: TestClient) -> None:
    preflight = client.options(
        DOCS + "/x",
        headers={
            "Origin": "http://localhost:3008",
            "Access-Control-Request-Method": "DELETE",
        },
    )

    assert "DELETE" in preflight.headers["access-control-allow-methods"]
