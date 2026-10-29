"""POST /api/v1/chat end to end: real storage and search in a temp folder,
word-overlap embeddings, and a fake language model."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.domain.answers import AnswerSegment, CitedSpan
from app.domain.errors import AnswerRefused, LLMRateLimited, LLMUnavailable
from app.main import create_app
from tests.fakes import FakeLLMProvider, WordOverlapEmbeddingProvider
from tests.retrieval_dataset import DOCUMENTS

CHAT = "/api/v1/chat"


def make_client(
    tmp_path: Path, llm: FakeLLMProvider, upload: bool = True
) -> TestClient:
    client = TestClient(
        create_app(
            Settings(_env_file=None, data_dir=tmp_path),
            embedding_provider=WordOverlapEmbeddingProvider(),
            llm_provider=llm,
        )
    )
    if upload:
        for name, text in DOCUMENTS.items():
            assert (
                client.post(
                    "/api/v1/documents", files={"file": (name, text.encode())}
                ).status_code
                == 200
            )
    return client


def test_answers_with_citations_and_sources(tmp_path: Path) -> None:
    client = make_client(tmp_path, FakeLLMProvider())

    body = client.post(
        CHAT, json={"question": "What is the name of the office cat?"}
    ).json()

    assert body["insufficient_evidence"] is False
    assert body["answer"].endswith("[1]")
    assert body["model"] == "fake-llm"
    [citation] = body["citations"]
    assert citation["number"] == 1
    assert citation["filename"] == "office.txt"
    assert "Biscuit" in citation["quote"]
    assert citation["page_number"] is None
    assert len(body["sources"]) == len(DOCUMENTS)
    assert [s["cited_as"] for s in body["sources"]].count(1) == 1


def test_fabricated_citations_never_reach_the_client(tmp_path: Path) -> None:
    llm = FakeLLMProvider([AnswerSegment("Made up.", (CitedSpan(9, "invented text"),))])
    client = make_client(tmp_path, llm)

    body = client.post(CHAT, json={"question": "office cat"}).json()

    assert body["citations"] == []
    assert body["insufficient_evidence"] is True
    assert all(s["cited_as"] is None for s in body["sources"])


def test_empty_library_is_insufficient_evidence(tmp_path: Path) -> None:
    llm = FakeLLMProvider()
    client = make_client(tmp_path, llm, upload=False)

    body = client.post(CHAT, json={"question": "Anything?"}).json()

    assert body["insufficient_evidence"] is True
    assert body["sources"] == []
    assert llm.calls == []


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMUnavailable("down"), 503, "llm_unavailable"),
        (LLMRateLimited("busy", 29.5), 503, "llm_rate_limited"),
        (AnswerRefused("no"), 422, "answer_refused"),
    ],
)
def test_model_errors(tmp_path: Path, error: Exception, status: int, code: str) -> None:
    client = make_client(tmp_path, FakeLLMProvider(error=error))

    response = client.post(CHAT, json={"question": "office cat"})

    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    if code == "llm_rate_limited":
        assert response.headers["Retry-After"] == "30"


@pytest.mark.parametrize(
    "payload", [{}, {"question": ""}, {"question": "   "}, {"question": "x" * 2001}]
)
def test_invalid_questions(tmp_path: Path, payload: dict[str, str]) -> None:
    response = make_client(tmp_path, FakeLLMProvider(), upload=False).post(
        CHAT, json=payload
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_request"
