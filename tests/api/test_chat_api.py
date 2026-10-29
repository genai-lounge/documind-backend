"""POST /api/v1/chat end to end: real storage and search in a temp folder,
word-overlap embeddings, and a fake language model."""

import contextlib
import json
from pathlib import Path
from typing import Any

import anyio
import httpx2
import pytest
from fastapi.testclient import TestClient
from starlette.types import Message, Scope

from app.core.config import Settings
from app.domain.answers import AnswerSegment, ChatTurn, CitedSpan
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


# --- Streaming: POST /chat/stream ---

STREAM = f"{CHAT}/stream"


def read_events(response: httpx2.Response) -> list[tuple[str, Any]]:
    events = []
    for block in response.text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_streams_sources_text_citations_and_done(tmp_path: Path) -> None:
    client = make_client(tmp_path, FakeLLMProvider())

    response = client.post(
        STREAM, json={"question": "What is the name of the office cat?"}
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    events = read_events(response)
    assert [name for name, _ in events] == [
        "sources",
        "citation",
        "delta",
        "delta",
        "done",
    ]
    sources = events[0][1]
    assert len(sources) == len(DOCUMENTS)
    assert all(s["cited_as"] is None for s in sources)
    citation = events[1][1]
    assert citation["number"] == 1
    assert citation["filename"] == "office.txt"
    assert events[3][1] == {"text": "[1]"}
    assert events[-1][1] == {"insufficient_evidence": False, "model": "fake-llm"}


def test_streaming_answer_text_matches_the_whole_answer(tmp_path: Path) -> None:
    client = make_client(tmp_path, FakeLLMProvider())
    question = {"question": "What is the name of the office cat?"}

    streamed = "".join(
        data["text"]
        for name, data in read_events(client.post(STREAM, json=question))
        if name == "delta"
    )

    assert streamed.strip() == client.post(CHAT, json=question).json()["answer"]


def test_streaming_with_no_documents(tmp_path: Path) -> None:
    client = make_client(tmp_path, FakeLLMProvider(), upload=False)

    events = read_events(client.post(STREAM, json={"question": "Anything?"}))

    assert [name for name, _ in events] == ["sources", "delta", "done"]
    assert events[0][1] == []
    assert events[-1][1] == {"insufficient_evidence": True, "model": None}


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMUnavailable("down"), 503, "llm_unavailable"),
        (LLMRateLimited("busy", 29.5), 503, "llm_rate_limited"),
    ],
)
def test_errors_before_streaming_are_normal_http_errors(
    tmp_path: Path, error: Exception, status: int, code: str
) -> None:
    client = make_client(tmp_path, FakeLLMProvider(error=error))

    response = client.post(STREAM, json={"question": "office cat"})

    assert response.status_code == status
    assert response.json()["detail"]["code"] == code


def test_invalid_stream_requests_are_normal_http_errors(tmp_path: Path) -> None:
    response = make_client(tmp_path, FakeLLMProvider(), upload=False).post(
        STREAM, json={"question": " "}
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_request"


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (LLMUnavailable("The connection dropped."), "llm_unavailable"),
        (AnswerRefused("Declined."), "answer_refused"),
        (RuntimeError("bug with secret details"), "internal_error"),
    ],
)
def test_errors_after_streaming_started_become_an_error_event(
    tmp_path: Path, error: Exception, code: str
) -> None:
    llm = FakeLLMProvider(error_after=(2, error))
    client = make_client(tmp_path, llm)

    response = client.post(STREAM, json={"question": "office cat"})

    assert response.status_code == 200
    events = read_events(response)
    assert [name for name, _ in events] == ["sources", "citation", "delta", "error"]
    assert events[-1][1]["code"] == code
    assert "secret" not in response.text
    assert llm.streams[0].closed


# --- Conversation history ---


def test_history_is_passed_to_the_model(tmp_path: Path) -> None:
    llm = FakeLLMProvider()
    client = make_client(tmp_path, llm)
    history = [{"question": "What is the office cat called?", "answer": "Biscuit.[1]"}]

    response = client.post(
        STREAM, json={"question": "How old is it?", "history": history}
    )

    assert response.status_code == 200
    _, _, sent = llm.calls[0]
    assert sent == [ChatTurn("What is the office cat called?", "Biscuit.[1]")]


@pytest.mark.parametrize(
    "history",
    [
        [{"question": "", "answer": "a"}],
        [{"question": "q"}],
        [{"question": "q", "answer": "a" * 20001}],
        [{"question": "q", "answer": "a"}] * 11,
    ],
)
def test_invalid_history_is_rejected(
    tmp_path: Path, history: list[dict[str, str]]
) -> None:
    response = make_client(tmp_path, FakeLLMProvider(), upload=False).post(
        STREAM, json={"question": "q", "history": history}
    )

    assert response.status_code == 422


def test_a_client_disconnect_closes_the_model_stream_straight_away(
    tmp_path: Path,
) -> None:
    llm = FakeLLMProvider([AnswerSegment(f"Part {i}.") for i in range(50)])
    app = make_client(tmp_path, llm).app
    sent: list[Message] = []

    async def send(message: Message) -> None:
        if len(sent) == 3:
            raise OSError("client went away")
        sent.append(message)

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": STREAM,
        "raw_path": STREAM.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json"), (b"host", b"testserver")],
        "client": ("127.0.0.1", 1),
        "server": ("testserver", 80),
    }
    body = json.dumps({"question": "office cat"}).encode()
    requests: list[Message] = [
        {"type": "http.request", "body": body, "more_body": False}
    ]

    async def receive() -> Message:
        if requests:
            return requests.pop()
        await anyio.sleep_forever()
        return {"type": "http.disconnect"}

    async def run() -> None:
        with contextlib.suppress(Exception):
            await app(scope, receive, send)

    anyio.run(run)

    [stream] = llm.streams
    assert stream.closed
    assert stream.yielded < len(stream.events)
