from collections.abc import Iterator

import httpx2
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_ingest_document
from app.application.ingest_document import IngestDocument
from app.container import Container
from app.core.config import Settings
from app.main import create_app
from tests.factories import make_pdf
from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider

SETTINGS = Settings(_env_file=None)
EMBEDDER = FakeEmbeddingProvider(dimensions=16)
app = create_app(SETTINGS, embedding_provider=EMBEDDER, llm_provider=FakeLLMProvider())
client = TestClient(app)

URL = "/api/v1/documents"


def upload(filename: str, data: bytes) -> httpx2.Response:
    return client.post(URL, files={"file": (filename, data)})


def assert_error(response: httpx2.Response, status_code: int, code: str) -> None:
    assert response.status_code == status_code
    body = response.json()
    assert body["detail"]["code"] == code
    assert body["detail"]["message"]


def test_upload_pdf(two_page_pdf: bytes) -> None:
    response = upload("report.pdf", two_page_pdf)

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "report.pdf"
    assert body["file_type"] == "pdf"
    assert body["size_bytes"] == len(two_page_pdf)
    assert body["status"] == "extracted"
    assert body["page_count"] == 2
    assert body["metadata"] == {"title": "Sample", "author": "Tester"}
    assert [s["page_number"] for s in body["sections"]] == [1, 2]
    assert "Alpha page one" in body["sections"][0]["text"]
    assert body["sections"][0]["char_count"] == len(body["sections"][0]["text"])
    assert body["warnings"] == []
    assert len(body["id"]) == 32


def test_upload_docx(sample_docx: bytes) -> None:
    response = upload("notes.docx", sample_docx)

    assert response.status_code == 200
    body = response.json()
    assert body["file_type"] == "docx"
    assert body["page_count"] is None
    assert "Name | Score" in body["sections"][0]["text"]


def test_upload_txt() -> None:
    response = upload("notes.txt", b"plain text content")

    assert response.status_code == 200
    body = response.json()
    assert body["file_type"] == "txt"
    assert body["sections"] == [
        {"page_number": None, "text": "plain text content", "char_count": 18}
    ]


def test_extension_is_case_insensitive() -> None:
    assert upload("NOTES.TXT", b"hello").status_code == 200


def test_filename_path_is_stripped() -> None:
    response = upload("../../etc/notes.txt", b"hello")

    assert response.json()["filename"] == "notes.txt"


def test_each_upload_gets_a_new_id() -> None:
    first = upload("a.txt", b"x").json()["id"]
    second = upload("a.txt", b"x").json()["id"]

    assert first != second


def test_scanned_style_pdf_reports_no_text() -> None:
    response = upload("scan.pdf", make_pdf(["", ""]))

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_text"
    assert "OCR" in body["warnings"][0]


def test_pdf_with_some_blank_pages_warns_per_page() -> None:
    body = upload("mixed.pdf", make_pdf(["text", "", "more"])).json()

    assert body["status"] == "extracted"
    assert len(body["warnings"]) == 1
    assert "Page 2" in body["warnings"][0]


def test_whitespace_only_txt_reports_no_text() -> None:
    body = upload("blank.txt", b"   \n  ").json()

    assert body["status"] == "no_text"
    assert body["warnings"] == ["No text was found in this file."]


@pytest.mark.parametrize("filename", ["image.png", "archive.zip", "README", "doc.doc"])
def test_unsupported_extension(filename: str) -> None:
    assert_error(upload(filename, b"data"), 415, "unsupported_file_type")


def test_pdf_extension_with_other_content() -> None:
    assert_error(
        upload("fake.pdf", b"MZ\x90\x00 executable"), 415, "file_type_mismatch"
    )


def test_docx_extension_with_other_content() -> None:
    assert_error(upload("fake.docx", b"just text"), 415, "file_type_mismatch")


def test_empty_file() -> None:
    assert_error(upload("empty.txt", b""), 400, "empty_file")


@pytest.fixture
def ten_byte_limit() -> Iterator[None]:
    """Swap in a use case with a 10-byte limit, via dependency injection."""
    container = Container(
        SETTINGS, embedding_provider=EMBEDDER, llm_provider=FakeLLMProvider()
    )
    app.dependency_overrides[get_ingest_document] = lambda: IngestDocument(
        parsers=container.parser_registry,
        chunker=container.chunker,
        embedder=container.embedding_provider,
        files=container.files,
        documents=container.documents,
        vectors=container.vector_store,
        max_upload_bytes=10,
    )
    yield
    app.dependency_overrides.clear()


def test_file_over_limit(ten_byte_limit: None) -> None:
    assert_error(upload("big.txt", b"x" * 11), 413, "file_too_large")


def test_file_exactly_at_limit_is_accepted(ten_byte_limit: None) -> None:
    assert upload("ok.txt", b"x" * 10).status_code == 200


def test_damaged_pdf() -> None:
    assert_error(upload("broken.pdf", b"%PDF-1.7 garbage"), 422, "malformed_document")


def test_password_protected_pdf() -> None:
    response = upload("locked.pdf", make_pdf(["secret"], password="pw"))

    assert_error(response, 422, "malformed_document")
    assert "password" in response.json()["detail"]["message"]


def test_non_utf8_txt() -> None:
    assert_error(
        upload("latin.txt", "café".encode("latin-1")), 422, "malformed_document"
    )


def test_missing_file_field() -> None:
    assert_error(client.post(URL), 422, "invalid_request")


def test_cors_allows_frontend_origin() -> None:
    response = client.options(
        URL,
        headers={
            "Origin": "http://localhost:3008",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.headers["access-control-allow-origin"] == "http://localhost:3008"


def test_cors_rejects_unknown_origin() -> None:
    response = client.options(
        URL,
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert "access-control-allow-origin" not in response.headers


# --- Chunks and embeddings ---


def test_upload_returns_embedded_chunks(two_page_pdf: bytes) -> None:
    body = upload("report.pdf", two_page_pdf).json()

    assert body["embedding"] == {"model": "fake-embedder", "dimensions": 16}
    assert [c["chunk_index"] for c in body["chunks"]] == [0, 1]
    assert [c["page_number"] for c in body["chunks"]] == [1, 2]
    first = body["chunks"][0]
    page_text = body["sections"][0]["text"]
    assert first["text"] == page_text[first["start_char"] : first["end_char"]]
    assert first["char_count"] == len(first["text"])
    assert len(first["embedding_preview"]) == 8


def test_document_without_text_has_no_chunks() -> None:
    body = upload("scan.pdf", make_pdf(["", ""])).json()

    assert body["chunks"] == []
    assert body["embedding"] is None


def test_long_text_is_split_into_several_chunks() -> None:
    body = upload("long.txt", ("A sentence about revenue. " * 200).encode()).json()

    assert len(body["chunks"]) > 1
    assert all(c["char_count"] <= SETTINGS.chunk_size for c in body["chunks"])
