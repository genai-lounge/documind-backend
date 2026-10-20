import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app
from tests.conftest import make_pdf

client = TestClient(app)

URL = "/api/v1/documents"


def upload(filename: str, data: bytes):
    return client.post(URL, files={"file": (filename, data)})


def assert_error(response, status_code: int, code: str) -> None:
    assert response.status_code == status_code
    body = response.json()
    assert body["detail"]["code"] == code
    assert body["detail"]["message"]


# --- Success cases ---


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


# --- Textless documents ---


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


# --- Rejected uploads ---


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


def test_file_over_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "MAX_UPLOAD_SIZE_BYTES", 10)

    assert_error(upload("big.txt", b"x" * 11), 413, "file_too_large")


def test_file_exactly_at_limit_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "MAX_UPLOAD_SIZE_BYTES", 10)

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


# --- CORS ---


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
