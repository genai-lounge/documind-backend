"""Integration tests: the PDF adapter against the real PyMuPDF library."""

import pytest

from app.domain.errors import DocumentParseError
from app.infrastructure.parsing.pymupdf_parser import (
    NO_TEXT_IN_PDF_WARNING,
    PyMuPdfParser,
)
from tests.factories import make_pdf

parser = PyMuPdfParser()


def test_keeps_page_numbers_and_text(two_page_pdf: bytes) -> None:
    result = parser.parse(two_page_pdf)

    assert result.page_count == 2
    assert [s.page_number for s in result.sections] == [1, 2]
    assert "Alpha page one" in result.sections[0].text
    assert "Beta page two" in result.sections[1].text
    assert result.warnings == ()


def test_reads_title_and_author(two_page_pdf: bytes) -> None:
    result = parser.parse(two_page_pdf)

    assert result.metadata.title == "Sample"
    assert result.metadata.author == "Tester"


def test_missing_metadata_becomes_none() -> None:
    result = parser.parse(make_pdf(["text"]))

    assert result.metadata.title is None
    assert result.metadata.author is None


def test_blank_page_gives_empty_section_and_warning() -> None:
    result = parser.parse(make_pdf(["has text", ""]))

    assert result.sections[1].page_number == 2
    assert result.sections[1].text.strip() == ""
    assert result.warnings == (
        "Page 2 has no extractable text (it may be a scanned image or blank).",
    )


def test_pdf_without_any_text_warns_about_ocr() -> None:
    result = parser.parse(make_pdf(["", ""]))

    assert not result.has_text
    assert result.warnings == (NO_TEXT_IN_PDF_WARNING,)


def test_damaged_raises_parse_error() -> None:
    with pytest.raises(DocumentParseError, match="damaged"):
        parser.parse(b"%PDF-1.7 this is not really a pdf")


def test_password_protected_raises_parse_error() -> None:
    with pytest.raises(DocumentParseError, match="password"):
        parser.parse(make_pdf(["secret"], password="pw"))


@pytest.mark.parametrize(
    ("data", "expected"),
    [(b"%PDF-1.7 ...", True), (b"\n\n%PDF-1.4", True), (b"MZ executable", False)],
)
def test_matches_content(data: bytes, expected: bool) -> None:
    assert parser.matches_content(data) is expected
