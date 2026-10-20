"""Integration tests: the plain-text adapter."""

import pytest

from app.domain.errors import DocumentParseError
from app.infrastructure.parsing.plain_text_parser import PlainTextParser

parser = PlainTextParser()


def test_decodes_utf8() -> None:
    result = parser.parse("Héllo wörld".encode())

    assert result.sections[0].text == "Héllo wörld"
    assert result.sections[0].page_number is None


def test_strips_byte_order_mark() -> None:
    assert parser.parse(b"\xef\xbb\xbfwith BOM").sections[0].text == "with BOM"


def test_normalizes_windows_line_endings() -> None:
    assert parser.parse(b"one\r\ntwo").sections[0].text == "one\ntwo"


def test_rejects_non_utf8() -> None:
    with pytest.raises(DocumentParseError, match="UTF-8"):
        parser.parse("café".encode("latin-1"))


def test_rejects_binary_content() -> None:
    with pytest.raises(DocumentParseError, match="binary"):
        parser.parse(b"abc\x00def")
